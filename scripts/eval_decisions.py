"""Fast decision benchmark: does the model pick the right *first* tool for a situation?

One model call per case (no simulation), so prompt and tool-description changes can be measured
in about a minute. Complements scripts/eval_scenarios.py, which runs whole tasks in the simulator.

    python scripts/eval_decisions.py            # all cases, prints a table and the pass rate
    python scripts/eval_decisions.py -k behind  # only cases whose name contains "behind"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.body.hal import BodyPose  # noqa: E402
from robot.cognition.agent import recover_text_tool_calls  # noqa: E402
from robot.cognition.llamacpp import LlamaCppBackend  # noqa: E402
from robot.cognition.prompts import PromptBuilder  # noqa: E402
from robot.core.clock import RealClock  # noqa: E402
from robot.core.config import RobotConfig  # noqa: E402
from robot.face.animator import EYE_COLOR_THEMES  # noqa: E402
from robot.skills import build_default_registry  # noqa: E402
from robot.world.spatial_map import SpatialMap, TrackedObject  # noqa: E402

SCENE = [("box_1", "brown cardboard box", 950, 420, 120), ("ball_1", "red ball", 680, 160, 48),
         ("cube_1", "blue cube", 620, -210, 60), ("cylinder_1", "green cylinder", 840, -30, 43)]
DUCK_BEHIND = ("duck_1", "yellow rubber duck", -700, 300, 45)

Check = Callable[[list[tuple[str, dict[str, Any]]], str], bool]


def first(name: str, **want: Any) -> Check:
    def check(calls, _text):
        if not calls or calls[0][0] != name:
            return False
        args = calls[0][1]
        return all(str(args.get(k, "")).lower() == str(v).lower() for k, v in want.items())
    return check


def any_of(*checks: Check) -> Check:
    return lambda calls, text: any(c(calls, text) for c in checks)


def no_tool(calls, text) -> bool:
    return not calls and bool(text.strip())


@dataclass
class Case:
    name: str
    request: str
    check: Check
    objects: list[tuple] = field(default_factory=list)
    pose: BodyPose = field(default_factory=BodyPose)
    background: bool = False
    core: list[str] = field(default_factory=list)
    after_observe: bool = False  # the model already observed this turn; it should now answer


CASES = [
    Case("see/empty-map", "what do you see?", first("observe")),
    Case("see/known", "what do you see?", any_of(first("observe"), no_tool), SCENE),  # fresh map is fine
    Case("see/after-observe", "what do you see?", no_tool, SCENE, after_observe=True),
    Case("behind/box", "go behind that box", first("go_to", target="box_1", relation="behind"), SCENE),
    Case("go/ball", "go to the red ball", first("go_to", target="ball_1"), SCENE),
    Case("between", "go between the ball and the cube", first("go_to", relation="between"), SCENE),
    Case("start/return", "now come back to where you started", first("go_to", target="start"), SCENE,
         BodyPose(1100, 700, -120)),
    Case("turn/around", "turn around", first("turn", direction="around"), SCENE),
    Case("turn/left", "look to your left", first("turn", direction="left"), SCENE),
    Case("behind-you", "what's behind you?", any_of(first("turn", direction="around"), first("look_around")),
         SCENE),
    Case("find/unknown", "find the yellow duck", first("look_around"), SCENE),
    Case("find/known-behind", "find the yellow duck",
         any_of(first("turn_to", target="duck_1"), first("go_to", target="duck_1"), first("look_around")),
         [*SCENE, DUCK_BEHIND]),
    Case("look-at", "look at the ball", any_of(first("turn_to", target="ball_1"), first("turn", direction="right")),
         SCENE, BodyPose(0, 0, 90)),
    Case("move/little", "move forward a little", first("move", direction="forward"), SCENE),
    Case("greet", "hi!", no_tool),
    Case("chat/feel", "how are you feeling?", no_tool),
    Case("remember/name", "my name is Deniz", first("remember")),
    Case("recall/name", "what's my name?", any_of(no_tool, first("recall")), core=["The user's name is Deniz."]),
    Case("dance", "do a little dance", first("express", gesture="dance")),
    Case("eyes", "make your eyes green", first("set_eye_color", color="green")),
    Case("bg/curious", "You feel curious (curiosity 0.8). About 70% of the area around you is still unexplored.",
         any_of(first("explore"), first("look_around"), first("observe")), SCENE, background=True),
]


def build_messages(case: Case, prompts: PromptBuilder, image_url: str) -> list[dict[str, Any]]:
    clock = RealClock()
    smap = SpatialMap(clock)
    now = clock.monotonic()
    for oid, label, x, y, r in case.objects:
        smap.objects[oid] = TrackedObject(oid, label, x, y, r, now - 5, now - 5, 3, 0.9)
    smap.places["start"] = (0.0, 0.0)
    situation = prompts.situation(world=smap.render_for_prompt(case.pose), core_facts=case.core, memories=[],
                                  recent=[], battery_pct=86.0)
    messages = prompts.build(case.request, situation, background=case.background)
    if case.after_observe:
        objs = [smap.describe(o, case.pose).replace(", seen 5s ago", "") for o in smap.objects.values()]
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "observe", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(
                {"ok": True, "result": "I can see 4 thing(s). If that answers the request, reply now.",
                 "objects": objs})},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}},
                                         {"type": "text", "text": "(camera view after observe)"}]},
        ]
    return messages


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", default="", help="only run cases whose name contains this")
    parser.add_argument("--config", default="config/robot.yaml")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args(argv)
    cfg = RobotConfig.from_yaml(args.config)
    backend = LlamaCppBackend(cfg.llm)
    if not await backend.is_available():
        print(f"llama-server is not answering at {cfg.llm.base_url}")
        return 2
    prompts = PromptBuilder(cfg.soul_path, cfg.context_budget, cfg.name)
    tools = build_default_registry(sorted(EYE_COLOR_THEMES)).tool_schemas()
    names = {t["function"]["name"] for t in tools}
    image = Path(__file__).resolve().parents[1] / "tests" / "data" / "scene_front.png"
    image_url = ("data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()) if image.exists() else ""
    passed = total = 0
    try:
        for case in [c for c in CASES if args.k in c.name]:
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                result = await backend.chat(build_messages(case, prompts, image_url), tools=tools)
                calls = [(c.name, c.arguments) for c in result.tool_calls]
                if not calls:
                    calls = [(c.name, c.arguments) for c in recover_text_tool_calls(result.content, names)]
                ok = case.check(calls, result.content if not calls else "")
                passed += ok
                total += 1
                shown = ", ".join(f"{n}({json.dumps(a)})" for n, a in calls) or repr(result.content[:70])
                print(f"{'PASS' if ok else 'FAIL'}  {case.name:<20} {time.perf_counter() - t0:4.1f}s  {shown}")
    finally:
        await backend.close()
    print(f"\n{passed}/{total} passed ({100 * passed / max(1, total):.0f}%)")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
