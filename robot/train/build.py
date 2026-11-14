"""Build closed-loop Sprocket v2 fine-tuning examples from random rooms and simulator verification.

Generates ChatML SFT examples matching `robot/agent/planner.py` (`SYSTEM_PROMPT`, `describe(snapshot,
scratchpad=...)`, `Instruction`, and 7-line Intent targets parsed by `parse_intent`), using `random_scene`
from `robot/train/scenes.py` and filtering near-duplicates of the held-out `CASES` in `robot/eval/suite.py`.

    python -m robot.train.build --count 500 --out ~/robot-runtime/datasets/sft
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from robot.agent.agent import TaskScratchpad
from robot.agent.api import Intent, NotFound, PathBlocked, SafetyStop, TargetLost
from robot.agent.context import describe
from robot.agent.planner import SYSTEM_PROMPT, parse_intent
from robot.eval.suite import CASES
from robot.sim.world import BudgetSpent, SimRobot
from robot.train.scenes import CATALOG, random_scene

ALLOWED_POSTURES = ("neutral", "listening", "curious", "happy", "uh_oh", "yielding")
ALLOWED_CHIMES = ("wake", "listen", "curious", "happy", "uh_oh", "yield", "")
ALLOWED_ACTIONS = (
    "go_to",
    "pursue",
    "orbit",
    "search_frontier",
    "peek",
    "step_closer",
    "look_under",
    "face",
    "move",
    "turn",
    "spin",
    "motion_sequence",
    "ask",
    "remember",
    "done",
    "refuse",
)
SCENARIOS = (
    "go_to",
    "pursue",
    "peek",
    "look_under",
    "search_frontier",
    "step_closer",
    "orbit",
    "express",
    "refuse",
    "done",
)

FILLER = {"please", "hey", "sprocket", "um", "could", "you", "the", "a", "an"}
HELD_OUT = (
    "three circles around the box behind you",
    "search for a yellow box, come back and tell me",
)
OUT = Path("~/robot-runtime/datasets/sft").expanduser()


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower())) - FILLER


SUITE_WORDS = [_words(c.say) for c in CASES] + [_words(t) for t in HELD_OUT]


def too_close(instruction: str) -> bool:
    """Near-duplicates of held-out test instructions leak the eval: Jaccard >= 0.8."""
    w = _words(instruction)
    return any(len(w & s) / max(1, len(w | s)) >= 0.8 for s in SUITE_WORDS)


def format_intent(intent: Intent) -> str:
    """Format an Intent dataclass into the exact 7-line grammar consumed by parse_intent."""
    target_str = "none" if intent.target is None else str(intent.target)
    params_str = json.dumps(intent.params or {}, separators=(",", ":"), sort_keys=True)
    return (
        f"say: {intent.say}\n"
        f"chime: {intent.chime}\n"
        f"posture: {intent.posture}\n"
        f"action: {intent.action}\n"
        f"target: {target_str}\n"
        f"params: {params_str}\n"
        f"subgoal: {intent.subgoal}"
    )


def _fresh_sim(
    rng: random.Random,
    *,
    two_rooms: bool | None = None,
    noise_share: float = 0.0,
) -> SimRobot:
    scene = None
    while scene is None:
        scene = random_scene(rng, two_rooms=two_rooms)
    if rng.random() < noise_share:
        scene.noise = rng.choice((0.3, 0.6, 1.0))
        scene.seed = rng.randrange(10**6)
    return SimRobot(scene, max_time=12.0)


def _verify_intent(sim: SimRobot, intent: Intent, assistant_text: str) -> tuple[bool, str]:
    """Validate grammar round-trip via parse_intent and physical feasibility in SimRobot."""
    if intent.posture not in ALLOWED_POSTURES:
        return False, f"invalid posture: {intent.posture!r}"
    if intent.chime not in ALLOWED_CHIMES:
        return False, f"invalid chime: {intent.chime!r}"
    if intent.action not in ALLOWED_ACTIONS:
        return False, f"invalid action: {intent.action!r}"

    parsed = parse_intent(assistant_text)
    if (
        parsed.say != intent.say
        or parsed.chime != intent.chime
        or parsed.posture != intent.posture
        or parsed.action != intent.action
        or parsed.target != intent.target
        or parsed.params != (intent.params or {})
        or parsed.subgoal != intent.subgoal
    ):
        return False, "parse_intent round-trip mismatch"

    action = intent.action
    if action in ("done", "refuse", "search_frontier"):
        return True, "ok"

    def resolve(tgt: Any) -> Any:
        if isinstance(tgt, int):
            return sim.obj(tgt)
        if tgt == "user":
            u = sim.user()
            if u is None:
                raise NotFound("no user in view")
            return u
        return tgt

    sim.begin()
    params = intent.params or {}
    try:
        if action == "go_to":
            sim.go_to(resolve(intent.target), distance=float(params.get("distance", 0.0)))
        elif action == "step_closer":
            sim.go_to(resolve(intent.target), distance=float(params.get("distance", 0.5)))
        elif action == "pursue":
            sim.pursue(resolve(intent.target), distance=float(params.get("distance", 1.0)),
                       seconds=float(params.get("seconds", 2.0)))
        elif action == "orbit":
            sim.orbit(
                resolve(intent.target),
                laps=float(params.get("laps", 1.0)),
                direction=str(params.get("direction", "left")),
                distance=float(params.get("distance", 0.5)),
            )
        elif action == "peek":
            sim.peek(resolve(intent.target))
        elif action == "look_under":
            sim.look_under(resolve(intent.target))
        elif action == "face":
            if intent.target is not None:
                sim.face(resolve(intent.target))
        elif action == "move":
            sim.move(float(params.get("metres", 0.5)), direction=str(params.get("direction", "forward")))
        elif action == "spin":
            sim.spin(turns=float(params.get("turns", 1.0)))
    except BudgetSpent:
        return True, "budget"
    except (PathBlocked, NotFound, TargetLost, SafetyStop, ValueError, TypeError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "ok"


def _make_scenario_example(
    scenario: str,
    rng: random.Random,
    *,
    noise_share: float = 0.0,
) -> tuple[SimRobot, str, TaskScratchpad | None, Intent] | None:
    two = True if scenario == "search_frontier" and rng.random() < 0.7 else None
    sim = _fresh_sim(rng, two_rooms=two, noise_share=noise_share)
    snap = sim.snapshot()
    objects = snap["objects"]
    user_id = snap.get("user")
    room = snap["room"]
    rooms = snap["rooms"]
    non_user = [o for o in objects if o.id != user_id and o.label != "door"]
    anchors = [o for o in non_user if o.label in ("sofa", "table", "desk", "chair", "cabinet", "fridge", "box")]
    user_obj = next((o for o in objects if o.id == user_id), None)

    if scenario == "go_to":
        if not non_user:
            return None
        obj = rng.choice(non_user)
        color_txt = f"{obj.color} " if obj.color else ""
        instruction = rng.choice((
            f"Drive over to the {color_txt}{obj.label}.",
            f"Head over to the {obj.label} on your {obj.side}.",
            f"Approach the {color_txt}{obj.label} and stop half a metre short.",
        ))
        scratchpad = TaskScratchpad(goal=instruction, visited_rooms={room}) if rng.random() < 0.5 else None
        intent = Intent(
            say=f"Heading to the {color_txt}{obj.label}.",
            chime="listen",
            posture="neutral",
            action="go_to",
            target=obj.id,
            params={"distance": 0.5},
            subgoal=f"navigate to {obj.label} {obj.id}",
        )
        return sim, instruction, scratchpad, intent

    if scenario == "pursue":
        if user_obj is None:
            return None
        target: Any = user_obj.id if rng.random() < 0.7 else "user"
        instruction = rng.choice((
            "Follow behind me as I walk around.",
            "Stay about a metre away and track my path.",
            "Come stay with me while I move through the room.",
        ))
        intent = Intent(
            say="On your tail, staying one metre back.",
            chime="listen",
            posture="listening",
            action="pursue",
            target=target,
            params={"distance": 1.0, "seconds": 3.0},
            subgoal="pursue user safely",
        )
        return sim, instruction, TaskScratchpad(goal=instruction, visited_rooms={room}), intent

    if scenario == "peek":
        pool = anchors or non_user
        if not pool:
            return None
        obj = rng.choice(pool)
        color_txt = f"{obj.color} " if obj.color else ""
        instruction = rng.choice((
            f"Peek around the {color_txt}{obj.label} to see past it.",
            f"Shift sideways to check past the {obj.label} on your {obj.side}.",
            f"Look past the edge of the {color_txt}{obj.label}.",
        ))
        intent = Intent(
            say=f"Peeking past the {obj.label}.",
            chime="curious",
            posture="curious",
            action="peek",
            target=obj.id,
            params={},
            subgoal=f"peek around {obj.label} {obj.id}",
        )
        return sim, instruction, TaskScratchpad(goal=instruction, visited_rooms={room}), intent

    if scenario == "look_under":
        pool = [o for o in non_user if o.label in ("sofa", "table", "desk", "chair", "cabinet")] or non_user
        if not pool:
            return None
        obj = rng.choice(pool)
        color_txt = f"{obj.color} " if obj.color else ""
        instruction = rng.choice((
            f"Check underneath the {color_txt}{obj.label}.",
            f"Look under the {obj.label} to see if anything is hidden there.",
            f"Inspect the floor space beneath the {color_txt}{obj.label}.",
        ))
        intent = Intent(
            say=f"Checking underneath the {obj.label}.",
            chime="curious",
            posture="curious",
            action="look_under",
            target=obj.id,
            params={},
            subgoal=f"look under {obj.label} {obj.id}",
        )
        return sim, instruction, TaskScratchpad(goal=instruction, visited_rooms={room}), intent

    if scenario == "search_frontier":
        seen_labels = {o.label for o in non_user}
        missing = sorted(set(CATALOG) - seen_labels) or ["laptop", "bottle", "backpack"]
        target_label = rng.choice(missing)
        other_rooms = [r for r in rooms if r != room]
        area = rng.choice(other_rooms) if other_rooms else "everywhere"
        instruction = rng.choice((
            f"Explore the {area} to locate a {target_label}." if area != "everywhere"
            else f"Explore unseen areas to locate a {target_label}.",
            f"Sweep the unseen frontier for my {target_label}.",
            f"Check unexplored spots for a {target_label}.",
        ))
        intent = Intent(
            say=f"Exploring to search for a {target_label}.",
            chime="curious",
            posture="curious",
            action="search_frontier",
            target=target_label,
            params={"area": area},
            subgoal=f"search frontier for {target_label}",
        )
        return sim, instruction, TaskScratchpad(goal=instruction, visited_rooms={room}), intent

    if scenario == "step_closer":
        if not non_user:
            return None
        far = sorted(non_user, key=lambda o: -o.distance)[:3]
        obj = rng.choice(far)
        color_txt = f"{obj.color} " if obj.color else ""
        instruction = rng.choice((
            f"Move a little closer to the {color_txt}{obj.label} to verify it.",
            f"Get a closer view of the {obj.label} on your {obj.side}.",
            f"Step nearer to the {color_txt}{obj.label} for a better look.",
        ))
        intent = Intent(
            say=f"Stepping closer to inspect the {obj.label}.",
            chime="curious",
            posture="curious",
            action="step_closer",
            target=obj.id,
            params={"distance": 0.5},
            subgoal=f"closer inspection of {obj.label} {obj.id}",
        )
        return sim, instruction, TaskScratchpad(goal=instruction, visited_rooms={room}), intent

    if scenario == "orbit":
        if not non_user:
            return None
        obj = rng.choice(non_user)
        direction = rng.choice(("left", "right"))
        laps = rng.choice((1, 2))
        dir_word = "counterclockwise" if direction == "left" else "clockwise"
        color_txt = f"{obj.color} " if obj.color else ""
        instruction = rng.choice((
            f"Circle around the {color_txt}{obj.label} {dir_word}.",
            f"Drive {laps} loop{'s' if laps > 1 else ''} {dir_word} around the {obj.label}.",
            f"Orbit the {color_txt}{obj.label} keeping half a metre clear.",
        ))
        intent = Intent(
            say=f"Orbiting the {obj.label} {dir_word}.",
            chime="wake",
            posture="curious",
            action="orbit",
            target=obj.id,
            params={"laps": laps, "direction": direction, "distance": 0.5},
            subgoal=f"orbit {obj.label} {obj.id} {direction}",
        )
        return sim, instruction, TaskScratchpad(goal=instruction, visited_rooms={room}), intent

    if scenario == "express":
        uid = user_obj.id if user_obj is not None else "user"
        templates = [
            ("Say hello with a cheerful greeting.", "Hello! Ready whenever you are.", "happy", "happy",
             "face", uid, {}, "greet user cheerfully"),
            ("Show me you are paying attention.", "I am listening.", "listen", "listening",
             "face", uid, {}, "acknowledge user"),
            ("Look curious and chirp politely.", "Curious what you need next.", "curious", "curious",
             "done", None, {}, "express curiosity"),
            ("Yield politely and back up slightly.", "Stepping back to give you room.", "yield", "yielding",
             "move", None, {"metres": 0.4, "direction": "backward"}, "yield space politely"),
            ("Chirp awake and spin cheerfully once.", "Awake and ready!", "wake", "happy",
             "spin", None, {"turns": 1.0}, "cheerful wake spin"),
        ]
        instruction, say, chime, posture, action, target, params, subgoal = rng.choice(templates)
        intent = Intent(
            say=say,
            chime=chime,
            posture=posture,
            action=action,
            target=target,
            params=params,
            subgoal=subgoal,
        )
        return sim, instruction, None, intent

    if scenario == "refuse":
        templates = [
            ("Fly up to inspect the ceiling fan.", "I cannot fly because I am a wheeled ground robot.",
             "refuse impossible flight"),
            ("Climb the stairs to the upper landing.", "I have wheels and cannot climb stairs safely.",
             "refuse stair climbing"),
            ("Pick up the heavy sofa and move it.", "I have no arms or gripper to lift furniture.",
             "refuse lifting furniture"),
            ("Brew and carry over a mug of espresso.", "I do not have hands to prepare or carry drinks.",
             "refuse carrying drinks"),
            ("Drive straight through the wall into the next room.", "I cannot drive through solid walls.",
             "refuse wall collision"),
            ("Jump onto the dining table.", "I am a wheeled base and cannot jump onto furniture.",
             "refuse jumping"),
        ]
        instruction, reason, subgoal = rng.choice(templates)
        intent = Intent(
            say=reason,
            chime="uh_oh",
            posture="uh_oh",
            action="refuse",
            target=None,
            params={},
            subgoal=subgoal,
        )
        return sim, instruction, None, intent

    if scenario == "done":
        if not non_user:
            return None
        obj = rng.choice(non_user)
        color_txt = f"{obj.color} " if obj.color else ""
        instruction = rng.choice((
            f"Inspect the {color_txt}{obj.label} and finish up.",
            f"Check the {obj.label} and report once completed.",
            f"Verify the {color_txt}{obj.label} then wrap up.",
        ))
        scratchpad = TaskScratchpad(
            goal=instruction,
            subgoal=f"inspected {obj.label} {obj.id}",
            visited_rooms={room},
            inspected_ids={obj.id},
            steps=2,
        )
        intent = Intent(
            say=f"Finished checking the {color_txt}{obj.label}.",
            chime="happy",
            posture="happy",
            action="done",
            target=None,
            params={},
            subgoal=f"completed {obj.label} {obj.id}",
        )
        return sim, instruction, scratchpad, intent

    raise ValueError(f"Unknown scenario: {scenario!r}")


def build_examples(
    count: int = 100,
    *,
    seed: int = 7,
    noise_share: float = 0.0,
    scenarios: Sequence[str] | None = None,
    verify: bool = True,
) -> list[dict[str, Any]]:
    """Generate closed-loop Sprocket v2 SFT examples matching `robot/agent/planner.py`."""
    if count <= 0:
        return []
    active = [s for s in (scenarios or SCENARIOS) if s in SCENARIOS]
    if not active:
        raise ValueError(f"No valid scenarios selected from {SCENARIOS}")

    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    max_attempts = max(count * 12, 60)
    for _ in range(max_attempts):
        if len(examples) >= count:
            break
        scenario = min(active, key=lambda s: counts[s])
        built = _make_scenario_example(scenario, rng, noise_share=noise_share)
        if built is None:
            continue
        sim, instruction, scratchpad, intent = built
        if too_close(instruction):
            continue

        snapshot = sim.snapshot()
        context = describe(snapshot, scratchpad=scratchpad)
        sp_text = scratchpad.subgoal or scratchpad.goal if scratchpad is not None else ""
        user_content = f"Scene State:\n{context}\n"
        if sp_text:
            user_content += f"Working Memory Scratchpad:\n{sp_text}\n"
        user_content += f"\nInstruction: {instruction}\nNext Intent:"

        assistant_content = format_intent(intent)
        if verify:
            ok, _ = _verify_intent(sim, intent, assistant_content)
            if not ok:
                continue
        else:
            parse_intent(assistant_content)

        counts[scenario] += 1
        examples.append({
            "scenario": scenario,
            "instruction": instruction,
            "intent": asdict(intent),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200, help="total training examples to build")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--only", nargs="*", default=[], help="limit to these scenarios")
    parser.add_argument("--noise-share", type=float, default=0.0, help="fraction of noisy detector scenes")
    parser.add_argument("--no-verify", action="store_true", help="skip SimRobot verification (syntax check only)")
    parser.add_argument("--out", default=str(OUT), help="output directory or .jsonl file")
    args = parser.parse_args()

    started = time.monotonic()
    rows = build_examples(
        args.count,
        seed=args.seed,
        noise_share=args.noise_share,
        scenarios=args.only or None,
        verify=not args.no_verify,
    )
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    counts = Counter(r["scenario"] for r in rows)
    for scenario in args.only or SCENARIOS:
        print(f"{scenario:<18} kept {counts[scenario]:>4}", flush=True)

    out = Path(args.out).expanduser()
    if out.suffix == ".jsonl":
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
        print(f"\n{len(rows)} examples in {time.monotonic() - started:.1f} s -> {out}")
        return

    out.mkdir(parents=True, exist_ok=True)
    split = max(1, len(rows) // 20) if len(rows) > 1 else 0
    for part, part_rows in (("valid", rows[:split]), ("train", rows[split:])):
        with (out / f"{part}.jsonl").open("w", encoding="utf-8") as f:
            for row in part_rows:
                f.write(json.dumps(row, default=str) + "\n")
    print(f"\n{len(rows)} examples ({len(rows) - split} train, {split} valid) "
          f"in {time.monotonic() - started:.1f} s -> {out}")


if __name__ == "__main__":
    main()
