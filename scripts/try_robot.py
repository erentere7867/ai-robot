"""Talk to the simulated robot from the command line and see every tool call it makes.

    python scripts/try_robot.py "what do you see?" "go behind that box"
    python scripts/try_robot.py --keep-memory "my name is Deniz" "what's my name?"
    python scripts/try_robot.py --physics "go to the red ball"

Each argument is one human turn. Uses a throwaway memory/map unless --keep-memory is given.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.core.config import RobotConfig  # noqa: E402
from robot.main import RobotHarness  # noqa: E402


def _fmt_pose(p) -> str:
    return f"x={p.x_mm:.0f} y={p.y_mm:.0f} heading={p.heading_deg:.0f}"


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("messages", nargs="+")
    parser.add_argument("--config", default="config/robot.yaml")
    parser.add_argument("--backend", default="llamacpp", choices=["auto", "llamacpp", "null"])
    parser.add_argument("--keep-memory", action="store_true", help="use the normal memory DB and map")
    parser.add_argument("--physics", action="store_true", help="MuJoCo dynamics body (slip, odometry drift)")
    args = parser.parse_args(argv)

    cfg = RobotConfig.from_yaml(args.config)
    cfg.llm.backend = args.backend
    cfg.record_events = False
    cfg.autonomy.enabled = False
    if args.physics:
        cfg.body_backend = "physics"
    if not args.keep_memory:
        tmp = Path(tempfile.mkdtemp(prefix="robot-try-"))
        cfg.database_path, cfg.spatial_map_path = tmp / "memory.db", tmp / "map.json"
    harness = RobotHarness(cfg)

    def on_call(event) -> None:
        p = event.payload
        calls = ", ".join(f"{c['name']}({c['args']})" for c in p.get("tool_calls", [])) or "-"
        text = (p.get("content") or "").replace("\n", " ")[:120]
        print(f"   model #{p['step']} {p['latency_ms'] / 1000:.1f}s ptok={p['prompt_tokens']}: {calls} {text!r}")

    def on_skill(event) -> None:
        p = event.payload
        print(f"      -> {p['tool']}: {'ok' if p['ok'] else 'FAILED'} - {p['message']}")
        for line in (p.get("data") or {}).get("objects", [])[:8]:
            print(f"         . {line}")

    harness.bus.subscribe("cognition.model_call", on_call, immediate=True)
    harness.bus.subscribe("skill.completed", on_skill, immediate=True)
    await harness.start()
    try:
        for text in args.messages:
            truth = getattr(harness.body, "true_pose", harness.body.pose)()
            print(f"\nHUMAN: {text}    [{_fmt_pose(truth)}]")
            t0 = time.perf_counter()
            reply = await harness.submit_message(text, timeout_s=600)
            await harness.arbiter.wait_until_done()
            truth = getattr(harness.body, "true_pose", harness.body.pose)()
            print(f"ROBOT: {reply.get('say', '')!r}  ({time.perf_counter() - t0:.1f}s, {reply.get('status')})"
                  f"  [{_fmt_pose(truth)}]")
        print("\n" + harness.spatial_map.render_for_prompt(harness.body.pose()))
        for obj in harness.sandbox.list_objects():
            pose = getattr(harness.body, "true_pose", harness.body.pose)()
            d = math.hypot(obj["x_mm"] - pose.x_mm, obj["y_mm"] - pose.y_mm)
            print(f"   truth: {obj['label']:<16} at ({obj['x_mm']:.0f}, {obj['y_mm']:.0f}), {d:.0f} mm away")
    finally:
        await harness.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
