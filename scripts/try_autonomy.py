"""Leave the simulated pet alone and watch what it decides to do by itself.

    python scripts/try_autonomy.py --seconds 180 --speed 20

`--speed` multiplies how fast drives (curiosity, social, play) build up, so a demo does not take an hour.
Prints every background decision, tool call and anything the robot says.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.core.config import RobotConfig  # noqa: E402
from robot.main import RobotHarness  # noqa: E402


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--speed", type=float, default=20.0, help="drive build-up multiplier")
    parser.add_argument("--config", default="config/robot.yaml")
    args = parser.parse_args(argv)

    cfg = RobotConfig.from_yaml(args.config)
    cfg.record_events = False
    cfg.autonomy.enabled = True
    cfg.autonomy.idle_after_s = 3.0
    cfg.autonomy.think_interval_s = 10.0
    cfg.autonomy.drive_rate_scale = args.speed
    tmp = Path(tempfile.mkdtemp(prefix="robot-autonomy-"))
    cfg.database_path, cfg.spatial_map_path = tmp / "memory.db", tmp / "map.json"
    harness = RobotHarness(cfg)
    t0 = time.monotonic()

    def stamp() -> str:
        return f"[{time.monotonic() - t0:6.1f}s]"

    def on_skill(event) -> None:
        p = event.payload
        print(f"{stamp()}   {p['tool']}({p['args']}) -> {'ok' if p['ok'] else 'FAILED'}: {p['message'][:110]}")

    def on_response(event) -> None:
        p = event.payload
        if p.get("origin") == "autonomous":
            print(f"{stamp()} decided: {p.get('status')}; says: {p.get('say')!r}")

    harness.bus.subscribe("skill.completed", on_skill, immediate=True)
    harness.bus.subscribe("robot.response", on_response, immediate=True)
    await harness.start()
    last_trigger = None
    try:
        while time.monotonic() - t0 < args.seconds:
            await asyncio.sleep(0.5)
            snap = harness.autonomy.snapshot()
            if snap["last_trigger"] != last_trigger and snap["last_trigger"]:
                last_trigger = snap["last_trigger"]
                levels = {k: v["level"] for k, v in snap["drives"].items()}
                print(f"{stamp()} woke up because: {last_trigger}  drives={levels}")
        print("\n" + harness.spatial_map.render_for_prompt(harness.body.pose()))
        print(f"coverage of the area around me: {1 - harness.spatial_map.unseen_fraction_near(0, 0, 1500):.0%}")
    finally:
        await harness.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
