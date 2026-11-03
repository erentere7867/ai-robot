"""Control-loop timing under load: does steering stay on time while the rest of the robot is busy?

Runs the simulated robot (null model backend, real clock, the trained detector streaming, simulated
lidar) through a continuous behaviour, while a "cognition" task on the main event loop does what a real
turn does: bursts of CPU-bound Python (parsing a model reply, building a prompt, encoding a dashboard
frame). Reports how late control ticks started and how many deadlines were missed, with the control loop
in the main event loop (the old way) and in its own real-time thread.

    python scripts/bench_control.py                    # both modes, 50 Hz, 20 s each
    python scripts/bench_control.py --hz 100 --seconds 30 --stall-ms 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.control.behaviors import Orbit  # noqa: E402
from robot.core.config import RobotConfig  # noqa: E402
from robot.main import RobotHarness  # noqa: E402


def burn(ms: float) -> None:
    """CPU-bound Python holding the GIL, like json.loads of a big reply or a prompt rebuild."""
    end = time.perf_counter() + ms / 1000.0
    payload = {"k": list(range(200))}
    while time.perf_counter() < end:
        json.loads(json.dumps(payload))


async def cognition_load(stall_ms: float, every_s: float, stop: asyncio.Event) -> int:
    n = 0
    while not stop.is_set():
        burn(stall_ms)  # blocks the main event loop, as synchronous work in a turn does
        n += 1
        await asyncio.sleep(every_s)
    return n


async def run(threaded: bool, tune: bool, hz: float, seconds: float, stall_ms: float, every_s: float) -> dict:
    cfg = RobotConfig.from_yaml("config/robot.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        cfg.database_path = Path(tmp) / "memory.db"
        cfg.spatial_map_path = Path(tmp) / "map.json"
        cfg.record_events = False
        cfg.autonomy.enabled = False
        cfg.perception.loop_hz = hz
        cfg.perception.control_thread = threaded
        cfg.perception.lidar = "sim"
        cfg.perception.tune_process = tune
        cfg.llm.backend = "null"
        harness = RobotHarness(cfg)
        await harness.start()
        try:
            fast = harness.fast
            await asyncio.sleep(1.0)  # detector and loop warm up
            fast.stats.__init__()
            stop = asyncio.Event()
            load = asyncio.create_task(cognition_load(stall_ms, every_s, stop))
            pose = harness.body.pose()
            orbit = Orbit((pose.x_mm + 500.0, pose.y_mm), 500.0, 50.0, 3.14159, "a point", False)
            fast.start_behavior(orbit)
            await asyncio.sleep(seconds)
            stop.set()
            bursts = await load
            fast.cancel_behavior(reason="benchmark over")
            timing = fast.stats.summary()
            return {"mode": fast.realtime + (", tuned" if tune else ""), "hz": hz, "ticks_per_s": round(timing["ticks"] / seconds, 1), "stall_ms": stall_ms, "stalls": bursts,
                    "detect_hz": round(fast.detect_hz, 1), **timing}
        finally:
            await harness.stop()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--stall-ms", type=float, default=30.0, help="length of each burst of main-loop work")
    parser.add_argument("--every", type=float, default=0.15, help="seconds between bursts")
    parser.add_argument("--out", default="artifacts/bench-control.json")
    args = parser.parse_args()
    results = []
    for threaded, tune in ((False, False), (True, False), (True, True)):
        r = await run(threaded, tune, args.hz, args.seconds, args.stall_ms, args.every)
        results.append(r)
        print(json.dumps(r))
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
