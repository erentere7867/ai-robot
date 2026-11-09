"""Does lidar localization keep the pose honest? Drive loops on the physics body (wheel slip, noisy encoders
and gyro) and compare the odometry pose and the corrected pose with the simulator's ground truth.

    python scripts/bench_localization.py --seconds 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.control.behaviors import GoToPoint  # noqa: E402
from robot.core.angles import wrap_deg  # noqa: E402
from robot.core.config import RobotConfig  # noqa: E402
from robot.main import RobotHarness  # noqa: E402


def err(a, b) -> tuple[float, float]:
    d = math.hypot(a.x_mm - b.x_mm, a.y_mm - b.y_mm)
    h = abs(wrap_deg(a.heading_deg - b.heading_deg))
    return d, h


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out", default="artifacts/bench-localization.json")
    args = parser.parse_args()
    cfg = RobotConfig.from_yaml("config/robot.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        cfg.database_path, cfg.spatial_map_path = Path(tmp) / "m.db", Path(tmp) / "map.json"
        cfg.record_events, cfg.autonomy.enabled, cfg.llm.backend = False, False, "null"
        cfg.body_backend = "physics"
        cfg.perception.lidar, cfg.perception.localization = "sim", True
        h = RobotHarness(cfg)
        await h.start()
        try:
            true = h.body.true_pose
            await asyncio.sleep(1.5)  # a few scans of the room from the start pose
            waypoints = [(700.0, 0.0), (700.0, 700.0), (-700.0, 700.0), (-700.0, -700.0), (700.0, -700.0)]
            samples, k = [], 0
            loop = asyncio.get_running_loop()
            end = loop.time() + args.seconds
            while loop.time() < end:
                goal = waypoints[k % len(waypoints)]
                k += 1
                done = await h.fast.run(GoToPoint(goal, "a waypoint"))
                t = true()
                samples.append({"leg": k, "ok": done.ok, "odom": err(h.body.odometry_pose(), t), "corrected": err(h.body.pose(), t)})
            last = samples[-1]
            summary = {
                "legs": len(samples),
                "odometry_error_mm_deg": [round(v, 1) for v in last["odom"]],
                "corrected_error_mm_deg": [round(v, 1) for v in last["corrected"]],
                "max_odometry_error_mm": round(max(s["odom"][0] for s in samples), 1),
                "max_corrected_error_mm": round(max(s["corrected"][0] for s in samples), 1),
                "localizer": h.localizer.snapshot(),
            }
            print(json.dumps(summary))
            Path(args.out).write_text(json.dumps({"summary": summary, "samples": samples}, indent=1), encoding="utf-8")
        finally:
            await h.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
