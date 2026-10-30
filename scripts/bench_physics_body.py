"""Measure PhysicsBody cost per simulated second, with and without the bound sandbox."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from robot.body.hal import BodyCommand, BodyPose
from robot.body.physics_body import PhysicsBody
from robot.core.bus import EventBus
from robot.core.clock import ReplayClock
from robot.world.sandbox import SimulationSandbox

ROUTE = (
    ("spin", {"angle_deg": -90}),
    ("drive", {"distance_mm": 600}),
    ("spin", {"angle_deg": 180}),
    ("drive", {"distance_mm": 400, "speed_mm_s": 400}),
    ("perform_trick", {"trick": "dance"}),
)


async def measure(with_sandbox: bool, laps: int) -> dict[str, float]:
    bus = EventBus(ReplayClock())
    body = PhysicsBody(bus)
    if with_sandbox:
        SimulationSandbox(bus).bind(body)
    await bus.start()
    await body.start()
    started = time.perf_counter()
    for lap in range(laps):
        body.reset_pose(BodyPose())
        for i, (tool, args) in enumerate(ROUTE):
            await body.execute(BodyCommand(f"{lap}-{i}", tool, args))
    wall_s = time.perf_counter() - started
    sim_s = bus.clock.monotonic()
    await body.stop()
    await bus.stop()
    return {
        "sandbox": with_sandbox,
        "simulated_s": round(sim_s, 2),
        "wall_s": round(wall_s, 3),
        "real_time_factor": round(sim_s / wall_s, 1),
        "physics_ms_per_sim_s": round(body.physics_ms_per_sim_s(), 2),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laps", type=int, default=5)
    args = parser.parse_args()
    for with_sandbox in (False, True):
        print(json.dumps(await measure(with_sandbox, args.laps)))


if __name__ == "__main__":
    asyncio.run(main())
