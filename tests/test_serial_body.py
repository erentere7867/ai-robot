"""SerialBody against a simulated motor microcontroller speaking the firmware's line protocol."""

from __future__ import annotations

import asyncio
import math
import threading
import time

import pytest

from robot.body.hal import BodyCommand
from robot.body.serial_body import SerialBody
from robot.core.bus import EventBus
from robot.core.clock import RealClock


class FakeMCU:
    """What firmware/base_controller does: speed loops, watchdog, reflexes, 50 Hz telemetry."""

    TICKS_PER_MM = 720 / (math.pi * 120.0)

    def __init__(self, wall_x_mm: float = 10_000.0) -> None:
        self.x = self.y = self.heading = 0.0
        self.v = self.w = 0.0
        self.target = (0.0, 0.0)
        self.last_cmd = time.monotonic()
        self.ticks = [0.0, 0.0]
        self.flags = 0
        self.estop = False
        self.wall_x = wall_x_mm
        self.lines: list[str] = []
        self._t = time.monotonic()
        self._lock = threading.Lock()

    def write(self, data: bytes) -> None:
        with self._lock:
            for line in data.decode().splitlines():
                self.lines.append(line)
                parts = line.split()
                if parts[0] == "V":
                    self.target = (float(parts[1]), float(parts[2]) / 1000.0)
                    self.last_cmd = time.monotonic()
                elif parts[0] == "S":
                    self.target = (0.0, 0.0)
                elif parts[0] == "E":
                    self.estop = parts[1] == "1"

    def tof(self) -> float:
        if abs(self.heading) > 30:
            return 2000.0
        return max(0.0, (self.wall_x - self.x) / math.cos(math.radians(self.heading)) - 141.0)

    def readline(self) -> bytes:
        time.sleep(0.02)
        with self._lock:
            now = time.monotonic()
            dt, self._t = now - self._t, now
            v, w = self.target
            if now - self.last_cmd > 0.3 or self.estop:  # watchdog
                v = w = 0.0
            blocked = v > 0 and self.tof() <= 120.0
            if blocked:
                v = 0.0
            self.v += max(-900 * dt, min(900 * dt, v - self.v))
            self.w += max(-600 * dt, min(600 * dt, w - self.w))
            self.heading = (self.heading + self.w * dt + 180) % 360 - 180
            ds = self.v * dt
            self.x += ds * math.cos(math.radians(self.heading))
            self.y += ds * math.sin(math.radians(self.heading))
            arc = math.radians(self.w * dt) * 104.0 * 2.4  # skid steer: wheels slip when turning
            self.ticks[0] += (ds - arc) * self.TICKS_PER_MM
            self.ticks[1] += (ds + arc) * self.TICKS_PER_MM
            flags = (32 if blocked else 0) | (4 if self.estop else 0)
            return (f"T {int(now * 1000)} {int(self.ticks[0])} {int(self.ticks[1])} {int(self.w * 1000)} "
                    f"{int(self.tof())} {flags} 8000\n").encode()


@pytest.fixture
async def rig():
    bus = EventBus(RealClock())
    mcu = FakeMCU(wall_x_mm=900.0)
    body = SerialBody(bus, transport=mcu)
    reflexes: list[str] = []
    bus.subscribe("reflex.triggered", lambda e: reflexes.append(e.payload["reflex"]), immediate=True)
    await bus.start()
    await body.start()
    await asyncio.sleep(0.1)
    try:
        yield mcu, body, reflexes
    finally:
        await body.stop()
        await bus.stop()


async def test_odometry_follows_the_wheels_and_gyro(rig):
    mcu, body, _ = rig
    result = await body.execute(BodyCommand("t", "spin", {"angle_deg": 90}))
    assert result.ok, result.message
    await asyncio.sleep(0.2)
    assert abs(body.pose().heading_deg - mcu.heading) < 2.0 and abs(body.pose().heading_deg - 90) < 4.0
    result = await body.execute(BodyCommand("d", "drive", {"direction": "forward", "distance_mm": 300}))
    assert result.ok, result.message
    await asyncio.sleep(0.2)
    assert math.hypot(body.pose().x_mm - mcu.x, body.pose().y_mm - mcu.y) < 25.0
    assert abs(body.pose().y_mm - 300) < 30.0


async def test_speed_commands_and_the_mcu_watchdog(rig):
    mcu, body, _ = rig
    for _ in range(20):
        await body.set_velocity(150.0, 0.0)
        await asyncio.sleep(0.05)
    assert mcu.x > 60
    x = mcu.x
    await asyncio.sleep(0.8)  # host stops sending: the MCU must stop the wheels by itself
    assert mcu.x - x < 80 and abs(mcu.v) < 1.0


async def test_forward_block_flag_becomes_an_obstacle_reflex(rig):
    mcu, body, reflexes = rig
    result = await body.execute(BodyCommand("d", "drive", {"direction": "forward", "distance_mm": 1500}))
    assert not result.ok and result.status == "reflex_stopped"
    assert "obstacle_stop" in reflexes and mcu.x < 900 - 141 - 100


async def test_cancel_and_estop_stop_motion(rig):
    mcu, body, _ = rig
    task = asyncio.ensure_future(body.execute(BodyCommand("s", "spin", {"angle_deg": 360})))
    await asyncio.sleep(0.4)
    await body.cancel()
    result = await task
    assert result.status == "cancelled" and mcu.lines[-1] == "S"
    body.set_estop(True)
    assert mcu.lines[-1] == "E 1"
