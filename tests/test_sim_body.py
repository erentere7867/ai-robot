"""Kinematic body in the sandbox: bumper reflex, range reflex and battery."""

from __future__ import annotations

import pytest

from robot.body.hal import BodyCommand
from robot.body.sim_body import SimBody
from robot.core.bus import EventBus
from robot.core.clock import ReplayClock
from robot.world.sandbox import SimulationSandbox
from robot_fakes import clear_scene


@pytest.fixture
async def rig():
    clock = ReplayClock()
    bus = EventBus(clock)
    body = SimBody(bus)
    sandbox = SimulationSandbox(bus)
    sandbox.bind(body)
    clear_scene(sandbox, keep=("cardboard_box", "charging_dock"))
    reflexes: list[str] = []
    bus.subscribe("reflex.triggered", lambda e: reflexes.append(e.payload["reflex"]), immediate=True)
    await bus.start()
    await body.start()
    try:
        yield clock, body, sandbox, reflexes
    finally:
        await body.stop()
        sandbox.unbind()
        await bus.stop()


def drive(distance_mm: float, direction: str = "forward", cid: str = "c") -> BodyCommand:
    return BodyCommand(cid, "drive", {"direction": direction, "distance_mm": distance_mm})


async def test_range_reflex_stops_before_a_box(rig):
    _, body, sandbox, reflexes = rig
    sandbox.move_object("cardboard_box", 600.0, 0.0)
    result = await body.execute(drive(1000))
    assert result.status == "reflex_stopped" and reflexes == ["obstacle_stop"]
    assert body.pose().x_mm < 600 - 110 - 140  # stopped before touching


async def test_bumper_stops_driving_into_a_box_when_the_range_sensor_misses_it(rig):
    _, body, sandbox, reflexes = rig
    sandbox.move_object("cardboard_box", 600.0, 0.0)
    body.set_simulated_obstacle(front_tof_mm=2000.0)
    result = await body.execute(drive(1000))
    assert not result.ok and result.status == "reflex_stopped"
    assert result.message == "Bumper pressed: I bumped into something."
    assert reflexes == ["bumper"]
    contact = 600 - 110 - 140
    assert contact - 15 <= body.pose().x_mm <= contact
    assert body.pose().linear_vel_mm_s == 0
    backed = await body.execute(drive(100, "backward", "c2"))
    assert backed.ok  # moving away from the contact is allowed


async def test_bumper_stops_backing_into_a_wall(rig):
    _, body, _, reflexes = rig
    result = await body.execute(drive(2000, "backward"))
    assert result.status == "reflex_stopped" and reflexes == ["bumper"]
    wall = -(1500 - 20 - 140)
    assert wall <= body.pose().x_mm <= wall + 15


async def test_dock_is_a_ramp_not_an_obstacle(rig):
    _, body, _, reflexes = rig
    await body.execute(BodyCommand("t", "spin", {"angle_deg": 180}))
    result = await body.execute(drive(650))
    assert result.ok and reflexes == []


async def test_battery_drains_with_distance_and_idle_time(rig):
    clock, body, _, _ = rig
    assert body.sensors().battery_pct == pytest.approx(100.0)
    await body.execute(BodyCommand("t", "spin", {"angle_deg": 90}))  # face open floor (+y)
    level = body.sensors().battery_pct
    started = clock.monotonic()
    await body.execute(drive(1000))
    elapsed_min = (clock.monotonic() - started) / 60.0
    assert body.sensors().battery_pct == pytest.approx(level - 1.0 - 0.5 * elapsed_min, abs=1e-6)
    clock.advance(120.0)
    assert body.sensors().battery_pct == pytest.approx(level - 1.0 - 0.5 * elapsed_min - 1.0, abs=1e-6)
    assert body.sensors().charging is False


async def test_battery_charges_on_the_dock(rig):
    clock, body, sandbox, _ = rig
    body._sensors.battery_pct = 40.0
    await body.execute(BodyCommand("t", "spin", {"angle_deg": 180}))
    await body.execute(drive(680))
    assert sandbox.is_charging(body.pose())
    readings = body.sensors()
    assert readings.charging is True
    level = readings.battery_pct
    clock.advance(60.0)
    assert body.sensors().battery_pct == pytest.approx(level + 20.0)
    clock.advance(3600.0)
    assert body.sensors().battery_pct == 100.0
    await body.execute(drive(400, cid="off"))  # facing away from the dock now: drive off it
    assert body.sensors().charging is False
