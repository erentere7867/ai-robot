"""The robot's bodies: the kinematic and MuJoCo simulators, and the VENTUNO Q link (with a fake MCU)."""

from __future__ import annotations

import asyncio
import math
import re
import time

import pytest
from robot_fakes import clear_scene

from robot.body.hal import BodyPose, load_capabilities
from robot.core.bus import EventBus
from robot.core.clock import RealClock, ReplayClock
from robot.core.config import PROJECT_ROOT
from robot.hardware.ventuno_body import VentunoBody
from robot.sim.physics_body import PhysicsBody, SensorNoise
from robot.sim.sandbox import SimulationSandbox
from robot.sim.sim_body import SimBody

CAPS = load_capabilities()


async def control(clock, body, v, w, seconds, hz=20.0):
    """Send a speed command every tick, like the fast control loop."""
    reasons = []
    for _ in range(int(seconds * hz)):
        reasons.append(await body.set_velocity(v, w))
        clock.advance(1.0 / hz)
    reasons.append(await body.update_motion())
    return reasons


@pytest.fixture
async def rig():
    clock = ReplayClock()
    bus = EventBus(clock)
    body = SimBody(bus, CAPS)
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


async def test_range_reflex_blocks_driving_forward_but_not_turning_away(rig):
    clock, body, sandbox, reflexes = rig
    sandbox.move_object("cardboard_box", 600.0, 0.0)
    reasons = await control(clock, body, 300.0, 0.0, 4.0)
    assert "obstacle_stop" in reasons and reflexes.count("obstacle_stop") == 1  # reported once, not per tick
    assert body.pose().x_mm < 600 - 110 - 140
    before = body.pose().heading_deg
    await control(clock, body, 0.0, 90.0, 1.0)
    assert body.pose().heading_deg - before > 60


async def test_bumper_stops_driving_into_a_box_when_the_range_sensor_misses_it(rig):
    clock, body, sandbox, reflexes = rig
    sandbox.move_object("cardboard_box", 600.0, 0.0)
    body.set_simulated_obstacle(front_tof_mm=2000.0)
    await control(clock, body, 300.0, 0.0, 4.0)
    assert set(reflexes) == {"bumper"}
    contact = 600 - 110 - 140
    assert contact - 25 <= body.pose().x_mm <= contact
    await control(clock, body, -200.0, 0.0, 0.5)
    assert body.pose().x_mm < contact - 40  # moving away from the contact is allowed


async def test_bumper_stops_backing_into_a_wall(rig):
    clock, body, _, reflexes = rig
    await control(clock, body, -300.0, 0.0, 8.0)
    assert set(reflexes) == {"bumper"}
    wall = -(1500 - 20 - 140)
    assert wall <= body.pose().x_mm <= wall + 25


async def test_dock_is_a_ramp_and_charges_the_battery(rig):
    clock, body, sandbox, reflexes = rig
    body._sensors.battery_pct = 40.0
    dock = sandbox.get_object("charging_dock")
    body.reset_pose(BodyPose(dock.x_mm + 500.0, dock.y_mm, 180.0))
    await control(clock, body, 200.0, 0.0, 2.6)
    assert reflexes == [] and sandbox.is_charging(body.pose())
    readings = body.sensors()
    assert readings.charging is True
    level = readings.battery_pct
    clock.advance(60.0)
    assert body.sensors().battery_pct == pytest.approx(level + 20.0, abs=0.5)


async def test_velocity_control_tracks_speed_and_turn_rate(rig):
    clock, body, sandbox, _ = rig
    clear_scene(sandbox)
    await control(clock, body, 200.0, 0.0, 2.0)
    assert 330 < body.pose().x_mm < 410  # 2 s at 200 mm/s, less the acceleration ramp
    start = body.pose().heading_deg
    await control(clock, body, 0.0, 90.0, 1.0)
    assert 70 < body.pose().heading_deg - start < 95


async def test_velocity_watchdog_stops_the_robot_when_commands_stop(rig):
    clock, body, sandbox, _ = rig
    clear_scene(sandbox)
    await control(clock, body, 300.0, 0.0, 1.0)
    clock.advance(5.0)  # the control loop died
    x = body.pose().x_mm
    await body.update_motion()
    assert body.pose().x_mm - x < 0.35 * 300 + 60 and body.pose().linear_vel_mm_s == 0.0


async def test_estop_stops_motion_and_refuses_new_commands(rig):
    clock, body, sandbox, reflexes = rig
    clear_scene(sandbox)
    await control(clock, body, 200.0, 0.0, 1.0)
    body.set_estop(True)
    x = body.pose().x_mm
    await control(clock, body, 200.0, 0.0, 1.0)
    assert body.pose().x_mm == pytest.approx(x, abs=1.0) and (await body.health()).status == "estop_active"


async def test_head_moves_at_its_speed_limit_and_stays_inside_its_range(rig):
    clock, body, _, _ = rig
    body.set_head(500.0)
    clock.advance(0.1)
    await body.update_motion()
    assert 0 < body.pose().head_tilt_deg <= CAPS.head_tilt.max_speed_deg_s * 0.1 + 1e-6
    for _ in range(40):
        clock.advance(0.1)
        await body.update_motion()
    assert body.pose().head_tilt_deg == CAPS.head_tilt.max_deg


def position_error_mm(a: BodyPose, b: BodyPose) -> float:
    return math.hypot(a.x_mm - b.x_mm, a.y_mm - b.y_mm)


def settle(body: PhysicsBody, seconds: float = 1.0) -> None:
    body.bus.clock.advance(seconds)
    body.advance_to_now()


@pytest.fixture
async def bus():
    bus = EventBus(ReplayClock())
    await bus.start()
    yield bus
    await bus.stop()


@pytest.mark.parametrize("seed", [0, 1])
async def test_odometry_tracks_the_truth_while_driving_and_turning(bus, seed):
    body = PhysicsBody(bus, CAPS, seed=seed)
    await control(bus.clock, body, 200.0, 0.0, 2.0)
    settle(body)
    assert position_error_mm(body.pose(), body.true_pose()) < 15
    start = body.true_pose().heading_deg
    await control(bus.clock, body, 0.0, 90.0, 1.0)
    settle(body)
    assert 60 < body.true_pose().heading_deg - start < 100
    assert abs(body.pose().heading_deg - body.true_pose().heading_deg) < 2.0


async def test_physics_watchdog_stops_the_wheels(bus):
    body = PhysicsBody(bus, CAPS, noise=SensorNoise.off())
    body.reset_pose(BodyPose(x_mm=-600, y_mm=-700))
    await control(bus.clock, body, 200.0, 0.0, 2.0)
    assert 300 < body.true_pose().x_mm + 600 < 430
    settle(body, 2.0)
    assert abs(body.true_pose().linear_vel_mm_s) < 5 and abs(body.true_pose().angular_vel_deg_s) < 3


async def test_speed_control_obeys_the_range_reflex_and_the_bumper(bus):
    body = PhysicsBody(bus, CAPS, noise=SensorNoise.off())
    reflexes: list[str] = []
    bus.subscribe("reflex.triggered", lambda e: reflexes.append(e.payload["reflex"]))
    body.reset_pose(BodyPose(x_mm=1000, y_mm=-900))
    reasons = await control(bus.clock, body, 300.0, 0.0, 3.0)
    await bus.drain()
    assert "obstacle_stop" in reasons and reflexes.count("obstacle_stop") == 1
    assert 1500 - (body.true_pose().x_mm + 150) > 40
    body.set_simulated_obstacle(front_tof_mm=2000)  # blinded range sensor: the bumper must catch it
    reasons = await control(bus.clock, body, 300.0, 0.0, 3.0)
    await bus.drain()
    assert "bumper" in reasons and body.true_pose().x_mm <= 1500 - 150 + 5


async def test_physics_estop_stops_the_robot(bus):
    body = PhysicsBody(bus, CAPS)
    body.reset_pose(BodyPose(heading_deg=-90))
    await control(bus.clock, body, 200.0, 0.0, 1.0)
    body.set_estop(True)
    await body.update_motion()
    settle(body)
    stopped = body.true_pose()
    settle(body)
    assert position_error_mm(stopped, body.true_pose()) < 15 and abs(body.true_pose().linear_vel_mm_s) < 5


async def test_pushing_ball_moves_it_and_updates_sandbox(bus):
    sandbox = SimulationSandbox()
    body = PhysicsBody(bus, CAPS, sandbox=sandbox)
    sandbox.bind(body)
    sandbox.move_object("red_ball", 300, -700)
    body.reset_pose(BodyPose(x_mm=300 + 48 + 150 + 40, y_mm=-700))  # ball 40 mm behind the robot
    assert body.object_xy_mm("red_ball") == pytest.approx((300, -700))
    await control(bus.clock, body, -200.0, 0.0, 1.5)
    ball = sandbox.get_object("red_ball")
    assert ball.x_mm < 300 - 100 and abs(ball.y_mm + 700) < 60  # a 60 g ball does not trip the bumper
    assert math.dist(body.object_xy_mm("red_ball"), (ball.x_mm, ball.y_mm)) <= 5.0 + 1e-6

    sandbox.move_object("red_ball", 500, -1000)  # a drag in the sandbox teleports the physics body
    body.advance_to_now()
    assert body.object_xy_mm("red_ball") == pytest.approx((500, -1000))


async def test_sandbox_edits_are_mirrored_into_physics(bus):
    sandbox = SimulationSandbox()
    body = PhysicsBody(bus, CAPS)
    sandbox.bind(body)
    crate = sandbox.add_object(label="Toy Crate", shape="box", x_mm=-200, y_mm=-700, size_mm=120)
    body.advance_to_now()
    assert body.object_xy_mm(crate.id) == pytest.approx((-200, -700))
    sandbox.remove_object(crate.id)
    sandbox.remove_object("blue_cube")
    body.advance_to_now()
    assert body.object_xy_mm(crate.id) is None and body.object_xy_mm("blue_cube") is None
    sandbox.reset_default_scene()
    body.advance_to_now()
    for obj in sandbox.list_objects():
        assert body.object_xy_mm(obj["id"]) == pytest.approx((obj["x_mm"], obj["y_mm"]), abs=0.5)


class FakeMcu:
    """The VENTUNO Q's STM32H5 as the router delivers it: a kinematic base with its watchdog and range envelope,
    answering cmd_vel/stop/estop notifications with 50 Hz telemetry."""

    def __init__(self, wall_x_mm: float = 900.0) -> None:
        self.handlers: dict = {}
        self.x = self.y = self.heading = self.v = self.w = 0.0
        self.head_ddeg = 0
        self.cmd = (0.0, 0.0)
        self.last_cmd = 0.0
        self.estop = False
        self.wall = wall_x_mm
        self.sent: list = []
        self._task = None

    def open(self):
        pass

    def provide(self, method, handler):
        self.handlers[method] = handler
        self._task = self._task or asyncio.get_running_loop().create_task(self._run())

    def notify(self, method, *params):
        self.sent.append((method, params))
        if method == "cmd_vel":
            self.cmd, self.last_cmd = (params[0], params[1] / 1000.0), time.monotonic()
        elif method == "stop":
            self.cmd = (0.0, 0.0)
        elif method == "estop":
            self.estop = bool(params[0])
        elif method == "head":
            self.head_ddeg = params[0]

    def close(self):
        if self._task is not None:
            self._task.cancel()

    async def _run(self):
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.02)
            now = time.monotonic()
            dt, last = now - last, now
            v, w = self.cmd if now - self.last_cmd < 0.3 and not self.estop else (0.0, 0.0)
            range_mm = (self.wall - self.x - 141.0) / max(1e-6, math.cos(math.radians(self.heading)))
            if v > 0 and 0 < range_mm <= 120.0:
                v = 0.0
            self.v, self.w = v, w
            self.heading += w * dt
            self.x += v * dt * math.cos(math.radians(self.heading))
            self.y += v * dt * math.sin(math.radians(self.heading))
            flags = (32 if v == 0 and self.cmd[0] > 0 and range_mm <= 120.0 else 0) | (4 if self.estop else 0)
            self.handlers["telemetry"](int(now * 1000), int(self.x), int(self.y), int(self.heading * 1000),
                                       int(v), int(w * 1000), 0, 0, int(max(0.0, range_mm)), flags, 8000,
                                       self.head_ddeg)


async def test_ventuno_body_drives_on_mcu_odometry_and_reports_reflexes():
    bus = EventBus(RealClock())
    mcu = FakeMcu(wall_x_mm=900.0)
    body = VentunoBody(bus, CAPS, link=mcu)
    reflexes = []
    bus.subscribe("reflex.triggered", lambda e: reflexes.append(e.payload["reflex"]), immediate=True)
    await bus.start()
    await body.start()
    try:
        for _ in range(100):
            await body.set_velocity(0.0, 90.0)
            await asyncio.sleep(0.01)
            if body.pose().heading_deg > 45.0:
                break
        assert 0.0 < body.pose().heading_deg < 90.0
        body.set_head(20.0)
        assert mcu.sent[-1] == ("head", (200,))
        await asyncio.sleep(0.05)
        assert body.pose().head_tilt_deg == 20.0
        mcu.heading = 0.0
        for _ in range(400):
            await body.set_velocity(300.0, 0.0)
            await asyncio.sleep(0.01)
            if "obstacle_stop" in reflexes:
                break
        assert "obstacle_stop" in reflexes and mcu.x < 900 - 141 - 100
        body.set_estop(True)
        assert mcu.sent[-1] == ("estop", (True,)) and (await body.health()).healthy
    finally:
        await body.stop()
        await bus.stop()


def test_firmware_agrees_with_body_yaml():
    """The STM32 sketch cannot read the YAML; a robot built from one and driven by the other would misjudge distance."""
    source = (PROJECT_ROOT / "firmware" / "ventuno_mcu" / "ventuno_mcu.ino").read_text(encoding="utf-8")

    def constant(name: str) -> float:
        return float(re.search(rf"\b{name} = ([0-9.]+)f?\b", source).group(1))

    assert constant("WHEEL_DIAMETER_MM") == CAPS.wheel_diameter_mm
    assert constant("TRACK_MM") == CAPS.track_width_mm
    assert constant("TICKS_PER_REV") == CAPS.encoder_ticks_per_rev
    assert constant("MAX_V") == CAPS.max_linear_speed_mm_s
    assert constant("MAX_W") == CAPS.max_angular_speed_deg_s
