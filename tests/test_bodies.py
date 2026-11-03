"""The robot's bodies: the kinematic and MuJoCo simulators, and the VENTUNO Q link (with a fake MCU)."""

from __future__ import annotations

import asyncio
import math
import time

import pytest
from robot_fakes import clear_scene

from robot.body.hal import BodyCommand, BodyPose
from robot.body.physics_body import PhysicsBody, SensorNoise
from robot.body.sim_body import SimBody
from robot.core.bus import EventBus
from robot.core.clock import RealClock, ReplayClock
from robot.world.sandbox import SimulationSandbox


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


async def control(clock, body, v, w, seconds, hz=20.0):
    """Send a speed command every tick, like the fast control loop."""
    reasons = []
    for _ in range(int(seconds * hz)):
        reasons.append(await body.set_velocity(v, w))
        clock.advance(1.0 / hz)
    reasons.append(await body.update_motion())
    return reasons


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


async def test_range_reflex_blocks_driving_forward_but_not_turning_away(rig):
    clock, body, sandbox, reflexes = rig
    sandbox.move_object("cardboard_box", 600.0, 0.0)
    reasons = await control(clock, body, 300.0, 0.0, 4.0)
    assert "obstacle_stop" in reasons and reflexes.count("obstacle_stop") == 1  # reported once, not per tick
    assert body.pose().x_mm < 600 - 110 - 140
    before = body.pose().heading_deg
    await control(clock, body, 0.0, 90.0, 1.0)
    assert body.pose().heading_deg - before > 60


SQUARE_PATH = (
    ("spin", {"angle_deg": -90}),
    ("drive", {"distance_mm": 500}),
    ("spin", {"angle_deg": 90}),
    ("drive", {"distance_mm": 400}),
)


def make_body(**kwargs) -> PhysicsBody:
    return PhysicsBody(EventBus(ReplayClock()), **kwargs)


def run(tool: str, **args) -> BodyCommand:
    return BodyCommand(tool, tool, args)


async def run_path(body: PhysicsBody, path=SQUARE_PATH) -> None:
    for tool, args in path:
        result = await body.execute(run(tool, **args))
        assert result.status == "completed", result


def position_error_mm(a: BodyPose, b: BodyPose) -> float:
    return math.hypot(a.x_mm - b.x_mm, a.y_mm - b.y_mm)


async def wait_until(predicate, limit: int = 5000) -> None:
    for _ in range(limit):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


def settle(body: PhysicsBody, seconds: float = 1.0) -> None:
    body.bus.clock.advance(seconds)
    body.advance_to_now()


@pytest.fixture
async def bus():
    bus = EventBus(ReplayClock())
    await bus.start()
    yield bus
    await bus.stop()


@pytest.mark.parametrize("seed", [0, 1, 2])
async def test_drive_500mm_accuracy(seed):
    body = make_body(seed=seed)
    result = await body.execute(run("drive", distance_mm=500))
    assert result.ok and result.status == "completed"
    assert result.message == "Drove forward 500 mm."
    odom, truth = body.pose(), body.true_pose()
    # Closed loop on the odometry estimate; truth differs by the seeded ~1 % wheel calibration error.
    assert odom.x_mm == pytest.approx(500, abs=1.5)
    assert truth.x_mm == pytest.approx(500, abs=10)
    assert abs(truth.y_mm) < 5 and abs(truth.heading_deg) < 1


@pytest.mark.parametrize("seed", [0, 1])
async def test_spin_90_degrees(seed):
    body = make_body(seed=seed)
    result = await body.execute(run("spin", angle_deg=90))
    assert result.status == "completed" and result.message == "Spun 90 degrees."
    truth = body.true_pose()
    assert body.pose().heading_deg == pytest.approx(90, abs=0.5)
    assert truth.heading_deg == pytest.approx(90, abs=1.5)
    assert math.hypot(truth.x_mm, truth.y_mm) < 15  # skid-steer spins (nearly) in place

    await body.execute(run("spin", angle_deg=-90))
    assert body.true_pose().heading_deg == pytest.approx(0, abs=2.0)


async def test_pose_is_odometry_and_matches_truth_without_noise():
    body = make_body(noise=SensorNoise.off())
    await body.execute(run("drive", distance_mm=300))
    assert position_error_mm(body.pose(), body.true_pose()) < 2.0
    await run_path(body)
    odom, truth = body.pose(), body.true_pose()
    assert odom.heading_deg == pytest.approx(truth.heading_deg, abs=0.05)  # exact IMU
    assert position_error_mm(odom, truth) < 15  # only wheel slip while spinning in place
    assert odom != truth
    sensors = body.sensors()
    assert sensors.wheel_ticks_left > 0 and sensors.wheel_ticks_right > 0
    assert sensors.imu_yaw_deg == pytest.approx(odom.heading_deg)
    assert body.sensors().front_tof_mm == sensors.front_tof_mm  # no ToF noise


async def test_wall_contact_is_bumper_reflex(bus):
    body = PhysicsBody(bus)
    reflexes: list[str] = []
    bus.subscribe("reflex.triggered", lambda e: reflexes.append(e.payload["reflex"]))

    # Backing up has no ToF protection: the bumper stops the robot at the wall.
    body.reset_pose(BodyPose(x_mm=-1100, y_mm=-800))
    result = await body.execute(run("drive", distance_mm=600, direction="backward"))
    await bus.drain()
    assert result.status == "reflex_stopped" and not result.ok
    assert result.details["reflex"] == "bumper" and reflexes == ["bumper"]
    stopped = body.true_pose()
    assert -1500 + 150 - 5 <= stopped.x_mm < -1300  # chassis rear (150 mm) at the wall, not in it
    settle(body)
    assert position_error_mm(stopped, body.true_pose()) < 10
    assert abs(body.true_pose().linear_vel_mm_s) < 5

    # Driving forward, the ToF obstacle reflex fires before contact...
    body.reset_pose(BodyPose(x_mm=1000, y_mm=-900))
    result = await body.execute(run("drive", distance_mm=600))
    await bus.drain()
    assert result.status == "reflex_stopped" and reflexes[-1] == "obstacle_stop"
    settle(body)
    assert 1500 - (body.true_pose().x_mm + 150) > 50

    # ...unless the ToF is blinded (fault injection); then the bumper catches it.
    body.set_simulated_obstacle(front_tof_mm=2000)
    body.reset_pose(BodyPose(x_mm=1000, y_mm=-900))
    result = await body.execute(run("drive", distance_mm=600))
    await bus.drain()
    assert result.status == "reflex_stopped" and reflexes[-1] == "bumper"
    assert body.true_pose().x_mm <= 1500 - 150 + 5


async def test_estop_and_cancel_stop_motion(bus):
    body = PhysicsBody(bus)
    body.reset_pose(BodyPose(heading_deg=-90))  # open floor toward -y
    task = asyncio.create_task(body.execute(run("drive", distance_mm=800)))
    await wait_until(lambda: body.true_pose().y_mm < -150)
    body.set_estop(True)
    result = await task
    assert result.status == "reflex_stopped" and result.message == "Emergency stop is active"
    stopped = body.true_pose()
    assert stopped.y_mm > -400
    settle(body)
    assert position_error_mm(stopped, body.true_pose()) < 15
    assert abs(body.true_pose().linear_vel_mm_s) < 5
    blocked = await body.execute(run("drive", distance_mm=100))
    assert blocked.status == "reflex_stopped"
    assert blocked.message == "Motion blocked because emergency stop is active."
    assert (await body.health()).status == "estop_active"
    body.set_estop(False)

    start_heading = body.true_pose().heading_deg
    task = asyncio.create_task(body.execute(run("spin", angle_deg=180)))
    await wait_until(lambda: abs(body.true_pose().heading_deg - start_heading) > 30)
    await body.cancel()
    result = await task
    assert result.status == "cancelled" and not result.ok
    heading = body.true_pose().heading_deg
    settle(body)
    assert abs(body.true_pose().heading_deg - heading) < 5
    assert abs(body.true_pose().heading_deg - start_heading) < 120

    resumed = await body.execute(run("drive", distance_mm=100))
    assert resumed.status == "completed"


async def test_pushing_ball_moves_it_and_updates_sandbox(bus, monkeypatch):
    sandbox = SimulationSandbox()
    body = PhysicsBody(bus, sandbox=sandbox)
    sandbox.bind(body)  # also attaches the sandbox raycasts as the sensor model
    sandbox.move_object("red_ball", 300, -700)
    body.reset_pose(BodyPose(x_mm=300 + 48 + 150 + 40, y_mm=-700))  # ball 40 mm behind the robot
    assert body.object_xy_mm("red_ball") == pytest.approx((300, -700))

    reported: list[tuple[str, float, float]] = []
    move_object = sandbox.move_object

    def spy(obj_id, x_mm, y_mm):
        reported.append((obj_id, x_mm, y_mm))
        return move_object(obj_id, x_mm, y_mm)

    monkeypatch.setattr(sandbox, "move_object", spy)
    result = await body.execute(run("drive", distance_mm=300, direction="backward"))
    assert result.status == "completed"  # a 60 g ball does not trip the bumper
    ball = sandbox.get_object("red_ball")
    assert ball.x_mm < 300 - 100 and abs(ball.y_mm + 700) < 60
    assert {obj_id for obj_id, _, _ in reported} == {"red_ball"}
    assert all(
        math.dist(p[1:], q[1:]) > 5 for p, q in zip(reported, reported[1:], strict=False)
    )  # only moves above the 5 mm threshold are written back
    assert math.dist(body.object_xy_mm("red_ball"), (ball.x_mm, ball.y_mm)) <= 5.0 + 1e-6

    # A drag in the sandbox teleports the physics body.
    move_object("red_ball", 500, -1000)
    body.advance_to_now()
    assert body.object_xy_mm("red_ball") == pytest.approx((500, -1000))


async def test_sandbox_edits_are_mirrored_into_physics(bus):
    sandbox = SimulationSandbox()
    body = PhysicsBody(bus)
    sandbox.bind(body)
    crate = sandbox.add_object(label="Toy Crate", shape="box", x_mm=-200, y_mm=-700, size_mm=120)
    body.reset_pose(BodyPose(x_mm=100, y_mm=-700))
    assert body.object_xy_mm(crate.id) == pytest.approx((-200, -700))
    result = await body.execute(run("drive", distance_mm=250, direction="backward"))
    assert result.status == "completed"
    assert sandbox.get_object(crate.id).x_mm < -250

    sandbox.remove_object(crate.id)
    sandbox.remove_object("blue_cube")
    body.advance_to_now()
    assert body.object_xy_mm(crate.id) is None and body.object_xy_mm("blue_cube") is None
    sandbox.reset_default_scene()
    body.advance_to_now()
    for obj in sandbox.list_objects():
        assert body.object_xy_mm(obj["id"]) == pytest.approx((obj["x_mm"], obj["y_mm"]), abs=0.5)


class RecordingSensorModel:
    def __init__(self, cliff_beyond_x_mm: float = math.inf) -> None:
        self.poses: list[BodyPose] = []
        self.cliff_beyond_x_mm = cliff_beyond_x_mm
        self.charging = False

    def compute_front_tof_mm(self, pose: BodyPose) -> float:
        self.poses.append(pose)
        return 1500.0

    def check_cliff_triggered(self, pose: BodyPose) -> bool:
        return pose.x_mm > self.cliff_beyond_x_mm

    def is_charging(self, pose: BodyPose) -> bool:
        return self.charging


async def test_battery_drains_with_travel_and_charges_on_dock():
    body = make_body(
        noise=SensorNoise.off(), battery_idle_pct_per_min=0.0, battery_pct_per_m=10.0, charge_pct_per_min=4.0
    )
    model = RecordingSensorModel()
    body.attach_sensor_model(model)
    assert (await body.execute(run("drive", distance_mm=500))).status == "completed"
    sensors = body.sensors()
    assert sensors.battery_pct == pytest.approx(95.0, abs=0.3) and not sensors.charging
    model.charging = True
    body.bus.clock.advance(60.0)
    sensors = body.sensors()
    assert sensors.charging and sensors.battery_pct == pytest.approx(99.0, abs=0.3)

    idle = make_body(battery_idle_pct_per_min=6.0)
    idle.bus.clock.advance(60.0)
    assert idle.sensors().battery_pct == pytest.approx(94.0, abs=0.01)
    with pytest.raises(ValueError):
        make_body(battery_pct_per_m=-1.0)


async def drive_by_speed(body, v, w, seconds, hz=20.0):
    reasons = []
    for _ in range(int(seconds * hz)):
        reasons.append(await body.set_velocity(v, w))
        body.bus.clock.advance(1.0 / hz)
    reasons.append(await body.update_motion())
    return reasons


async def test_speed_control_drives_turns_and_stops_on_the_watchdog(bus):
    body = PhysicsBody(bus, noise=SensorNoise.off())
    body.reset_pose(BodyPose(x_mm=-600, y_mm=-700))
    await drive_by_speed(body, 200.0, 0.0, 2.0)
    assert 300 < body.true_pose().x_mm + 600 < 430
    start = body.true_pose().heading_deg
    await drive_by_speed(body, 0.0, 90.0, 1.0)
    assert 60 < body.true_pose().heading_deg - start < 100
    settle(body, 2.0)  # no more commands: the watchdog stops the wheels
    assert abs(body.true_pose().linear_vel_mm_s) < 5 and abs(body.true_pose().angular_vel_deg_s) < 3


async def test_speed_control_obeys_the_range_reflex_and_the_bumper(bus):
    body = PhysicsBody(bus, noise=SensorNoise.off())
    reflexes: list[str] = []
    bus.subscribe("reflex.triggered", lambda e: reflexes.append(e.payload["reflex"]))
    body.reset_pose(BodyPose(x_mm=1000, y_mm=-900))
    reasons = await drive_by_speed(body, 300.0, 0.0, 3.0)
    await bus.drain()
    assert "obstacle_stop" in reasons and reflexes.count("obstacle_stop") == 1
    assert 1500 - (body.true_pose().x_mm + 150) > 40
    body.set_simulated_obstacle(front_tof_mm=2000)  # blinded range sensor: the bumper must catch it
    reasons = await drive_by_speed(body, 300.0, 0.0, 3.0)
    await bus.drain()
    assert "bumper" in reasons and body.true_pose().x_mm <= 1500 - 150 + 5


class FakeMcu:
    """The VENTUNO Q's STM32H5 as the router delivers it: a kinematic base with its watchdog and range envelope,
    answering cmd_vel/stop/estop notifications with 50 Hz telemetry."""

    def __init__(self, wall_x_mm: float = 900.0) -> None:
        self.handlers: dict = {}
        self.x = self.y = self.heading = self.v = self.w = 0.0
        self.cmd = (0.0, 0.0)
        self.last_cmd = 0.0
        self.estop = False
        self.wall = wall_x_mm
        self.sent: list = []
        self._task = None

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
                                       int(v), int(w * 1000), 0, 0, int(max(0.0, range_mm)), flags, 8000, 0)


async def test_ventuno_body_drives_on_mcu_odometry_and_reports_reflexes():
    from robot.body.ventuno_body import VentunoBody

    bus = EventBus(RealClock())
    mcu = FakeMcu(wall_x_mm=900.0)
    body = VentunoBody(bus, link=mcu)
    reflexes = []
    bus.subscribe("reflex.triggered", lambda e: reflexes.append(e.payload["reflex"]), immediate=True)
    await bus.start()
    await body.start()
    try:
        spun = await body.execute(BodyCommand("s", "spin", {"angle_deg": 90}))
        assert spun.ok and abs(body.pose().heading_deg - 90.0) < 4.0
        await body.execute(BodyCommand("s2", "spin", {"angle_deg": -90}))
        blocked = await body.execute(BodyCommand("d", "drive", {"direction": "forward", "distance_mm": 1500}))
        assert not blocked.ok and blocked.status == "reflex_stopped" and "obstacle_stop" in reflexes
        assert mcu.x < 900 - 141 - 100
        await body.set_velocity(150.0, 0.0)
        assert mcu.sent[-1] == ("cmd_vel", (150, 0))
        body.set_estop(True)
        assert mcu.sent[-1] == ("estop", (True,)) and (await body.health()).healthy
    finally:
        await body.stop()
        await bus.stop()
