"""Rigid-body PhysicsBody: accuracy, odometry vs truth, bumper, safety, pushing and sandbox sync."""

from __future__ import annotations

import asyncio
import math
import time

import pytest

from robot.body.hal import BodyCommand, BodyPose
from robot.body.physics_body import PhysicsBody, SensorNoise
from robot.body.sim_body import SimBody
from robot.core.bus import EventBus
from robot.core.clock import RealClock, ReplayClock
from robot.world.sandbox import SimulationSandbox

# Open floor (no default sandbox object or MJCF body within reach): from the origin toward -y.
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


async def test_sensor_noise_is_seeded_bounded_and_plausible():
    async def final_poses(seed: int) -> tuple[BodyPose, BodyPose]:
        body = make_body(seed=seed)
        await run_path(body)
        return body.pose(), body.true_pose()

    a, b, c = await final_poses(1), await final_poses(1), await final_poses(2)
    assert a == b  # same seed, same run
    assert a[0] != c[0]
    heading_errors = []
    for odom, truth in (a, c):
        heading_errors.append(abs(odom.heading_deg - truth.heading_deg))
        assert heading_errors[-1] < 2.0
        assert position_error_mm(odom, truth) < 30
    assert max(heading_errors) > 0.05  # gyro bias/noise actually reaches the estimate

    body = make_body(seed=4)
    readings = [body.sensors().front_tof_mm for _ in range(20)]
    assert len(set(readings)) > 1  # ToF noise
    assert max(readings) - min(readings) < 30


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


async def test_reset_pose_interrupts_motion(bus):
    body = PhysicsBody(bus)
    task = asyncio.create_task(body.execute(run("spin", angle_deg=-180)))
    await wait_until(lambda: body.true_pose().heading_deg < -20)
    body._pose = BodyPose(x_mm=-200, y_mm=-600)  # what the dashboard's scene reset does
    result = await asyncio.wait_for(task, timeout=5)
    assert result.status == "cancelled"
    assert body.pose().x_mm == pytest.approx(-200, abs=0.5)
    assert body.true_pose().y_mm == pytest.approx(-600, abs=0.5)


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


async def test_raycast_sensor_model_is_fed_true_pose():
    body = make_body(seed=3)
    model = RecordingSensorModel(cliff_beyond_x_mm=200.0)
    body.attach_sensor_model(model)
    assert (await body.execute(run("drive", distance_mm=150))).status == "completed"
    body.sensors()
    assert model.poses[-1] == body.true_pose()
    assert model.poses[-1] != body.pose()
    body.set_sandbox_sensors(front_tof_mm=50.0)  # odometry-based pushes are ignored
    assert body.sensors().front_tof_mm > 1400

    result = await body.execute(run("drive", distance_mm=300))
    assert result.status == "reflex_stopped" and result.message == "Cliff sensor triggered"
    assert 200 < body.true_pose().x_mm < 240


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


async def test_command_semantics_match_sim_body():
    commands = [
        ("tilt_head", {"angle_deg": 20}),
        ("tilt_head", {"angle_deg": 50}),
        ("drive", {"distance_mm": 200}),
        ("drive", {"distance_mm": 150, "direction": "backward"}),
        ("spin", {"angle_deg": 90}),
        ("spin", {"angle_deg": -90}),
        ("perform_trick", {"trick": "nod"}),
        ("perform_trick", {"trick": "wiggle"}),
        ("perform_trick", {"trick": "moonwalk"}),
        ("set_gaze", {"target": "left"}),
        ("set_eye_color", {"color": "amber"}),
        ("drive", {"distance_mm": -5}),
        ("fly", {}),
        ("stop", {}),
    ]
    outcomes = {}
    for cls in (SimBody, PhysicsBody):
        bus = EventBus(ReplayClock())
        events: list[str] = []
        bus.subscribe("*", lambda e, events=events: events.append(e.type))
        await bus.start()
        body = cls(bus)
        await body.start()
        rows = []
        for tool, args in commands:
            result = await body.execute(run(tool, **args))
            rows.append((result.tool, result.ok, result.status, result.message, result.details))
        await bus.drain()
        await body.stop()
        await bus.stop()
        outcomes[cls] = (rows, sorted(set(events)))
        if cls is PhysicsBody:
            assert body.pose().head_tilt_deg == pytest.approx(0.0)  # nod ends at neutral
            assert body.true_pose().head_tilt_deg == pytest.approx(0.0, abs=1.0)
    assert outcomes[PhysicsBody] == outcomes[SimBody]
    assert outcomes[SimBody][1] == ["body.pose", "expression.updated", "face.theme"]


async def test_pose_events_carry_true_pose_and_battery(bus):
    body = PhysicsBody(bus)
    events = []
    bus.subscribe("body.pose", events.append)
    await body.execute(run("tilt_head", angle_deg=10))
    await bus.drain()
    payload = events[-1].payload
    assert payload["pose"]["head_tilt_deg"] == pytest.approx(10.0)
    assert payload["true_pose"]["head_tilt_deg"] == pytest.approx(10.0, abs=1.0)
    assert payload["sensors"]["battery_pct"] <= 100.0 and payload["sensors"]["charging"] is False


async def test_physics_is_faster_than_real_time():
    body = make_body()
    started = time.perf_counter()
    await run_path(body)
    wall_s = time.perf_counter() - started
    sim_s = body.bus.clock.monotonic()
    assert sim_s > 5.0
    assert body.physics_ms_per_sim_s() < 250.0  # >= 4x real time for stepping alone (~25 ms measured)
    assert wall_s < 0.5 * sim_s  # including the 20 Hz supervision loop and event emission


async def test_real_clock_lifecycle_runs_idle_physics():
    bus = EventBus(RealClock())
    await bus.start()
    body = PhysicsBody(bus, step_dt_s=0.02)
    await body.start()
    try:
        result = await body.execute(run("drive", distance_mm=40))
        assert result.status == "completed"
        assert body.true_pose().x_mm == pytest.approx(40, abs=5)
        sim_time = body._sim_time
        await asyncio.sleep(0.2)
        assert body._sim_time > sim_time + 0.1  # the idle task keeps the world evolving
        health = await body.health()
        assert health.status == "ok" and "true_pose" in health.details
    finally:
        await body.stop()
        await bus.stop()
    assert body._idle_task is None
