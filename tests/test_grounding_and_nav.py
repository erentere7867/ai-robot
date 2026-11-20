"""Tests for semantic grounding, user resolution, synonym fallback, and Nav2 tolerance / heading."""

from __future__ import annotations

import math
from concurrent.futures import Future
from threading import Lock
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robot.agent.api import NotFound, Object, PathBlocked, Pose, Preempted, TargetLost
from robot.interfaces.live import PAGE
from robot.perception.tracker import Track, Tracker
from robot.ros.perceive import Perception
from robot.ros.real import RealRobot
from robot.ros.robot import RosRobot
from robot.sim.world import Scene, SimRobot, Thing


def _make_scene(things: list[Thing] | None = None) -> Scene:
    return Scene(
        width=10.0,
        height=10.0,
        things=things or [],
        rooms={"lounge": (-4.0, -4.0, 4.0, 4.0)},
        start=Pose(0.0, 0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# 1. SimRobot._named & synonym fallback
# ---------------------------------------------------------------------------


def test_named_resolves_user_aliases_when_present():
    user_thing = Thing("person", 1.5, 0.5, radius=0.25, user=True)
    sim = SimRobot(_make_scene([user_thing]))

    for alias in ["user", "me", "you", "the user", "person(user)", "human", "speaker", "person"]:
        target = sim._named(alias)
        assert isinstance(target, Object)
        assert target.id == user_thing.id


def test_named_raises_not_found_on_user_aliases_when_no_user():
    sim = SimRobot(_make_scene([Thing("chair", 2.0, 0.0, radius=0.3)]))
    with pytest.raises(NotFound):
        sim._named("user")
    with pytest.raises(NotFound):
        sim._named("me")
    with pytest.raises(NotFound):
        sim._named("speaker")


def test_named_extracts_embedded_integer_ids():
    desk = Thing("desk", 2.0, 1.0, radius=0.4, id=7)
    table = Thing("table", -2.0, 1.0, radius=0.5, id=10)
    sim = SimRobot(_make_scene([desk, table]))

    # Leading integer
    assert sim._named("7 desk").id == 7
    assert sim._named("10  table").id == 10
    # Trailing integer
    assert sim._named("desk 7").id == 7
    assert sim._named("table 10").id == 10
    # Hash integer
    assert sim._named("#7").id == 7


def test_synonym_fallback_when_exact_match_absent():
    # Only a desk is in the world, user asks for "table"
    desk = Thing("desk", 2.0, 0.0, radius=0.4, color="brown")
    sim = SimRobot(_make_scene([desk]))

    matched = sim._named("table")
    assert isinstance(matched, Object)
    assert matched.label == "desk"

    where_results = sim.where("table")
    assert len(where_results) > 0
    assert where_results[0].label == "desk"

    # Only a sofa is in the world, user asks for "couch"
    sofa = Thing("sofa", 3.0, 0.0, radius=0.6, color="grey")
    sim2 = SimRobot(_make_scene([sofa]))
    assert sim2._named("couch").label == "sofa"

    # Only a cup is in the world, user asks for "mug"
    cup = Thing("cup", 1.0, 0.0, radius=0.1, color="red")
    sim3 = SimRobot(_make_scene([cup]))
    assert sim3._named("mug").label == "cup"


def test_colour_request_uses_the_named_track_crop_not_the_background():
    real = object.__new__(RealRobot)
    tracker = Tracker()
    track = Track(7, "bike", 1.1, -0.5, 0.3, 0, 0, hits=4)
    tracker.tracks[track.id] = track
    thing = Thing("bike", track.x, track.y, color="brown", id=7)
    real.perception = SimpleNamespace(lock=Lock(), tracker=tracker, crops={7: np.zeros((36, 36, 3), np.uint8)})
    questions = []

    def decide(messages, choices):
        questions.append(messages[0]["content"][-1]["text"])
        return {w: float(w == "blue") for w in choices}

    real.brain = SimpleNamespace(decide=decide)
    assert not real._matching("red bicycle", [thing])
    assert real._matching("the blue bicycle", [thing]) == [thing]
    assert track.color == thing.color == "blue"
    assert len(questions) == 1 and "red" not in questions[0] and "blue" not in questions[0]


def test_exact_match_prioritized_over_synonyms():
    # Both table and desk exist: asking for table must yield table
    desk = Thing("desk", 2.0, 0.0, radius=0.4, id=1)
    table = Thing("table", -2.0, 0.0, radius=0.5, id=2)
    sim = SimRobot(_make_scene([desk, table]))

    assert sim._named("table").id == 2
    assert sim._named("desk").id == 1


# ---------------------------------------------------------------------------
# 3. RosRobot._goal_point & _go tolerance and retry
# ---------------------------------------------------------------------------


class MockRosRobot(RosRobot):
    """Subclass of RosRobot that mocks ROS2 action communication for testing."""

    def __init__(self, scene: Scene) -> None:
        super(RosRobot, self).__init__(scene)
        self.nav_calls: list[tuple[str, dict[str, Any]]] = []
        self.rotate_calls: list[float] = []
        self.fail_nav_counts: int = 0
        self.sync_calls: int = 0

    def _sync(self) -> None:
        self.sync_calls += 1

    def _step(self, v: float, w: float, dt: float = 0.05) -> None:
        super(RosRobot, self)._step(v, w, dt)

    def _act(self, name: str, **fields: Any) -> None:
        self.nav_calls.append((name, fields))
        if name == "navigate_to_pose" and self.fail_nav_counts > 0:
            self.fail_nav_counts -= 1
            raise PathBlocked("Nav2 aborted near obstacle footprint")

    def _rotate(self, angle: float, rate: float = 1.0) -> None:
        self.rotate_calls.append(angle)
        self.yaw += angle

    def _pose_msg(self, x: float, y: float, yaw: float) -> Any:
        return {"x": x, "y": y, "yaw": yaw}


def test_ros_robot_goal_point_heading_faces_object():
    scene = _make_scene()
    robot = MockRosRobot(scene)
    target = Object(
        id=1,
        label="chair",
        position=(3.0, 0.0),
        distance=3.0,
        bearing=0.0,
        side="ahead",
        radius=0.3,
        color="brown",
        last_seen=0.0,
    )

    where = robot._goal_point(target, distance=0.5)
    assert where is not None
    (end_x, end_y), arrival_yaw = where

    # Robot must arrive facing the target object (3.0, 0.0) from (end_x, end_y)
    expected_yaw = math.atan2(0.0 - end_y, 3.0 - end_x)
    assert abs(arrival_yaw - expected_yaw) < 1e-3


@pytest.mark.parametrize("when", ["before acceptance", "while driving"])
def test_pending_and_active_nav_goals_are_cancelled_when_interrupted(when):
    robot = MockRosRobot(_make_scene([Thing("bag", 1.5, 0.0)]))
    target, = robot.detect("bag")
    sent, result = Future(), Future()
    cancelled, velocities = [], []

    def cancel():
        cancelled.append(True)
        result.set_result(SimpleNamespace(status=5))

    handle = SimpleNamespace(accepted=True, cancel_goal_async=cancel, get_result_async=lambda: result)
    robot.actions = {"navigate_to_pose": SimpleNamespace(wait_for_server=lambda **_: True,
                                                        send_goal_async=lambda goal: sent)}
    robot.kinds = {"navigate_to_pose": SimpleNamespace(Goal=SimpleNamespace)}
    robot._publish = lambda v, w: velocities.append((v, w))

    def step(v, w):
        if when == "before acceptance":
            raise Preempted("stop")
        robot.things.pop(target.id)  # perception withdrew the ID after Nav2 accepted it

    robot._step = step
    if when == "while driving":
        sent.set_result(handle)
        with pytest.raises(TargetLost, match="stopped approaching"):
            RosRobot._act(robot, "navigate_to_pose", watch=target)
        assert velocities[-1] == (0.0, 0.0) and robot._goal is None
    else:
        with pytest.raises(Preempted):
            RosRobot._act(robot, "navigate_to_pose", watch=target)
        sent.set_result(handle)
    assert cancelled == [True]


def test_ros_robot_go_arrival_tolerance_absorbs_nav2_abort():
    scene = _make_scene()
    robot = MockRosRobot(scene)
    target = Object(
        id=1,
        label="table",
        position=(2.0, 0.0),
        distance=2.0,
        bearing=0.0,
        side="ahead",
        radius=0.4,
        color="brown",
        last_seen=0.0,
    )

    # Place robot close enough to target: actual_dist <= desired_reach + 0.35
    # desired_reach = 0.4 + max(0.5, 0.35) = 0.9. desired_reach + 0.35 = 1.25.
    robot.x = 1.0  # dist to target is 1.0 <= 1.25
    robot.y = 0.0
    robot.yaw = 0.0
    robot.fail_nav_counts = 1  # Nav2 aborts

    # Should NOT raise PathBlocked because it's within tolerance
    robot._go(target, distance=0.5)
    assert robot.sync_calls >= 1


def test_nav2_and_slam_parameters_fit_this_robot(tmp_path):
    from pathlib import Path

    import pytest
    import yaml

    from robot.ros import bringup
    from robot.ros.model import BODY_R

    if not (bringup.NAV2 / "params" / "nav2_params.yaml").exists() or not Path(bringup.SLAM_DEFAULTS).exists():
        pytest.skip("Nav2 and slam_toolbox are not installed here")
    cfg = yaml.safe_load(bringup.params(tmp_path).read_text())
    for name in ("local_costmap", "global_costmap"):
        assert cfg[name][name]["ros__parameters"]["robot_radius"] == BODY_R
    assert cfg["amcl"]["ros__parameters"]["max_beams"] == bringup.AMCL["max_beams"]
    assert cfg["controller_server"]["ros__parameters"]["FollowPath"]["vx_max"] == bringup.CRUISE
    slam = cfg["slam_toolbox"]["ros__parameters"]
    assert slam["use_sim_time"] and slam["base_frame"] == "base_link" and slam["max_laser_range"] <= 5.0


def test_ros_robot_shutdown_cleans_up_action_node_and_executor():
    scene = _make_scene()
    robot = MockRosRobot(scene)

    class FakeNode:
        def __init__(self):
            self.destroyed = False

        def destroy_node(self):
            self.destroyed = True

    class FakeExecutor:
        def __init__(self):
            self.shutdown_called = False

        def shutdown(self):
            self.shutdown_called = True

    robot._action_node = FakeNode()
    robot._action_executor = FakeExecutor()
    robot.node = FakeNode()
    robot._executor = FakeExecutor()

    robot.shutdown()

    assert robot._action_node.destroyed
    assert robot._action_executor.shutdown_called
    assert robot.node.destroyed
    assert robot._executor.shutdown_called


def test_ros_robot_act_recreates_action_client_on_timeout():
    import sys
    import types

    scene = _make_scene()
    robot = MockRosRobot(scene)

    client1_waits = []
    client2_waits = []

    class FakeClient1:
        def wait_for_server(self, timeout_sec=10.0):
            client1_waits.append(timeout_sec)
            return False  # Times out after 10s

    class FakeClient2:
        def wait_for_server(self, timeout_sec=20.0):
            client2_waits.append(timeout_sec)
            return False  # Times out after 20s

    created_clients = []

    class FakeActionClient:
        def __init__(self, node, kind, name):
            c = FakeClient2()
            created_clients.append((node, kind, name, c))
            self._client = c

        def wait_for_server(self, timeout_sec=20.0):
            return self._client.wait_for_server(timeout_sec)

    fake_rclpy = types.ModuleType("rclpy")
    fake_action = types.ModuleType("rclpy.action")
    fake_action.ActionClient = FakeActionClient
    fake_rclpy.action = fake_action
    orig_rclpy = sys.modules.get("rclpy")
    orig_action = sys.modules.get("rclpy.action")
    sys.modules["rclpy"] = fake_rclpy
    sys.modules["rclpy.action"] = fake_action

    try:
        robot._action_node = object()
        robot.kinds = {"test_act": object}
        robot.actions = {"test_act": FakeClient1()}

        with pytest.raises(RuntimeError, match="the test_act action server is not up"):
            RosRobot._act(robot, "test_act")

        assert client1_waits == [10.0]
        assert client2_waits == [20.0]
        assert len(created_clients) == 1
    finally:
        if orig_rclpy is not None:
            sys.modules["rclpy"] = orig_rclpy
        else:
            sys.modules.pop("rclpy", None)
        if orig_action is not None:
            sys.modules["rclpy.action"] = orig_action
        else:
            sys.modules.pop("rclpy.action", None)


# ---------------------------------------------------------------------------
# 4. live.py HTML reply banner & speak aloud checkbox
# ---------------------------------------------------------------------------


def test_live_html_reply_banner_and_speech_toggle():
    assert '<div class="reply" id="reply" hidden>' in PAGE
    assert '<span class="reply-tag">SPROCKET</span>' in PAGE
    assert '<span id="reply-text"></span>' in PAGE
    assert '.reply{' in PAGE
    assert '<input type="checkbox" id="speak" checked onchange="toggleVoice()"' in PAGE


# ---------------------------------------------------------------------------
# 5. Regression: SimRobot.orbit does not trigger false cable snag
# ---------------------------------------------------------------------------


def test_orbit_around_furniture_no_false_cable_snag():
    desk = Thing("desk", 0.0, 0.0, radius=0.5)
    scene = Scene(
        width=10.0,
        height=10.0,
        things=[desk],
        rooms={"lounge": (-4.0, -4.0, 4.0, 4.0)},
        start=Pose(1.2, 0.0, 90.0),
        snag_at=None,
    )
    sim = SimRobot(scene)
    # Must orbit cleanly without raising PathBlocked: cable snag detected
    sim.orbit("desk", laps=1.0, distance=0.5)
    assert sim.t > 1.0
    assert len(sim.trace.poses) > 20


def test_snag_detection_still_triggers_when_snag_at_present():
    scene = Scene(
        width=10.0,
        height=10.0,
        things=[],
        rooms={"lounge": (-4.0, -4.0, 4.0, 4.0)},
        start=Pose(0.0, 0.0, 0.0),
        snag_at=(0.5, 0.0),
    )
    sim = SimRobot(scene)
    with pytest.raises(PathBlocked, match="cable snag detected"):
        sim.go_to((2.0, 0.0))


def test_sim_robot_begin_resets_task_budget():
    from robot.sim.world import BudgetSpent

    sim = SimRobot(_make_scene(), max_time=600.0)
    sim.t = 590.0
    sim._interrupts()  # does not raise

    # Calling begin() resets task_t0 to 590.0
    sim.begin()
    assert sim._task_t0 == 590.0
    sim.t = 700.0  # total t is 700.0, but task time is only 110.0 s
    sim._interrupts()  # does not raise

    # Once task time exceeds 600.0s (i.e. t > 590.0 + 600.0 = 1190.0)
    sim.t = 1191.0
    with pytest.raises(BudgetSpent):
        sim._interrupts()


def test_ros_robot_begin_and_sync_bounds_poses():
    robot = MockRosRobot(_make_scene())
    robot.t0 = 0.0
    robot._now = lambda: 10.0
    robot._read_pose = lambda: True
    robot._perceive = lambda: None
    robot._interrupts = lambda: None

    robot.trace.poses = [(float(i), 0.0, 0.0, 0.0) for i in range(1000)]
    robot.begin()
    assert len(robot.trace.poses) == 500

    # RosRobot._sync bounds trace.poses when > 3000
    robot.trace.poses = [(float(i), 0.0, 0.0, 0.0) for i in range(3001)]
    RosRobot._sync(robot)
    assert len(robot.trace.poses) == 1500  # bounded to last 1500 by _sync


def test_scan_sectors_made_earlier_are_moved_to_where_the_robot_is_now():
    import math

    import numpy as np

    from robot.ros.perceive import combine

    ranges = np.full(360, np.nan, np.float32)
    ranges[180] = 2.0  # something 2 m dead ahead, seen when the robot faced along +x
    scan, angle_min, step = combine([(0.0, (0.0, 0.0, 0.0), ranges)], (0.0, 0.0, math.pi / 2))
    k = int(np.nanargmin(scan))
    assert abs(math.degrees(angle_min + (k + 0.5) * step) + 90) < 1.5  # now it is on the right
    assert abs(float(scan[k]) - 2.0) < 0.01
    scan, _, _ = combine([(0.0, (0.0, 0.0, 0.0), ranges)], (1.0, 0.0, 0.0))
    assert abs(float(np.nanmin(scan)) - 1.0) < 0.01  # driven 1 m toward it


def test_wheels_that_do_not_follow_the_drive_put_the_touched_spot_in_the_scan():
    import numpy as np

    from robot.ros.model import BODY_R
    from robot.ros.perceive import SyntheticScanPublisher

    scan = SyntheticScanPublisher(publish=True)
    t = 0.0
    for _ in range(50):  # driving, the wheels following: nothing touched
        scan.drive(t, 0.3)
        assert not scan.wheels(t, 0.3, (0.0, 0.0, 0.0))
        t += 0.02
    new = []
    for _ in range(25):  # held: the motors driven, the wheels still
        scan.drive(t, 0.3)
        new.append(scan.wheels(t, 0.0, (0.0, 0.0, 0.0)))
        t += 0.02
    assert sum(new) == 1 and new.index(True) >= 14  # once, after STALL_S
    _, _, ranges = scan._contact_part(t + 60.0, (-0.3, 0.0, 0.0))  # a minute later, backed off 0.3 m
    k = int(np.nanargmin(ranges))
    assert abs(float(ranges[k]) - (BODY_R + 0.03 + 0.3)) < 0.02 and abs(k + 0.5 - 180) < 2  # still there, ahead
    assert np.all(np.isnan(scan._contact_part(t + 200.0, (0.0, 0.0, 0.0))[2]))  # forgotten after CONTACT_KEEP


def test_camera_for_ambiguous_ids_uses_distinct_visible_marks_not_old_or_rejected_ones():
    real = object.__new__(RealRobot)
    real._now = lambda: 10.0
    tracks = {i: SimpleNamespace(present=True, rejected=i == 3) for i in (1, 2, 3)}
    real.perception = SimpleNamespace(
        lock=Lock(), tracker=SimpleNamespace(tracks=tracks),
        detected={"front": (9.9, None), "left": (9.5, None), "right": (6.0, None)},
        boxes={"front": [((), 1), ((), 1), ((), 3)], "left": [((), 1), ((), 2)], "right": [((), 4)]},
    )
    assert real.marked_cameras({1, 2, 3}) == ["left"]
    tracks[4] = SimpleNamespace(present=True, rejected=False)
    assert real.marked_cameras({3, 4}) == []
    real.perception.detected["right"] = (10.0, None)
    assert real.marked_cameras({1, 2, 4}) == ["right", "left"]


def test_a_stale_detection_cannot_put_its_badge_on_a_new_exposure():
    p = object.__new__(Perception)
    p.lock, p.tracker = Lock(), Tracker()
    cam = "front_camera"
    p.tracker.tracks[1] = Track(1, "bag", 1, 0, 0.2, 0, 0)
    current = np.full((90, 160, 3), 73, dtype=np.uint8)
    p.detected = {cam: (2.0, np.zeros_like(current))}
    p.frames = {cam: (SimpleNamespace(sec=8, nanosec=0), current)}
    p.boxes = {cam: [((10, 12, 45, 70), 1)]}
    assert np.array_equal(p.render_annotated_view(cam, 160, 90, 7.0), current)
    assert p.render_annotated_view(cam, 160, 90, 9.0) is None


def test_real_robot_wall_bounds_rejection():
    from unittest.mock import MagicMock

    from robot.perception.tracker import Track

    # Scene 10x10 -> bounds are +/- 4.88 (width/2 - 0.12)
    scene = _make_scene()
    real = object.__new__(RealRobot)
    real._ready = True
    real.scene = scene
    real.store = None
    real._cleared_tracks = set()
    real.t = 10.0
    real.things = {}
    real.detected = set()
    real._rooms = {}
    real._named_at = 0.0

    # Track 1 inside room, Track 2 outside room wall (e.g. x=5.1 > 4.88)
    tr_inside = Track(1, "chair", 2.0, 1.0, 0.3, 0.0, 10.0, present=True, hits=3)
    tr_outside = Track(2, "couch", 5.2, 1.0, 0.5, 0.0, 10.0, present=True, hits=3)

    real.perception = MagicMock()
    real.perception.objects.return_value = [tr_inside, tr_outside]

    real._perceive_objects()

    # Only track 1 should be perceived as a Thing
    assert 1 in real.things
    assert 2 not in real.things


def test_resize_rgb_bilinear():
    import numpy as np

    from robot.perception.camera import resize_rgb as _resize_rgb

    # 100x100 input image with a horizontal gradient
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for c in range(100):
        img[:, c] = c * 2

    # Downscale to 50x50
    resized = _resize_rgb(img, 50, 50)
    assert resized.shape == (50, 50, 3)
    # Check monotonic increase
    assert np.all(np.diff(resized[25, :, 0]) >= 0)

    # Upscale to 200x200
    upscaled = _resize_rgb(img, 200, 200)
    assert upscaled.shape == (200, 200, 3)


def test_ros_robot_begin_refreshes_clock():
    import threading

    from robot.ros.robot import RosRobot
    from robot.sim.world import Trace

    robot = object.__new__(RosRobot)
    robot.t0 = 100.0
    robot.t = 0.0
    robot._now = lambda: 800.0
    robot.trace = Trace()
    robot.places = {}
    robot.x = 0.0
    robot.y = 0.0
    robot.yaw = 0.0
    robot._cancel = threading.Event()

    # Call begin()
    robot.begin()

    # Clock should have been refreshed to 800 - 100 = 700
    assert robot.t == 700.0
    assert robot._task_t0 == 700.0
    # No time budget spent on task start
    assert robot.t - robot._task_t0 == 0.0


def test_real_robot_outdoor_scene_does_not_reject_distant_tracks():
    from unittest.mock import MagicMock

    from robot.perception.tracker import Track
    from robot.ros.real import RealRobot
    from robot.sim.world import Scene

    # Open outdoor scene without enclosed walls
    outdoor_scene = Scene(
        width=10.0,
        height=10.0,
        things=[],
        rooms={},
        walls=[],
    )
    outdoor_scene.outdoor = True

    real = object.__new__(RealRobot)
    real._ready = True
    real.scene = outdoor_scene
    real.store = None
    real._cleared_tracks = set()
    real.t = 10.0
    real.things = {}
    real.detected = set()
    real._rooms = {}
    real._named_at = 0.0

    # Distant track at x=15.0 m (> width / 2)
    tr_distant = Track(1, "tree", 15.0, 0.0, 0.5, 0.0, 10.0, present=True, hits=3)
    real.perception = MagicMock()
    real.perception.objects.return_value = [tr_distant]

    real._perceive_objects()

    # Distant object should NOT be rejected in an outdoor scene
    assert 1 in real.things


def test_a_new_word_gets_enough_frames_of_its_own_before_nothing_counts_as_in_view(monkeypatch):
    import time
    from collections import Counter

    from robot.perception.rig import RIG
    from robot.ros.real import LOOKS

    real = object.__new__(RealRobot)
    vocabulary, used = ["person"], {n: ["person"] for n in RIG}  # each camera's frame in flight has the old words
    looks = {n: Counter() for n in RIG}
    steps = []

    def step(v: float, w: float) -> None:  # one frame from each camera finishes; the next starts with the words now
        steps.append(1)
        for n in RIG:
            looks[n].update(used[n])
            used[n] = list(vocabulary)

    real.perception = SimpleNamespace(looks=looks, detector=SimpleNamespace(watch=vocabulary.append))
    real._call = real._perceive_objects = lambda *a, **k: None
    real._step, real.brain, real.things, real._detected, real.memory = step, None, {}, set(), {}
    assert real.detect("blue bicycle") == []
    assert all(looks[n]["bicycle"] == LOOKS for n in RIG) and len(steps) == LOOKS + 1
    monkeypatch.setattr("robot.ros.real.LOOK_S", 0.05)
    real._step = lambda v, w: time.sleep(0.01)  # the vision server is down: no frame is looked at
    with pytest.raises(RuntimeError, match="vision server"):
        real.detect("chair")


def test_go_toward_hands_go_to_a_point_it_plans_to():
    from robot.eval.suite import two_rooms
    from robot.sim.world import BEARINGS

    sim = SimRobot(two_rooms())
    refused = []
    for x, y in ((i * 0.5, j * 0.5) for i in range(-8, 9) for j in range(-6, 7)):
        if sim._occ(x, y, sim.inflated):
            continue
        sim.x, sim.y = x, y
        for direction in BEARINGS:
            goal, reach = sim._reach(sim.toward(direction, 1.5), 0.0)
            if math.dist(goal, (x, y)) > reach and sim._plan(goal, reach) is None:
                refused.append(((x, y), direction))
    assert refused == []


def test_a_thing_mapped_into_a_walls_margin_is_approached_from_the_nearest_floor():
    from robot.sim.world import NEARBY

    sim = SimRobot(Scene(10.0, 4.0, [], {}, walls=[((2.0, -2.0), (2.0, 2.0))]))
    bike = Object(1, "bike", (2.4, 0.0), 2.4, 0.0, "ahead", 0.2, "", 0.0)  # ranged through it, onto the wall
    goal, reach = sim._reach(bike, 0.0)
    assert sim._plan(goal, reach) is None
    path = sim._route(bike, goal, reach)
    assert path and reach < math.dist(path[-1], goal) <= reach + NEARBY
    far = Object(2, "bike", (4.5, 0.0), 4.5, 0.0, "ahead", 0.2, "", 0.0)  # well behind the wall: not this way
    assert sim._route(far, far.position, reach) is None
    with pytest.raises(PathBlocked):
        sim._go(far, 0.0)


def test_a_search_steps_back_from_a_likely_way_with_no_floor_to_see_the_thing_whole():
    real = object.__new__(RealRobot)
    SimRobot.__init__(real, Scene(10.0, 10.0, [], {}))
    real.search_log, asked, floor = [], [], {"front_camera": []}

    def spots(camera: str, **_: Any) -> tuple:
        asked.append(camera)
        return np.zeros((4, 4, 3), np.uint8), floor.get(camera, [(-2.0, 2.0)])

    real.floor_spots = spots
    real.brain = SimpleNamespace(yes_probability=lambda _: 0.95 if asked[-1] == "front_camera" else 0.07)
    stop = real._next_stop("blue bicycle", None, [])  # right in front: the bicycle itself leaves no floor that way
    assert isinstance(stop, Pose) and (round(stop.x, 2), round(stop.y, 2), stop.yaw) == (-1.0, 0.0, 0.0)
    assert real.search_log[-1].endswith("stepped back to see front whole")
    assert real._next_stop("blue bicycle", None, [(-1.0, 0.0)]) == (-2.0, 2.0)  # stepped back already: floor elsewhere
    assert real.search_log[-1].endswith("went rear-left")
    floor.update({"rear_left_camera": [], "rear_right_camera": []})
    assert real._next_stop("blue bicycle", None, [(-1.0, 0.0)]) is None  # nowhere new to go: the search ends
    floor["front_camera"] = [(1.5, 0.0), (3.0, 0.5)]
    assert real._next_stop("blue bicycle", None, []) == (3.0, 0.5)  # floor that way: the farthest, as before
