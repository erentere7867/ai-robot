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
        from robot.ros.motion import MotionGate

        super(RosRobot, self).__init__(scene)
        self.nav_calls: list[tuple[str, dict[str, Any]]] = []
        self.rotate_calls: list[float] = []
        self.fail_nav_counts: int = 0
        self.sync_calls: int = 0
        self._goal = None
        self._publish = lambda *_: None
        self._motion = MotionGate(lambda: round(self.t * 1e9), lambda *_: None,
                                  lambda v, w, _: self._publish(v, w), 0.35, 1.8)

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
    from robot.agent.api import SafetyStop

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

    def step(goal):
        if when == "before acceptance":
            raise Preempted("stop")
        robot.things.pop(target.id)  # perception withdrew the ID after Nav2 accepted it

    robot._nav_tick = step
    if when == "while driving":
        sent.set_result(handle)
        with pytest.raises(TargetLost, match="stopped approaching"):
            RosRobot._act(robot, "navigate_to_pose", watch=target)
        assert velocities[-1] == (0.0, 0.0) and robot._goal is None
    else:
        with pytest.raises(Preempted):
            RosRobot._act(robot, "navigate_to_pose", watch=target)
        assert robot._motion.owner is None and velocities[-1] == (0.0, 0.0)
        with pytest.raises(SafetyStop, match="previous motion goal"):
            RosRobot._act(robot, "navigate_to_pose")
        sent.set_result(handle)
        robot._motion_available()
    assert cancelled == [True]


def test_a_nav2_goal_that_gets_no_nearer_is_given_up_and_cancelled():
    from robot.ros.robot import PROGRESS_S

    robot = MockRosRobot(_make_scene())
    sent, result, cancelled = Future(), Future(), []

    def cancel():
        cancelled.append(robot.t)
        result.set_result(SimpleNamespace(status=5))

    sent.set_result(SimpleNamespace(accepted=True, cancel_goal_async=cancel, get_result_async=lambda: result))
    robot.actions = {"navigate_to_pose": SimpleNamespace(wait_for_server=lambda **_: True,
                                                        send_goal_async=lambda goal: sent)}
    robot.kinds = {"navigate_to_pose": SimpleNamespace(Goal=SimpleNamespace)}
    robot._publish = lambda v, w: None

    def step(goal):  # Nav2 backs up, spins and replans: a second passes and the robot is no nearer
        robot.t += 1.0
        robot.x = 0.3 * math.sin(robot.t)

    robot._nav_tick = step
    goal = SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=3.0, y=0.0)))
    with pytest.raises(PathBlocked, match="no nearer"):
        RosRobot._act(robot, "navigate_to_pose", pose=goal)
    assert len(cancelled) == 1 and PROGRESS_S < cancelled[0] <= PROGRESS_S + 2


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
    cm = cfg["collision_monitor"]["ros__parameters"]
    assert (cm["cmd_vel_in_topic"], cm["cmd_vel_out_topic"]) == (bringup.SELECTED, bringup.CHECKED)
    assert cm["enable_stamped_cmd_vel"] and cm["source_timeout"] == 1.0
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
    real.search_log, asked, floor, real._stepped_back = [], [], {"front_camera": []}, False

    def spots(camera: str, **_: Any) -> tuple:
        asked.append(camera)
        return np.zeros((4, 4, 3), np.uint8), floor.get(camera, [(-2.0, 2.0)])

    real.floor_spots = spots
    odds = {"front_camera": 0.95}
    real.brain = SimpleNamespace(yes_probability=lambda _: odds.get(asked[-1], 0.07))
    stop = real._next_stop("blue bicycle", None, [])  # right in front: the bicycle itself leaves no floor that way
    assert isinstance(stop, Pose) and (round(stop.x, 2), round(stop.y, 2), stop.yaw) == (-1.0, 0.0, 0.0)
    assert real.search_log[-1].endswith("stepped back to see front whole")
    assert real._next_stop("blue bicycle", None, []) == (-2.0, 2.0)  # once a search: no second step back
    real._stepped_back = False
    odds["front_camera"] = 0.3  # likeliest, but unlikely: nothing worth backing off to see
    assert real._next_stop("blue bicycle", None, []) == (-2.0, 2.0) and real.search_log[-1].endswith("went rear-left")
    odds["front_camera"] = 0.95
    assert real._next_stop("blue bicycle", None, [(-1.0, 0.0)]) == (-2.0, 2.0)  # stepped back already: floor elsewhere
    assert real.search_log[-1].endswith("went rear-left")
    floor.update({"rear_left_camera": [], "rear_right_camera": []})
    assert real._next_stop("blue bicycle", None, [(-1.0, 0.0)]) is None  # nowhere new to go: the search ends
    floor["front_camera"] = [(1.5, 0.0), (3.0, 0.5)]
    assert real._next_stop("blue bicycle", None, []) == (3.0, 0.5)  # floor that way: the farthest, as before


@pytest.mark.parametrize(("seen_ago", "answer", "confirmed"), [(0.4, None, True), (6.6, 0.9, True), (6.6, 0.1, False)])
def test_an_arrival_is_confirmed_by_the_detector_or_the_vision_model_on_the_things_place(seen_ago, answer, confirmed):
    from robot.perception.rig import RIG

    real = object.__new__(RealRobot)
    cam = RIG["front_camera"]
    real.perception = SimpleNamespace(frames={"front_camera": (0, np.zeros((cam.height, cam.width, 3), np.uint8))})
    real._pose_at = lambda stamp: (0.0, 0.0, 0.0)
    asked = []

    def yes_probability(messages: list) -> float:
        asked.append(messages[0]["content"][-1]["text"])
        return answer

    real.brain = SimpleNamespace(yes_probability=yes_probability)
    beanbag = Object(3, "beanbag", (1.0, 0.0), 1.0, 0.0, "ahead", 0.5, "grey", seen_ago)  # parked facing it, 1 m off
    assert real.sees_now(beanbag) is confirmed
    assert asked == ([] if answer is None else ["Is there a beanbag in the middle of this picture? Answer yes or no."])


def test_detect_falls_back_to_vision_model_places_visual_track_and_sees_now_confirms_or_drops():
    from collections import Counter, deque

    from robot.perception.rig import RIG

    real = object.__new__(RealRobot)
    SimRobot.__init__(real, _make_scene())
    real._ready = True
    real.store, real._cleared_tracks, real._habit_recorded, real._rooms, real._named_at = None, set(), {}, [], 0.0
    real.speaker_profile, real._user = None, None
    tracker = Tracker()
    cam = RIG["front_camera"]
    rgb = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
    rgb[80:320, 480:780] = 255
    depth = np.full((360, 640), 2.0, dtype=np.float32)
    stamp = SimpleNamespace(sec=1, nanosec=0)
    looks = {n: Counter() for n in RIG}
    real.perception = SimpleNamespace(
        lock=Lock(), tracker=tracker, crops={}, looks=looks, scans=deque(), scan=None,
        frames={"front_camera": (stamp, rgb)},
        scan_bridge=SimpleNamespace(cam_depths={"front_camera": deque([(1_000_000_000, depth)])}),
        detector=SimpleNamespace(watch=lambda w: None),
        objects=lambda: [t for t in tracker.objects()],
    )
    real._pose_at = lambda s: (0.0, 0.0, 0.0)

    def step(v: float, w: float) -> None:
        for n in RIG:
            looks[n]["computer monitor"] += 1

    real._step = step
    # Vision model says yes wherever the white patch is present
    import base64

    import cv2

    def yes_prob(messages):
        url = messages[0]["content"][0]["image_url"]["url"]
        arr = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
        return min(0.98, 0.05 + 12.0 * float((arr > 200).mean()))

    real.brain = SimpleNamespace(yes_probability=yes_prob)
    found = real.detect("computer monitor")
    assert len(found) == 1 and found[0].label == "computer monitor" and 1.7 <= found[0].distance <= 2.3
    tid = found[0].id
    assert tracker.tracks[tid].visual is True and tid in real.perception.crops

    # Second call updates the existing track instead of duplicating it
    found2 = real.detect("computer monitor")
    assert len(found2) == 1 and found2[0].id == tid and len(tracker.tracks) == 1

    # On arrival (last_seen > SEEN_NOW), visual target confirms against the 768px front picture
    stale = Object(tid, "computer monitor", found[0].position, found[0].distance, 0.0, "ahead", 0.25, "", 5.0)
    assert real.sees_now(stale) is True and tracker.tracks[tid].present is True

    # If the picture no longer shows it up close, sees_now rejects and withdraws the visual track
    real.perception.frames["front_camera"] = (stamp, np.zeros_like(rgb))
    assert real.sees_now(stale) is False and tracker.tracks[tid].present is False


    # Elevated visual target occluded nose-close (e.g. monitor on desk) confirms via closer approach snapshot
    tracker.tracks[tid].present = True
    tracker.tracks[tid].z = 0.85
    tracker.tracks[tid].spotted_from = (0.0, 0.0)
    real._trail_views = deque([(tracker.tracks[tid].first_seen + 1.0, (0.4, 0.0, 0.0),
                                {"front_camera": (stamp, rgb, depth, (0.4, 0.0, 0.0))})])
    assert real.sees_now(stale) is True and tracker.tracks[tid].present is True

    # Mid-leg corridor search recovers a target passed between stops from _trail_views
    real._searching, real._search_snap_after = True, 0.0
    real.perception.frames["front_camera"] = (stamp, np.zeros_like(rgb))
    real._trail_views = deque([(1.0, (1.0, 0.0, 0.0),
                                {"front_camera": (stamp, rgb, depth, (1.0, 0.0, 0.0))})])
    passed = real._see_target("umbrella")
    assert passed is not None and passed.label == "umbrella" and passed.x > 2.5

    # Visible-side approach: when spotted from (1.0, 0.0) and now around the side at (2.8, 1.0), return toward spotted
    umb = Object(passed.id, "umbrella", (passed.x, passed.y), 1.0, 0.0, "ahead", 0.2, "", 0.0)
    tracker.tracks[passed.id].spotted_from = (passed.x - 0.8, passed.y)
    real.x, real.y, real.yaw = passed.x, passed.y - 1.0, 0.0
    gp = real._goal_point(umb, 0.0)
    assert gp is not None and math.dist(gp[0], (passed.x - 0.8, passed.y)) < 0.2


def test_floor_flat_visual_object_drives_onto_free_floor():
    sim = SimRobot(_make_scene())
    rug = Object(1, "rug", (2.0, 0.0), 2.0, 0.0, "ahead", 0.0, "", 0.0)
    _, reach_rug = sim._reach(rug, 0.0)
    assert reach_rug == 0.2
    chair = Object(2, "chair", (2.0, 0.0), 2.0, 0.0, "ahead", 0.2, "", 0.0)
    _, reach_chair = sim._reach(chair, 0.0)
    assert reach_chair > 0.6


def test_search_uses_unseen_map_frontiers_instead_of_zigzagging_when_no_camera_has_a_lead():
    # Two-room scene with an interior dividing wall so the far room is genuinely uncovered until visited.
    scene = Scene(12.0, 6.0, [], {}, walls=[((0.0, -3.0), (0.0, 1.0))], start=Pose(-3.0, 0.0, 0.0))
    real = object.__new__(RealRobot)
    SimRobot.__init__(real, scene)
    real.search_log, real._stepped_back, real._searching = [], False, True
    real.coverage = np.zeros_like(real.occupied)
    real._cover()
    real.floor_spots = lambda camera, **_: (np.zeros((4, 4, 3), np.uint8), [(-1.5, 0.0)])  # already-covered spot
    real.brain = SimpleNamespace(yes_probability=lambda _: 0.15)  # no visual lead in any camera
    stop = real._next_stop("umbrella", None, [])
    assert stop is not None and stop[0] > -3.0 and stop != (-1.5, 0.0)
    assert math.dist(stop, (-3.0, 0.0)) <= 3.3  # capped to a <=3.2 m path hop along the way around the wall
    assert real.search_log[-1].endswith("explored unseen area")


def test_search_toward_and_lean_stationary_look():
    from robot.agent.agent import Agent
    from robot.agent.tools import Tools
    from robot.mind.brain import Reply, ToolCall

    scene = Scene(10.0, 6.0, [Thing("mug", 3.5, 0.0, radius=0.1)], {}, start=Pose(-3.0, 0.0, 0.0))
    sim = SimRobot(scene)
    tools = Tools(sim)
    out, ok = tools.run("search", {"what": "mug", "area": "everywhere", "toward": "front"})
    assert ok and "Found candidate mug" in out and sim.x > -1.0

    # Stationary look(what=...) returns only matching detections + pictures without re-dumping full scene text.
    seen_msgs = []
    replies = [
        Reply("", tool_calls=[ToolCall("c1", "look", {"direction": "front", "what": "mug"})]),
        Reply("I see the mug."),
    ]
    brain = SimpleNamespace(chat=lambda m, *a, **kw: (seen_msgs.append(list(m)), replies.pop(0))[1])
    Agent(brain, cameras="front").do("Check the mug.", sim)
    tool_msg = [m for m in seen_msgs[1] if m["role"] == "tool"][-1]["content"]
    assert isinstance(tool_msg, list)
    assert "Seen: mug" in tool_msg[0]["text"]
    assert not any("What you sense now:" in p.get("text", "") for p in tool_msg if isinstance(p, dict))

    # After 3 stationary look(..., what=...) calls, "look" is hidden from schemas until the robot moves.
    schema_names: list[list[str]] = []
    replies2 = [
        Reply("", tool_calls=[ToolCall("l1", "look", {"direction": "front", "what": "mug"})]),
        Reply("", tool_calls=[ToolCall("l2", "look", {"direction": "rear-left", "what": "mug"})]),
        Reply("", tool_calls=[ToolCall("l3", "look", {"direction": "rear-right", "what": "mug"})]),
        Reply("Done."),
    ]
    brain2 = SimpleNamespace(
        chat=lambda m, schemas, **kw: (schema_names.append([s["function"]["name"] for s in schemas]), replies2.pop(0))[1]
    )
    Agent(brain2, cameras="front").do("Find the mug.", sim)
    assert "look" in schema_names[0] and "look" in schema_names[2]
    assert "look" not in schema_names[3] and "look_back" not in schema_names[3]


def test_small_object_depth_refinement_and_arrived_reached_wording():
    from robot.agent.tools import Tools

    # RosRobot._check_goal tolerates <=0.6 m same-label depth drift on small objects (r=0.06 m), rejects >0.6 m or rename
    robot = MockRosRobot(_make_scene())
    shoe = Object(20, "shoes", (2.28, 4.39), 1.5, 0.0, "ahead", 0.06, "black", 0.0)
    robot.things[20] = Thing("shoes", 2.50, 4.55, radius=0.06)  # 0.27 m drift (> 2*r = 0.12 m, <= 0.6 m gate)
    robot.things[20].id = 20
    robot.memory[20] = ((2.50, 4.55), 0.0)
    RosRobot._check_goal(robot, shoe)  # must not raise
    robot.memory[20] = ((3.10, 4.55), 0.0)  # 0.84 m drift -> TargetLost
    with pytest.raises(TargetLost, match="mapped footprint"):
        RosRobot._check_goal(robot, shoe)

    # Tools._arrived says "Facing" when stopped farther away (>0.6 m gap), "Reached" within normal stopping gap
    sim = SimRobot(Scene(10.0, 6.0, [Thing("umbrella", 3.0, 0.0, radius=0.15)], {}, start=Pose(0.0, 0.0, 0.0)))
    tools = Tools(sim)
    out_far, ok = tools.run("stop_near", {"target": "umbrella", "metres": 1.2})
    assert ok and "Facing umbrella" in out_far and "Reached" not in out_far
    out_near, ok = tools.run("go_to", {"target": "umbrella"})
    assert ok and "Reached umbrella" in out_near


def test_floor_sees_now_and_spotted_standpoint_pull_in():
    import threading

    from robot.perception.tracker import Track, Tracker

    real = object.__new__(RealRobot)
    SimRobot.__init__(real, _make_scene())
    tracker = Tracker()
    tracker.tracks[1] = Track(1, "big rug", 0.6, 0.0, 0.0, 0.0, 0.0, hits=2, verified=True, sure=True, visual=True)
    rgb = np.zeros((432, 768, 3), dtype=np.uint8)
    stamp = SimpleNamespace(sec=1, nanosec=0)
    real.perception = SimpleNamespace(lock=threading.Lock(), tracker=tracker, frames={"front_camera": (stamp, rgb)})
    real._pose_at = lambda _: (0.0, 0.0, 0.0)
    orig_calls = []

    def brain_yes(msgs):
        q = msgs[0]["content"][1]["text"]
        orig_calls.append(q)
        if "in the middle of this picture" in q:
            return 0.1 if len(orig_calls) == 1 else 0.9  # 1st: narrow crop fails; 3rd: floor half succeeds
        return 0.47  # 2nd: whole 768px frame fails (cabinet dominates top half)

    real.brain = SimpleNamespace(yes_probability=brain_yes)
    rug = Object(1, "big rug", (0.6, 0.0), 0.6, 0.0, "ahead", 0.0, "", 10.0)
    assert real.sees_now(rug) is True and tracker.tracks[1].present is True

    # RealRobot._go steps straight onto _spotted_standpoint when Nav2 stops 0.25 m short
    tracker.tracks[2] = Track(2, "umbrella", 2.0, 0.0, 0.15, 0.0, 0.0, hits=2, verified=True, sure=True, visual=True,
                              spotted_from=(1.3, 0.0))
    umb = Object(2, "umbrella", (2.0, 0.0), 0.8, 0.0, "ahead", 0.15, "", 0.0)
    cand = real._spotted_standpoint(umb, 0.0)
    assert cand is not None
    real.x, real.y, real.yaw = cand[0] - 0.25, cand[1], 0.0
    moved: list[float] = []
    real._rotate = lambda a: setattr(real, "yaw", real.yaw + a)
    real._move = lambda d: (moved.append(d), setattr(real, "x", real.x + d * math.cos(real.yaw)))[0]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RosRobot, "_go", lambda self, *a, **kw: None)
        real._go(umb, 0.0)
    assert len(moved) == 1 and 0.20 <= moved[0] <= 0.30

    # RealRobot.obj refreshes self.things when perception merges track 15 -> 11 between _perceive_objects() syncs
    t11 = Track(11, "shoes", 2.0, 0.0, 0.1, 0.0, 0.0, hits=1, verified=False, sure=False)
    t15 = Track(15, "shoes", 2.05, 0.0, 0.1, 0.0, 0.0, hits=2, verified=True, sure=True, visual=True)
    tracker.tracks[11], tracker.tracks[15] = t11, t15
    tracker._merge([])
    assert tracker.current(15) == 11 and t11.verified and t11.sure and t11.visual and t11.confirmed
    synced = []
    real.things.pop(11, None)
    real._perceive_objects = lambda: (synced.append(1), real.things.__setitem__(11, Thing("shoes", 2.0, 0.0, 0.1, id=11)),
                                      real.memory.__setitem__(11, ((2.0, 0.0), 0.0)))
    got = real.obj(15)
    assert got.id == 11 and synced == [1]


def test_says_no_contractions_and_search_near_direction_or_here():
    from robot.agent.tools import Tools
    from robot.eval.home3d import says_no

    assert says_no("I'm sorry, but I don't see any elephant here.")
    assert says_no("I don\u2019t see one.")
    assert not says_no("Yes, I don't think it moved.")

    scene = Scene(10.0, 6.0, [Thing("mug", 3.5, 0.0, radius=0.1)], {}, start=Pose(-3.0, 0.0, 0.0))
    sim = SimRobot(scene)
    seen_detects: list[str] = []
    orig_detect = sim.detect
    sim.detect = lambda desc: (seen_detects.append(desc), orig_detect(desc))[1]
    tools = Tools(sim)
    out, ok = tools.run("search", {"what": "mug", "near": "front"})
    assert ok and "Found candidate mug" in out and "front" not in seen_detects
    seen_detects.clear()
    tools.run("search", {"what": "mug", "near": "here"})
    assert "here" not in seen_detects


def test_sync_vlm_vetting_cluster_pick_and_semantic_places(tmp_path):
    from threading import Lock

    from robot.agent.agent import Agent
    from robot.agent.tools import ToolError, Tools
    from robot.ros.explore import describe_place, load_places, save_places

    # 1. Synchronous VLM vetting in RealRobot._matching rejects false YOLOE tracks and preserves synonym tracks
    real = object.__new__(RealRobot)
    tracker = Tracker()
    cloth = Track(1, "shoes", 1.5, 0.0, 0.15, 0.0, 0.0, hits=2, verified=False, sure=False)
    chair = Track(2, "chair", 2.0, 0.0, 0.3, 0.0, 0.0, hits=2, verified=False, sure=False)
    table = Track(3, "table", 2.5, 0.0, 0.4, 0.0, 0.0, hits=2, verified=True, sure=True, named_from=400)
    tracker.tracks = {1: cloth, 2: chair, 3: table}
    crop = np.zeros((20, 20, 3), dtype=np.uint8)
    real.perception = SimpleNamespace(lock=Lock(), tracker=tracker, crops={1: crop, 2: crop, 3: crop})
    asked: list[str] = []

    def yes_prob(msgs):
        q = msgs[0]["content"][-1]["text"]
        asked.append(q)
        if "chair" in q:
            return 0.9
        return 0.05  # cloth is not shoes, coffee table is not desk

    real.brain = SimpleNamespace(yes_probability=yes_prob)
    assert real._matching("shoes", [Thing("shoes", 1.5, 0.0, 0.15, id=1)]) == []
    assert cloth.rejected is True
    assert len(real._matching("chair", [Thing("chair", 2.0, 0.0, 0.3, id=2)])) == 1
    assert chair.sure is True and chair.named_from == 400
    before_calls = len(asked)
    assert len(real._matching("chair", [Thing("chair", 2.0, 0.0, 0.3, id=2)])) == 1
    assert len(asked) == before_calls  # cached sure=True until crop grows 2x
    assert real._matching("desk", [Thing("table", 2.5, 0.0, 0.4, id=3)]) == []
    assert table.rejected is False  # coffee table stays valid as a table

    # 2. Co-located same-label targets (<= 0.7 m) auto-pick the nearest; separated ones (> 0.7 m) still ask
    pair = SimRobot(Scene(8.0, 6.0, [Thing("shoes", 1.5, 0.0, 0.1, id=1), Thing("shoes", 1.8, 0.1, 0.1, id=2)],
                          {}, start=Pose(0.0, 0.0, 0.0)))
    pair.detect("shoes")
    assert Tools(pair)._resolve("shoes").id == 1
    apart = SimRobot(Scene(8.0, 6.0, [Thing("shoes", 1.5, 0.0, 0.1, id=1), Thing("shoes", 1.5, 2.0, 0.1, id=2)],
                           {}, start=Pose(0.0, 0.0, 0.0)))
    apart.detect("shoes")
    with pytest.raises(ToolError, match="could be any of"):
        Tools(apart)._resolve("shoes")

    # 3. Semantic place captions, persistence, prompt ordering, fuzzy search(near=...), and out-of-view go_to hint
    fake_brain = SimpleNamespace(describe=lambda _u, _p: SimpleNamespace(text="Kitchen: cabinets, counter (bright)."))
    cap = describe_place(fake_brain, {"front_camera": np.zeros((180, 320, 3), dtype=np.uint8)})
    assert cap == "kitchen: cabinets, counter"
    places_path = tmp_path / "home.places.json"
    save_places(places_path, {
        "kitchen: cabinets, counter": Pose(3.0, 0.0, 0.0),
        "hallway: doorway, wooden door": Pose(-3.0, 0.0, 180.0),
    })
    loaded = load_places(places_path)
    assert set(loaded) == {"kitchen: cabinets, counter", "hallway: doorway, wooden door"}

    sim = SimRobot(Scene(10.0, 6.0, [Thing("mug", -3.2, 0.0, 0.1)], {}, start=Pose(0.0, 0.0, 0.0)))
    sim.places.update(loaded)
    sim.places["where you were before"] = Pose(1.0, 0.0, 0.0)
    txt = Agent._places_text(sim, sim.pose())
    assert txt.startswith("Places you can go_to by name (or search near): where you were before")
    assert "kitchen: cabinets, counter" in txt and "hallway: doorway, wooden door" in txt

    tools = Tools(sim)
    out, ok = tools.run("search", {"what": "mug", "near": "hallway"})
    assert ok and "Found candidate mug" in out and sim.x < -1.0

    # Out-of-view go_to("kitchen") resolves to mapped "kitchen: cabinets, counter" and drives there without blind search
    out2, ok2 = tools.run("go_to", {"target": "kitchen"})
    assert ok2 and "Arrived at kitchen: cabinets, counter" in out2 and sim.x > 2.0

    # 4. Unvisited mapped areas hint + Agent blocks repeating identical search() after moving
    from robot.mind.brain import Reply, ToolCall

    sim2 = SimRobot(Scene(10.0, 6.0, [], {}, start=Pose(0.0, 0.0, 0.0)))
    sim2.places["far hall: doors"] = Pose(4.5, 0.0, 0.0)
    sim2.search = lambda *a, **kw: None
    t2 = Tools(sim2)
    miss, ok3 = t2.run("search", {"what": "umbrella"})
    assert ok3 and "Areas not visited closely yet: far hall: doors." in miss

    calls = iter([
        Reply("", tool_calls=[ToolCall("1", "search", {"what": "umbrella"})]),
        Reply("", tool_calls=[ToolCall("2", "search", {"what": "umbrella"})]),
        Reply("No umbrella found."),
    ])
    fake_llm = SimpleNamespace(chat=lambda *_a, **_kw: next(calls))
    rec = Agent(fake_llm, cameras="none").do("Find the umbrella.", sim2, speak=False)
    assert rec.steps[1].calls[0][3] is False and "made this exact call" in rec.steps[1].calls[0][2]


def test_color_qualified_sees_now_and_visual_target_color_check():
    from collections import deque

    from robot.perception.rig import RIG

    real = object.__new__(RealRobot)
    SimRobot.__init__(real, _make_scene())
    real._ready = False
    tracker = Tracker()
    cam = RIG["front_camera"]
    rgb = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
    stamp = SimpleNamespace(sec=1, nanosec=0)
    real.perception = SimpleNamespace(
        lock=Lock(), tracker=tracker, frames={"front_camera": (stamp, rgb)},
        scans=deque(), scan=None, crops={},
        scan_bridge=SimpleNamespace(cam_depths={"front_camera": deque([(1_000_000_000, np.full((360, 640), 2.0, np.float32))])}),
    )
    real._pose_at = lambda _s: (0.0, 0.0, 0.0)

    # 1. Visual track "red bike" with last_seen=0.0 must still verify post-arrival with "red bike" (not bare "bike")
    t3 = Track(3, "bike", 0.8, 0.0, 0.19, 0.0, 0.0, hits=2, verified=True, sure=True, visual=True, visual_color="red")
    tracker.tracks[3] = t3
    asked: list[str] = []

    def yes_prob(msgs):
        q = msgs[0]["content"][-1]["text"]
        asked.append(q)
        return 0.85 if ("bike" in q and "red" not in q) else 0.10

    real.brain = SimpleNamespace(yes_probability=yes_prob)
    bike3 = Object(3, "bike", (0.8, 0.0), 0.8, 0.0, "ahead", 0.19, "red", 0.1)
    assert real.sees_now(bike3) is False and t3.present is False
    assert all("red bike" in q for q in asked)

    # 2. Non-visual track parked close up with wrong colour clears visual_color without dropping a real bike
    t4 = Track(4, "bike", 0.8, 0.0, 0.25, 0.0, 0.0, hits=4, verified=True, sure=True, visual=False,
               visual_color="red", color_from=500)
    tracker.tracks[4] = t4
    bike4 = Object(4, "bike", (0.8, 0.0), 0.8, 0.0, "ahead", 0.25, "red", 5.0)
    assert real.sees_now(bike4) is False and t4.visual_color == "" and t4.present is True

    # 3. _see_target("red bike") rejects a crop when _ask_color says "black"
    rgb_bright = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
    rgb_bright[80:320, 480:780] = 255
    real.perception.frames["front_camera"] = (stamp, rgb_bright)
    real.brain = SimpleNamespace(
        yes_probability=lambda _m: 0.9,
        decide=lambda _m, choices: {c: (0.98 if c == "black" else 0.0) for c in choices},
    )
    assert real._see_target("red bike") is None


def test_location_phrases_rejected_front_feedback_after_move_and_detector_clear():
    from robot.agent.agent import Agent
    from robot.agent.tools import Tools
    from robot.mind.brain import Reply, ToolCall
    from robot.perception.detector import Detector

    # 1. Location phrases ("chair in living room") and object IDs ("bike 7") never leak into YOLOE vocabulary
    sim = SimRobot(Scene(8.0, 6.0, [Thing("chair", 2.0, 0.0, 0.3)], {}, start=Pose(0.0, 0.0, 0.0)))
    vocab_before = sim.vocabulary.copy()
    tools = Tools(sim)
    for name, args in (
        ("go_to", {"target": "chair in living room"}),
        ("search", {"what": "chair in living room"}),
        ("look", {"direction": "front", "what": "cup on table"}),
        ("search", {"what": "bike 7"}),
        ("look", {"direction": "front", "what": "25 CHAIR"}),
    ):
        msg, ok = tools.run(name, args)
        assert not ok and ("search(" in msg or "object ID" in msg)
    assert sim.vocabulary == vocab_before

    # 2. Default see_after_moves="front" attaches 1 fresh front_camera frame after open-loop moves without full scene dump
    seen_turns = []
    replies = [
        Reply("", tool_calls=[ToolCall("m1", "go_toward", {"direction": "front", "metres": 0.5})]),
        Reply("Done."),
    ]
    fake_llm = SimpleNamespace(chat=lambda m, *a, **kw: (seen_turns.append(list(m)), replies.pop(0))[1])
    Agent(fake_llm, cameras="all").do("Step forward.", sim, speak=False)
    tool_content = [m for m in seen_turns[1] if m["role"] == "tool"][-1]["content"]
    assert isinstance(tool_content, list)
    assert sum(p.get("type") == "image_url" for p in tool_content) == 1
    assert any(p.get("text") == "front camera:" for p in tool_content)
    assert not any("What you sense now:" in p.get("text", "") for p in tool_content)

    # 3. Detector.clear() drops previous request's asked words while preserving `always`
    det = Detector(server=SimpleNamespace(slots=8), always=("person",))
    det.watch("shoes", "umbrella")
    assert det.vocabulary == ["person", "shoes", "umbrella"]
    det.clear()
    assert det.vocabulary == ["person"]

    # 4. _trim_images removes evicted images along with their captions (no '(older picture removed)' text mash)
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "What you sense now:"},
        {"type": "text", "text": "front camera:"}, {"type": "image_url", "image_url": {"url": "u1"}},
        {"type": "text", "text": "rear-left camera:"}, {"type": "image_url", "image_url": {"url": "u2"}},
        {"type": "text", "text": "rear-right camera:"}, {"type": "image_url", "image_url": {"url": "u3"}},
        {"type": "text", "text": "\nThe user says: \"hi\""},
    ]}, {"role": "tool", "content": [
        {"type": "text", "text": "Moved."},
        {"type": "text", "text": "front camera:"}, {"type": "image_url", "image_url": {"url": "u4"}},
    ]}]
    Agent._trim_images(msgs)
    texts = [p.get("text", "") for p in msgs[0]["content"] if p.get("type") == "text"]
    assert "front camera:" not in texts and not any("older picture removed" in t for t in texts)
    assert sum(p.get("type") == "image_url" for m in msgs for p in m["content"]) == 3

    # 5. render_annotated_view hides absent tracks (present=False) so dead IDs don't lure go_to retries
    from robot.ros.perceive import Perception

    perc = object.__new__(Perception)
    perc.lock = Lock()
    perc.tracker = Tracker()
    perc.tracker.tracks[25] = Track(25, "chair", 2.0, 0.0, 0.3, 0.0, 0.0, hits=2, present=False)
    perc.detected = {"front_camera": (10.0, np.zeros((360, 640, 3), dtype=np.uint8))}
    perc.boxes = {"front_camera": [((100, 100, 200, 200), 25)]}
    perc.frames = {}
    out_absent = perc.render_annotated_view("front_camera", 640, 360, 5.0)
    assert out_absent is not None and int(out_absent.sum()) == 0
    perc.tracker.tracks[25].present = True
    out_present = perc.render_annotated_view("front_camera", 640, 360, 5.0)
    assert out_present is not None and int(out_present.sum()) > 0
