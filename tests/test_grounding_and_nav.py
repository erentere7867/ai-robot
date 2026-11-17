"""Tests for semantic grounding, user resolution, synonym fallback, and Nav2 tolerance / heading."""

from __future__ import annotations

import math
from typing import Any

import pytest

from robot.agent.api import NotFound, Object, PathBlocked, Pose
from robot.interfaces.live import PAGE
from robot.ros.real import NAMES
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


def test_exact_match_prioritized_over_synonyms():
    # Both table and desk exist: asking for table must yield table
    desk = Thing("desk", 2.0, 0.0, radius=0.4, id=1)
    table = Thing("table", -2.0, 0.0, radius=0.5, id=2)
    sim = SimRobot(_make_scene([desk, table]))

    assert sim._named("table").id == 2
    assert sim._named("desk").id == 1


# ---------------------------------------------------------------------------
# 2. RealRobot.NAMES bidirectional synonyms
# ---------------------------------------------------------------------------


def test_real_robot_names_aliases():
    assert "desk" in NAMES["table"]
    assert "table" in NAMES["desk"]
    assert "coffee table" in NAMES["desk"]
    assert "coffee table" in NAMES["table"]
    assert "couch" in NAMES["sofa"]
    assert "seat" in NAMES["chair"]
    assert "human" in NAMES["person"]
    assert "mug" in NAMES["cup"]
    assert "bag" in NAMES["backpack"]
    assert "trash can" in NAMES["bin"]


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


def test_ros_robot_go_retries_with_larger_distance():
    scene = _make_scene()
    robot = MockRosRobot(scene)
    target = Object(
        id=1,
        label="table",
        position=(3.0, 0.0),
        distance=3.0,
        bearing=0.0,
        side="ahead",
        radius=0.4,
        color="brown",
        last_seen=0.0,
    )

    robot.x = 0.0
    robot.y = 0.0
    # First attempt fails; second attempt (retry with standoff +0.25) succeeds
    robot.fail_nav_counts = 1
    robot._go(target, distance=0.4)
    # navigate_to_pose called twice (initial + retry)
    assert len(robot.nav_calls) == 2


def test_ros_robot_go_falls_back_to_a_star_cmd_vel_when_nav2_fails():
    table = Thing("table", 3.0, 0.0, radius=0.4, color="brown", id=1)
    scene = _make_scene([table])
    robot = MockRosRobot(scene)
    target = Object(
        id=1,
        label="table",
        position=(3.0, 0.0),
        distance=3.0,
        bearing=0.0,
        side="ahead",
        radius=0.4,
        color="brown",
        last_seen=0.0,
    )

    robot.x = 0.0
    robot.y = 0.0
    robot.yaw = 0.0
    # Both initial and retry fail Nav2
    robot.fail_nav_counts = 5
    robot._go(target, distance=0.5)

    desired_reach = 0.4 + max(0.5, 0.35)
    actual_dist = math.dist((3.0, 0.0), (robot.x, robot.y))
    assert actual_dist <= desired_reach + 0.1
    assert robot.nav_calls  # Nav2 was attempted first


def test_nav2_params_and_amcl_tuning(tmp_path):
    import yaml

    from robot.ros.stage import AMCL, _nav2_params

    assert AMCL["transform_tolerance"] == 1.0

    out_file = _nav2_params(tmp_path)
    assert out_file.exists()

    cfg = yaml.safe_load(out_file.read_text())
    assert cfg["amcl"]["ros__parameters"]["transform_tolerance"] == 1.0
    assert cfg["controller_server"]["ros__parameters"]["transform_tolerance"] == 1.0
    assert cfg["controller_server"]["ros__parameters"]["FollowPath"]["transform_tolerance"] == 1.0
    assert cfg["controller_server"]["ros__parameters"]["FollowPath"]["batch_size"] == 800
    assert cfg["local_costmap"]["local_costmap"]["ros__parameters"]["transform_tolerance"] == 1.0
    assert cfg["global_costmap"]["global_costmap"]["ros__parameters"]["transform_tolerance"] == 1.0
    assert cfg["behavior_server"]["ros__parameters"]["transform_tolerance"] == 1.0
    assert cfg["collision_monitor"]["ros__parameters"]["transform_tolerance"] == 1.0
    assert cfg["collision_monitor"]["ros__parameters"]["source_timeout"] == 5.0


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


def test_synthetic_scan_publisher_external_scan_skips_duplicate_raycasting():
    from types import SimpleNamespace
    from unittest.mock import patch

    import numpy as np

    from robot.ros.perceive import SyntheticScanPublisher

    pub = SyntheticScanPublisher(node=None, publish=False)
    assert not pub._external_scan

    raw = np.ones((60, 80), dtype=np.float32)
    stamp = SimpleNamespace(sec=10, nanosec=0)

    # When _external_scan is False, synthetic_scan runs
    dummy_scan = (np.zeros(360, dtype=np.float32), 0.0, 0.01)
    with patch("robot.ros.perceive.synthetic_scan", return_value=dummy_scan) as mock_scan:
        pub.ingest_depth("front_camera", stamp, raw)
        assert mock_scan.called
        assert pub.scan is not None

    # When _external_scan is True and publish is False, synthetic_scan is skipped
    pub._external_scan = True
    with patch("robot.ros.perceive.synthetic_scan") as mock_scan:
        pub.ingest_depth("front_camera", SimpleNamespace(sec=11, nanosec=0), raw)
        assert not mock_scan.called


def test_page_refresh_survives_preemption():
    import time
    from types import SimpleNamespace
    from unittest.mock import patch

    from robot.agent.api import Preempted
    from robot.interfaces.live import Live
    from robot.ros.run3d import page

    class FlakyRobot:
        def __init__(self):
            self.calls = 0
            self.perception = object()
            self.posture = "neutral"

        def _perceive_objects(self):
            self.calls += 1
            if self.calls == 1:
                raise Preempted("stopped during cancel")

    robot = FlakyRobot()
    holder = SimpleNamespace(sim=robot)

    with patch.object(Live, "serve"), patch("robot.ros.run3d.scan_points", return_value=[]):
        live = page(holder, port=9999)
        deadline = time.monotonic() + 2.0
        while robot.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.05)

        assert robot.calls >= 2, "refresh() thread died after Preempted exception!"

        # Also test camera() fallback catches Preempted / Exception
        with patch("robot.ros.run3d.annotated", side_effect=Preempted("cancelled")):
            assert live.camera("front_camera") is None


def test_real_robot_wall_bounds_rejection():
    from unittest.mock import MagicMock

    from robot.perception.tracker import Track
    from robot.ros.real import RealRobot

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

    from robot.ros.robot import _resize_rgb

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





