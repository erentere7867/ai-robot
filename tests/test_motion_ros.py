"""Run a real Nav2 motion and stop against synthetic sensors, without a robot or models."""

import math
import os
import signal
import subprocess
import threading
import time

import pytest

from robot.agent.api import Pose, Preempted
from robot.ros import bringup
from robot.ros.motion import CHECKED, NAV, WHEELS
from robot.ros.robot import RosRobot
from robot.sim.habitat_ros import Hardware
from robot.sim.world import Scene


def wait_for(f, why, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if f():
            return
        time.sleep(0.01)
    raise AssertionError(why)


@pytest.mark.skipif(os.environ.get("RUN_ROS_MOTION_SMOKE") != "1", reason="requires an isolated ROS 2/Nav2 graph")
def test_live_nav2_motion_handoff(tmp_path):
    import rclpy
    from geometry_msgs.msg import Point, TransformStamped, TwistStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    from tf2_ros import StaticTransformBroadcaster

    assert os.environ.get("ROS_DOMAIN_ID", "0") != "0", "use a separate ROS domain"
    rclpy.init()
    node = rclpy.create_node("motion_smoke_driver")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    transforms = []
    for parent, child in [("map", "odom"), ("odom", "base_link"), ("base_link", "base_footprint")]:
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = parent, child
        tf.transform.rotation.w = 1.0
        transforms.append(tf)
    broadcaster = StaticTransformBroadcaster(node)
    broadcaster.sendTransform(transforms)
    clock = node.create_publisher(Clock, "/clock", 10)
    scan = node.create_publisher(LaserScan, "/scan", 10)
    maps = node.create_publisher(OccupancyGrid, "/map", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    grid = OccupancyGrid()
    grid.header.frame_id = "map"
    grid.info.width = grid.info.height = 100
    grid.info.resolution = 0.1
    grid.info.origin.position.x = grid.info.origin.position.y = -5.0
    grid.info.origin.orientation.w = 1.0
    grid.data = [0] * 10000
    maps.publish(grid)
    fake_base = Hardware.__new__(Hardware)
    fake_base.cmd, fake_base.t = (0, 0.0, 0.0), 10.0
    nav, checked, wheels = [], [], []
    node.create_subscription(TwistStamped, NAV, lambda m: nav.append(RosRobot._command(m)), 10)
    node.create_subscription(TwistStamped, CHECKED, lambda m: checked.append(RosRobot._command(m)), 10)

    def receive(msg):
        fake_base._command(msg)
        wheels.append((RosRobot._command(msg), fake_base._want()))

    node.create_subscription(TwistStamped, WHEELS, receive, 10)

    def tick():
        fake_base.t = round(fake_base.t + 0.01, 2)
        tick = Clock()
        tick.clock = fake_base._stamp()
        clock.publish(tick)
        if tick.clock.nanosec % 50_000_000 == 0:
            msg = LaserScan()
            msg.header.frame_id = "base_footprint"
            msg.header.stamp = tick.clock
            msg.angle_min, msg.angle_max, msg.angle_increment = -math.pi, math.pi, math.pi / 90
            msg.range_min, msg.range_max = 0.05, 4.0
            msg.ranges = [3.9] * 180
            scan.publish(msg)

    node.create_timer(0.01, tick)
    threading.Thread(target=executor.spin, daemon=True).start()
    with (tmp_path / "nav2.log").open("w") as log:
        params = bringup.params(tmp_path)
        launch = subprocess.Popen(
            ["ros2", "launch", "nav2_bringup", "bringup_launch.py", "use_sim_time:=True",
             "use_localization:=False", "use_keepout_zones:=False", "use_speed_zones:=False",
             f"params_file:={params}", "autostart:=True", "use_composition:=True"],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        robot = None
        try:
            wait_for(lambda: node.count_publishers(NAV) >= 2, "Nav2 nodes did not remap their command output")
            robot = RosRobot(Scene(10, 10, [], {}, start=Pose(0, 0, 0)), synthetic_scan=False)
            assert [pub.node_name for pub in node.get_publishers_info_by_topic(WHEELS)] == ["sprocket_api"]
            errors = []

            def drive():
                try:
                    robot._act("drive_on_heading", target=Point(x=1.0), speed=0.20, time_allowance=robot._allow(20))
                except BaseException as exc:
                    errors.append(exc)

            action = threading.Thread(target=drive, daemon=True)
            action.start()
            wait_for(lambda: any(v > 0 for (_, _, _), (v, _) in wheels), "no checked Nav2 motion reached the base")
            moving = next(raw for raw, want in wheels if want[0] > 0)
            wait_for(lambda: moving in checked and moving in nav, "wheel command lost its original stamp")
            robot.stop()
            wait_for(lambda: bool(errors), "Nav2 motion did not preempt")
            assert isinstance(errors[0], Preempted), errors
            wait_for(lambda: wheels[-1][1] == (0.0, 0.0), "wheels did not receive stop")
            tail = len(wheels)
            checked_before = checked.count(moving)
            stale = node.create_publisher(TwistStamped, CHECKED, 1)
            wait_for(lambda: stale.get_subscription_count() > 0, "checked topic missing")
            stale.publish(robot._twist_msg(*moving))
            wait_for(lambda: checked.count(moving) > checked_before, "the late reply was not delivered")
            wait_for(lambda: robot._goal is None or robot._goal.terminal(), "Nav2 never confirmed stop")
            time.sleep(0.1)
            assert all(raw[:2] == (0.0, 0.0) for raw, _ in wheels[tail:]), wheels[tail:]
            assert fake_base._want() == (0.0, 0.0)
            robot._step(0.12, 0.0)
            wait_for(lambda: any(want[0] == 0.12 for _, want in wheels[tail:]), "direct drive did not take over")
            robot.stop()
            wait_for(lambda: wheels[-1][1] == (0.0, 0.0), "direct drive did not stop")
        finally:
            try:
                if robot is not None:
                    robot.shutdown()
            finally:
                os.killpg(launch.pid, signal.SIGINT)
                launch.wait(timeout=15)
                executor.shutdown()
                node.destroy_node()
                rclpy.shutdown()
