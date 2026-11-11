"""The robot API on ROS 2: Nav2 moves the robot, TF says where it is, the lidar is the real /scan.

Perception is not built yet (phase 3), so objects come from the scene the 2D world was made from: `RosRobot` keeps
a `SimRobot` in step with the real pose and lets it answer find, detect, ask and unseen as if a perfect detector
were running. Everything that moves or measures goes through ROS. The suite's checks read the same trace in both
backends, so one program can be judged in the 2D world and in Gazebo by the same rules.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from robot.agent.api import Object, PathBlocked, Point, Pose, Preempted, choose, wrap
from robot.sim.world import CRUISE, DT, RADIUS, Scene, SimRobot

DONE, CANCELED, ABORTED = 4, 5, 6  # action_msgs/GoalStatus


class RosRobot(SimRobot):
    def __init__(self, scene: Scene, max_time: float = 600.0, nudge: Any = None) -> None:
        import rclpy
        from geometry_msgs.msg import Twist, TwistStamped
        from nav2_msgs.action import DriveOnHeading, FollowPath, NavigateToPose, Spin
        from rclpy.action import ActionClient
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.parameter import Parameter
        from sensor_msgs.msg import LaserScan
        from tf2_ros import Buffer, TransformListener

        super().__init__(scene, max_time=max_time)
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("sprocket_api", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.tf = Buffer()
        self._tf_listener = TransformListener(self.tf, self.node)
        self.actions = {name: ActionClient(self.node, kind, name) for name, kind in (
            ("navigate_to_pose", NavigateToPose), ("follow_path", FollowPath), ("spin", Spin),
            ("drive_on_heading", DriveOnHeading))}
        self.kinds = {"navigate_to_pose": NavigateToPose, "follow_path": FollowPath, "spin": Spin,
                      "drive_on_heading": DriveOnHeading}
        stamped = any("TwistStamped" in t for n, ts in self.node.get_topic_names_and_types() if n.endswith("cmd_vel_nav")
                      for t in ts)
        self._twist = TwistStamped if stamped else Twist
        self.cmd = self.node.create_publisher(self._twist, "cmd_vel_nav", 10)  # smoothed, then collision-monitored
        self._scan: Any = None
        self.node.create_subscription(LaserScan, "scan", lambda m: setattr(self, "_scan", m), 10)
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        threading.Thread(target=self._executor.spin, daemon=True).start()
        self._goal: Any = None
        self.t0 = None
        deadline, next_nudge = time.monotonic() + 120.0, time.monotonic() + 10.0
        while not self._read_pose():
            if time.monotonic() > deadline:
                raise RuntimeError("no map -> base_link transform after 120 s: is Nav2 running and localised?")
            if nudge is not None and time.monotonic() > next_nudge:  # e.g. send the initial pose again
                nudge()
                next_nudge = time.monotonic() + 10.0
            time.sleep(0.2)
        self.t0 = self._now()
        self.t = 0.0
        self.trace.poses = [(0.0, self.x, self.y, self.yaw)]
        self._last_perceive = -1.0
        self._perceive()

    # ---- keeping the shadow world in step ---------------------------------------------------------------------
    def _now(self) -> float:
        return self.node.get_clock().now().nanoseconds / 1e9

    def _read_pose(self) -> bool:
        from rclpy.time import Time

        try:
            tr = self.tf.lookup_transform("map", "base_link", Time()).transform
        except Exception:  # noqa: BLE001 - not there yet
            return False
        q = tr.rotation
        self.x, self.y = tr.translation.x, tr.translation.y
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return True

    def _sync(self) -> None:
        self._read_pose()
        self.t = self._now() - self.t0
        self.trace.poses.append((round(self.t, 3), self.x, self.y, self.yaw))
        self._perceive()
        self._interrupts()

    def _wait(self, seconds: float) -> None:
        end = self._now() + seconds
        while self._now() < end:
            time.sleep(0.01)

    def _step(self, v: float, w: float, dt: float = DT) -> None:
        """One control period: say, listen, sleep and drive in the 2D world all advance time through this."""
        self._interrupts()
        if self._behaviour is not None and not v and not w:  # a background skill drives while the program waits
            v, w = self._behaviour()
        if v or w or self.v_now or self.w_now:
            self._publish(v, w)
        self.v_now, self.w_now = v, w
        self._wait(dt)
        self._sync()

    def _publish(self, v: float, w: float) -> None:
        msg = self._twist()
        twist = msg.twist if hasattr(msg, "twist") else msg
        if hasattr(msg, "header"):
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
        twist.linear.x, twist.angular.z = float(v), float(w)
        self.cmd.publish(msg)

    def cancel(self) -> None:
        super().cancel()
        if self._goal is not None:
            self._goal.cancel_goal_async()
        self._publish(0.0, 0.0)

    # ---- Nav2 actions -----------------------------------------------------------------------------------------
    def _act(self, name: str, **fields: Any) -> None:
        """Send one action goal and wait for it, keeping the trace and the stop checks running meanwhile."""
        client = self.actions[name]
        if not client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError(f"the {name} action server is not up")
        goal = self.kinds[name].Goal()
        for key, value in fields.items():
            setattr(goal, key, value)
        sent = client.send_goal_async(goal)
        while not sent.done():
            self._step(0.0, 0.0)
        handle = sent.result()
        if not handle.accepted:
            raise PathBlocked(f"{name} refused the goal")
        self._goal, result = handle, handle.get_result_async()
        try:
            while not result.done():
                self._step(0.0, 0.0)
        except BaseException:
            handle.cancel_goal_async()
            raise
        finally:
            self._goal = None
        status = result.result().status
        if status == CANCELED:
            raise Preempted("stopped")
        if status != DONE:
            raise PathBlocked(f"{name} could not finish (status {status})")

    def _pose_msg(self, x: float, y: float, yaw: float) -> Any:
        from geometry_msgs.msg import PoseStamped

        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = float(x), float(y)
        msg.pose.orientation.z, msg.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return msg

    @staticmethod
    def _allow(seconds: float) -> Any:
        from builtin_interfaces.msg import Duration

        return Duration(sec=int(seconds))

    def _rotate(self, angle: float, rate: float = 1.0) -> None:
        """Spin, then check the turn against the map and spin the rest. Nav2's spin counts wheel odometry, and wheels
        slip when turning on the spot: in Gazebo a 360 degree spin came up about 12 degrees short."""
        done = 0.0
        for _ in range(3):
            left = angle - done
            if abs(left) < math.radians(3):
                return
            first = len(self.trace.poses) - 1
            self._act("spin", target_yaw=float(left), time_allowance=self._allow(10 + abs(left) * 4))
            yaws = [p[3] for p in self.trace.poses[first:]]
            done += sum(wrap(b - a) for a, b in zip(yaws, yaws[1:], strict=False))

    def _goal_point(self, target: Point | Pose | Object, distance: float) -> tuple[Point, float] | None:
        """A free spot on the static map within reach of the target, and the heading to arrive with."""
        goal, reach = self._reach(target, distance)
        if math.dist(goal, (self.x, self.y)) <= reach:
            return None
        path = self._plan(goal, reach)
        if path is None:
            raise PathBlocked(f"no free path to ({goal[0]:.1f}, {goal[1]:.1f})")
        end = path[-1] if isinstance(target, Object) or self._occ(*goal, self.inflated) else goal
        if isinstance(target, Pose):
            return end, math.radians(target.yaw)
        before = path[-2] if len(path) > 1 else (self.x, self.y)
        return end, math.atan2(end[1] - before[1], end[0] - before[0])

    # ---- motion: the 2D world's skills call these, so every skill drives through Nav2 ---------------------------
    def _go(self, target: Point | Pose | Object, distance: float) -> None:
        where = self._goal_point(target, distance)
        if where is not None:
            (gx, gy), yaw = where
            self._act("navigate_to_pose", pose=self._pose_msg(gx, gy, yaw))
        if isinstance(target, Pose):
            self._rotate(wrap(math.radians(target.yaw) - self.yaw))

    def _drive_path(self, points: list[Point], speed: float, record: bool) -> None:
        """Nav2's controller follows each lap on its own: a path that crosses itself confuses its lookahead."""
        from nav_msgs.msg import Path

        free = [p for p in points if not self._occ(*p, self.inflated)]
        if record:
            for p in points:
                if self._occ(*p, self.inflated):
                    self.trace.violations.append(f"the path passes through an obstacle at ({p[0]:.2f}, {p[1]:.2f})")
        if not free:
            return
        if math.dist((self.x, self.y), free[0]) > 0.3:
            nxt = free[1] if len(free) > 1 else free[0]
            self._act("navigate_to_pose", pose=self._pose_msg(*free[0], math.atan2(nxt[1] - free[0][1],
                                                                                  nxt[0] - free[0][0])))
        laps, start, run = [], 0, 0.0
        for i in range(1, len(free)):
            run += math.dist(free[i - 1], free[i])
            if run > 1.0 and math.dist(free[i], free[start]) < 0.2:
                laps.append(free[start:i + 1])
                start, run = i, 0.0
        laps.append(free[start:])
        for lap in (lap for lap in laps if len(lap) > 1):
            msg = Path()
            msg.header.frame_id = "map"
            for k, (px, py) in enumerate(lap):
                nxt = lap[min(k + 1, len(lap) - 1)]
                prv = lap[max(k - 1, 0)]
                msg.poses.append(self._pose_msg(px, py, math.atan2(nxt[1] - prv[1], nxt[0] - prv[0])))
            self._act("follow_path", path=msg)

    def _move(self, distance: float) -> None:
        from geometry_msgs.msg import Point as RosPoint

        if abs(distance) > 0.01:
            speed = math.copysign(min(CRUISE, 0.25), distance)
            self._act("drive_on_heading", target=RosPoint(x=distance), speed=speed,
                      time_allowance=self._allow(10 + abs(distance) / 0.1))

    def _halt(self) -> None:
        """Every end of motion sends an explicit stop: Nav2's velocity smoother keeps the last command until its
        timeout, so silence would let the robot coast on for up to a second (measured: about 25 cm after a drive)."""
        super()._halt()
        if hasattr(self, "cmd"):
            self._publish(0.0, 0.0)

    def stop(self) -> None:
        self._call("stop")
        if self._goal is not None:
            self._goal.cancel_goal_async()
        self._halt()

    # ---- state and sensing that come from ROS -------------------------------------------------------------------
    def pose(self) -> Pose:
        self._call()
        self._sync()
        return Pose(round(self.x, 3), round(self.y, 3), round(math.degrees(self.yaw), 1))

    def clearance(self, direction: str = "ahead") -> float:
        """From the real lidar: the nearest return within 15 degrees of the direction, less the robot's radius."""
        self._call()
        offset = {"ahead": 0.0, "left": 90.0, "behind": 180.0, "right": -90.0}[
            choose(direction, ("ahead", "behind", "left", "right"), "direction")]
        msg = self._scan
        if msg is None:
            return super().clearance(direction)
        near = [r for k, r in enumerate(msg.ranges) if msg.range_min <= r <= msg.range_max
                and abs(math.degrees(wrap(msg.angle_min + k * msg.angle_increment - math.radians(offset)))) <= 15]
        return round(max(0.0, min(near) - RADIUS), 2) if near else 12.0

    def shutdown(self) -> None:
        self._executor.shutdown()
        self.node.destroy_node()
