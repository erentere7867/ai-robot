"""Sprocket v2 Robot API on ROS 2: Nav2 navigation, TF pose, synthetic /scan, and /cmd_vel Twist control.

Every skill from the 2D simulator (`go_to`, `pursue`, `peek`, `look_under`, `express`, `wake_turn`, `motion_sequence`,
`render_view`, etc.) runs through ROS 2: Nav2 actions for map-level goals, BodyController 30-50 Hz pursuit &
procedural aliveness on /cmd_vel, 3-CSI calibrated synthetic /scan for AMCL/clearance without hardware lidar, and the
front MIPI-CSI camera for live RGB frames.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np

from robot.agent.api import Object, PathBlocked, Point, Pose, Preempted, choose, wrap, xy
from robot.interfaces.voice import chime as synth_chime
from robot.sim.world import CRUISE, DT, RADIUS, Scene, SimRobot

DONE, CANCELED, ABORTED = 4, 5, 6  # action_msgs/GoalStatus


def _resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image array to `(height, width, 3)` uint8 using pure NumPy nearest-neighbor indexing."""
    if width <= 0 or height <= 0:
        return np.zeros((max(0, height), max(0, width), 3), dtype=np.uint8)
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    h, w = arr.shape[:2]
    if h == height and w == width and arr.shape[2] == 3:
        return arr.copy()
    ys = (np.arange(height) * h) // height
    xs = (np.arange(width) * w) // width
    return arr[np.ix_(ys, xs)][:, :, :3].copy()


class RosRobot(SimRobot):
    def __init__(self, scene: Scene, max_time: float = 600.0, nudge: Any = None,
                 synthetic_scan: bool = True) -> None:
        import rclpy
        from geometry_msgs.msg import Twist, TwistStamped
        from nav2_msgs.action import DriveOnHeading, FollowPath, NavigateToPose, Spin
        from rclpy.action import ActionClient
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.parameter import Parameter
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, LaserScan
        from tf2_ros import Buffer, TransformListener

        from robot.ros.perceive import TOPICS, SyntheticScanPublisher, _image

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
        stamped = any("TwistStamped" in t for n, ts in self.node.get_topic_names_and_types()
                      if n in ("cmd_vel", "/cmd_vel") for t in ts)
        self._twist = TwistStamped if stamped else Twist
        self.cmd = self.node.create_publisher(self._twist, "/cmd_vel", 10)
        self._scan: Any = None
        self._front_rgb: np.ndarray | None = None
        self.last_chime: bytes | None = None
        self.on_chime: Any = None
        self._scan_bridge = SyntheticScanPublisher(self.node) if synthetic_scan else None
        self.node.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "_scan", m), 10)
        self.node.create_subscription(Image, TOPICS["front_camera"],
                                      lambda m: setattr(self, "_front_rgb", _image(m)), qos_profile_sensor_data)
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
            start = (self.x, self.y)
            try:
                self._act("drive_on_heading", target=RosPoint(x=distance), speed=speed,
                          time_allowance=self._allow(10 + abs(distance) / 0.1))
            except PathBlocked:
                self._sync()
                if abs(distance) - math.dist(start, (self.x, self.y)) > min(0.1, 0.1 * abs(distance)):
                    raise  # stopped well short; within 10% (10 cm at most) is done, as in the 2D world

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

    # ---- Sprocket v2 skills on ROS 2 --------------------------------------------------------------------------
    def _people_with_vel(self) -> list[tuple[float, float, float, float]]:
        vel_map = getattr(self, "_people_velocities", {})
        out: list[tuple[float, float, float, float]] = []
        for t, p in self._people():
            vx, vy = vel_map.get(t.id, (0.0, 0.0))
            if vx == 0.0 and vy == 0.0 and getattr(t, "route", None) and getattr(t, "speed", 0.0) > 0:
                p_next = t.at(self.t + 0.5)
                if p_next is not None:
                    vx = (p_next[0] - p[0]) / 0.5
                    vy = (p_next[1] - p[1]) / 0.5
            out.append((p[0], p[1], vx, vy))
        return out

    def pursue(self, target: Any, distance: float = 1.0, seconds: float | None = None) -> None:
        self._call("pursue", target=target, distance=distance, seconds=seconds)
        is_user = isinstance(target, str) and target.lower().strip() in ("user", "the user", "me")
        tgt = (self.user() if is_user else None) or self._target(target)
        end = self.t + float(seconds) if seconds is not None else math.inf
        try:
            while self.t < end:
                self._interrupts()
                if is_user:
                    u = self.user()
                    if not u:
                        break
                    pt = u.position
                elif isinstance(tgt, Object) and tgt.id in self.things:
                    th = self.things[tgt.id]
                    pt = self._where(th) or (th.x, th.y)
                else:
                    pt = xy(tgt)
                d = math.hypot(pt[0] - self.x, pt[1] - self.y)
                if d <= distance + 0.15:
                    err = wrap(math.atan2(pt[1] - self.y, pt[0] - self.x) - self.yaw)
                    if abs(err) > 0.15:
                        self._step(0.0, max(-1.8, min(1.8, 2.0 * err)))
                    else:
                        break
                else:
                    route = self._plan(pt, distance) or [pt]
                    v, w, _, _ = self.body.step_pursuit(
                        (self.x, self.y, self.yaw),
                        route,
                        0,
                        blend_r=0.28,
                        v_cruise=CRUISE,
                        occupied_fn=lambda x, y: self._occ(x, y, self.inflated),
                        people=self._people_with_vel(),
                    )
                    self._step(v, w)
        finally:
            self._halt()

    def peek(self, target: Any) -> None:
        self._call("peek", target=target)
        target_pt = xy(self._target(target))
        peek_pt = self.body.plan_peek(
            (self.x, self.y, self.yaw), target_pt, lambda x, y: self._occ(x, y, self.inflated)
        )
        if peek_pt is not None:
            self.go_to(peek_pt)
            self.face(target_pt)

    def look_under(self, anchor: Any) -> None:
        self._call("look_under", anchor=anchor)
        t = self._target(anchor)
        anchor_radius = t.radius if isinstance(t, Object) else 0.5
        anchor_xy = xy(t)
        look_pt = self.body.plan_look_under(
            (self.x, self.y, self.yaw), anchor_xy, anchor_radius, lambda x, y: self._occ(x, y, self.inflated)
        )
        if look_pt is not None:
            self.go_to(look_pt)
            self.face(anchor_xy)

    def express(self, posture: str, chime_kind: str = "", seconds: float = 0.6) -> None:
        self._call("express", posture=posture, chime_kind=chime_kind)
        self.posture = posture
        self.trace.postures.append((round(self.t, 2), posture))
        if chime_kind:
            self.trace.chimes.append((round(self.t, 2), chime_kind))
            try:
                wav = synth_chime(chime_kind)
                self.last_chime = wav
                if callable(self.on_chime):
                    self.on_chime(wav)
            except ValueError:
                pass
        steps = max(1, round(float(seconds) / DT))
        try:
            for s in range(steps):
                v, w = self.body.aliveness_velocity(posture, s * DT, (self.x, self.y, self.yaw))
                self._step(v, w)
        finally:
            self._halt()

    def wake_turn(self, doa_rad: float) -> None:
        self._call("wake_turn", doa_rad=doa_rad)
        self._rotate(wrap(doa_rad), rate=2.5)
        self.express("listening", chime_kind="wake", seconds=0.3)

    def motion_sequence(self, steps: list[Any]) -> None:
        self._call("motion_sequence", count=len(steps))
        try:
            for step in steps:
                if isinstance(step, dict):
                    v = float(step.get("v", 0.0))
                    w = float(step.get("w", 0.0))
                    sec = float(step.get("seconds", step.get("s", 0.5)))
                elif isinstance(step, (list, tuple)) and len(step) >= 3:
                    v, w, sec = float(step[0]), float(step[1]), float(step[2])
                else:
                    continue
                for _ in range(max(1, round(sec / DT))):
                    self._step(v, w)
        finally:
            self._halt()

    def render_view(self, width: int = 320, height: int = 240) -> np.ndarray:
        """Latest RGB frame from the front CSI camera, or a zero array before frames arrive."""
        rgb = self._front_rgb
        if rgb is None or rgb.size == 0:
            return np.zeros((max(0, height), max(0, width), 3), dtype=np.uint8)
        return _resize_rgb(rgb, width, height)

    # ---- state and sensing that come from ROS -------------------------------------------------------------------
    def pose(self) -> Pose:
        self._call()
        self._sync()
        return Pose(round(self.x, 3), round(self.y, 3), round(math.degrees(self.yaw), 1))

    def clearance(self, direction: str = "ahead") -> float:
        """From synthetic or external /scan: nearest return within 15 degrees of `direction`, less robot radius."""
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
        self._halt()
        self._executor.shutdown()
        self.node.destroy_node()
