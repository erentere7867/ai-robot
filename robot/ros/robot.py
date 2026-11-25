"""Sprocket v2 Robot API on ROS 2: Nav2 navigation, TF pose, synthetic /scan, and /cmd_vel Twist control.

Every skill from the 2D simulator (`go_to`, `follow`, `express`, etc.) runs through ROS 2: Nav2 actions for map-level
goals, BodyController 30-50 Hz pursuit and procedural aliveness on /cmd_vel, and 3-CSI calibrated synthetic /scan for
AMCL/clearance without hardware lidar.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from robot.agent.api import (
    NotFound,
    Object,
    PathBlocked,
    Point,
    Pose,
    Preempted,
    SafetyStop,
    TargetLost,
    choose,
    wrap,
    xy,
)
from robot.interfaces.voice import chime as synth_chime
from robot.sim.world import CRUISE, DT, RADIUS, Scene, SimRobot

DONE, CANCELED, ABORTED = 4, 5, 6  # action_msgs/GoalStatus
# A search stop Nav2 could not reach kept it backing up, spinning and replanning along one 2 m stretch for 490 s.
PROGRESS_S, PROGRESS_M = 30.0, 0.25


class RosRobot(SimRobot):
    def __init__(self, scene: Scene, max_time: float = 600.0, nudge: Any = None,
                 synthetic_scan: bool = True) -> None:
        import rclpy
        from geometry_msgs.msg import TwistStamped
        from nav2_msgs.action import DriveOnHeading, FollowPath, NavigateToPose, Spin
        from rclpy.action import ActionClient
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.parameter import Parameter
        from sensor_msgs.msg import LaserScan
        from tf2_ros import Buffer, TransformListener

        from robot.ros.perceive import SyntheticScanPublisher

        super().__init__(scene, max_time=max_time)
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("sprocket_api", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self._action_node = rclpy.create_node(
            "sprocket_actions", parameter_overrides=[Parameter("use_sim_time", value=True)]
        )
        self.tf = Buffer()
        self._tf_listener = TransformListener(self.tf, self.node)
        self.kinds = {"navigate_to_pose": NavigateToPose, "follow_path": FollowPath, "spin": Spin,
                      "drive_on_heading": DriveOnHeading}
        self.actions: dict[str, Any] = {}
        # Direct motion (drive, express) enters Nav2's own command chain where its controller's does: the velocity
        # smoother, then the collision monitor, which stops the robot short of anything in the way. Nothing reaches
        # the wheels around that safety stop.
        self._twist = TwistStamped
        self.cmd = self.node.create_publisher(TwistStamped, "/cmd_vel_nav", 10)
        self._scan: Any = None
        self.last_chime: bytes | None = None
        self.on_chime: Any = None
        self._scan_bridge = SyntheticScanPublisher(self.node, publish=False) if synthetic_scan else None
        self.node.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "_scan", m), 10)
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        threading.Thread(target=self._executor.spin, daemon=True).start()
        self._action_executor = MultiThreadedExecutor()
        self._action_executor.add_node(self._action_node)
        threading.Thread(target=self._action_executor.spin, daemon=True).start()
        self._goal: Any = None
        self.t0 = None
        deadline, next_nudge = time.monotonic() + 300.0, time.monotonic() + 10.0  # Gazebo can take minutes
        while not self._read_pose():
            if time.monotonic() > deadline:
                raise RuntimeError("no map -> base_link transform after 300 s: is Nav2 running and localised?")
            if nudge is not None and time.monotonic() > next_nudge:  # e.g. send the initial pose again
                nudge()
                next_nudge = time.monotonic() + 10.0
            time.sleep(0.2)
        self._wait_for_nav2(deadline)
        # Create action clients after Nav2's lifecycle managers are active: clients built while behavior_server was
        # still inactive occasionally missed FastDDS SHM discovery on drive_on_heading (E2 reached 0.84 m then failed).
        self.actions = {name: ActionClient(self._action_node, kind, name) for name, kind in self.kinds.items()}
        self.t0 = self._now()
        self.t = 0.0
        self.trace.poses = [(0.0, self.x, self.y, self.yaw)]
        self._last_perceive = -1.0
        self._perceive()

    def _wait_for_nav2(self, deadline: float) -> None:
        """Until every Nav2 lifecycle manager says its nodes are active: before that, its action servers exist but
        refuse goals (a spin sent while the planner was still activating was turned down)."""
        from std_srvs.srv import Trigger

        clients: dict[str, Any] = {}
        while time.monotonic() < deadline:
            for name, types in self.node.get_service_names_and_types():
                if name.endswith("/is_active") and "std_srvs/srv/Trigger" in types and name not in clients:
                    clients[name] = self.node.create_client(Trigger, name)
            if clients:
                answers = [c.call_async(Trigger.Request()) for c in clients.values()]
                end = time.monotonic() + 5.0
                while not all(a.done() for a in answers) and time.monotonic() < end:
                    time.sleep(0.05)
                if all(a.done() and a.result().success for a in answers):
                    return
            time.sleep(1.0)
        raise RuntimeError("Nav2 did not become active")

    # ---- keeping the shadow world in step ---------------------------------------------------------------------
    def _now(self) -> float:
        return self.node.get_clock().now().nanoseconds / 1e9

    def begin(self) -> None:
        if getattr(self, "t0", None) is not None:
            try:
                self.t = self._now() - self.t0
            except Exception:
                pass
        if hasattr(self, "tf"):
            try:
                self._read_pose()
            except Exception:
                pass
        super().begin()
        self.trace.poses = self.trace.poses[-500:]

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
        if len(self.trace.poses) > 3000:
            self.trace.poses = self.trace.poses[-1500:]
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
        if self._scan_bridge is not None and v:
            hazard = self._scan_bridge.hazards.get("front" if v > 0 else "rear")
            if hazard is not None:
                pitch = self._scan_bridge.tilts[-1][2] if self._scan_bridge.tilts else 0.0
                if v < 0 and getattr(hazard, "kind", "") == "cliff":
                    v, w = 0.0, 0.0
                else:
                    _, v, w, _ = self.body.check_proprioception(
                        v, w, self.v_now, self.w_now, dt, pitch_rad=pitch, hazard=hazard)
        if v or w or self.v_now or self.w_now:
            self._publish(v, w)
        self.v_now, self.w_now = v, w
        self._wait(dt)
        self._sync()

    def _publish(self, v: float, w: float) -> None:
        msg = self._twist()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        self.cmd.publish(msg)

    def cancel(self) -> None:
        super().cancel()
        if self._goal is not None:
            self._goal.cancel_goal_async()
        self._publish(0.0, 0.0)

    # ---- Nav2 actions -----------------------------------------------------------------------------------------
    def _check_goal(self, target: Object) -> None:
        try:
            current = self.obj(target.id)
        except NotFound as exc:
            raise TargetLost(f"stopped approaching {target.label} {target.id}: {exc}") from exc
        # Match Tracker.gate (0.6 m): small objects (e.g. shoes, r=0.04-0.11 m) routinely refine depth by 0.16-0.26 m
        # while approaching, which falsely aborted 4 consecutive go_to calls against 2*r (0.08-0.22 m).
        limit = max(0.6, target.radius + current.radius)
        if current.label != target.label or math.dist(current.position, target.position) > limit:
            raise TargetLost(f"stopped: the label or mapped footprint of {target.label} {target.id} changed")

    def _act(self, name: str, *, watch: Object | None = None, **fields: Any) -> None:
        """Send one action goal and wait for it, keeping the trace and the stop checks running meanwhile."""
        client = self.actions[name]
        if not client.wait_for_server(timeout_sec=10.0):
            from rclpy.action import ActionClient

            if hasattr(client, "destroy"):
                client.destroy()
            self.actions[name] = ActionClient(self._action_node, self.kinds[name], name)
            client = self.actions[name]
            if not client.wait_for_server(timeout_sec=20.0):
                raise RuntimeError(f"the {name} action server is not up")
        goal = self.kinds[name].Goal()
        for key, value in fields.items():
            setattr(goal, key, value)
        if watch is not None:
            self._check_goal(watch)
        sent = client.send_goal_async(goal)
        try:
            while not sent.done():
                self._step(0.0, 0.0)
        except BaseException:
            def cancel_when_accepted(future: Any) -> None:
                accepted = future.result()
                if accepted and accepted.accepted:
                    accepted.cancel_goal_async()
            sent.add_done_callback(cancel_when_accepted)
            raise
        handle = sent.result()
        if not handle.accepted:
            raise PathBlocked(f"{name} refused the goal")
        self._goal, result = handle, handle.get_result_async()
        next_check = 0.0
        visited = [(self.x, self.y)]
        since = self.t
        try:
            while not result.done():
                self._step(0.0, 0.0)
                if watch is not None and time.monotonic() >= next_check:
                    self._check_goal(watch)
                    next_check = time.monotonic() + 0.2
                if name == "navigate_to_pose" and not result.done():
                    # Replanning can change path length without motion, and repeated recovery can retrace the same
                    # stretch. Only reaching new ground counts, including a detour that leads away from the goal.
                    here = (self.x, self.y)
                    if all(math.dist(here, point) >= PROGRESS_M for point in reversed(visited)):
                        visited.append(here)
                        since = self.t
                    if self.t - since > PROGRESS_S:
                        raise PathBlocked(f"no navigation progress for {PROGRESS_S:g} s")
        except BaseException as exc:
            handle.cancel_goal_async()
            self._publish(0.0, 0.0)
            if isinstance(exc, (TargetLost, PathBlocked)):
                # Do not hand the model another motion turn while the old controller can still drive.
                deadline = time.monotonic() + 2.0
                while not result.done() and time.monotonic() < deadline:
                    time.sleep(0.02)
                if not result.done():
                    raise SafetyStop(f"Nav2 did not confirm that the {name} goal it was told to drop had stopped") from exc
            raise
        finally:
            self._goal = None
        status = result.result().status
        if status == CANCELED:
            raise Preempted("stopped")
        if status != DONE:
            detail = getattr(result.result().result, "error_msg", "") or f"status {status}"
            raise PathBlocked(f"{name} could not finish: {detail}")

    def _pose_msg(self, x: float, y: float, yaw: float) -> Any:
        from geometry_msgs.msg import PoseStamped

        msg = PoseStamped()
        msg.header.frame_id = "map"
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
            t_start = self.t
            self._act("spin", target_yaw=float(left), time_allowance=self._allow(10 + abs(left) * 4))
            yaws = [p[3] for p in self.trace.poses if p[0] >= t_start]
            done += sum(wrap(b - a) for a, b in zip(yaws, yaws[1:], strict=False))

    def _goal_point(self, target: Point | Pose | Object, distance: float) -> tuple[Point, float] | None:
        """A free spot on the static map within reach of the target, and the heading to arrive with."""
        goal, reach = self._reach(target, distance)
        if math.dist(goal, (self.x, self.y)) <= reach:
            return None
        path = self._route(target, goal, reach)
        if path is None:
            raise PathBlocked(f"no free path to ({goal[0]:.1f}, {goal[1]:.1f})")
        if not path:  # already on the nearest floor there is
            return None
        end = path[-1] if isinstance(target, Object) or self._occ(*goal, self.inflated) else goal
        if isinstance(target, Pose):
            return end, math.radians(target.yaw)
        if isinstance(target, Object) and distance > 0 and math.dist(end, goal) > 0.05:
            return end, math.atan2(goal[1] - end[1], goal[0] - end[0])
        before = path[-2] if len(path) > 1 else (self.x, self.y)
        return end, math.atan2(end[1] - before[1], end[0] - before[0])

    # ---- motion: the 2D world's skills call these, so every skill drives through Nav2 ---------------------------
    def _face_goal(self, goal: Point) -> None:
        target_yaw = math.atan2(goal[1] - self.y, goal[0] - self.x)
        err = wrap(target_yaw - self.yaw)
        if abs(err) > 0.22:
            try:
                self._rotate(err)
            except Exception:
                pass

    def _go(self, target: Point | Pose | Object, distance: float, *, side: str = "") -> None:
        """Nav2 drives there; when it gives up short of the target, so does the robot, and the caller hears why."""
        watch = target if isinstance(target, Object) else None
        target, distance = self._destination(target, distance, side)
        goal = xy(target)
        reach = (target.radius if isinstance(target, Object) else 0.0) + max(distance, 0.35)
        where = self._goal_point(target, distance)
        if where is not None:
            (gx, gy), yaw = where
            try:
                self._act("navigate_to_pose", watch=watch, pose=self._pose_msg(gx, gy, yaw))
            except PathBlocked:
                self._sync()
                if math.dist(goal, (self.x, self.y)) > reach + 0.35 and math.dist((gx, gy), (self.x, self.y)) > 0.35:
                    raise
                self._face_goal(goal)  # stopped just short (or at nearest free cell beside furniture): face it
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
        steps = round(float(seconds) / DT)  # 0: just the posture and the chime, no motion
        try:
            for s in range(steps):
                v, w = self.body.aliveness_velocity(posture, s * DT, (self.x, self.y, self.yaw))
                self._step(v, w)
        finally:
            self._halt()

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
        for client in getattr(self, "actions", {}).values():
            try:
                client.destroy()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(self, "_action_executor"):
            self._action_executor.shutdown()
        if hasattr(self, "_action_node"):
            self._action_node.destroy_node()
        if hasattr(self, "_executor"):
            self._executor.shutdown()
        if hasattr(self, "node"):
            self.node.destroy_node()
