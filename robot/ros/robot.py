"""Sprocket's ROS 2 API: Nav2 for navigation, and one checked velocity output for every skill."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator
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
    Timeout,
    choose,
    wrap,
    xy,
)
from robot.body.controller import Progress
from robot.interfaces.voice import chime as synth_chime
from robot.ros import motion
from robot.ros.bringup import CRUISE as ROS_CRUISE
from robot.ros.bringup import TURN
from robot.sim.world import CRUISE, DT, RADIUS, Scene, SimRobot

# A search stop Nav2 could not reach kept it backing up, spinning and replanning along one 2 m stretch for 490 s.
PROGRESS_S, PROGRESS_M = 30.0, 0.25
CANCEL_WAIT_S = 2.0
DIRECT = "direct"
PLAN_TIMEOUT_S, PLAN_ENDPOINT_M = 2.0, 0.10


class RosRobot(SimRobot):
    def __init__(self, scene: Scene, max_time: float = 600.0, nudge: Any = None,
                 synthetic_scan: bool = True) -> None:
        import rclpy
        from geometry_msgs.msg import TwistStamped
        from nav2_msgs.action import ComputePathToPose, DriveOnHeading, FollowPath, NavigateToPose, Spin
        from rclpy.action import ActionClient
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
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
                      "drive_on_heading": DriveOnHeading, "compute_path_to_pose": ComputePathToPose}
        self.actions: dict[str, Any] = {}
        self._twist = TwistStamped
        self.cmd = self.node.create_publisher(TwistStamped, motion.WHEELS, 1)
        selected = self.node.create_publisher(TwistStamped, motion.SELECTED, 1)
        self._motion = motion.MotionGate(lambda: self.node.get_clock().now().nanoseconds,
                                          lambda v, w, t: selected.publish(self._twist_msg(v, w, t)),
                                          self._publish, ROS_CRUISE, TURN)
        self._goal: motion.NavGoal | None = None
        self._scan: Any = None
        self.last_chime: bytes | None = None
        self.on_chime: Any = None
        self._scan_bridge = SyntheticScanPublisher(self.node, publish=False) if synthetic_scan else None
        motion_group = MutuallyExclusiveCallbackGroup()
        self.node.create_subscription(TwistStamped, motion.NAV, self._nav_command, 1, callback_group=motion_group)
        self.node.create_subscription(TwistStamped, motion.CHECKED, self._checked_command, 1,
                                      callback_group=motion_group)
        self.node.create_timer(motion.CONTROL_PERIOD, self._motion.expire, callback_group=motion_group)
        self.node.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "_scan", m), 10)
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        threading.Thread(target=self._executor.spin, daemon=True).start()
        self._action_executor = MultiThreadedExecutor()
        self._action_executor.add_node(self._action_node)
        threading.Thread(target=self._action_executor.spin, daemon=True).start()
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
            self._interrupts()
            time.sleep(0.01)

    def _step(self, v: float, w: float, dt: float = DT) -> None:
        """Direct motion and background skills; Nav2 waits without running either."""
        try:
            self._interrupts()
            if self._behaviour is not None and not v and not w:
                v, w = self._behaviour()
            with self._motion.lock:
                self._interrupts()
                if v or w:
                    self._motion_available()
                    self._arm_motion(DIRECT)
                    self._motion.submit(DIRECT, v, w)
                elif self._motion.owner == DIRECT:
                    self._halt()
            self._wait(dt)
            self._sync()
        except BaseException:
            self._halt()
            raise

    def _twist_msg(self, v: float, w: float, stamp: int) -> Any:
        msg = self._twist()
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        msg.header.frame_id = "base_link"
        msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        return msg

    def _publish(self, v: float, w: float, stamp: int) -> None:
        if self._scan_bridge is not None and v:
            hazard = self._scan_bridge.hazards.get("front" if v > 0 else "rear")
            if getattr(hazard, "kind", "") == "cliff":
                v = w = 0.0  # do not turn a cliff stop into motion toward the opposite side
        self.v_now, self.w_now = v, w
        self.cmd.publish(self._twist_msg(v, w, stamp))

    def _nav_command(self, msg: Any) -> None:
        self._motion.submit(motion.NAV, *motion.command(msg))

    def _checked_command(self, msg: Any) -> None:
        self._motion.checked(*motion.command(msg))

    def _drop_motion(self, *, cancel_task: bool = False) -> motion.NavGoal | None:
        with self._motion.lock:
            if cancel_task:
                super().cancel()
            else:
                self._halt()
            goal = self._goal
            if goal is not None:
                goal.cancel()
            return goal

    def cancel(self) -> None:
        self._drop_motion(cancel_task=True)

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

    def _motion_available(self) -> None:
        self._retire_goal()
        if self._goal is not None:
            raise SafetyStop("Nav2 has not released the previous motion goal")

    def _arm_motion(self, owner: str) -> None:
        if not self._motion.arm(owner):
            raise SafetyStop("ROS time went backwards; restart the motion process before driving again")

    def _nav_tick(self, goal: motion.NavGoal) -> None:
        if goal.cancelled:
            raise Preempted("stopped")
        self._wait(DT)
        self._sync()

    def _client(self, name: str, timeout: float, retry: float = 0.0) -> Any:
        client = self.actions[name]
        if client.wait_for_server(timeout_sec=timeout):
            return client
        if retry:
            from rclpy.action import ActionClient

            if hasattr(client, "destroy"):
                client.destroy()
            self.actions[name] = client = ActionClient(self._action_node, self.kinds[name], name)
            if client.wait_for_server(timeout_sec=retry):
                return client
        return None

    def _act(self, name: str, *, watch: Object | None = None, **fields: Any) -> None:
        """Send one action goal and wait for it, keeping the trace and the stop checks running meanwhile."""
        self._interrupts()
        with self._motion.lock:
            self._motion_available()
            self._halt()
        client = self._client(name, 10.0, retry=20.0)
        if client is None:
            raise RuntimeError(f"the {name} action server is not up")
        goal = self.kinds[name].Goal()
        for key, value in fields.items():
            setattr(goal, key, value)
        if watch is not None:
            self._check_goal(watch)
        with self._motion.lock:
            self._interrupts()
            self._motion_available()
            self._arm_motion(motion.NAV)
            try:
                pending = self._goal = motion.NavGoal(client.send_goal_async(goal), self._motion.lock)
            except BaseException:
                self._halt()
                raise
        try:
            while not pending.sent.done():
                self._nav_tick(pending)
            result = pending.result_future()
            if result is None:
                raise PathBlocked(f"{name} refused the goal")
            next_check = 0.0
            progress = Progress((self.x, self.y), self.t, PROGRESS_M, PROGRESS_S)
            while not result.done():
                self._nav_tick(pending)
                if watch is not None and time.monotonic() >= next_check:
                    self._check_goal(watch)
                    next_check = time.monotonic() + 0.2
                if name == "navigate_to_pose" and not result.done() and progress.stalled((self.x, self.y), self.t):
                    raise PathBlocked(f"no navigation progress for {PROGRESS_S:g} s")
            self._interrupts()
            if pending.cancelled:
                raise Preempted("stopped")
            outcome = result.result()
            if outcome.status == motion.CANCELED:
                raise Preempted("stopped")
            if outcome.status != motion.DONE:
                detail = getattr(outcome.result, "error_msg", "") or f"status {outcome.status}"
                raise PathBlocked(f"{name} could not finish: {detail}")
        except BaseException as exc:
            self._drop_motion()
            if isinstance(exc, (TargetLost, PathBlocked)) and not pending.wait_terminal(CANCEL_WAIT_S):
                raise SafetyStop(f"Nav2 did not confirm that the {name} goal it was told to drop had stopped") from exc
            raise
        finally:
            with self._motion.lock:
                self._halt()
                pending.released = True
                self._retire_goal()

    def _retire_goal(self) -> None:
        if self._goal is not None and self._goal.released and self._goal.terminal():
            self._goal = None

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

    def _nav_reaches(self, end: Point, yaw: float = 0.0) -> bool:
        """Check the live costmap without arming motion. Nav2 can report success for a path 0.20 m short."""
        name = "compute_path_to_pose"
        self._interrupts()
        client = self._client(name, PLAN_TIMEOUT_S)
        if client is None:
            raise Timeout("navigation planner unavailable: Nav2's planner did not become ready")
        goal = self.kinds[name].Goal()
        goal.goal = self._pose_msg(*end, yaw)
        pending = motion.NavGoal(client.send_goal_async(goal), self._motion.lock)
        try:
            if not pending.wait_terminal(PLAN_TIMEOUT_S, self._interrupts):
                raise Timeout("navigation planner unavailable: Nav2's planner did not answer in time")
        except BaseException:
            pending.cancel()
            raise
        result = pending.result_future()
        if result is None:
            return False
        outcome = result.result()
        if outcome.status != motion.DONE:
            return False
        path = outcome.result.path.poses
        if not path:
            return False
        last = path[-1].pose.position
        return math.dist(end, (last.x, last.y)) <= PLAN_ENDPOINT_M

    def _goal_point(self, target: Point | Pose | Object, distance: float, *,
                    rejected: set[Point] | None = None) -> tuple[Point, float] | None:
        if rejected is None:
            rejected = set()
        for end, yaw in self._goal_candidates(target, distance):
            if end not in rejected:
                if self._nav_reaches(end, yaw):
                    return end, yaw
                rejected.add(end)
        if rejected:
            raise PathBlocked("Nav2 found no path to a usable goal on its live costmap")
        return None

    def _goal_candidates(self, target: Point | Pose | Object,
                         distance: float) -> Iterator[tuple[Point, float]]:
        """A free spot on the static map within reach of the target, and the heading to arrive with."""
        goal, reach = self._reach(target, distance)
        if math.dist(goal, (self.x, self.y)) <= reach:
            return
        path = self._route(target, goal, reach)
        if path is None:
            raise PathBlocked(f"no free path to ({goal[0]:.1f}, {goal[1]:.1f})")
        if not path:  # already on the nearest floor there is
            return
        end = path[-1] if isinstance(target, Object) or self._occ(*goal, self.inflated) else goal
        if isinstance(target, Pose):
            yield end, math.radians(target.yaw)
        elif isinstance(target, Object) and distance > 0 and math.dist(end, goal) > 0.05:
            yield end, math.atan2(goal[1] - end[1], goal[0] - end[0])
        else:
            before = path[-2] if len(path) > 1 else (self.x, self.y)
            yield end, math.atan2(end[1] - before[1], end[0] - before[0])

    # ---- motion: the 2D world's skills call these, so every skill drives through Nav2 ---------------------------
    def _face_goal(self, goal: Point) -> None:
        target_yaw = math.atan2(goal[1] - self.y, goal[0] - self.x)
        err = wrap(target_yaw - self.yaw)
        if abs(err) > 0.22:
            try:
                self._rotate(err)
            except PathBlocked:
                pass

    def _finish_approach(self, target: Object, distance: float, rejected: set[Point]) -> None:
        """Refine the view without retrying a point Nav2 rejected."""

    def _go(self, target: Point | Pose | Object, distance: float, *, side: str = "") -> None:
        """Nav2 drives there; when it gives up short of the target, so does the robot, and the caller hears why."""
        watch = target if isinstance(target, Object) else None
        target, distance = self._destination(target, distance, side)
        goal = xy(target)
        reach = (target.radius if isinstance(target, Object) else 0.0) + max(distance, 0.35)
        rejected: set[Point] = set()
        where = self._goal_point(target, distance, rejected=rejected)
        if where is not None:
            (gx, gy), yaw = where
            try:
                self._act("navigate_to_pose", watch=watch, pose=self._pose_msg(gx, gy, yaw))
            except PathBlocked:
                rejected.add((gx, gy))
                self._sync()
                if math.dist(goal, (self.x, self.y)) > reach + 0.35 and math.dist((gx, gy), (self.x, self.y)) > 0.35:
                    raise
                self._face_goal(goal)  # stopped just short (or at nearest free cell beside furniture): face it
        if isinstance(target, Object):
            self._finish_approach(target, distance, rejected)
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
        super()._halt()
        if hasattr(self, "_motion"):
            self._motion.stop()

    def stop(self) -> None:
        self._call("stop")
        self._drop_motion()

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
        confirmed = True
        if hasattr(self, "_motion"):
            goal = self._drop_motion(cancel_task=True)
            if goal is not None:
                confirmed = goal.wait_terminal(CANCEL_WAIT_S)
        if hasattr(self, "_action_executor"):
            self._action_executor.shutdown()
        for client in getattr(self, "actions", {}).values():
            try:
                client.destroy()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(self, "_action_node"):
            self._action_node.destroy_node()
        if hasattr(self, "_executor"):
            self._executor.shutdown()
        if hasattr(self, "node"):
            self.node.destroy_node()
        if not confirmed:
            raise SafetyStop("Nav2 did not confirm the previous goal stopped; restart Nav2 before driving again")
