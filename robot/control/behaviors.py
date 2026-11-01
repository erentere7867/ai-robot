"""Continuous behaviours: closed-loop controllers the fast loop ticks about 20 times a second.

A behaviour reads the live world (pose, range sensor, the spatial map that perception keeps current)
and returns a speed command, or Done when it has succeeded or failed. Goals are recomputed every tick
from the latest estimate, so a target that moves, or whose position becomes more accurate as the robot
gets closer, is followed without any re-planning by the language model.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot.body.hal import BodyPose
from robot.navigation.planner import GridPlanner
from robot.world.spatial_map import SpatialMap, TrackedObject, label_score

ROBOT_CLEARANCE_MM = 300.0  # object surface to my centre when I stop "near" it
ARRIVE_MM = 60.0
WAYPOINT_MM = 150.0
FACE_TOLERANCE_DEG = 3.0
MAX_V = 320.0
MAX_W = 120.0
REFLEX_WORDS = {"obstacle_stop": "something is right in front of me",
                "bumper": "Bumper pressed: I bumped into something", "cliff_detected": "there is a drop ahead",
                "estop_active": "the emergency stop is on"}


def wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def bearing_to(pose: BodyPose, x: float, y: float) -> float:
    return wrap(math.degrees(math.atan2(y - pose.y_mm, x - pose.x_mm)) - pose.heading_deg)


@dataclass
class Tick:
    now: float
    dt: float
    pose: BodyPose
    tof_mm: float
    blocked: str | None  # reflex that limited the last speed command
    world: SpatialMap
    planner: GridPlanner


@dataclass
class Drive:
    v: float  # mm/s
    w: float  # deg/s (+ = left)


@dataclass
class Done:
    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


def turn_rate(error_deg: float, *, gain: float = 3.0, min_rate: float = 0.0) -> float:
    rate = max(-MAX_W, min(MAX_W, gain * error_deg))
    if min_rate and 0 < abs(rate) < min_rate:
        rate = math.copysign(min_rate, rate)
    return rate


def steer_to(pose: BodyPose, x: float, y: float, *, v_max: float, slow_mm: float) -> Drive:
    """Smooth pursuit of a point: turn in place when it is well off to the side, curve toward it otherwise."""
    dist = math.hypot(x - pose.x_mm, y - pose.y_mm)
    err = bearing_to(pose, x, y)
    if abs(err) > 50.0:
        return Drive(0.0, turn_rate(err))
    v = v_max * min(1.0, dist / max(1.0, slow_mm)) * math.cos(math.radians(err)) ** 2
    return Drive(max(40.0, v) if dist > 20.0 else 0.0, turn_rate(err))


def occupied_points(world: SpatialMap, exclude: set[str]) -> np.ndarray:
    """Range-sensor hits as planner points, minus those on objects I am allowed to approach."""
    idx = np.nonzero(world.occupied_mask())
    if not len(idx[0]):
        return np.zeros((0, 2))
    cell = 50.0
    points = np.stack([-world.half + (idx[0] + 0.5) * cell, -world.half + (idx[1] + 0.5) * cell], axis=1)
    for obj_id in exclude:
        obj = world.objects.get(obj_id)
        if obj is not None:
            points = points[np.hypot(points[:, 0] - obj.x_mm, points[:, 1] - obj.y_mm) > obj.radius_mm + 120.0]
    return points


class PathFollower:
    """Plans around known obstacles (re-planning as the map changes) and pursues the path smoothly."""

    REPLAN_S = 0.8

    def __init__(self, exclude: set[str] | None = None, *, ignore_hits_on: set[str] | None = None) -> None:
        self.exclude = set(exclude or ())  # objects I may drive onto or between: not obstacles at all
        # Objects whose range-sensor hits are ignored (still avoided as objects): approaching something
        # means the sensor sees it, and those hits must not keep me from getting near.
        self.ignore_hits_on = self.exclude | set(ignore_hits_on or ())
        self.waypoints: list[tuple[float, float]] = []
        self.goal: tuple[float, float] | None = None
        self.planned_at = -math.inf
        self.blocked_count = 0
        self.end: tuple[float, float] | None = None  # where the plan really ends (nearest free spot)

    def at_end(self, pose: BodyPose) -> bool:
        return self.end is not None and math.hypot(pose.x_mm - self.end[0], pose.y_mm - self.end[1]) <= ARRIVE_MM

    def drive(self, t: Tick, goal: tuple[float, float], *, v_max: float = MAX_V, slow_mm: float = 350.0,
              fallbacks: list[tuple[float, float]] | None = None) -> Drive | str:
        """`fallbacks` are acceptable alternatives (in order) for when the goal itself is blocked."""
        pose = t.pose
        if t.blocked == "obstacle_stop":
            # Something is right in front that the map did not know: remember it so the plan goes around.
            self.blocked_count += 1
            if self.blocked_count > 8:
                return "the way is blocked by something I keep running into"
            yaw = math.radians(pose.heading_deg)
            for lateral in (-60.0, 0.0, 60.0):
                t.world.add_obstacle_point(pose.x_mm + 260.0 * math.cos(yaw) - lateral * math.sin(yaw),
                                           pose.y_mm + 260.0 * math.sin(yaw) + lateral * math.cos(yaw))
            self.planned_at = -math.inf
        elif t.blocked in ("bumper", "cliff_detected", "estop_active"):
            return REFLEX_WORDS[t.blocked]
        moved = self.goal is None or math.hypot(goal[0] - self.goal[0], goal[1] - self.goal[1]) > 120.0
        if moved or not self.waypoints or t.now - self.planned_at > self.REPLAN_S:
            # The object I am approaching gets its plain radius: its padded outline would swallow the very spot
            # I am heading for, and the front sensor plus slowing down guard the last few centimetres.
            obstacles = [dataclasses.replace(ob, radius_mm=t.world.objects[ob.label].radius_mm + 10.0)
                         if ob.label in self.ignore_hits_on and ob.label in t.world.objects else ob
                         for ob in t.world.obstacles(exclude=self.exclude)]
            points = occupied_points(t.world, self.ignore_hits_on)
            target = goal
            if fallbacks:
                target = t.planner.first_free([goal, *fallbacks], obstacles, points=points) or goal
            plan = t.planner.plan((pose.x_mm, pose.y_mm), target, obstacles, points=points)
            if not plan.ok:
                if "outside" in plan.message:
                    return "that spot is outside the area I can map"
                return "every way there is blocked by walls or objects; there may not be room for me"
            self.waypoints, self.goal, self.planned_at = list(plan.waypoints), goal, t.now
            self.end = self.waypoints[-1] if self.waypoints else goal
        while len(self.waypoints) > 1 and math.hypot(pose.x_mm - self.waypoints[0][0],
                                                     pose.y_mm - self.waypoints[0][1]) < WAYPOINT_MM:
            self.waypoints.pop(0)
        target = self.waypoints[0] if self.waypoints else goal
        final = len(self.waypoints) <= 1
        return steer_to(pose, *target, v_max=v_max, slow_mm=slow_mm if final else 150.0)


class Behavior:
    name = "behavior"
    persistent = False  # keeps running after the turn that started it (e.g. following)
    timeout_s = 90.0

    def __init__(self) -> None:
        self.started_at: float | None = None

    def tick(self, t: Tick) -> Drive | Done:
        if self.started_at is None:
            self.started_at = t.now
        if not self.persistent and t.now - self.started_at > self.timeout_s:
            return Done(False, f"{self.describe()} took too long, so I stopped")
        return self.step(t)

    def step(self, t: Tick) -> Drive | Done:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class Turn(Behavior):
    name = "turn"

    def __init__(self, angle_deg: float) -> None:
        super().__init__()
        self.angle = float(angle_deg)
        self.turned = 0.0
        self.last: float | None = None

    def step(self, t: Tick) -> Drive | Done:
        if self.last is not None:
            self.turned += wrap(t.pose.heading_deg - self.last)
        self.last = t.pose.heading_deg
        remaining = self.angle - self.turned
        if abs(remaining) < 1.5:
            return Done(True, f"Turned {abs(self.angle):.0f} degrees.")
        return Drive(0.0, turn_rate(remaining, gain=2.5, min_rate=20.0))

    def describe(self) -> str:
        return f"turning {abs(self.angle):.0f} degrees"


class Move(Behavior):
    name = "move"

    def __init__(self, distance_mm: float) -> None:
        super().__init__()
        self.distance = float(distance_mm)
        self.origin: BodyPose | None = None

    def step(self, t: Tick) -> Drive | Done:
        if self.origin is None:
            self.origin = t.pose
        o = self.origin
        yaw = math.radians(o.heading_deg)
        done = (t.pose.x_mm - o.x_mm) * math.cos(yaw) + (t.pose.y_mm - o.y_mm) * math.sin(yaw)
        remaining = self.distance - done
        direction = "forward" if self.distance > 0 else "backward"
        if abs(remaining) < 10.0:
            return Done(True, f"Moved {direction} {abs(self.distance) / 10:.0f} cm.")
        if t.blocked and (self.distance > 0 or t.blocked != "obstacle_stop"):
            return Done(False, f"Moved only {abs(done) / 10:.0f} of {abs(self.distance) / 10:.0f} cm: "
                               f"{REFLEX_WORDS.get(t.blocked, t.blocked)}.")
        speed = math.copysign(min(250.0, max(60.0, 2.0 * abs(remaining))), remaining)
        return Drive(speed, turn_rate(wrap(o.heading_deg - t.pose.heading_deg)))


class Face(Behavior):
    name = "face"

    def __init__(self, obj_id: str) -> None:
        super().__init__()
        self.obj_id = obj_id

    def step(self, t: Tick) -> Drive | Done:
        obj = t.world.objects.get(self.obj_id)
        if obj is None:
            return Done(False, f"I don't know where {self.obj_id} is any more")
        err = bearing_to(t.pose, obj.x_mm, obj.y_mm)
        if abs(err) < FACE_TOLERANCE_DEG:
            return Done(True, f"Facing {self.obj_id}.", {"target": self.obj_id})
        return Drive(0.0, turn_rate(err, min_rate=20.0))


@dataclass
class Relation:
    """Where to stand relative to an object; "behind"/"left of" are as seen from where the task began."""

    kind: str  # near | in_front_of | behind | left_of | right_of | between | on
    gap_mm: float = ROBOT_CLEARANCE_MM
    other_id: str | None = None
    view: tuple[float, float] = (1.0, 0.0)  # unit vector from the robot to the object at the start

    def goal(self, world: SpatialMap, obj: TrackedObject, gap_mm: float | None = None, swing_deg: float = 0.0
             ) -> tuple[float, float] | None:
        """`swing_deg` turns the spot around the object (keeping its side), for when walls block it."""
        ux, uy = self.view
        if swing_deg:
            c, s = math.cos(math.radians(swing_deg)), math.sin(math.radians(swing_deg))
            ux, uy = ux * c - uy * s, ux * s + uy * c
        gap = obj.radius_mm + (self.gap_mm if gap_mm is None else gap_mm)
        if self.kind == "between":
            other = world.objects.get(self.other_id or "")
            if other is None:
                return None
            return 0.5 * (obj.x_mm + other.x_mm), 0.5 * (obj.y_mm + other.y_mm)
        if self.kind == "on":
            return obj.x_mm, obj.y_mm
        if self.kind == "behind":
            return obj.x_mm + ux * gap, obj.y_mm + uy * gap
        if self.kind in ("left_of", "right_of"):
            side = 1.0 if self.kind == "left_of" else -1.0
            return obj.x_mm - uy * gap * side, obj.y_mm + ux * gap * side
        return obj.x_mm - ux * gap, obj.y_mm - uy * gap


class GoToObject(Behavior):
    name = "go_to"
    LOST_S = 3.0

    def __init__(self, obj_id: str, relation: Relation) -> None:
        super().__init__()
        self.obj_id = obj_id
        self.relation = relation
        exclude = {obj_id, relation.other_id} if relation.kind == "between" else \
            {obj_id} if relation.kind == "on" else set()
        self.follower = PathFollower({e for e in exclude if e}, ignore_hits_on={obj_id})
        self.phase = "drive"
        self.missing_since: float | None = None
        self.closest = False

    def describe(self) -> str:
        return f"going {self.relation.kind.replace('_', ' ')} {self.obj_id}"

    def step(self, t: Tick) -> Drive | Done:
        obj = t.world.objects.get(self.obj_id)
        if obj is None:
            self.missing_since = self.missing_since or t.now
            if t.now - self.missing_since > self.LOST_S:
                return Done(False, f"I lost track of {self.obj_id}")
            return Drive(0.0, 0.0)
        self.missing_since = None
        goal = self.relation.goal(t.world, obj)
        if goal is None:
            return Done(False, f"I lost track of {self.relation.other_id}")
        pose = t.pose
        if self.phase == "drive":
            near = math.hypot(pose.x_mm - goal[0], pose.y_mm - goal[1]) <= ARRIVE_MM
            if near or (self.follower.at_end(pose) and not self.follower.waypoints[1:]):
                self.closest = not near
                self.phase = "face" if self.relation.kind != "on" else "done"
            else:
                # If the exact spot is blocked (a wall behind the box), try a little closer, then swing the spot
                # around the object at the same distance, so "behind" stays behind rather than drifting to
                # whatever free cell is nearest.
                fallbacks = [g for g in (self.relation.goal(t.world, obj, gap, swing)
                                         for swing in (0.0, 15.0, -15.0, 30.0, -30.0, 40.0, -40.0)
                                         for gap in (self.relation.gap_mm, 260.0))
                             if g is not None] if self.relation.kind in ("behind", "near", "in_front_of",
                                                                         "left_of", "right_of") else None
                drive = self.follower.drive(t, goal, fallbacks=fallbacks)
                if isinstance(drive, str):
                    if self.relation.kind == "on" and ("keep running into" in drive or "Bumper" in drive) and \
                            math.hypot(pose.x_mm - obj.x_mm, pose.y_mm - obj.y_mm) < obj.radius_mm + 450.0:
                        return Done(True, f"{self.obj_id} is solid, so I can't drive onto it; I'm right next to it.",
                                    {"target": self.obj_id})
                    return Done(False, f"I couldn't get {self.relation.kind.replace('_', ' ')} {self.obj_id}: "
                                       f"{drive}")
                return drive
        if self.phase == "face":
            err = bearing_to(pose, obj.x_mm, obj.y_mm)
            if abs(err) > FACE_TOLERANCE_DEG:
                return Drive(0.0, turn_rate(err, min_rate=20.0))
        place = self.relation.kind.replace("_", " ")
        target = self.obj_id + (f" and {self.relation.other_id}" if self.relation.kind == "between" else "")
        dist = math.hypot(pose.x_mm - obj.x_mm, pose.y_mm - obj.y_mm) / 1000.0
        seen = t.now - obj.last_seen < 2.0
        note = f" I can see {self.obj_id} {dist:.1f} m away." if seen else \
            f" I can't see {self.obj_id} from here right now."
        return Done(True, f"Arrived {place} {target}" + (" (as close as the space allows)." if self.closest else ".")
                    + note, {"target": self.obj_id, "confirmed": seen})


class GoToPoint(Behavior):
    name = "go_to_point"

    def __init__(self, point: tuple[float, float], label: str = "there") -> None:
        super().__init__()
        self.point = point
        self.label = label
        self.follower = PathFollower()

    def describe(self) -> str:
        return f"going to {self.label}"

    def step(self, t: Tick) -> Drive | Done:
        if math.hypot(t.pose.x_mm - self.point[0], t.pose.y_mm - self.point[1]) <= ARRIVE_MM:
            return Done(True, f"I'm at {self.label}.")
        if self.follower.at_end(t.pose) and not self.follower.waypoints[1:]:
            return Done(True, f"I'm as close to {self.label} as the space allows.")
        drive = self.follower.drive(t, self.point)
        return Done(False, f"I couldn't get to {self.label}: {drive}") if isinstance(drive, str) else drive


class Follow(Behavior):
    """Stay a set distance from a (possibly moving) object, facing it, until told otherwise."""

    name = "follow"
    persistent = True
    LOST_S = 5.0

    def __init__(self, obj_id: str, distance_mm: float = 600.0) -> None:
        super().__init__()
        self.obj_id = obj_id
        self.distance = distance_mm
        self.follower = PathFollower()

    def describe(self) -> str:
        return f"following {self.obj_id}"

    def step(self, t: Tick) -> Drive | Done:
        obj = t.world.objects.get(self.obj_id)
        if obj is None or t.now - obj.last_seen > self.LOST_S:
            return Done(False, f"I lost sight of {self.obj_id} while following it")
        pose = t.pose
        dx, dy = obj.x_mm - pose.x_mm, obj.y_mm - pose.y_mm
        dist = math.hypot(dx, dy)
        keep = self.distance + obj.radius_mm
        err = bearing_to(pose, obj.x_mm, obj.y_mm)
        if abs(dist - keep) < 120.0:
            return Drive(0.0, turn_rate(err) if abs(err) > 4.0 else 0.0)
        if dist < keep:  # too close: back off while still facing it
            return Drive(-min(200.0, 1.5 * (keep - dist)), turn_rate(err))
        goal = (obj.x_mm - dx / dist * keep, obj.y_mm - dy / dist * keep)
        drive = self.follower.drive(t, goal, v_max=min(MAX_V + 60.0, 160.0 + 0.6 * (dist - keep)), slow_mm=250.0)
        return Drive(0.0, turn_rate(err)) if isinstance(drive, str) else drive


class Search(Behavior):
    """Turn smoothly while the fast detector watches, until the thing turns up (then face it)."""

    name = "search"
    SPEED = 55.0

    def __init__(self, query: str, matches: Callable[[TrackedObject], int] | None = None) -> None:
        super().__init__()
        self.query = query
        self.matches = matches or (lambda obj: label_score(obj.label, query))
        self.turned = 0.0
        self.last: float | None = None
        self.found: str | None = None
        self.synonym: str | None = None  # asked for a box, saw a cube: settle for it if nothing better turns up

    def describe(self) -> str:
        return f"looking for {self.query}"

    def step(self, t: Tick) -> Drive | Done:
        if self.found is None:
            fresh = [o for o in t.world.objects.values()
                     if o.last_seen >= (self.started_at or 0.0) and o.confidence >= 0.45 and self.matches(o) >= 2]
            if fresh:
                self.found = min(fresh, key=lambda o: o.relative(t.pose)[0]).id
            elif self.synonym is None:
                near = [o for o in t.world.objects.values()
                        if o.last_seen >= (self.started_at or 0.0) and o.confidence >= 0.45 and self.matches(o) == 1]
                self.synonym = near[0].id if near else None
        if self.found is not None:
            obj = t.world.objects.get(self.found)
            if obj is None:
                self.found = None
            else:
                err = bearing_to(t.pose, obj.x_mm, obj.y_mm)
                if abs(err) < FACE_TOLERANCE_DEG:
                    return Done(True, f"Found it and turned to face it: {t.world.describe(obj, t.pose)}.",
                                {"found": obj.id})
                return Drive(0.0, turn_rate(err, min_rate=20.0))
        if self.last is not None:
            self.turned += abs(wrap(t.pose.heading_deg - self.last))
        self.last = t.pose.heading_deg
        remaining = 360.0 - self.turned
        if remaining <= 1.0:
            if self.synonym is not None and self.synonym in t.world.objects:
                self.found = self.synonym
                return Drive(0.0, 0.0)
            return Done(False, f"I looked all around but did not find any {self.query}.")
        # Ease into exactly one turn, so a search that finds nothing leaves me facing the way I started.
        return Drive(0.0, min(self.SPEED, max(15.0, 2.5 * remaining)))


class Scan(Behavior):
    """One full, smooth turn so perception sees the whole room; ends facing the way it started."""

    name = "scan"

    def __init__(self) -> None:
        super().__init__()
        self.turn = Turn(360.0)

    def describe(self) -> str:
        return "looking all around"

    def step(self, t: Tick) -> Drive | Done:
        out = self.turn.step(t)
        if isinstance(out, Done):
            return Done(True, "I looked all around.")
        return Drive(0.0, max(-75.0, min(75.0, out.w)))


class Orbit(Behavior):
    """Drive a loop through points on a circle around a centre, skipping points that are blocked."""

    name = "circle_around"

    def __init__(self, center: tuple[float, float], radius_mm: float, laps: float, start_angle: float,
                 label: str) -> None:
        super().__init__()
        self.label = label
        cx, cy = center
        points = max(8, int(round(2 * math.pi * radius_mm / 350.0)))
        total = int(round(points * max(0.25, min(3.0, laps))))
        self.center = center
        self.points = [(cx + radius_mm * math.cos(start_angle + 2 * math.pi * k / points),
                        cy + radius_mm * math.sin(start_angle + 2 * math.pi * k / points)) for k in range(1, total + 1)]
        self.total = total
        self.skipped = 0
        self.follower = PathFollower()

    def describe(self) -> str:
        return f"circling {self.label}"

    def step(self, t: Tick) -> Drive | Done:
        while self.points:
            target = self.points[0]
            reached_closest = self.follower.goal == target and self.follower.at_end(t.pose)
            if math.hypot(t.pose.x_mm - target[0], t.pose.y_mm - target[1]) < 180.0 or reached_closest:
                self.points.pop(0)  # there, or as close as walls allow (a point inside the wall margin)
                continue
            drive = self.follower.drive(t, target, v_max=240.0, slow_mm=1.0)
            if isinstance(drive, str):
                self.points.pop(0)
                self.skipped += 1
                self.follower = PathFollower()
                continue
            return drive
        if self.skipped >= self.total:
            return Done(False, f"I couldn't drive around {self.label}: walls or obstacles block the way.")
        err = bearing_to(t.pose, *self.center)
        if abs(err) > FACE_TOLERANCE_DEG:
            return Drive(0.0, turn_rate(err, min_rate=20.0))
        note = f" ({self.skipped} of {self.total} points were blocked, so I went past them)" if self.skipped else ""
        return Done(True, f"Drove a loop around {self.label}{note}.")
