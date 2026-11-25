"""A 2D kinematic world behind the robot API: fast enough for the suite, honest enough to trust its checks.

The floor is an occupancy grid with walls; things are labelled discs with facts, and people can walk routes. The
robot is a 0.30 m disc under the firmware's speed limits, and the collision monitor, camera range, live vocabulary,
object memory and camera coverage behave as on the robot, simplified. `ask` stands in for the vision model by
answering from each thing's facts (its real accuracy is measured separately on photos).
"""

from __future__ import annotations

import heapq
import math
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot.agent.api import (
    Matches,
    NotFound,
    Object,
    PathBlocked,
    Point,
    Pose,
    Preempted,
    SafetyStop,
    TargetLost,
    choose,
    geom,
    side,
    wrap,
    xy,
)
from robot.body.controller import BodyController, Progress
from robot.perception.tracker import canonical

RES = 0.05  # occupancy cell, m
SEARCH_STOPS = 20  # most places one search drives to before it gives up
BEARINGS = {"front": 0.0, "front-left": 45.0, "left": 90.0, "behind-left": 135.0, "behind": 180.0,
            "behind-right": -135.0, "right": -90.0, "front-right": -45.0,  # degrees from the robot's heading, + left
            "rear-left": 120.0, "rear-right": -120.0}  # where those pictures look: the model named them for go_toward
PLAN = 2  # planning cell = PLAN * RES
NEARBY = 1.0  # metres beyond its usual reach a thing may be approached when no free floor is nearer (see _route)
DT = 0.05  # simulation step, s
RADIUS = 0.15  # the robot
V_MAX, W_MAX, TURN_RATE, CRUISE = 0.5, 1.5, 1.0, 0.4  # firmware ceilings and normal speeds
STOP_ZONE, SLOW_ZONE, PERSON_ZONE = 0.25, 0.6, 1.5  # collision monitor
SEE_RANGE, PERCEIVE_EVERY, STALL_S, ASK_S = 4.5, 0.2, 10.0, 2.0
SPIN_RATES = {"slow": 0.5, "normal": 1.0, "fast": 1.5}  # rad/s
MOTION = {
    "move",
    "turn",
    "spin",
    "face",
    "go_to",
    "drive",
    "drive_path",
    "stop",
    "orbit",
    "follow",
    "express",
}
CAMERA_YAWS = {"front_camera": 0.0, "rear_left_camera": 2 * math.pi / 3, "rear_right_camera": -2 * math.pi / 3}
CAMERA_HFOV, CAMERA_HEIGHT, WALL_H = 2.44, 0.22, 2.4  # as on the robot (robot/ros/model.py)
HEIGHTS = {  # metres, for drawing things in the simulated cameras
    "person": 1.7, "chair": 0.9, "sofa": 0.85, "couch": 0.85, "table": 0.75, "desk": 0.75, "box": 0.35, "ball": 0.22,
    "backpack": 0.45, "bag": 0.45, "plant": 0.9, "lamp": 1.5, "door": 2.0, "tv": 1.0, "fridge": 1.8, "bed": 0.55,
    "cup": 0.1, "mug": 0.1, "bottle": 0.25, "shelf": 1.8, "cabinet": 1.0, "bin": 0.5, "keys": 0.05, "shoe": 0.1,
    "book": 0.05, "laptop": 0.05, "phone": 0.03, "cat": 0.3, "dog": 0.55, "toy": 0.2, "pillow": 0.2, "charger": 0.05,
}
RGB = {
    "red": (200, 40, 40), "orange": (235, 130, 30), "yellow": (230, 200, 40), "green": (60, 150, 60),
    "blue": (50, 90, 200), "purple": (130, 60, 170), "pink": (230, 130, 170), "brown": (125, 82, 45),
    "black": (35, 35, 38), "white": (238, 238, 232), "grey": (128, 128, 128), "gray": (128, 128, 128),
    "person": (60, 70, 110),
}
BASE_VOCABULARY = frozenset("""person chair sofa couch table desk box ball backpack bag cup mug bottle plant lamp tv bed
door shoe book laptop phone cat dog charger bin toy pillow shelf cabinet fridge sink window basket""".split())
COLORS = frozenset("red orange yellow green blue purple pink brown black white grey gray".split())
SYNONYMS = {
    "table": ("desk", "coffee table", "work table"),
    "desk": ("table", "coffee table", "work table"),
    "couch": ("sofa", "chair"),
    "sofa": ("couch", "seat"),
    "mug": ("cup",),
    "cup": ("mug",),
    "bag": ("backpack",),
    "backpack": ("bag",),
    "trash can": ("bin",),
    "bin": ("trash can",),
}


def _words(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words}


def visual_name(description: str) -> str:
    """The object name used for detection; colour is measured separately on the candidate."""
    words = description.lower().split()
    if words and words[0] in ("a", "an", "the"):
        words = words[1:]
    return " ".join(w for w in words if w not in COLORS)


@dataclass
class Thing:
    label: str
    x: float
    y: float
    radius: float = 0.2
    color: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    names: tuple[str, ...] = ()  # other phrases a person might use: "seat", "something to sit on"
    solid: bool = True
    route: tuple[Point, ...] = ()  # people walk it once at `speed`, then stay
    speed: float = 0.5
    appear_at: float = 0.0
    user: bool = False
    id: int = 0

    @property
    def person(self) -> bool:
        return self.label == "person"

    def at(self, t: float) -> Point | None:
        """Where it is at time t; None before it appears."""
        if t < self.appear_at:
            return None
        here, left = (self.x, self.y), (t - self.appear_at) * self.speed
        for nxt in self.route:
            step = math.dist(here, nxt)
            if left <= step:
                f = left / step if step else 0.0
                return here[0] + (nxt[0] - here[0]) * f, here[1] + (nxt[1] - here[1]) * f
            here, left = nxt, left - step
        return here

    def score(self, phrase: str) -> int:
        """How well a phrase names this thing: 3 whole name, 2 label word, 1 other name word, 0 no match."""
        p = phrase.lower().strip()
        if p in self.names or p == self.label:
            return 3
        words = _words(p)
        if words & _words(self.label):
            return 2
        return 1 if any(words & _words(n) for n in self.names) else 0


@dataclass
class Scene:
    width: float
    height: float  # the floor spans x in [-width/2, width/2] and y in [-height/2, height/2]
    things: list[Thing]
    rooms: dict[str, tuple[float, float, float, float]]  # id -> x0, y0, x1, y1
    walls: list[tuple[Point, Point]] = field(default_factory=list)  # inside the outer walls
    start: Pose = Pose(0.0, 0.0, 0.0)
    replies: list[tuple[float, str]] = field(default_factory=list)  # what the user says, and from when
    notes: list[str] = field(default_factory=list)  # remembered from earlier tasks
    seen: list[tuple[str, Point]] = field(default_factory=list)  # in the object map from earlier, maybe stale
    stop_at: float | None = None  # the user says "stop" at this time
    noise: float = 0.0  # 0 sees perfectly; 1 is a poor detector: misses grow with distance, positions jitter
    seed: int = 0  # for the noise, so a run can be repeated
    snag_at: tuple[float, float] | None = None
    cliffs: list[tuple[float, float, float]] = field(default_factory=list)  # (x, y, radius)
    steps: list[tuple[float, float, float, float]] = field(default_factory=list)  # (x, y, radius, delta_z)


@dataclass
class Event:
    """One API call, for the readback: when, what, and where the robot was when it started."""

    t: float
    name: str
    args: dict[str, Any]
    pose: tuple[float, float, float]  # x, y, yaw in radians
    result: str = ""


@dataclass
class Trace:
    poses: list[tuple[float, float, float, float]] = field(default_factory=list)  # t, x, y, yaw
    said: list[tuple[float, str]] = field(default_factory=list)
    asked: list[tuple[float, str, str]] = field(default_factory=list)
    listened: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    bumps: int = 0
    calls: int = 0
    events: list[Event] = field(default_factory=list)
    chimes: list[tuple[float, str]] = field(default_factory=list)
    postures: list[tuple[float, str]] = field(default_factory=list)


class BudgetSpent(Preempted):
    """A task used up its simulated time (time running out is a kind of preemption)."""


class SimRobot:
    """The robot API on a simulated floor. `realtime` paces motion like the real robot (for preemption tests)."""

    def __init__(self, scene: Scene, *, realtime: bool = False, max_time: float = 600.0) -> None:
        self.scene = scene
        self.realtime, self.max_time = realtime, max_time
        for i, thing in enumerate(scene.things, 1):
            thing.id = thing.id or i
        self.things = {t.id: t for t in scene.things}
        self.t = 0.0
        self._task_t0: float = 0.0
        self.x, self.y, self.yaw = scene.start.x, scene.start.y, math.radians(scene.start.yaw)
        self.v_now = self.w_now = 0.0
        self.held = False  # the collision monitor stopped the last step
        self._rng = random.Random(scene.seed)
        self._detected: set[int] = set()  # ids the detector reported at the last perception tick
        self._behaviour: Any = None  # a background skill's controller: () -> (v, w)
        self.ended_by = ""  # why the program was stopped: "cancel", "stop word" or "budget"
        self.vocabulary = set(BASE_VOCABULARY)
        self.memory: dict[int, tuple[Point, float]] = {}  # id -> where and when last seen
        for label, where in scene.seen:
            match = max(self.things.values(), key=lambda t: t.score(label))
            self.memory[match.id] = (where, -60.0)
        self.notes = list(scene.notes)
        self.replies = sorted(scene.replies)
        self.places: dict[str, Pose] = {"start": Pose(self.x, self.y, math.degrees(self.yaw))}
        self.trace = Trace(poses=[(0.0, self.x, self.y, self.yaw)])
        self.body = BodyController()
        self.posture = "neutral"
        self._cancel = threading.Event()
        self._last_perceive = -1.0
        self._build_grid()
        for _ in range(10 if scene.noise else 1):  # with noise, the robot has watched for 2 s before the task
            self._last_perceive = -1.0
            self._perceive()

    # ---- world geometry ---------------------------------------------------------------------------------------
    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x + self.scene.width / 2) / RES), int((y + self.scene.height / 2) / RES)

    def _centre(self, i: int, j: int, res: float = RES) -> Point:
        return i * res + res / 2 - self.scene.width / 2, j * res + res / 2 - self.scene.height / 2

    def _build_grid(self) -> None:
        w, h = self.scene.width, self.scene.height
        nx, ny = round(w / RES), round(h / RES)
        walls = np.zeros((nx, ny), bool)
        walls[[0, -1], :] = walls[:, [0, -1]] = True
        for a, b in self.scene.walls:
            for k in range(int(math.dist(a, b) / (RES / 2)) + 1):
                f = k * RES / 2 / max(math.dist(a, b), 1e-9)
                i, j = self._cell(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
                walls[max(0, i - 1):i + 2, max(0, j - 1):j + 2] = True
        self.walls = walls
        xs, ys = np.meshgrid(*(np.arange(n) * RES + RES / 2 - size / 2 for n, size in ((nx, w), (ny, h))), indexing="ij")
        self.occupied = walls.copy()
        for t in self.things.values():
            if t.solid and not t.route and not t.person:
                self.occupied |= (xs - t.x) ** 2 + (ys - t.y) ** 2 <= t.radius ** 2
        self.set_occupied(self.occupied)
        self.coverage = np.zeros_like(walls)

    def set_occupied(self, occupied: np.ndarray) -> None:
        """Use this occupancy grid (a saved map, say) and derive the planning grids from it: a cell is blocked when
        any occupied cell lies within the robot's radius plus a margin."""
        nx, ny = occupied.shape
        self.occupied = occupied
        k = math.ceil((RADIUS + 0.03) / RES)
        inflated = np.zeros_like(occupied)
        padded = np.pad(occupied, k)
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                if di * di + dj * dj <= k * k:
                    inflated |= padded[k + di:k + di + nx, k + dj:k + dj + ny]
        self.inflated = inflated
        self.blocked = inflated[:nx - nx % PLAN, :ny - ny % PLAN].reshape(nx // PLAN, PLAN, ny // PLAN, PLAN).any(axis=(1, 3))

    def _occ(self, x: float, y: float, grid: np.ndarray | None = None) -> bool:
        grid = self.occupied if grid is None else grid
        i, j = self._cell(x, y)
        return not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]) or bool(grid[i, j])

    def _people(self) -> list[tuple[Thing, Point]]:
        return [(t, p) for t in self.things.values() if t.person and (p := t.at(self.t)) is not None]

    def _where(self, thing: Thing) -> Point | None:
        return thing.at(self.t)

    def _line_of_sight(self, a: Point, b: Point) -> bool:
        n = max(1, int(math.dist(a, b) / RES))
        return not any(self._occ(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n, self.walls)
                       for k in range(1, n))

    # ---- perception ---------------------------------------------------------------------------------------------
    def _visible(self) -> list[Thing]:
        """Everything a camera can see now: 360 degrees, within range, not behind a wall."""
        out = []
        for t in self.things.values():
            p = self._where(t)
            if p is not None and math.dist(p, (self.x, self.y)) <= SEE_RANGE and self._line_of_sight((self.x, self.y), p):
                out.append(t)
        return out

    def _detectable(self, t: Thing) -> bool:
        words = _words(t.label) | {w for n in t.names for w in _words(n)}
        return bool(words & self.vocabulary)

    def _perceive(self) -> None:
        if self.t - self._last_perceive < PERCEIVE_EVERY - 1e-9:
            return
        self._last_perceive = self.t
        self._perceive_objects()
        self._cover()

    def _perceive_objects(self) -> None:
        """What the detector reports this tick, into memory: here from the scene (the RealRobot uses its cameras)."""
        noise, self._detected = self.scene.noise, set()
        for t in self._visible():
            if not self._detectable(t):
                continue
            p = self._where(t)
            if noise:  # a real detector misses things, more often far away, and places them a little off
                d = math.dist(p, (self.x, self.y))
                if self._rng.random() < noise * (0.15 + 0.6 * d / SEE_RANGE):
                    continue
                s = noise * (0.03 + 0.03 * d)
                p = (p[0] + self._rng.gauss(0.0, s), p[1] + self._rng.gauss(0.0, s))
            self._detected.add(t.id)
            self.memory[t.id] = (p, self.t)

    def _cover(self) -> None:
        """Camera coverage: floor cells within range along unobstructed rays."""
        angles = np.linspace(-math.pi, math.pi, 720, endpoint=False)  # 4 cm apart at full range
        steps = np.arange(RADIUS, SEE_RANGE, RES)
        px = self.x + np.outer(np.cos(angles), steps)
        py = self.y + np.outer(np.sin(angles), steps)
        ii = np.clip(((px + self.scene.width / 2) / RES).astype(int), 0, self.walls.shape[0] - 1)
        jj = np.clip(((py + self.scene.height / 2) / RES).astype(int), 0, self.walls.shape[1] - 1)
        hit = np.cumsum(self.walls[ii, jj], axis=1) > 0
        self.coverage[ii[~hit], jj[~hit]] = True

    def _target(self, target: Any) -> Any:
        """A where or detect result passed whole means its best match; a str names a saved place, a room or a thing."""
        if isinstance(target, str):
            return self._named(target)
        if isinstance(target, list) and target and all(isinstance(v, (Object, Pose)) for v in target):
            return target[0]
        xy(target)  # raises NotFound for an empty list and a clear TypeError for anything else
        return target

    def _named(self, name: str) -> Pose | Point | Object:
        raw = str(name).strip()
        lowered = raw.lower()
        key = lowered.removeprefix("the ").strip()

        # 1. User alias resolution
        user_aliases = {"user", "me", "you", "the user", "person(user)", "human", "speaker"}
        if (
            lowered in user_aliases
            or key in {"user", "me", "you", "person(user)", "human", "speaker"}
            or (key in ("person", "the person") and self.user() is not None)
        ):
            u = self.user()
            if u is not None:
                return u
            raise NotFound(f"{name!r} is the user, but no user is present")

        # 2. Saved place or room
        if key in self.places:
            return self.places[key]
        if key in self.scene.rooms:  # the free spot nearest the room's middle that can be reached
            x0, y0, x1, y1 = self.scene.rooms[key]
            reach, mid = self._reachable(), ((x0 + x1) / 2, (y0 + y1) / 2)
            cells = [
                (i, j)
                for i, j in zip(*np.nonzero(reach), strict=True)
                if x0 < self._centre(i, j, RES * PLAN)[0] < x1 and y0 < self._centre(i, j, RES * PLAN)[1] < y1
            ]
            if cells:
                i, j = min(cells, key=lambda c: math.dist(self._centre(*c, RES * PLAN), mid))
                return self._centre(i, j, RES * PLAN)
            raise PathBlocked(f"no reachable spot in the {key}")

        # 3. Embedded integer ID e.g. "10 table", "10  table", "desk 7", "#7"
        m = re.search(r"^#?(\d+)\b", raw) or re.search(r"\b(\d+)$", raw)
        if m:
            oid = int(m.group(1))
            if oid in self.things:
                return self.obj(oid)

        # 4. Seen/remembered object query
        seen = self._remembered(key)
        if seen:
            return seen[0]
        places = ", ".join(repr(n) for n in [*self.places, *self.scene.rooms])
        raise NotFound(f"{name!r} is not a saved place, a room ({places}) or anything seen")

    def _object(self, t: Thing, where: Point | None = None, seen_at: float | None = None) -> Object:
        p = where or self._where(t) or (t.x, t.y)
        bearing = round(math.degrees(wrap(math.atan2(p[1] - self.y, p[0] - self.x) - self.yaw)), 1)
        return Object(t.id, t.label, (round(p[0], 3), round(p[1], 3)), round(math.dist(p, (self.x, self.y)), 3),
                      bearing, side(bearing), t.radius, t.color, round(self.t - (self.t if seen_at is None else seen_at), 1))

    # ---- time and motion ----------------------------------------------------------------------------------------
    def _halt(self) -> None:
        """End of a motion: the robot stands still (the ROS backend also sends the stop)."""
        self.v_now = self.w_now = 0.0

    def cancel(self) -> None:
        """Stop the running program now (the runner calls this from another thread)."""
        self._cancel.set()
        self._halt()

    def _interrupts(self) -> None:
        if self._cancel.is_set():
            self._halt()
            self.ended_by = self.ended_by or "cancel"
            raise Preempted("stopped")
        if self.scene.stop_at is not None and self.t >= self.scene.stop_at:
            self._halt()
            self.ended_by = "stop word"
            raise Preempted('the user said "stop"')
        if self.t - getattr(self, "_task_t0", 0.0) > self.max_time:
            self.ended_by = "budget"
            raise BudgetSpent(f"used {self.max_time:.0f} s of simulated time")

    def _clearance(self, direction: float) -> float:
        """Free distance from the robot's edge to the nearest obstacle in the direction of travel."""
        c, s = math.cos(self.yaw) * direction, math.sin(self.yaw) * direction
        people = [p for _, p in self._people()]
        for d in np.arange(0.0, SLOW_ZONE + RES, RES):
            for lat in (-RADIUS, 0.0, RADIUS):
                px, py = self.x + c * (RADIUS + d) - s * lat, self.y + s * (RADIUS + d) + c * lat
                if self._occ(px, py) or any(math.dist((px, py), q) < 0.25 for q in people):
                    return d
        return math.inf

    def _step(self, v: float, w: float, dt: float = DT) -> None:
        self._interrupts()
        if self._behaviour is not None and not v and not w:  # a background skill drives while the program waits
            v, w = self._behaviour()
        v, w = max(-V_MAX, min(V_MAX, v)), max(-W_MAX, min(W_MAX, w))
        self.held = False
        if v:
            gap = self._clearance(1.0 if v > 0 else -1.0)
            self.held = gap < STOP_ZONE
            v = 0.0 if self.held else v * 0.5 if gap < SLOW_ZONE else v
            if any(math.dist(p, (self.x, self.y)) < PERSON_ZONE for _, p in self._people()):
                v *= 0.5
        yaw = self.yaw + w * dt
        nx, ny = self.x + v * dt * math.cos(yaw), self.y + v * dt * math.sin(yaw)
        if v and any(self._occ(nx + RADIUS * math.cos(a), ny + RADIUS * math.sin(a)) for a in np.linspace(0, 6.28, 12)):
            self.trace.bumps += 1
            v, nx, ny = 0.0, self.x, self.y
        self.x, self.y, self.yaw = nx, ny, wrap(yaw)
        self.v_now, self.w_now = v, w
        self.t += dt
        self.trace.poses.append((round(self.t, 3), self.x, self.y, self.yaw))
        self._perceive()
        if self.realtime:
            time.sleep(dt)

    def _rotate(self, angle: float, rate: float = TURN_RATE) -> None:
        n = max(1, math.ceil(abs(angle) / (rate * DT)))
        for _ in range(n):
            self._step(0.0, angle / (n * DT))
        self._halt()

    def _plan(self, goal: Point, reach: float) -> list[Point] | None:
        """A* on the planning grid to any free cell within `reach` of the goal, with the path shortened."""
        blocked = self.blocked.copy()
        res = RES * PLAN
        nx, ny = blocked.shape
        for _, (px, py) in self._people():
            i0, j0 = int((px + self.scene.width / 2) / res), int((py + self.scene.height / 2) / res)
            k = math.ceil((0.25 + RADIUS + 0.1) / res)  # a person plus the robot plus a margin, as the monitor needs
            blocked[max(0, i0 - k):i0 + k + 1, max(0, j0 - k):j0 + k + 1] = True
        si, sj = int((self.x + self.scene.width / 2) / res), int((self.y + self.scene.height / 2) / res)
        blocked[max(0, si - 1):si + 2, max(0, sj - 1):sj + 2] = False

        def centre(i: int, j: int) -> Point:
            return self._centre(i, j, res)

        def h(i: int, j: int) -> float:
            return max(0.0, math.dist(centre(i, j), goal) - reach)

        frontier, came, cost = [(h(si, sj), si, sj)], {(si, sj): None}, {(si, sj): 0.0}
        end = None
        while frontier:
            _, i, j = heapq.heappop(frontier)
            if math.dist(centre(i, j), goal) <= reach:
                end = (i, j)
                break
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and not blocked[a, b]:
                    c = cost[(i, j)] + res * math.hypot(di, dj)
                    if c < cost.get((a, b), math.inf):
                        cost[(a, b)], came[(a, b)] = c, (i, j)
                        heapq.heappush(frontier, (c + h(a, b), a, b))
        if end is None:
            return None
        cells = []
        while end is not None:
            cells.append(end)
            end = came[end]
        points = [centre(i, j) for i, j in reversed(cells)][1:]

        def clear(a: Point, b: Point) -> bool:
            n = max(1, int(math.dist(a, b) / (res / 2)))
            return not any(blocked[min(nx - 1, int((a[0] + (b[0] - a[0]) * k / n + self.scene.width / 2) / res)),
                                   min(ny - 1, int((a[1] + (b[1] - a[1]) * k / n + self.scene.height / 2) / res))]
                           for k in range(n + 1))

        short, here = [], (self.x, self.y)
        k = 0
        while k < len(points):
            far = k
            while far + 1 < len(points) and clear(here, points[far + 1]):
                far += 1
            short.append(points[far])
            here, k = points[far], far + 1
        return short

    def _check_hazard(self) -> Any:
        try:
            from robot.perception.depth import StepHazard
        except ImportError:

            @dataclass
            class StepHazard:
                kind: str
                delta_z: float
                distance: float
                torque_boost: bool = False

        for cx, cy, r in self.scene.cliffs:
            d = math.hypot(cx - self.x, cy - self.y)
            if d <= r + RADIUS:
                return StepHazard(kind="cliff", delta_z=-0.06, distance=d)

        for sx, sy, r, dz in self.scene.steps:
            d = math.hypot(sx - self.x, sy - self.y)
            if d <= r + RADIUS:
                is_trav = abs(dz) <= 0.022
                kind = "traversable" if is_trav else ("cliff" if dz < -0.04 else "obstacle")
                return StepHazard(kind=kind, delta_z=dz, distance=d, torque_boost=(is_trav and dz >= 0.008))
        return None

    def _pursue(
        self,
        points: list[Point],
        speed: float,
        end_tol: float = 0.1,
        goal: Point | None = None,
        reach: float = 0.0,
    ) -> None:
        """Drive through points in order using BodyController.step_pursuit (non-stop waypoint spline pursuit)."""
        progress = Progress((self.x, self.y), self.t, 0.05, STALL_S)
        idx = 0
        waypoints = [(float(p[0]), float(p[1])) for p in points]
        while idx < len(waypoints):
            self._interrupts()
            tx, ty = waypoints[idx]
            d = math.hypot(tx - self.x, ty - self.y)
            last = idx == len(waypoints) - 1
            if (goal is not None and math.dist(goal, (self.x, self.y)) <= reach) or (last and self.held and d < 0.5):
                break

            # Proprioception & cable snag / step hazards
            hazard = self._check_hazard()
            is_snagged = self.scene.snag_at is not None and math.dist((self.x, self.y), self.scene.snag_at) < 0.35
            v_actual = 0.0 if (is_snagged and abs(self.v_now) > 0.08) else self.v_now
            v_expected = (self.v_now or speed) if is_snagged else self.v_now
            status, v_safe, w_safe, torque = self.body.check_proprioception(
                v_expected, self.w_now, v_actual, self.w_now, DT, pitch_rad=0.0, hazard=hazard
            )
            if status == "cliff_stop":
                self._step(v_safe, 0.0)
                self._halt()
                raise SafetyStop("cliff detected by ToF")
            if status == "snag_recovery":
                self._step(v_safe, 0.0)
                self._halt()
                raise PathBlocked("cable snag detected, backed off")

            # Extract moving people for predictive social yielding
            people = []
            for t, p in self._people():
                vx = vy = 0.0
                if t.route and t.speed > 0:
                    p_next = t.at(self.t + 0.5)
                    if p_next:
                        vx = (p_next[0] - p[0]) / 0.5
                        vy = (p_next[1] - p[1]) / 0.5
                people.append((p[0], p[1], vx, vy))

            v_cmd, w_cmd, next_idx, arrived = self.body.step_pursuit(
                (self.x, self.y, self.yaw),
                waypoints,
                idx,
                blend_r=0.28,
                v_cruise=speed,
                occupied_fn=lambda x, y: self._occ(x, y, self.inflated),
                people=people,
            )
            if arrived:
                break

            if progress.stalled((self.x, self.y), self.t, advanced=next_idx > idx):
                self._halt()
                raise PathBlocked(f"no progress toward ({tx:.1f}, {ty:.1f}) for {STALL_S:.0f} s")

            idx = next_idx
            self._step(v_cmd * torque, w_cmd)
        self._halt()

    def _route(self, target: Point | Pose | Object, goal: Point, reach: float) -> list[Point] | None:
        """A path to within `reach` of `goal`; for a thing with no free floor that near, a path to the reachable floor
        nearest it, NEARBY metres further at most. Seen from about 4 m, a bicycle by a wall was mapped 0.6 m into the
        wall's margin; the nearest floor the robot could reach was 1.04 m from that centre and 0.51 m from the bike's
        later one, and go_to had answered "no free path" with the bike in view."""
        path = self._plan(goal, reach)
        if path is not None or not isinstance(target, Object):
            return path
        res = RES * PLAN
        xs, ys = self._centre(*np.nonzero(self._reachable()), res)
        gaps = np.hypot(xs - goal[0], ys - goal[1])
        k = int(np.argmin(gaps))
        return self._plan((float(xs[k]), float(ys[k])), res / 2) if gaps[k] <= reach + NEARBY else None

    def _reach(self, target: Point | Pose | Object, stop_dist: float) -> tuple[Point, float]:
        goal = xy(target)
        if isinstance(target, Object):
            thing = self.things.get(target.id)
            live = self._where(thing) if thing and thing.person else None
            if target.radius == 0.0:  # flat floor target (e.g. rug): drive onto it, or to nearest free floor beside it
                return goal, max(stop_dist, 0.2)
            return live or goal, max(stop_dist, target.radius + RADIUS + STOP_ZONE + 0.05)
        if self._occ(*goal, self.inflated):
            return goal, max(stop_dist, 0.4)
        return goal, max(stop_dist, 0.1)

    # ---- the API ------------------------------------------------------------------------------------------------
    def _call(self, call: str = "", /, **args: Any) -> None:
        """Every API call: count it, log it for the suite's checks, honour a stop."""
        self.trace.calls += 1
        if call:
            self.trace.events.append(Event(round(self.t, 2), call, args, (self.x, self.y, self.yaw)))
        if call in MOTION:  # any new motion replaces a background skill
            self._behaviour = None
        self._interrupts()

    def _result(self, text: str) -> None:
        if self.trace.events:
            self.trace.events[-1].result = text

    def pose(self) -> Pose:
        self._call()
        return Pose(round(self.x, 3), round(self.y, 3), round(math.degrees(self.yaw), 1))

    def battery(self) -> float:
        self._call()
        return round(max(0.0, 1.0 - self.t / 10800.0), 3)

    def user(self) -> Object | None:
        self._call()
        return next((self._object(t, p) for t, p in self._people() if t.user), None)

    def room(self) -> str:
        self._call()
        return self._room_at(self.x, self.y)

    def _room_at(self, x: float, y: float) -> str:
        return next((name for name, (x0, y0, x1, y1) in self.scene.rooms.items() if x0 <= x <= x1 and y0 <= y <= y1),
                    "room")

    def obj(self, id: int) -> Object:  # noqa: A002 - the name programs use
        self._call()
        if isinstance(id, list) and id:  # a where/detect result: refresh its best match
            id = id[0]  # noqa: A001
        t = self.things.get(id.id if isinstance(id, Object) else int(id))
        if t is None:
            raise NotFound(f"there is no object {id}")
        if t in self._visible():
            return self._object(t)
        if t.id in self.memory:
            where, when = self.memory[t.id]
            return self._object(t, where, when)
        raise NotFound(f"object {id} has not been seen")

    def sees_now(self, target: Object) -> bool:
        """Is the thing in view now, where it is mapped (an arrival is confirmed by it)."""
        return target.last_seen <= 0

    # ---- motion -------------------------------------------------------------------------------------------------
    def toward(self, direction: str, metres: float) -> Point:
        """A floor point the planner can reach, `metres` away in a direction from the robot's own (front, front-left,
        left, behind-left, behind, behind-right, right, front-right): that point itself, the path going round what is
        between; where it is inside something or cut off, the reachable floor nearest it, within a metre; else the
        farthest reachable floor on the way there. A point merely free on the map was often one the planner could
        not reach (beyond a table, in a pocket), and the robot answered "blocked" to most of these calls."""
        self._call()
        return self._floor_toward(self.yaw + math.radians(BEARINGS[choose(direction, tuple(BEARINGS), "direction")]),
                                  metres)

    def _floor_toward(self, heading: float, metres: float) -> Point:
        """toward() on a map heading in radians."""
        far = max(0.1, abs(float(metres)))
        target = (self.x + far * math.cos(heading), self.y + far * math.sin(heading))

        def reachable(p: Point) -> bool:
            """Planned as go_to will plan it. Planned to within 0.25 m here and 0.1 m there, 1.8% of these points on
            the apt_0 map were ones go_to then answered "no free path" to."""
            goal, reach = self._reach(p, 0.0)
            return not self._occ(*p, self.inflated) and (
                math.dist(goal, (self.x, self.y)) <= reach or self._plan(goal, reach) is not None)

        if reachable(target):
            return target
        for r in (0.25, 0.5, 0.75, 1.0):
            ring = [(target[0] + r * math.cos(a), target[1] + r * math.sin(a))
                    for a in (heading + math.pi + k * math.pi / 4 for k in range(8))]  # the near side first
            for p in ring:
                if reachable(p):
                    return p
        steps = max(1, int(far / 0.25))
        for k in range(steps - 1, 0, -1):
            p = (self.x + far * k / steps * math.cos(heading), self.y + far * k / steps * math.sin(heading))
            if reachable(p):
                return p
        return (self.x, self.y)

    def move(self, metres: float, direction: str = "forward") -> None:
        if isinstance(metres, str) and isinstance(direction, (int, float)):  # move("forward", 1.0): the types tell
            metres, direction = direction, metres
        self._call("move", metres=metres, direction=direction)
        sign = 1.0 if choose(direction, ("forward", "backward"), "direction") == "forward" else -1.0
        self._move(sign * abs(float(metres)))

    def _move(self, distance: float) -> None:
        direction, left, since = math.copysign(1.0, distance), abs(distance), self.t
        while left > 0.01:
            x, y = self.x, self.y
            self._step(direction * min(CRUISE, max(0.05, 2.0 * left)), 0.0)
            left -= math.dist((x, y), (self.x, self.y))
            if self.held and self.t - since > STALL_S:
                self._halt()
                if left <= min(0.1, 0.1 * abs(distance)):  # close enough: things are placed to a few centimetres
                    return  # (a 1 m reverse stopped 5 cm short of a box placed 0.3 m closer than it stood)
                raise PathBlocked(f"blocked with {left:.2f} m still to go")
            if not self.held:
                since = self.t
        self._halt()

    def turn(self, degrees: float = 90, direction: str = "") -> None:
        if isinstance(degrees, str) and isinstance(direction, (int, float)):  # turn("left", 90): the types tell
            degrees, direction = direction, degrees
        elif isinstance(degrees, str) and not direction:  # turn("left") means a quarter turn, as people say it
            degrees, direction = 90, degrees
        if not direction:
            raise ValueError(f'say which way: robot.turn({degrees}, "left") or robot.turn({degrees}, "right")')
        self._call("turn", degrees=degrees, direction=direction)
        sign = 1.0 if choose(direction, ("left", "right"), "direction") == "left" else -1.0
        self._rotate(sign * math.radians(abs(float(degrees))))

    def spin(self, turns: float = 1.0, seconds: float | None = None, speed: str = "normal",
             direction: str = "left") -> None:
        self._call("spin", turns=turns, seconds=seconds, speed=speed, direction=direction)
        if seconds is None and abs(float(turns)) > 10:
            raise ValueError(f"turns counts full turns (1 = 360 degrees), so {turns} is too many; for degrees use "
                             "robot.turn(degrees, direction)")
        rate = SPIN_RATES[choose(speed, tuple(SPIN_RATES), "speed")]
        sign = 1.0 if choose(direction, ("left", "right"), "direction") == "left" else -1.0
        if seconds is not None:
            for _ in range(max(0, round(float(seconds) / DT))):
                self._step(0.0, sign * rate)
            self._halt()
        else:
            self._rotate(sign * 2 * math.pi * abs(float(turns)), rate)

    def face(self, target: Point | Pose | Object | str, background: bool = False) -> None:
        self._call("face", target=target, background=background)
        target = self._target(target)
        if background:  # keep turning toward it, a moving person included, while the program goes on
            def keep_facing() -> tuple[float, float]:
                tx, ty = self._reach(target, 0.0)[0]
                return 0.0, 2.5 * wrap(math.atan2(ty - self.y, tx - self.x) - self.yaw)
            self._behaviour = keep_facing
            return
        tx, ty = self._reach(target, 0.0)[0]
        self._rotate(wrap(math.atan2(ty - self.y, tx - self.x) - self.yaw))

    def go_to(self, target: Point | Pose | Object | str, distance: float = 0.0, *, side: str = "") -> None:
        self._call("go_to", target=target, distance=distance, side=side)
        self._go(self._target(target), float(distance), side=side)

    def _destination(self, target: Point | Pose | Object, distance: float, side: str) -> tuple:
        return (self.spot(target, side, distance), 0.0) if side else (target, distance)

    def _go(self, target: Point | Pose | Object, distance: float, *, side: str = "") -> None:
        target, distance = self._destination(target, distance, side)
        goal, reach = self._reach(target, distance)
        if math.dist(goal, (self.x, self.y)) <= reach:
            if isinstance(target, Pose):
                self._rotate(wrap(math.radians(target.yaw) - self.yaw))
            return
        attempts = 0
        while True:
            path = self._route(target, goal, reach)
            if path is None:
                raise PathBlocked(f"no free path to ({goal[0]:.1f}, {goal[1]:.1f})")
            if not isinstance(target, Object) and not self._occ(*goal, self.inflated):
                path.append(goal)  # the grid gets close; the last stretch goes to the exact point
            try:
                self._pursue(path, CRUISE, end_tol=0.08, goal=goal, reach=reach)
                break
            except PathBlocked:  # something moved into the way: plan again around it, once
                attempts += 1
                if attempts > 1:
                    raise
        if isinstance(target, Pose):
            self._rotate(wrap(math.radians(target.yaw) - self.yaw))

    def drive(self, *, metres_per_second: float, degrees_per_second: float = 0.0, seconds: float) -> None:
        self._call("drive", metres_per_second=metres_per_second, degrees_per_second=degrees_per_second,
                   seconds=seconds)
        for _ in range(max(0, round(float(seconds) / DT))):
            self._step(float(metres_per_second), math.radians(float(degrees_per_second)))
        self._halt()

    def drive_path(self, points: list[Point], speed: float = 0.3) -> None:
        self._call("drive_path", points=len(points), speed=speed)
        self._drive_path([xy(p) for p in points], float(speed), record=True)

    def _drive_path(self, points: list[Point], speed: float, record: bool) -> None:
        free = []
        for p in points:
            if self._occ(*p, self.inflated):
                if record:
                    self.trace.violations.append(f"the path passes through an obstacle at ({p[0]:.2f}, {p[1]:.2f})")
            else:
                free.append(p)
        if not free:
            return
        # Approach the first point by planning; then track the rest, detouring locally if a stretch is cut off.
        if math.dist((self.x, self.y), free[0]) > 0.3:
            route = self._plan(free[0], 0.1)
            if route is None:
                raise PathBlocked(f"no free path to ({free[0][0]:.1f}, {free[0][1]:.1f})")
            self._pursue(route, CRUISE)
        self._pursue(free, max(0.05, min(speed, V_MAX)))

    def stop(self) -> None:
        self._call("stop")
        self._halt()

    # ---- skills -------------------------------------------------------------------------------------------------
    def search(self, description: str, area: str = "here", *, toward: str = "") -> Object | None:
        """In view now: done, without moving. Otherwise memory first, checked in person; then camera coverage,
        viewpoint by viewpoint, until it is in view."""
        self._call("search", description=description, area=area)
        region, started = self._area(area), self.t
        # looked for with its own words before anything moves: checking at once found nothing, as no camera had been
        # looked at with a new word yet, and the robot drove off exploring with the sofa right behind it
        in_view = [o for o in self.detect(description) if not region or self._room_at(*o.position) == region]
        if in_view:
            found = in_view[0]
            self._result(f"found {found.label} {found.id} in view")
            return found
        for o in self._remembered(description):
            if region and self._room_at(*o.position) != region:
                continue
            try:
                self._go(o, 0.0)
            except PathBlocked:
                continue
            found = self._found(description, started)
            if found:
                self._result(f"found {found.color} {found.label} {found.id}".replace("  ", " "))
                return found
            self.memory.pop(o.id, None)  # looked at its place with a clear view and it is not there
        tried: list[Point] = []  # where it drove to look, or could not reach
        if toward and toward in BEARINGS:
            hint = self._floor_toward(self.yaw + math.radians(BEARINGS[toward]), 3.0)
            if math.dist(hint, (self.x, self.y)) >= 0.8 and (not region or self._room_at(*hint) == region):
                tried.append(hint)
                try:
                    self._go(hint, 0.0)
                except PathBlocked:
                    pass
                found = self._found(description, started)
                if found:
                    self._result(f"found {found.color} {found.label} {found.id}".replace("  ", " "))
                    return found
        for _ in range(SEARCH_STOPS):
            stop = self._next_stop(description, region, tried)
            if stop is None:
                break
            tried.append(xy(stop))
            try:
                self._go(stop, 0.0)
            except PathBlocked:
                pass
            found = self._found(description, started)
            if found:
                self._result(f"found {found.color} {found.label} {found.id}".replace("  ", " "))
                return found
        self._result("not found")
        return None

    def _next_stop(self, description: str, region: str | None, tried: list[Point]) -> Any:
        """Where a search looks next: the nearest viewpoint the cameras have not covered yet."""
        views = [v for v in self._unseen(region or "all") if all(math.dist((v.x, v.y), s) > 0.5 for s in tried)]
        return views[0] if views else None

    def _area(self, area: str) -> str | None:
        """A room id, or None for everywhere. People say "the living room", "this room", "anywhere"."""
        key = str(area).lower().strip().removeprefix("the ").strip()
        if key in ("everywhere", "all", "anywhere", "house", "home", "whole house", "the house"):
            return None
        if key in ("here", "room", "this room", "current room", "current"):
            return self._room_at(self.x, self.y)
        for name in self.scene.rooms:
            if key in (name, f"{name} room") or key.removesuffix(" room") == name:
                return name
        rooms = ", ".join(repr(r) for r in self.scene.rooms)
        raise ValueError(f"area must be 'here', 'everywhere' or a room ({rooms}), not {area!r}")

    def _found(self, description: str, since: float) -> Object | None:
        """In view now, looked at as detect looks (a thing that came into view as the robot stopped has had too few
        frames to be one yet), or seen since the search started."""
        visible = self.detect(description)
        if visible:
            return visible[0]
        recent = [o for o in self._remembered(description) if self.t - o.last_seen >= since - 1e-6]
        return recent[0] if recent else None

    def orbit(self, target: Point | Pose | Object | str, laps: float = 1.0, direction: str = "left",
              distance: float = 0.5) -> None:
        self._call("orbit", target=target, laps=laps, direction=direction, distance=distance)
        target = self._target(target)
        ccw = choose(direction, ("left", "right"), "direction") == "left"
        center = self._reach(target, 0.0)[0]
        people = [p for _, p in self._people()]

        def blocked(r: float) -> float:  # share of the ring on obstacles or too close to someone
            ring = geom.circle(center, r, (self.x, self.y))
            return sum(self._occ(*p, self.inflated) or any(math.dist(p, q) < 0.6 for q in people)
                       for p in ring) / len(ring)

        edge = (target.radius if isinstance(target, Object) else 0.0) + RADIUS
        wanted = max(0.1, float(distance))
        r = min((edge + d for d in (wanted, wanted - 0.2, wanted + 0.3, wanted + 0.6) if d >= 0.1),
                key=lambda r: (blocked(r) > 0.05, abs(r - edge - wanted)))
        if blocked(r) > 0.2:
            raise PathBlocked(f"no free ring round it: {blocked(r):.0%} of the best ring is blocked by things or people")
        self._drive_path(geom.circle(center, r, (self.x, self.y), laps=abs(float(laps)), ccw=ccw), 0.3, record=False)

    def follow(self, person: Object, distance: float = 1.0, seconds: float | None = None,
               background: bool = False) -> None:
        self._call("follow", person=person, distance=distance, seconds=seconds, background=background)
        person = self._target(person)
        thing = self.things.get(person.id) if isinstance(person, Object) else None
        if thing is None:
            raise TypeError("follow needs a person Object, such as robot.user()")
        control = self._follower(thing, float(distance))
        if background:  # keep following while the program goes on, until the next motion or "stop"
            self._behaviour = control
            return
        end = self.t + float(seconds) if seconds is not None else math.inf
        while self.t < end:
            self._step(*control())
        self._halt()

    def _follower(self, thing: Thing, distance: float) -> Any:
        """A controller that keeps `distance` m from a person: a route round obstacles, renewed twice a second."""
        state: dict[str, Any] = {"lost": None, "route": [], "replan": 0.0}

        def control() -> tuple[float, float]:
            p = self._where(thing)
            if p is None or thing not in self._visible():
                state["lost"] = self.t if state["lost"] is None else state["lost"]
                if self.t - state["lost"] > 5.0:
                    self._behaviour = None
                    raise TargetLost("lost sight of them for 5 s")
                return 0.0, 0.0
            state["lost"] = None
            d = math.dist(p, (self.x, self.y))
            if d <= distance + 0.3:  # close enough: keep facing them
                state["route"] = []
                return (-0.15 if d < distance - 0.3 else 0.0), 2.5 * wrap(math.atan2(p[1] - self.y, p[0] - self.x)
                                                                          - self.yaw)
            if self.t >= state["replan"] or not state["route"]:
                state["route"], state["replan"] = self._plan(p, distance) or [], self.t + 0.5
            route = state["route"]
            while route and math.dist(route[0], (self.x, self.y)) < 0.2:
                route.pop(0)
            tx, ty = route[0] if route else p
            err = wrap(math.atan2(ty - self.y, tx - self.x) - self.yaw)
            return (min(CRUISE, 0.8 * (d - distance)) if abs(err) < 0.7 else 0.0), 2.5 * err
        return control

    def spot(self, target: Point | Pose | Object | str, side: str, distance: float = 0.5) -> Point:
        self._call()
        target = self._target(target)
        choose(side, ("behind", "front", "left", "right"), "side")
        cx, cy = self._reach(target, 0.0)[0]
        r = (target.radius if isinstance(target, Object) else 0.0) + RADIUS + max(0.05, float(distance))
        n = math.hypot(cx - self.x, cy - self.y) or 1.0
        ux, uy = (cx - self.x) / n, (cy - self.y) / n  # from the robot to the target
        dx, dy = {"behind": (ux, uy), "front": (-ux, -uy), "left": (-uy, ux), "right": (uy, -ux)}[side]
        base = math.atan2(dy, dx)
        for extra in (0.0, 0.15, 0.3, 0.5):
            for turn in (0, 10, -10, 20, -20, 30, -30, 45, -45, 60, -60):
                a = base + math.radians(turn)
                p = (cx + (r + extra) * math.cos(a), cy + (r + extra) * math.sin(a))
                if not self._occ(*p, self.inflated):
                    return round(p[0], 2), round(p[1], 2)
        raise NotFound(f"no free spot {side} it")

    def express(self, posture: str, chime_kind: str = "", seconds: float = 0.6) -> None:
        self._call("express", posture=posture, chime_kind=chime_kind)
        self.posture = posture
        self.trace.postures.append((round(self.t, 2), posture))
        if chime_kind:
            self.trace.chimes.append((round(self.t, 2), chime_kind))
        steps = round(float(seconds) / DT)  # 0: just the posture and the chime, no motion
        for s in range(steps):
            t_in = s * DT
            v, w = self.body.aliveness_velocity(posture, t_in, (self.x, self.y, self.yaw))
            self._step(v, w)
        self._halt()

    def camera_view(self, camera: str = "front_camera", width: int = 512, height: int = 288) -> np.ndarray:
        """What one of the three cameras sees: walls, floor and every thing in its view at its size, colour and
        bearing, drawn in perspective, with the tracker's ids marked on the things it has detected. Simple shapes,
        but the vision model reads colours, sizes, counts and directions from the picture as it would on the robot."""
        from robot.perception.overlay import draw_set_of_mark

        yaw0 = self.yaw + CAMERA_YAWS[camera]
        f = width / CAMERA_HFOV  # pixels per radian, equiangular across the wide lens
        horizon = height // 2
        img = np.empty((height, width, 3), np.uint8)
        img[:horizon], img[horizon:] = (196, 204, 214), (176, 158, 128)
        rows = np.arange(height)[:, None]
        # walls: one ray per column
        angles = yaw0 + (width / 2 - (np.arange(width) + 0.5)) / f
        steps = np.arange(RADIUS, 15.0, RES / 2)
        px, py = self.x + np.outer(np.cos(angles), steps), self.y + np.outer(np.sin(angles), steps)
        ii = np.clip(((px + self.scene.width / 2) / RES).astype(int), 0, self.walls.shape[0] - 1)
        jj = np.clip(((py + self.scene.height / 2) / RES).astype(int), 0, self.walls.shape[1] - 1)
        hits = self.walls[ii, jj]
        d = np.where(hits.any(axis=1), steps[hits.argmax(axis=1)], steps[-1]) * np.cos(angles - yaw0).clip(0.2)
        top = horizon - f * np.arctan((WALL_H - CAMERA_HEIGHT) / d)
        bottom = horizon + f * np.arctan(CAMERA_HEIGHT / d)
        shade = (232 - np.clip(d, 0, 8) * 9).astype(np.uint8)
        wall = (rows >= top[None, :]) & (rows <= bottom[None, :])
        img[wall] = np.stack([shade, shade - 6, shade - 14], axis=1)[np.nonzero(wall)[1]]
        # things, far to near; `owner` keeps which thing is in front at each pixel
        boxes: dict[int, tuple[int, int, int, int]] = {}
        owner = np.full((height, width), -1, np.int32)
        seen = [(t, self._where(t)) for t in self._visible()]
        for t, p in sorted(seen, key=lambda tp: -math.dist(tp[1], (self.x, self.y))):
            dist = max(0.2, math.dist(p, (self.x, self.y)))
            bearing = wrap(math.atan2(p[1] - self.y, p[0] - self.x) - yaw0)
            half = math.atan2(t.radius, dist)
            if abs(bearing) - half > CAMERA_HFOV / 2:
                continue
            x0, x1 = int(width / 2 - f * (bearing + half)), int(width / 2 - f * (bearing - half))
            y1 = int(horizon + f * math.atan2(CAMERA_HEIGHT, dist))
            y0 = int(horizon - f * math.atan2(HEIGHTS.get(t.label, 0.5) - CAMERA_HEIGHT, dist))
            xa, xb, ya, yb = max(0, x0), min(width, x1 + 1), max(0, y0), min(height, y1 + 1)
            if xa >= xb or ya >= yb:
                continue
            color = np.array(RGB.get(t.color, RGB.get(t.label, (150, 120, 90))), np.uint8)
            img[ya:yb, xa:xb] = color
            img[ya:yb, xa:min(xb, xa + 2)] = img[ya:yb, max(xa, xb - 2):xb] = (color * 0.55).astype(np.uint8)
            owner[ya:yb, xa:xb] = t.id
            if t.person:  # a head on top
                head = max(2, (x1 - x0) // 3)
                cx = (x0 + x1) // 2
                img[max(0, y0 - 2 * head):max(0, y0), max(0, cx - head):max(0, cx + head)] = (224, 180, 150)
            if t.id in self.memory and self._detectable(t):
                boxes[t.id] = (xa, ya, xb - 1, yb - 1)  # clipped to the picture, so the id stays on it
        # an id is marked only where its thing can be seen, not on whatever hides it
        marks = []
        for tid, (x0, y0, x1, y1) in boxes.items():
            area = max(1, (x1 + 1 - x0) * (y1 + 1 - y0))
            if (owner == tid).sum() >= 0.3 * area:
                marks.append(((x0, y0, x1, y1), tid))
        return draw_set_of_mark(img, marks) if marks else img

    def wait_for(self, description: str, seconds: float | None = None) -> Object | None:
        self._call("wait_for", description=description, seconds=seconds)
        self._watch(description)
        end = self.t + float(seconds) if seconds is not None else math.inf
        while True:
            seen = self._matching(description, self._in_view())
            if seen:
                return self._object(seen[0])
            if self.t >= end:
                return None
            self._step(0.0, 0.0)

    def wait(self, seconds: float) -> None:
        self._call("wait", seconds=seconds)
        for _ in range(max(0, round(float(seconds) / DT))):
            self._step(0.0, 0.0)

    def save_place(self, name: str) -> None:
        self._call("save_place", name=name)
        self.places[str(name).lower()] = Pose(self.x, self.y, math.degrees(self.yaw))

    # ---- senses -------------------------------------------------------------------------------------------------
    def _watch(self, description: str) -> None:
        words = _words(description) - COLORS
        for w in list(words):
            if w in SYNONYMS:
                for alt in SYNONYMS[w]:
                    words |= _words(alt)
        new = words - self.vocabulary
        if new:
            self.vocabulary |= new
            self._last_perceive = -1.0
            self._perceive()

    def _matching(self, description: str, things: list[Thing]) -> list[Thing]:
        """Things the description names, best first; colour words must match too (on the robot, the mask's colour)."""
        colors = _words(description) & COLORS
        found = [t for t in things if self._detectable(t) and t.score(description) and (not colors or t.color in colors)]
        if found:
            return sorted(found, key=lambda t: -t.score(description))

        name = canonical(visual_name(description))
        found = [t for t in things if self._detectable(t) and canonical(t.label) == name
                 and (not colors or t.color in colors)]
        if found:
            return found

        # Synonym fallback if direct matching yielded no results
        desc_words = _words(description)
        desc_lower = description.lower()
        for w, alts in SYNONYMS.items():
            if w in desc_words or w in desc_lower:
                for alt in alts:
                    alt_desc = desc_lower.replace(w, alt)
                    alt_found = [
                        t
                        for t in things
                        if self._detectable(t)
                        and (t.score(alt) or t.score(alt_desc))
                        and (not colors or t.color in colors)
                    ]
                    if alt_found:
                        return sorted(alt_found, key=lambda t: -(t.score(alt) or t.score(alt_desc)))

        return []

    def _remembered(self, description: str) -> list[Object]:
        things = [self.things[i] for i in self.memory]
        return [self._object(t, *self.memory[t.id]) for t in self._matching(description, things)]

    def detect(self, description: str) -> list[Object]:
        self._call()
        self._watch(description)
        found = self._matching(description, self._in_view())
        return Matches(sorted((self._object(t, *self.memory[t.id]) for t in found), key=lambda o: o.distance))

    def _in_view(self) -> list[Thing]:
        """What the detector reports now: everything visible, or with noise, what it did not miss this tick."""
        return [t for t in self._visible() if not self.scene.noise or t.id in self._detected]

    def where(self, description: str, area: str = "everywhere") -> list[Object]:
        self._call()
        region = self._area(area)
        return Matches(o for o in self._remembered(description) if not region or self._room_at(*o.position) == region)

    def check(self, question: str, target: Point | Pose | Object | str | None = None) -> bool:
        self._call("check", question=question, target=target)
        return self._vision(question, target) == "yes"

    def describe(self, target: Point | Pose | Object | str | None = None, question: str = "What is this?") -> str:
        self._call("describe", question=question, target=target)
        return self._vision(question, target)

    def _vision(self, question: str, target: Any) -> str:
        """The vision model, answered from the scene: its real accuracy is measured separately on photos (M0)."""
        target = self._target(target) if target is not None else None
        if isinstance(target, Object):
            things = [self.things[target.id]] if target.id in self.things else []
        else:
            things = self._visible()
        answer = self._answer(question, things, isinstance(target, Object))
        self.trace.asked.append((round(self.t, 2), question, answer))
        for _ in range(round(ASK_S / DT)):  # the model takes about 2 s; the world goes on meanwhile
            self._step(0.0, 0.0)
        return answer

    def _answer(self, question: str, things: list[Thing], single: bool) -> str:
        q, words = question.lower(), _words(question)
        if single and things:
            t = things[0]
            if "colo" in q:
                return t.color or "I can't tell"
            if "where" in q:
                others = [o for o in self.things.values() if o is not t and not o.person and o.label != "door"]
                if others:
                    here = self._where(t) or (t.x, t.y)
                    near = min(others, key=lambda o: math.dist((o.x, o.y), here))
                    return f"next to the {near.label}"
            if words & COLORS:
                return "yes" if t.color in words else "no"
            for key, value in t.facts.items():
                if key in words and isinstance(value, bool):
                    return "yes" if value else "no"
            if words & {"what", "which"}:
                return f"a {t.color} {t.label}".replace("  ", " ")
            return "I can't tell"
        named = [t for t in things if any(t.score(w) for w in words - {"a", "an", "the", "there", "is", "are", "any"})]
        if "how many" in q:
            return str(len(named))
        if words & {"what", "describe", "see"} and not q.startswith(("is ", "are ", "do ", "does ")):
            labels = [f"{t.color} {t.label}".strip() for t in things if not t.person]
            return ", ".join(labels) if labels else "nothing"
        for t in named:
            for key, value in t.facts.items():
                if key in words and isinstance(value, bool):
                    return "yes" if value else "no"
        if words & COLORS:
            named = [t for t in named if t.color in words]
        return "yes" if named else "no"

    def clearance(self, direction: str = "ahead") -> float:
        self._call()
        offset = {"ahead": 0.0, "left": math.pi / 2, "behind": math.pi, "right": -math.pi / 2}[
            choose(direction, ("ahead", "behind", "left", "right"), "direction")]
        c, s = math.cos(self.yaw + offset), math.sin(self.yaw + offset)
        people = [p for _, p in self._people()]
        for d in np.arange(RADIUS, 12.0, RES):
            px, py = self.x + c * d, self.y + s * d
            if self._occ(px, py) or any(math.dist((px, py), q) < 0.25 for q in people):
                return round(float(d - RADIUS), 2)
        return 12.0

    def _unseen(self, region: str = "room") -> list[Pose]:
        """Reachable viewpoints next to floor the cameras have not covered, nearest first."""
        region = self._room_at(self.x, self.y) if region == "room" else region
        bounds = self.scene.rooms.get(region) if region != "all" else None
        reach = self._reachable()
        nx, ny = self.coverage.shape
        blocks: dict[tuple[int, int], list[tuple[int, int]]] = {}
        size = round(1.0 / RES)
        people = [p for _, p in self._people()]
        for i, j in zip(*np.nonzero(~self.coverage & ~self.inflated), strict=True):
            x, y = self._centre(i, j)
            if bounds and not (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]):
                continue
            if any(math.dist((x, y), q) < 0.8 for q in people):  # never send the robot to stand on someone
                continue
            blocks.setdefault((i // size, j // size), []).append((i, j))
        views = []
        for cells in blocks.values():
            if len(cells) < 40:
                continue
            cx = sum(self._centre(i, j)[0] for i, j in cells) / len(cells)
            cy = sum(self._centre(i, j)[1] for i, j in cells) / len(cells)
            ok = [(i, j) for i, j in cells if reach[min(nx // PLAN - 1, i // PLAN), min(ny // PLAN - 1, j // PLAN)]]
            if not ok:
                continue
            i, j = min(ok, key=lambda c: math.dist(self._centre(*c), (cx, cy)))
            vx, vy = self._centre(i, j)
            views.append(Pose(round(float(vx), 2), round(float(vy), 2), round(math.degrees(math.atan2(cy - vy, cx - vx)), 1)))
        return sorted(views, key=lambda p: math.dist((p.x, p.y), (self.x, self.y)))[:12]

    def _reachable(self) -> np.ndarray:
        blocked = self.blocked
        res = RES * PLAN
        start = (int((self.x + self.scene.width / 2) / res), int((self.y + self.scene.height / 2) / res))
        seen = np.zeros_like(blocked)
        stack = [start]
        seen[start] = True
        while stack:
            i, j = stack.pop()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < blocked.shape[0] and 0 <= b < blocked.shape[1] and not seen[a, b] and not blocked[a, b]:
                    seen[a, b] = True
                    stack.append((a, b))
        return seen

    # ---- speech and memory --------------------------------------------------------------------------------------
    def say(self, text: str) -> None:
        self._call("say", text=text)
        self._say(str(text))

    def _say(self, text: str) -> None:
        self.trace.said.append((round(self.t, 2), text))
        for _ in range(round((0.3 + 0.3 * len(text.split())) / DT)):
            self._step(0.0, 0.0)

    def ask(self, question: str, seconds: float = 8.0) -> str | None:
        self._call("ask", question=question)
        self._say(str(question))
        return self._listen(float(seconds))

    def listen(self, seconds: float = 8.0) -> str | None:
        self._call("listen", seconds=seconds)
        return self._listen(float(seconds))

    def _listen(self, seconds: float) -> str | None:
        self.trace.listened.append(round(self.t, 2))
        end = self.t + seconds
        while self.t < end:
            if self.replies and self.replies[0][0] <= self.t:
                return self.replies.pop(0)[1]
            self._step(0.0, 0.0)
        return None

    def remember(self, fact: str) -> None:
        self._call("remember", fact=fact)
        self.notes.append(str(fact))
        self.trace.notes.append(str(fact))

    def recall(self, query: str) -> list[str]:
        self._call()
        words = _words(query) - {"the", "a", "my", "i", "you", "what", "did", "to", "that", "is"}
        scored = sorted(((len(words & _words(n)), n) for n in self.notes), reverse=True)
        return [n for s, n in scored if s > 0]

    def elapsed(self) -> float:
        self._call()
        return round(self.t, 2)

    # ---- for the agent and the suite (not tools) ------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """What the agent tells the model: pose, room, rooms, battery, objects, user, notes."""
        user = next((t for t, _ in self._people() if t.user), None)
        objects = [self._object(self.things[i], where, when) for i, (where, when) in self.memory.items()]
        if user is not None and user.id not in self.memory:
            objects.append(self._object(user))
        return {
            "pose": Pose(round(self.x, 2), round(self.y, 2), round(math.degrees(self.yaw), 1)),
            "room": self._room_at(self.x, self.y),
            "rooms": list(self.scene.rooms),
            "battery": max(0.0, 1.0 - self.t / 10800.0),
            "objects": objects,
            "user": user.id if user else None,
            "notes": list(self.notes),
            "posture": getattr(self, "posture", "neutral"),
        }

    def begin(self) -> None:
        """A new task starts here: "start" names this spot, "where you were before" the start of the task before it
        (so "come back" has somewhere to go), and an earlier stop no longer applies."""
        self._task_t0 = self.t
        last = self.places.get("start")
        if last is not None and math.dist((last.x, last.y), (self.x, self.y)) > 0.3:  # not just another name for here
            self.places["where you were before"] = last
        self.places["start"] = Pose(self.x, self.y, math.degrees(self.yaw))
        self._cancel.clear()
        self.ended_by = ""
        self._behaviour = None

    def resume(self) -> None:
        """Carry on after an interruption (the user spoke mid-step), keeping the task's start and places."""
        self._cancel.clear()
        self.ended_by = ""
