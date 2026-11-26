"""The robot API: Pose, Object, the Robot protocol, errors and geometry helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

Point = tuple[float, float]


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float  # degrees, map frame, + is anticlockwise


@dataclass(frozen=True)
class Object:
    id: int
    label: str
    position: Point
    distance: float
    bearing: float  # degrees from the robot's heading, + is left, ±180 is behind
    side: str  # the bearing in words: ahead, front-left, left, behind-left, behind, ...
    radius: float
    color: str
    last_seen: float  # seconds ago
    room: str = ""  # the room its position is in, when the robot knows rooms


class PathBlocked(Exception):
    """No progress toward the goal or no free path."""


class TargetLost(Exception):
    """The thing being tracked is no longer seen."""


class NotFound(Exception):
    """Nothing matches."""


class Timeout(Exception):
    """A wait ran out."""


class SafetyStop(BaseException):
    """A safety layer stopped the robot."""


class Preempted(BaseException):
    """The program was stopped: 'stop' was spoken, a new instruction arrived, or time ran out."""


class Matches(list):
    """List returned by where and detect: best match first. Used like an Object, it accesses the best match."""

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        if not self:
            raise NotFound("nothing matched")
        return getattr(self[0], name)


def side(bearing: float) -> str:
    """A bearing in degrees as words: ahead, front-left, left, behind-left, behind, behind-right, right, front-right."""
    a = abs(bearing)
    lr = "left" if bearing > 0 else "right"
    if a <= 22.5:
        return "ahead"
    if a <= 67.5:
        return f"front-{lr}"
    if a <= 112.5:
        return lr
    return f"behind-{lr}" if a <= 157.5 else "behind"


def xy(p: Point | Pose | Object | list) -> Point:
    """Any target as a map point. A list from where or detect means its best match."""
    if isinstance(p, Object):
        return p.position
    if isinstance(p, Pose):
        return p.x, p.y
    if isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p):
        return float(p[0]), float(p[1])
    if isinstance(p, list) and all(isinstance(v, (Object, Pose)) for v in p):
        if not p:
            raise NotFound("the list is empty: nothing matched")
        return xy(p[0])
    raise TypeError(f"expected a point (x, y), a Pose or an Object, not {type(p).__name__}")


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def choose(value: str, options: tuple[str, ...], what: str) -> str:
    """Check a word argument and say which words are allowed."""
    if value not in options:
        raise ValueError(f"{what} must be one of {', '.join(repr(o) for o in options)}, not {value!r}")
    return value


Target = Any


class geom:  # noqa: N801
    """Pure math, no robot access."""

    @staticmethod
    def dist(a: Point | Pose | Object, b: Point | Pose | Object) -> float:
        """Metres between them."""
        (ax, ay), (bx, by) = xy(a), xy(b)
        return math.hypot(bx - ax, by - ay)

    @staticmethod
    def between(a: Point | Pose | Object, b: Point | Pose | Object) -> Point:
        """The point halfway."""
        (ax, ay), (bx, by) = xy(a), xy(b)
        return (ax + bx) / 2, (ay + by) / 2

    @staticmethod
    def circle(
        center: Point | Pose | Object,
        radius: float,
        start: Point | Pose | Object,
        laps: float = 1.0,
        ccw: bool = True,
    ) -> list[Point]:
        """Points round center, starting nearest start."""
        (cx, cy), (sx, sy) = xy(center), xy(start)
        a0, sign = math.atan2(sy - cy, sx - cx), 1.0 if ccw else -1.0
        n = max(8, math.ceil(2 * math.pi * radius * laps / 0.15))
        return [
            (
                cx + radius * math.cos(a0 + sign * 2 * math.pi * laps * k / n),
                cy + radius * math.sin(a0 + sign * 2 * math.pi * laps * k / n),
            )
            for k in range(n + 1)
        ]


class Robot(Protocol):
    def pose(self) -> Pose:
        """Where you are; yaw in degrees."""

    def battery(self) -> float:
        """0.0 .. 1.0."""

    def user(self) -> Object | None:
        """The person talking to you."""

    def room(self) -> str:
        """The room you are in."""

    def obj(self, id: int) -> Object:
        """A table object (by id) as it is now."""

    def move(self, metres: float, direction: str = "forward") -> None:
        """Straight; direction 'forward' or 'backward'."""

    def turn(self, degrees: float = 90, direction: str = "") -> None:
        """On the spot; direction 'left' or 'right'."""

    def spin(
        self,
        turns: float = 1.0,
        seconds: float | None = None,
        speed: str = "normal",
        direction: str = "left",
    ) -> None:
        """Full turns (1 = 360°), or for seconds; speed 'slow', 'normal' or 'fast'."""

    def face(self, target: Target, background: bool = False) -> None:
        """Turn toward target."""

    def go_to(self, target: Target, distance: float = 0.0, *, side: str = "") -> None:
        """Approach within distance of the centre, or choose a side with distance measured edge to edge."""

    def drive(self, *, metres_per_second: float, degrees_per_second: float = 0.0, seconds: float) -> None:
        """Raw speed; + degrees_per_second turns left."""

    def drive_path(self, points: list[Point], speed: float = 0.3) -> None:
        """Drive through points in order."""

    def stop(self) -> None:
        """Stop moving."""

    def search(self, description: str, area: str = "here", *, toward: str = "") -> Object | None:
        """Memory, then explore; area 'here', a room or 'everywhere'."""

    def orbit(self, target: Target, laps: float = 1.0, direction: str = "left", distance: float = 0.5) -> None:
        """Circle target distance m from edge."""

    def follow(
        self,
        person: Object,
        distance: float = 1.0,
        seconds: float | None = None,
        background: bool = False,
    ) -> None:
        """Stay distance m away from person."""

    def spot(self, target: Target, side: str, distance: float = 0.5) -> Point:
        """A free point next to target."""

    def express(self, posture: str, chime_kind: str = "", seconds: float = 0.6) -> None:
        """Express posture and procedural chime."""

    def wait_for(self, description: str, seconds: float | None = None) -> Object | None:
        """Stay still until description is in view."""

    def wait(self, seconds: float) -> None:
        """Stay still."""

    def save_place(self, name: str) -> None:
        """Name current location."""

    def detect(self, description: str) -> list[Object]:
        """Objects in view now."""

    def where(self, description: str, area: str = "everywhere") -> list[Object]:
        """Matching objects seen so far."""

    def check(self, question: str, target: Target | None = None) -> bool:
        """Vision model yes/no check."""

    def describe(self, target: Target | None = None, question: str = "What is this?") -> str:
        """Vision model description."""

    def clearance(self, direction: str = "ahead") -> float:
        """Free distance in direction."""

    def say(self, text: str) -> None:
        """Speak words."""

    def ask(self, question: str, seconds: float = 8.0) -> str | None:
        """Ask question and return spoken reply."""

    def listen(self, seconds: float = 8.0) -> str | None:
        """Listen for user reply."""

    def remember(self, fact: str) -> None:
        """Keep a fact in memory."""

    def recall(self, query: str) -> list[str]:
        """Recall facts matching query."""

    def elapsed(self) -> float:
        """Seconds since task started."""
