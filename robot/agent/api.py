"""The robot API: the whole surface a generated program can touch, and the one manifest the planner's command card,
the gate's whitelist and the training data are generated from.

Every program gets `robot`, `geom`, `math` and the four exceptions it may catch, and nothing else: no imports, no
files, no network. Units are metres, seconds and degrees, the ones people use in instructions. Directions are words
("left", "backward"), never signs: the untrained planner wrote turn(-90) for "left" even with "+ is left" in its card.
Arguments that carry a unit are keyword-only and named after it (`move(metres=1.5)`), so `drive(1.5, 0, 1)` for
"1.5 metres" cannot be written by accident.

Skills are calls that need feedback faster than the planner can give (search, orbit, follow): they are tested code
with their own recovery. Sequencing, counting, conditions and simple geometry stay in the program.
"""

from __future__ import annotations

import builtins
import inspect
import math
from dataclasses import dataclass
from typing import Protocol

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


class PathBlocked(Exception):
    """No progress toward the goal for 10 s, or no free path at all."""


class TargetLost(Exception):
    """The thing being tracked is no longer seen."""


class NotFound(Exception):
    """Nothing matches."""


class Timeout(Exception):
    """A wait ran out."""


class SafetyStop(BaseException):
    """A safety layer stopped the robot. Always ends the program."""


class Preempted(BaseException):
    """The program was stopped: "stop" was spoken, a new instruction arrived, or time ran out."""


class Matches(list):
    """What `where` and `detect` return: a list, best match first. Used like an Object, it means its best match, and an
    empty one raises NotFound, so `robot.go_to(robot.where("ball"))` and `robot.where("ball").position` both work."""

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        if not self:
            raise NotFound("nothing matched")
        return getattr(self[0], name)


CATCHABLE = (PathBlocked, TargetLost, NotFound, Timeout)
EXCEPTIONS = {e.__name__: e for e in (*CATCHABLE, SafetyStop, Preempted)}
SAFE_BUILTINS = {name: getattr(builtins, name) for name in (
    "len", "range", "min", "max", "abs", "sum", "sorted", "round", "any", "all", "enumerate", "zip", "set", "list",
    "dict", "str", "int", "float", "bool")}
SIDES = ("ahead", "front-left", "left", "behind-left", "behind", "behind-right", "right", "front-right")


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
    """Any target as a map point. A list from where or detect means its best match; an empty one raises NotFound."""
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


class Robot(Protocol):
    def pose(self) -> Pose:
        """where you are; yaw in degrees"""

    def battery(self) -> float:
        """0.0 .. 1.0"""

    def user(self) -> Object | None:
        """the person talking to you"""

    def room(self) -> str:
        """the room you are in"""

    def obj(self, id: int) -> Object:
        """a table object (by id) as it is now; raises NotFound"""

    def move(self, metres: float, direction: str = "forward") -> None:
        """straight; direction "forward" or "backward\""""

    def turn(self, degrees: float = 90, direction: str = "") -> None:
        """on the spot; direction "left" or "right" (needed)"""

    def spin(self, turns: float = 1.0, seconds: float | None = None, speed: str = "normal",
             direction: str = "left") -> None:
        """full turns (1 = 360°), or for seconds; speed "slow", "normal" or "fast\""""

    def face(self, target: Target, background: bool = False) -> None:
        """turn toward it; background=True keeps at it"""

    def go_to(self, target: Target, distance: float = 0.0) -> None:
        """drive there round obstacles, stopping distance m short"""

    def drive(self, *, metres_per_second: float, degrees_per_second: float = 0.0, seconds: float) -> None:
        """raw speed; + degrees_per_second turns left"""

    def drive_path(self, points: list[Point], speed: float = 0.3) -> None:
        """drive through the points in order"""

    def stop(self) -> None:
        """stop moving"""

    def search(self, description: str, area: str = "here") -> Object | None:
        """memory, then explore; area "here", a room or "everywhere"; None if not found"""

    def orbit(self, target: Target, laps: float = 1.0, direction: str = "left", distance: float = 0.5) -> None:
        """circle it distance m from its edge; "left" = anticlockwise"""

    def follow(self, person: Object, distance: float = 1.0, seconds: float | None = None,
               background: bool = False) -> None:
        """stay distance m away until "stop" or seconds; background=True runs alongside"""

    def spot(self, target: Target, side: str, distance: float = 0.5) -> Point:
        """a free point next to it: side "behind", "front", "left" or "right" as you see it now"""

    def wait_for(self, description: str, seconds: float | None = None) -> Object | None:
        """stay still until it is in view; None if seconds run out"""

    def wait(self, seconds: float) -> None:
        """stay still"""

    def save_place(self, name: str) -> None:
        """name where you are; go_to(name) comes back"""

    def detect(self, description: str) -> list[Object]:
        """in view now, nearest first"""

    def where(self, description: str, area: str = "everywhere") -> list[Object]:
        """everything seen so far that matches, best first"""

    def check(self, question: str, target: Target | None = None) -> bool:
        """vision model's yes/no about it or the view, ~2 s"""

    def describe(self, target: Target | None = None, question: str = "What is this?") -> str:
        """the vision model's answer in words, ~2 s"""

    def clearance(self, direction: str = "ahead") -> float:
        """free metres "ahead", "behind", "left" or "right\""""

    def say(self, text: str) -> None:
        """speak"""

    def ask(self, question: str, seconds: float = 8.0) -> str | None:
        """say it, then return the user's answer, or None"""

    def listen(self, seconds: float = 8.0) -> str | None:
        """what the user says next, or None"""

    def remember(self, fact: str) -> None:
        """keep a fact across tasks"""

    def recall(self, query: str) -> list[str]:
        """facts matching the query"""

    def elapsed(self) -> float:
        """seconds since the program started"""


Target = "Point | Pose | Object | str"  # for the card only; a str is a saved place, a room or a thing seen


class geom:  # noqa: N801 - the name programs use
    """Pure math, no robot access."""

    @staticmethod
    def dist(a: Point | Pose | Object, b: Point | Pose | Object) -> float:
        """metres between them"""
        (ax, ay), (bx, by) = xy(a), xy(b)
        return math.hypot(bx - ax, by - ay)

    @staticmethod
    def between(a: Point | Pose | Object, b: Point | Pose | Object) -> Point:
        """the point halfway"""
        (ax, ay), (bx, by) = xy(a), xy(b)
        return (ax + bx) / 2, (ay + by) / 2

    @staticmethod
    def circle(center: Point | Pose | Object, radius: float, start: Point | Pose | Object, laps: float = 1.0,
               ccw: bool = True) -> list[Point]:
        """points round center, starting nearest start"""
        (cx, cy), (sx, sy) = xy(center), xy(start)
        a0, sign = math.atan2(sy - cy, sx - cx), 1.0 if ccw else -1.0
        n = max(8, math.ceil(2 * math.pi * radius * laps / 0.15))
        return [(cx + radius * math.cos(a0 + sign * 2 * math.pi * laps * k / n),
                 cy + radius * math.sin(a0 + sign * 2 * math.pi * laps * k / n)) for k in range(n + 1)]


def _calls(cls: type) -> tuple[str, ...]:
    return tuple(name for name, value in vars(cls).items()
                 if not name.startswith("_") and (inspect.isfunction(value) or isinstance(value, staticmethod)))


ROBOT_CALLS = _calls(Robot)
GEOM_CALLS = _calls(geom)
SIGNATURES = {name: inspect.signature(getattr(Robot, name)) for name in ROBOT_CALLS}
_SECTIONS = {"pose": "state", "move": "motion: blocks until done, always safety-filtered",
             "search": "skills: tested, with their own recovery", "detect": "senses",
             "say": "speech and memory"}


def _line(owner: str, name: str, fn: object) -> str:
    sig = inspect.signature(fn)
    params, starred = [], False
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind is p.KEYWORD_ONLY and not starred:
            params.append("*")
            starred = True
        text = f"*{p.name}" if p.kind is p.VAR_POSITIONAL else p.name
        if p.annotation is not p.empty:
            text += f": {p.annotation}"
        if p.default is not p.empty:
            text += f" = {p.default!r}"
        params.append(text)
    doc = (inspect.getdoc(fn) or "").splitlines()[0]
    returns = "" if sig.return_annotation in ("None", None) else f" -> {sig.return_annotation}"
    line = f"{owner}.{name}({', '.join(params)}){returns}  # {doc}"
    return line.replace("Point | Pose | Object", "Target")


def api_card() -> str:
    """The API as the planner reads it, generated from this module so the two cannot drift."""
    lines = ["Names you have: robot, geom, math, PathBlocked, TargetLost, NotFound, Timeout. No imports.",
             "Metres, seconds, degrees. Pose: x, y, yaw. Object: id, label, position, distance, bearing, side, radius, "
             "color, last_seen (s ago). Target: (x, y), Pose, Object, or a str: a saved place (\"start\" is where the task "
             "began), a room or a thing seen."]
    for name in ROBOT_CALLS:
        if name in _SECTIONS:
            lines.append(f"# {_SECTIONS[name]}")
        lines.append(_line("robot", name, getattr(Robot, name)))
    lines.append("# pure math")
    lines += [_line("geom", name, getattr(geom, name)) for name in GEOM_CALLS]
    lines.append("Every loop must call robot.")
    return "\n".join(lines)
