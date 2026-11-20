"""Sprocket's tools: what the model can do, as function schemas in its native tool-calling format.

Every call is checked before it runs (an unknown tool, a missing or out-of-range argument, a target that matches
nothing or several things) and every answer is a short sentence the model reads on its next turn. A failure says what
went wrong and what to try instead; nothing is silently changed or guessed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from robot.agent.api import NotFound, Object, PathBlocked, Pose, TargetLost, Timeout, geom
from robot.sim.world import BEARINGS, visual_name

CAMERAS = {  # every direction go_toward takes too: the model used each tool's words for the other ("front-right")
    "front": ("front_camera",),
    "front-left": ("front_camera",),  # the front camera sees 70 degrees either way
    "front-right": ("front_camera",),
    "rear-left": ("rear_left_camera",),
    "rear-right": ("rear_right_camera",),
    "left": ("rear_left_camera",),  # the rear cameras look 120 degrees round, 70 degrees either way
    "right": ("rear_right_camera",),
    "behind-left": ("rear_left_camera",),
    "behind-right": ("rear_right_camera",),
    "behind": ("rear_left_camera", "rear_right_camera"),
    "around": ("front_camera", "rear_left_camera", "rear_right_camera"),
}
CAMERA_NAMES = {"front_camera": "front", "rear_left_camera": "rear-left", "rear_right_camera": "rear-right"}
ROBOT_RADIUS = 0.15  # metres, from the robot's centre to its edge
RELATION = re.compile(r"\b(left of|right of|behind|in front of|front of|next to|beside|near|between|on top of|under"
                      r"neath|under|far side of|close to)\b")
RANGE_ORDER = re.compile(r"\b(?:nearest|closest|farthest|furthest)\b", re.IGNORECASE)
VIEW_ORDER = re.compile(r"\b(?:leftmost|rightmost|left|right)\b", re.IGNORECASE)
USER_WORDS = {"user", "me", "you", "the user", "person(user)", "human", "speaker", "my owner", "owner"}
NO_TARGET = {"", "none", "null", "nothing"}
NO_CLEARANCE = ("I cannot verify clearance under furniture. I did not drive underneath or beside it. "
                "Ask before substituting a different destination")
NUMBER = re.compile(r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*"
                    r"(?:m|metres?|meters?|deg(?:rees?)?|°|s|sec(?:onds?)?)?\s*")


def speakable(text: str) -> str:
    """Text as a voice should say it: markdown marks (bold, headings, bullets, code) removed, one line."""
    text = re.sub(r"[*_`#]+|[☀-➿️\U0001f000-\U0001faff]", "", text)  # and emoji ("✅ done")
    text = re.sub(r"(?m)^\s*(?:[-•]|\d+[.)])\s+", "", text)
    return " ".join(text.split())


class ToolError(Exception):
    """A call that cannot run as asked; the message tells the model what to do instead."""


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, dict[str, Any]]
    required: tuple[str, ...]
    run: Callable[..., str]
    moves: bool = False  # what the cameras see afterwards is new

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": self.params, "required": list(self.required)},
        }}


@dataclass
class Todo:
    """The model's own to-do list for a request with several parts, shown back to it every turn."""

    steps: list[str] = field(default_factory=list)
    done: list[bool] = field(default_factory=list)

    def write(self, steps: list[str]) -> None:
        self.steps = [str(s).strip() for s in steps if str(s).strip()]
        self.done = [False] * len(self.steps)

    def tick(self, number: int) -> None:
        if not 1 <= number <= len(self.steps):
            raise ToolError(f"there is no step {number}; the to-do list has {len(self.steps)} steps")
        self.done[number - 1] = True

    def open_steps(self) -> list[str]:
        return [s for s, d in zip(self.steps, self.done, strict=True) if not d]

    def text(self) -> str:
        if not self.steps:
            return "(no to-do list)"
        return "\n".join(f"{i}. [{'x' if d else ' '}] {s}" for i, (s, d) in enumerate(zip(self.steps, self.done,
                                                                                          strict=True), 1))


def _num(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("a number, not true/false")
    if isinstance(value, (int, float)):
        return float(value)
    m = NUMBER.fullmatch(str(value))
    if not m or not math.isfinite(number := float(m.group(1))):
        raise ValueError(f"a number, not {value!r}")
    return number


def _check(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    """Arguments as the tool needs them, or a ToolError saying exactly what is wrong."""
    if "_unparsed" in args:
        raise ToolError(f"{tool.name}: could not read the arguments {args['_unparsed']!r}")
    unknown = set(args) - set(tool.params)
    if unknown:
        raise ToolError(f"{tool.name} has no argument {', '.join(sorted(unknown))}; "
                        f"it takes {', '.join(tool.params) or 'nothing'}")
    out: dict[str, Any] = {}
    for name, spec in tool.params.items():
        if name not in args or args[name] is None or args[name] == "":
            if name in tool.required:
                raise ToolError(f"{tool.name} needs {name}: {spec.get('description', spec.get('type'))}")
            continue
        value, kind = args[name], spec.get("type")
        try:
            if kind == "number":
                value = _num(value)
            elif kind == "integer":
                value = int(_num(value))
            elif kind == "boolean":
                value = value if isinstance(value, bool) else str(value).strip().lower() in ("true", "yes", "1")
            elif kind == "array":
                value = value if isinstance(value, list) else [s.strip() for s in str(value).split(";") if s.strip()]
            else:
                value = str(value).strip()
        except ValueError as exc:
            raise ToolError(f"{tool.name}: {name} must be {exc}") from None
        if "enum" in spec and value not in spec["enum"]:
            low = "-".join(str(value).lower().replace("_", " ").split())
            if low in spec["enum"]:
                value = low
            else:
                raise ToolError(f"{tool.name}: {name} must be one of {', '.join(spec['enum'])}, not {value!r}")
        lo, hi = spec.get("minimum"), spec.get("maximum")
        if lo is not None and value < lo or hi is not None and value > hi:
            raise ToolError(f"{tool.name}: {name} must be between {lo} and {hi}, not {value:g}")
        out[name] = value
    return out


def _where_words(o: Object) -> str:
    return f"{o.distance:.1f} m {o.side}"


def _check_visual_name(text: str) -> None:
    if RANGE_ORDER.search(text):
        raise ToolError("a distance comparison is not a camera label. Call search with just the object name, "
                        "then choose an ID by its measured distance")


class Tools:
    """The tools bound to one robot for one task."""

    def __init__(self, robot: Any, memory: Any = None) -> None:
        self.robot, self.memory = robot, memory
        self.todo = Todo()
        self.looked: list[str] = []  # cameras the model asked to see since the last observation
        self.moved = False  # the robot moved since the last observation
        self.found: Object | None = None
        self.pictures: list[tuple[str, Any]] = []  # sensor evidence with its provenance, not a new live view
        self._tools = {t.name: t for t in self._define()}
        self.began = robot.pose()  # where this request began
        self.done: list[str] = []  # each motion of this request, as it went
        self.turned = 0.0  # degrees of turning in all, either way

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def run(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Run one call: (what to tell the model, whether it worked). Stops and preemptions propagate."""
        tool = self._tools.get(name)
        if tool is None:
            return f"There is no tool {name!r}. Tools: {', '.join(self._tools)}.", False
        self.found = None  # what this call found by searching
        try:
            checked = _check(tool, args)
            result = tool.run(**checked)
            if self.found is not None and name != "search":
                result = f"Searched for it: found {self.found.label} {self.found.id}. {result}"
        except ToolError as exc:
            return f"Error: {exc}.", False
        except NotFound as exc:
            return f"Not found: {exc}. Use search() to look for it, or ask the user.", False
        except PathBlocked as exc:
            return f"Blocked: {exc}. Try another way, a different target, or ask the user for help.", False
        except TargetLost as exc:
            self.looked.extend(CAMERAS["around"])
            return f"Target lost: {exc}. Inspect the current views before choosing another ID.", False
        except Timeout as exc:
            return f"Timed out: {exc}.", False
        except (ValueError, TypeError) as exc:
            return f"Error: {exc}.", False
        finally:
            if tool.moves:
                self.moved = True
        if tool.moves:
            result += "\n" + self._progress(name, checked, result)
        return result, True

    def _progress(self, name: str, args: dict[str, Any], result: str) -> str:
        """Every motion of this request so far and where that has left the robot, after each motion: a small model
        loses count over a long conversation (it spun twice, then kept turning; it rocked left and right, and lost
        track of how often), but reads its progress right in the latest result."""
        if name == "turn":
            degrees = args["degrees"]
            self.turned += abs(degrees)
            self.done.append(f"turned {degrees:g} degrees {args['direction']}")
        elif name == "move":
            self.done.append(f"moved {args['metres']:g} m {args['direction']}")
        else:
            self.done.append(result.split(":")[0].split(",")[0].rstrip(".").lower())  # "arrived", "drove for 2 s"
        now = self.robot.pose()
        heading = (now.yaw - self.began.yaw + 180) % 360 - 180
        away = math.dist((now.x, now.y), (self.began.x, self.began.y))
        facing = "facing the way it began" if abs(heading) < 5 else \
            f"facing {abs(heading):.0f} degrees {'left' if heading > 0 else 'right'} of the way it began"
        turning = f"; {self.turned:g} degrees of turning in all" if self.turned and name == "turn" else ""
        return (f"This request so far: {'; '.join(self.done[-8:])}{turning}. You are {away:.1f} m from where it "
                f"began, {facing}.")

    # ---- targets ------------------------------------------------------------------------------------------------
    def _choices(self, matches: list[Object]) -> str:
        """Keep the measured range attached to the badge; pixel size is misleading on a wide camera."""
        by_range = sorted(matches, key=lambda o: o.distance)
        marked = getattr(self.robot, "marked_cameras", None)
        if marked:
            self.looked.extend(marked({o.id for o in matches}))
        nearest = ", ".join(f"{o.label} {o.id}" for o in by_range
                            if round(o.distance, 1) == round(by_range[0].distance, 1))
        listed = "; ".join(f"{o.label} {o.id} ({_where_words(o)})" for o in by_range)
        result = f"{listed}. Closest by estimated range from here: {nearest}."
        if geom.dist(self.robot.pose(), self.began) >= 0.3:
            # A search can move before it finds anything. What's close now need not be what was close when asked.
            start = sorted((geom.dist(o, self.began), o.id) for o in matches)
            result += " From the request start: " + "; ".join(f"ID {i} about {d:.1f} m" for d, i in start) + "."
        return result

    def _resolve(self, target: str, *, people_only: bool = False) -> Any:
        """An id, 'user', a room or saved place, or a description, as something the robot can drive to."""
        r = self.robot
        text = str(target).strip().strip("'\"")
        low = re.sub(r"^(the|a|an|my)\s+", "", text.lower()).strip()
        if low in NO_TARGET:
            raise ToolError("say which target: an id from the object list, 'user', a room, or a description")
        if low in USER_WORDS or (people_only and low in ("person", "them", "him", "her")):
            user = r.user()
            if user is None:
                raise ToolError("the user is not in view. Use look('around') or search('person') to find them")
            return user
        if re.match(r"under(?:neath)?\s+", low):
            raise ToolError(NO_CLEARANCE)
        m = (re.fullmatch(r"(?:#|id\s*[:=]?\s*|object\s+)?(\d+)(?:\s+[a-z ()-]+)?", low)  # "4", "#4", "4 box"
             or re.fullmatch(r"[a-z ]+?\s*#?(\d+)", low))  # "box 4", "box #4"
        if m:
            try:
                return r.obj(int(m.group(1)))
            except NotFound as exc:  # the robot says why, when it knows (gone from where it was, not one after all)
                raise ToolError(str(exc) if str(exc) != f"there is no object {m.group(1)}"
                                else f"there is no object {m.group(1)} in the object list") from None
        if not people_only:
            places = {str(p).lower() for p in getattr(r, "places", {})}
            rooms = {str(p).lower() for p in getattr(getattr(r, "scene", None), "rooms", {})}
            name = re.sub(r"\s*\(.*\)$", "", low)  # a place as the observation lists it: "kitchen (2.1 m front)"
            key = name.removesuffix(" room").strip()
            if name in places or name in rooms:
                return name
            if key in rooms:
                return key
        _check_visual_name(text)
        if RELATION.search(low):  # "behind the sofa" would otherwise silently mean "the sofa"
            raise ToolError(f"{text!r} describes a spot, not a thing. To go to a spot you can see, use go_toward "
                            "with the direction of the picture it is in and about how far; for a side of a thing use "
                            "go_to_side, to stop some metres from it use stop_near")
        if VIEW_ORDER.search(text):  # name lookup matches shared words; "right chair" must not pick any chair
            raise ToolError("a left/right selection needs a viewpoint. Use look(direction, what) with the "
                            "visual name, then choose the matching ID in the marked picture")
        matches = list(r.where(text)) or list(r.detect(text))  # known already, or in view when looked for by name
        if not matches:
            label = visual_name(low)
            if label and label != low:
                candidates = [o for o in r.where(label) if o.last_seen <= 1.5]
                if candidates:
                    raise ToolError(f"I see {label}, but cannot confirm {text!r}. {self._choices(candidates)} "
                                    "Check the marked views and use an ID only if it matches, or ask the user")
            # "go to the bicycle" means finding it first: answering "use search" cost the model a turn, and a small
            # model need not know to
            self.found = r.search(text)
            if self.found is None:
                raise ToolError(f"searched here and found no {text!r}. Search another room, or ask the user")
            return self.found
        best = [o for o in matches if o.label == matches[0].label]
        if len(best) > 1:
            raise ToolError(f"{text!r} could be any of: {self._choices(best)} Use the id, or ask the user which one")
        return matches[0]

    def _name(self, t: Any) -> str:
        if isinstance(t, Object):
            user = self.robot.user()
            return "the user" if user is not None and user.id == t.id else f"{t.label} {t.id}"
        if isinstance(t, Pose):
            return "that place"
        return str(t) if str(t) in getattr(self.robot, "places", {}) else f"the {t}"  # "the kitchen", a place as named

    # ---- the tools ----------------------------------------------------------------------------------------------
    def _define(self) -> list[Tool]:
        target = {"type": "string", "description": "an object ID, \"user\", a room or place, or a visual name "
                                                   "like \"red ball\". Compare returned distances, not object names."}
        side = {"type": "string", "enum": ["left", "right"]}
        return [
            Tool("say", "Say something out loud now, while you keep working. Your final reply is spoken too.",
                 {"text": {"type": "string"}}, ("text",), self.say),
            Tool("go_to", "Park beside a target and face it. Use an ID or visual name; a name not yet seen is searched "
                          "for first. For a requested side or gap use go_to_side or stop_near. Parking beside "
                          "something does not go underneath.",
                 {"target": target}, ("target",), self.go_to, moves=True),
            Tool("go_toward", "Drive about some metres in a direction, around obstacles: to go somewhere you can see "
                              "in a picture but that is not one thing you can name. The front picture shows what is in "
                              "front of you; rear-left shows behind you on the left; rear-right behind you on the right.",
                 {"direction": {"type": "string", "enum": list(BEARINGS)},
                  "metres": {"type": "number", "minimum": 0.3, "maximum": 6}},
                 ("direction", "metres"), self.go_toward, moves=True),
            Tool("go_to_side", "Drive to one side of a target, when the user names a side: behind it (its far "
                               "side), in front of it, or its left or right.",
                 {"target": target, "side": {"type": "string", "enum": ["behind", "front", "left", "right", "under"],
                                             "description": "Relative to the approach. Under is refused without "
                                                            "verified clearance."},
                  "distance": {"type": "number", "minimum": 0.3, "maximum": 5,
                               "description": "Only if requested: gap in metres from the object's edge."}},
                 ("target", "side"), self.go_to_side, moves=True),
            Tool("stop_near", "Drive toward a target and stop with some metres of space between you and it, when the "
                              "user says how far.",
                 {"target": target, "metres": {"type": "number", "minimum": 0.3, "maximum": 5}},
                 ("target", "metres"), self.stop_near, moves=True),
            Tool("move", "Drive straight forward or backward.",
                 {"metres": {"type": "number", "minimum": 0.05, "maximum": 5},
                  "direction": {"type": "string", "enum": ["forward", "backward"]}},
                 ("metres", "direction"), self.move, moves=True),
            Tool("turn", "Turn on the spot. 90 is a quarter turn, 180 turns around, 360 is a full spin.",
                 {"degrees": {"type": "number", "minimum": 1, "maximum": 1800}, "direction": side},
                 ("degrees", "direction"), self.turn, moves=True),
            Tool("drive", "Free motion for anything the other tools don't cover (wiggle, rock, slow spin, arc).",
                 {"speed": {"type": "number", "minimum": -0.5, "maximum": 0.5,
                            "description": "metres per second, + forward, - backward"},
                  "turn_rate": {"type": "number", "minimum": -120, "maximum": 120,
                                "description": "degrees per second, + left, - right"},
                  "seconds": {"type": "number", "minimum": 0.1, "maximum": 60}},
                 ("speed", "turn_rate", "seconds"), self.drive, moves=True),
            Tool("orbit", "Drive in circles around a target.",
                 {"target": target, "laps": {"type": "number", "minimum": 0.25, "maximum": 10},
                  "direction": {"type": "string", "enum": ["clockwise", "anticlockwise"]},
                  "gap": {"type": "number", "minimum": 0.1, "maximum": 2,
                          "description": "metres between you and its edge; default 0.5"}},
                 ("target", "laps", "direction"), self.orbit, moves=True),
            Tool("follow", "Follow a person or something that moves (a pet, a toy car), staying about a metre "
                           "away, until the user says stop (or for `seconds`).",
                 {"target": {"type": "string", "description": "\"user\", an id, or a description like \"the cat\""},
                  "seconds": {"type": "number", "minimum": 1, "maximum": 3600}},
                 ("target",), self.follow, moves=True),
            Tool("face", "Turn once to face a target.", {"target": target}, ("target",), self.face, moves=True),
            Tool("keep_facing", "Keep turning to face a target as it moves, until the user says stop.",
                 {"target": target}, ("target",), self.keep_facing, moves=True),
            Tool("search", "Find something that is not in the object list: checks memory, then drives around the "
                           "area until the cameras see it.",
                 {"what": {"type": "string", "description": "a visual name, like \"red ball\" or \"keys\", "
                                                           "without a distance comparison"},
                  "area": {"type": "string", "description": "\"here\" (this room), a room name, or \"everywhere\". "
                                                           "Search the room where it most likely is first."},
                  "near": {"type": "string", "description": "optional: a thing to start beside, where it likely is "
                                                           "(an id or a name, like \"sofa 4\")"}},
                 ("what",), self.search, moves=True),
            Tool("look", "Inspect fresh camera pictures without driving. To label what you see, supply its visual name.",
                 {"direction": {"type": "string", "enum": list(CAMERAS)},
                  "what": {"type": "string", "description": "Optional object name to detect and mark, e.g. 'bag'."}},
                 ("direction",), self.look),
            Tool("look_back", "See what the front camera saw in the last seconds, for what happened or moved. "
                              "Up to four pictures come in the next message, oldest first.",
                 {"seconds": {"type": "number", "minimum": 1, "maximum": 30}}, ("seconds",), self.look_back),
            Tool("wait", "Stay still for some seconds, or until something matching `until` comes into view.",
                 {"seconds": {"type": "number", "minimum": 0.5, "maximum": 600},
                  "until": {"type": "string", "description": "optional: what to wait for, like \"person\""}},
                 ("seconds",), self.wait),
            Tool("ask", "Ask the user a question out loud and wait for the answer.",
                 {"question": {"type": "string"}}, ("question",), self.ask),
            Tool("remember", "Keep a fact in long-term memory.",
                 {"fact": {"type": "string"}}, ("fact",), self.remember),
            Tool("recall", "What you know about something: remembered facts and where it was last seen.",
                 {"query": {"type": "string"}}, ("query",), self.recall),
            Tool("todo", "Write a to-do list for a request with several parts, in order. It is shown back to you "
                         "after every step.",
                 {"steps": {"type": "array", "items": {"type": "string"}}}, ("steps",), self.write_todo),
            Tool("todo_done", "Tick off a finished step of the to-do list by its number.",
                 {"step": {"type": "integer", "minimum": 1, "maximum": 30}}, ("step",), self.todo_done),
        ]

    def say(self, text: str) -> str:
        self.robot.say(speakable(text))
        return "Said it."

    def go_to(self, target: str, distance: float | None = None, side: str = "") -> str:
        if side == "under":
            raise ToolError(NO_CLEARANCE)
        t = self._resolve(target)
        person = isinstance(t, Object) and t.label == "person"
        before = self.robot.pose()
        if side:
            self.robot.go_to(t, distance=distance if distance is not None else (0.8 if person else 0.2), side=side)
        else:
            edge = t.radius + ROBOT_RADIUS if isinstance(t, Object) else 0.0
            self.robot.go_to(t, distance=distance + edge if distance is not None else (0.8 if person else 0.0))
        after = self.robot.pose()
        if not isinstance(t, Object) and math.dist((before.x, before.y), (after.x, after.y)) < 0.1:
            return f"Already at {self._name(t)}: you did not move."  # "start" is where this request began
        return self._arrived(t)

    def go_to_side(self, target: str, side: str, distance: float | None = None) -> str:
        return self.go_to(target, side=side, distance=distance)

    def stop_near(self, target: str, metres: float) -> str:
        return self.go_to(target, distance=metres)

    def _arrived(self, t: Any) -> str:
        if isinstance(t, Object):
            try:
                self.robot.face(t)
                fresh = self.robot.obj(t.id)
                if fresh.last_seen > 0:
                    raise NotFound(f"last seen {fresh.last_seen:g} s ago")
            except NotFound as exc:
                self.looked = list(CAMERAS["front"])
                raise ToolError(f"drive finished {geom.dist(self.robot.pose(), t):.1f} m from the last mapped centre "
                                f"of {t.label} {t.id}. Target not confirmed: {exc}. Check the front view") from exc
            gap = max(0.0, fresh.distance - fresh.radius - ROBOT_RADIUS)
            return (f"Drive finished. Facing {self._name(fresh)}, estimated gap {gap:.1f} m "
                    f"({fresh.distance:.1f} m to centre).")
        return f"Arrived at {self._name(t)}."

    def move(self, metres: float, direction: str) -> str:
        self.robot.move(metres, direction=direction)
        return f"Moved {metres:g} m {direction}."

    def turn(self, degrees: float, direction: str) -> str:
        self.robot.turn(degrees, direction=direction)
        return f"Turned {degrees:g} degrees {direction}."

    def drive(self, speed: float, turn_rate: float, seconds: float) -> str:
        self.robot.drive(metres_per_second=speed, degrees_per_second=turn_rate, seconds=seconds)
        return f"Drove for {seconds:g} s."

    def orbit(self, target: str, laps: float, direction: str, gap: float = 0.5) -> str:
        t = self._resolve(target)
        self.robot.orbit(t, laps=laps, direction="right" if direction == "clockwise" else "left", distance=gap)
        return f"Circled {self._name(t)} {laps:g} times {direction}."

    def follow(self, target: str, seconds: float | None = None) -> str:
        thing = self._resolve(target, people_only=True)
        if not isinstance(thing, Object):
            raise ToolError(f"follow needs something you can see, and {target!r} is a place")
        self.robot.follow(thing, distance=1.0, seconds=seconds)
        return f"Followed for {seconds:g} s." if seconds else "Stopped following."

    def face(self, target: str) -> str:
        t = self._resolve(target)
        self.robot.face(t)
        return f"Facing {self._name(t)}."

    def keep_facing(self, target: str) -> str:
        t = self._resolve(target)
        self.robot.face(t, background=True)
        while True:  # until "stop", or words from the user, preempt it
            self.robot.wait(1.0)

    def search(self, what: str, area: str = "here", near: str = "") -> str:
        _check_visual_name(what)
        if near:  # start beside it: "the table is by the sofa" is the model's to know, the looking is the robot's
            start = self._resolve(near)
            self.robot.go_to(start, distance=(start.radius + 0.5) if isinstance(start, Object) else 0.0)
        self.found = found = self.robot.search(what, area=area or "here")
        log = getattr(self.robot, "search_log", [])
        looked = f" Where you looked: {'; '.join(log[-4:])}." if log else ""
        if found is None:
            where = "everywhere I can reach" if area == "everywhere" else f"in {area}" if area != "here" else "here"
            return f"Searched {where}: no {what} found.{looked}"
        crop = getattr(self.robot, "target_crop", lambda _: None)(found.id)
        if crop is not None:
            self.pictures.append((f"Recorded detection crop for {found.label} ID {found.id} (not a live view):", crop))
        others = [o for o in self.robot.where(what, area=area or "here") if o.id != found.id]
        if others:
            return f"Found candidate {found.label} {found.id}. Known matches: {self._choices([found, *others])}{looked}"
        marked = getattr(self.robot, "marked_cameras", None)
        self.looked.extend(marked({found.id}) if marked else CAMERAS["around"])
        evidence = "" if self.looked or self.pictures else " No matching image is available to verify this ID."
        return f"Found candidate {found.label} {found.id}, {_where_words(found)}.{looked}{evidence}"

    def go_toward(self, direction: str, metres: float) -> str:
        before = self.robot.pose()
        self.robot.go_to(self.robot.toward(direction, metres))
        after = self.robot.pose()
        went = math.dist((before.x, before.y), (after.x, after.y))
        short = "" if went >= 0.8 * metres - 0.2 else \
            f" (the floor {metres:g} m that way could not be reached: this is the nearest that could)"
        return f"Drove {went:.1f} m {direction}{short}."

    def look_back(self, seconds: float) -> str:
        if not hasattr(self.robot, "past_views"):
            raise ToolError("this robot keeps no pictures from before")
        past = self.robot.past_views(seconds)
        if not past:
            return "No pictures from before yet."
        self.pictures.extend((f"front camera {ago:.0f} s ago:", rgb) for ago, rgb in past)
        return f"The front camera over the last {seconds:g} s: {len(past)} pictures follow, oldest first."

    def look(self, direction: str, what: str = "") -> str:
        if what:
            _check_visual_name(what)
            self.robot.detect(what)  # activate the requested label without starting a travelling search
        self.looked = list(dict.fromkeys([*self.looked, *CAMERAS[direction]]))
        return f"Looking {direction}: the pictures follow."

    def wait(self, seconds: float, until: str = "") -> str:
        if until:
            _check_visual_name(until)
            seen = self.robot.wait_for(until, seconds=seconds)
            self.moved = True
            return f"{until} is in view: {seen.label} {seen.id}, {_where_words(seen)}." if seen else \
                f"Waited {seconds:g} s; no {until} came into view."
        self.robot.wait(seconds)
        self.moved = True
        return f"Waited {seconds:g} s."

    def ask(self, question: str) -> str:
        answer = self.robot.ask(question)
        return f'The user answered: "{answer}"' if answer else "No answer."

    def remember(self, fact: str) -> str:
        self.robot.remember(fact)
        if self.memory is not None and hasattr(self.memory, "remember_fact"):
            self.memory.remember_fact(fact)
        return "Remembered."

    def recall(self, query: str) -> str:
        lines = list(self.robot.recall(query))
        if self.memory is not None and hasattr(self.memory, "query_facts"):
            lines += [f for f in self.memory.query_facts(query) if f not in lines]
        room_at = getattr(self.robot, "_room_at", None)
        for o in self.robot.where(query, area="everywhere"):
            ago = "now" if o.last_seen < 2 else f"{o.last_seen / 60:.0f} min ago" if o.last_seen >= 60 else \
                f"{o.last_seen:.0f} s ago"
            room = room_at(*o.position) if room_at else ""
            lines.append(f"{o.label} {o.id} last seen {ago}{f' in the {room}' if room else ''}, "
                         f"{_where_words(o)} from here")
        return "\n".join(lines) if lines else f"Nothing remembered about {query!r}."

    def write_todo(self, steps: list[str]) -> str:
        self.todo.write(steps)
        if not self.todo.steps:
            raise ToolError("the to-do list needs at least one step")
        return "To-do list:\n" + self.todo.text()

    def todo_done(self, step: int) -> str:
        self.todo.tick(step)
        left = self.todo.open_steps()
        return f"Ticked step {step}. " + (f"Still to do: {'; '.join(left)}." if left else "All steps are done.")
