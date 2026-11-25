"""Sprocket's tools: what the model can do, as function schemas in its native tool-calling format.

Every call is checked before it runs (an unknown tool, a missing or out-of-range argument, a target that matches
nothing or several things) and every answer is a short sentence the model reads on its next turn. A failure says what
went wrong and what to try instead; nothing is silently changed or guessed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from robot.agent.api import NotFound, Object, PathBlocked, Pose, TargetLost, Timeout, geom, wrap
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
                      r"neath|under|far side of|close to|in|on|by|at)\b")
RANGE_ORDER = re.compile(r"\b(?:nearest|closest|farthest|furthest)\b", re.IGNORECASE)
VIEW_ORDER = re.compile(r"\b(?:leftmost|rightmost|left|right)\b", re.IGNORECASE)
USER_WORDS = {"user", "me", "you", "the user", "person(user)", "human", "speaker", "my owner", "owner"}
NO_TARGET = {"", "none", "null", "nothing"}
PLACE_STOP = frozenset({"the", "a", "an", "my", "to", "in", "at", "by", "on", "of", "near", "area", "room", "place"})
CLUSTER_GAP = 0.7  # metres: co-located same-label targets (e.g. a pair of shoes) auto-pick the nearest instead of asking
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


@dataclass(frozen=True)
class ActionResult:
    """Measured outcome of one tool call."""

    text: str
    ok: bool
    status: str = "ok"  # ok, already_there, reached, partial, blocked, target_lost, timeout, not_found, error
    moved_m: float = 0.0
    turned_deg: float = 0.0


def recalled_facts(memory: Any, query: str, extra: Iterable[str] = ()) -> list[str]:
    """Retrieve matching stored facts (oldest to newest when multiple match) plus any extra unstored notes."""
    dated: list[tuple[str, float]] = list(memory.query_facts(query)) if memory is not None else []
    if len(dated) > 1:
        dated.sort(key=lambda item: item[1])
        lines = [f"{text} ({'most recent' if i == len(dated) - 1 else 'earlier'})" for i, (text, _) in enumerate(dated)]
    else:
        lines = [text for text, _ in dated]
    seen = {text for text, _ in dated}
    for note in extra:
        if note not in seen:
            seen.add(note)
            lines.append(note)
    return lines


def _turn_span(robot: Any, pose_idx: int, before: Pose, after: Pose, commanded: float) -> float | None:
    """Measured degrees turned since `pose_idx` (handles multi-turn spins via trace poses)."""
    poses = getattr(getattr(robot, "trace", None), "poses", ())
    if len(poses) > pose_idx + 1:
        yaws = [p[3] for p in poses[pose_idx:]]
        return math.degrees(sum(abs(wrap(b - a)) for a, b in zip(yaws, yaws[1:], strict=False)))
    if commanded < 180:
        return abs((after.yaw - before.yaw + 180) % 360 - 180)
    return None


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, dict[str, Any]]
    required: tuple[str, ...]
    run: Callable[..., str | ActionResult]
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


def _object_id(text: str) -> int | None:
    """Numeric track ID from '4', '#4', '4 box', 'box 4', 'box #4', or None."""
    low = text.strip().lower()
    m = (re.fullmatch(r"(?:#|id\s*[:=]?\s*|object\s+)?(\d+)(?:\s+[a-z ()-]+)?", low)
         or re.fullmatch(r"[a-z ]+?\s*#?(\d+)", low))
    return int(m.group(1)) if m else None


def _check_visual_name(text: str, *, allow_relation: bool = False) -> None:
    if _object_id(text) is not None:
        raise ToolError(f"{text!r} is an object ID, not a plain object name. Pass just the object name in "
                        "what=..., or pass the ID directly to go_to/face")
    if RANGE_ORDER.search(text):
        raise ToolError("a distance comparison is not a camera label. Call search with just the object name, "
                        "then choose an ID by its measured distance")
    if not allow_relation and RELATION.search(text.lower()):
        raise ToolError(f"{text!r} is a location phrase, not a plain object name. Pass just the object name in "
                        "what=... and put the area or landmark in search(near=...)")


def _place_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words} - PLACE_STOP


def _match_place(query: str, places: dict[str, Any], pose: Any = None) -> str | None:
    """Best-matching mapped area for a free-form query (e.g. 'hallway' -> 'hallway: doorway, door'),
    ignoring system places ('start', 'where you were before')."""
    q = _place_tokens(query)
    if not q or not places:
        return None
    best: tuple[int, int, float, str] | None = None
    for raw, p in places.items():
        name = str(raw).lower()
        if name in ("start", "where you were before"):
            continue
        head, _, rest = name.partition(":")
        head_toks = _place_tokens(head)
        all_toks = head_toks | _place_tokens(rest)
        overlap = len(q & all_toks)
        if not overlap or (overlap < len(q) and not (q & head_toks)):
            continue
        dist = -math.dist((p.x, p.y), (pose.x, pose.y)) if pose is not None and hasattr(p, "x") else 0.0
        score = (len(q & head_toks), overlap, dist)
        if best is None or score > best[:3]:
            best = (*score, name)
    return best[3] if best is not None else None


class Tools:
    """The tools bound to one robot for one task."""

    def __init__(self, robot: Any, memory: Any = None) -> None:
        self.robot = robot
        self.memory = memory if memory is not None else getattr(robot, "store", None)
        robot_persists = (
            self.memory is not None
            and getattr(robot, "persists_facts", False)
            and getattr(robot, "store", None) is self.memory
        )
        self._fact_store = None if robot_persists else self.memory
        self.todo = Todo()
        self.looked: list[str] = []  # cameras the model asked to see since the last observation
        self.moved = False  # the robot moved since the last observation
        self.found: Object | None = None
        self.pictures: list[tuple[str, Any]] = []  # sensor evidence with its provenance, not a new live view
        self._tools = {t.name: t for t in self._define()}
        self.began = robot.pose()  # where this request began
        self._task_pose_start = len(getattr(getattr(robot, "trace", None), "poses", ()))
        self.done: list[str] = []  # each motion of this request, as it went
        self.turned = 0.0  # degrees of turning in all, either way

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any]) -> ActionResult:
        """Run one call and return its structured outcome. Stops and preemptions propagate."""
        tool = self._tools.get(name)
        if tool is None:
            return ActionResult(f"There is no tool {name!r}. Tools: {', '.join(self._tools)}.", False, "error")
        self.found = None  # what this call found by searching
        before = self.robot.pose() if tool.moves else None
        moved_m = 0.0

        def motion_note() -> str:
            nonlocal moved_m
            if not tool.moves or before is None:
                return ""
            now = self.robot.pose()
            moved_m = math.dist((now.x, now.y), (before.x, before.y))
            return f" (moved {moved_m:.1f} m before stopping)" if moved_m > 0.05 else ""

        try:
            checked = _check(tool, args)
            raw = tool.run(**checked)
            res = raw if isinstance(raw, ActionResult) else ActionResult(raw, True, "reached" if tool.moves else "ok")
            if self.found is not None and name != "search":
                res = replace(res, text=f"Searched for it: found {self.found.label} {self.found.id}. {res.text}")
            if tool.moves:
                self.moved = True
        except ToolError as exc:
            return ActionResult(f"Error: {exc}.{motion_note()}", False, "error", moved_m)
        except NotFound as exc:
            return ActionResult(f"Not found: {exc}. Use search() to look for it, or ask the user.", False, "not_found")
        except PathBlocked as exc:
            return ActionResult(
                f"Blocked: {exc}{motion_note()}. Try another way, a different target, or ask the user for help.",
                False, "blocked", moved_m,
            )
        except TargetLost as exc:
            self.looked.extend(CAMERAS["around"])
            return ActionResult(
                f"Target lost: {exc}{motion_note()}. Inspect the current views before choosing another ID.",
                False, "target_lost", moved_m,
            )
        except Timeout as exc:
            return ActionResult(f"Timed out: {exc}{motion_note()}.", False, "timeout", moved_m)
        except (ValueError, TypeError) as exc:
            return ActionResult(f"Error: {exc}.", False, "error")
        finally:
            if tool.moves and not self.moved and before is not None:
                now = self.robot.pose()
                if (math.dist((now.x, now.y), (before.x, before.y)) > 0.02
                        or abs((now.yaw - before.yaw + 180) % 360 - 180) > 2.0):
                    self.moved = True
        if tool.moves:
            progress = self._progress(name, checked, res)
            if progress:
                res = replace(res, text=res.text + "\n" + progress)
        return res

    def _progress(self, name: str, args: dict[str, Any], res: ActionResult) -> str:
        """Keep motion progress compact: always report for repetition-prone steps (turn/move/drive/orbit) or after
        multiple motions, while omitting boilerplate on a single-step go_to/search to keep context lean."""
        if name == "turn":
            degrees = round(res.turned_deg) if res.status == "partial" else args["degrees"]
            self.turned += abs(degrees)
            self.done.append(f"turned {degrees:g} degrees {args['direction']}")
        elif name == "move":
            metres = f"{res.moved_m:.2f}" if res.status == "partial" else f"{args['metres']:g}"
            self.done.append(f"moved {metres} m {args['direction']}")
        else:
            self.done.append(res.text.split(":")[0].split(",")[0].rstrip(".").lower())  # "arrived", "drove for 2 s"
        if name not in ("turn", "move", "drive", "orbit") and len(self.done) <= 1:
            return ""
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

    def _resolve(self, target: str, *, people_only: bool = False, find: bool = True, fuzzy_place: bool = False) -> Any:
        """An id, 'user', a room or saved place, or a description, as something the robot can drive to; `find`: a
        description neither known nor in view is searched for, which drives (not for "turn to face the sofa": it
        drove off to look for the sofa)."""
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
        oid = _object_id(low)
        if oid is not None:
            try:
                return r.obj(oid)
            except NotFound as exc:  # the robot says why, when it knows (gone from where it was, not one after all)
                raise ToolError(str(exc) if str(exc) != f"there is no object {oid}"
                                else f"there is no object {oid} in the object list") from None
        raw_places = getattr(r, "places", {})
        places_map = {str(k).lower(): v for k, v in raw_places.items()}
        name = re.sub(r"\s*\(.*\)$", "", low)  # a place as the observation lists it: "kitchen (2.1 m front)"
        if not people_only:
            rooms = {str(p).lower() for p in getattr(getattr(r, "scene", None), "rooms", {})}
            key = name.removesuffix(" room").strip()
            if name in places_map or name in rooms:
                return name
            if key in rooms:
                return key
        _check_visual_name(text, allow_relation=True)
        if RELATION.search(low):  # "behind the sofa" or "chair in living room" would otherwise pollute YOLOE
            raise ToolError(f"{text!r} mixes an object and a place or spot. To find a thing in an area or beside a "
                            "landmark, pass just the object name to search(what=..., near=...) (or go_to the area "
                            "first). To drive to a visible floor spot use go_toward; for a side of a thing use "
                            "go_to_side; to stop some metres from it use stop_near")
        if VIEW_ORDER.search(text):  # name lookup matches shared words; "right chair" must not pick any chair
            raise ToolError("a left/right selection needs a viewpoint. Use look(direction, what) with the "
                            "visual name, then choose the matching ID in the marked picture")

        def pick(found: list[Object]) -> Object:
            best = [o for o in found if o.label == found[0].label]
            if len(best) > 1 and max(geom.dist(o, best[0]) for o in best) > CLUSTER_GAP:
                raise ToolError(f"{text!r} could be any of: {self._choices(best)} "
                                "Use the id, or ask the user which one")
            return found[0]

        known = list(r.where(text))
        if known:
            return pick(known)
        if fuzzy_place and not people_only and (hit := _match_place(name, places_map, r.pose())) is not None:
            return hit
        matches = list(r.detect(text))  # in view when looked for by name
        if not matches:
            label = visual_name(low)
            if label and label != low:
                candidates = [o for o in r.where(label) if o.last_seen <= 1.5]
                if candidates:
                    raise ToolError(f"I see {label}, but cannot confirm {text!r}. {self._choices(candidates)} "
                                    "Check the marked views and use an ID only if it matches, or ask the user")
            if not find:
                raise ToolError(f"no {text!r} is known or in view. Use look('around', {text!r}) to look without "
                                "driving, or search to drive round looking for it")
            # If a mapped semantic place names this area/landmark, drive there before sweeping the home blindly.
            hint = _match_place(name, places_map, r.pose()) if not people_only else None
            if hint is not None:
                hp = places_map.get(hint)
                if hp is not None and math.dist((hp.x, hp.y), (r.pose().x, r.pose().y)) > 0.6:
                    try:
                        r.go_to(hint)
                    except (NotFound, PathBlocked):
                        pass
                    matches = list(r.where(text)) or list(r.detect(text))
                    if matches:
                        return pick(matches)
                if _place_tokens(name) <= _place_tokens(hint.partition(":")[0]):
                    return hint
            # "go to the bicycle" means finding it first: answering "use search" cost the model a turn, and a small
            # model need not know to
            self.found = r.search(text)
            if self.found is None:
                raise ToolError(f"searched here and found no {text!r}. Search another room, or ask the user")
            return self.found
        return pick(matches)

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
            Tool("go_to", "Go to a target and face it. Use an ID, or any visible thing, area or place by name even if "
                          "not in the Objects list yet (it will be located or searched for automatically). For a "
                          "requested side or gap use go_to_side or stop_near. Parking beside something does not go "
                          "underneath.",
                 {"target": target}, ("target",), self.go_to, moves=True),
            Tool("go_toward", "Drive about some metres in a direction, around obstacles, only when no named thing, "
                              "surface or area can be given to go_to (such as an unnamed gap or a plain direction). "
                              "The front picture shows what is in front of you; rear-left shows behind you on the "
                              "left; rear-right behind you on the right.",
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
            Tool("search", "Find something not in the object list: checks memory, optionally starts beside a likely "
                           "landmark/area (`near`) or heads a promising camera direction first (`toward`), then "
                           "sweeps unseen floor until the cameras see it.",
                 {"what": {"type": "string", "description": "a visual name, like \"red ball\" or \"keys\", "
                                                           "without a distance comparison"},
                  "area": {"type": "string", "description": "\"here\" (this room), a room name, or \"everywhere\", not "
                                                           "a direction."},
                  "near": {"type": "string", "description": "optional: a known or visible landmark/area to go beside "
                                                           "first, where the thing likely is (e.g. \"sofa 4\", "
                                                           "\"kitchen\", \"stairs\")"},
                  "toward": {"type": "string", "enum": list(BEARINGS),
                             "description": "optional: camera direction to head first when a doorway or likely area "
                                            "lies that way"}},
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
            Tool("say", "Speak mid-task ONLY when you still have more tool calls to make after speaking. Never call "
                        "this for your final reply: end with plain text and no tool call instead.",
                 {"text": {"type": "string"}}, ("text",), self.say),
        ]

    def say(self, text: str) -> str:
        self.robot.say(speakable(text))
        return "Said it. If the request is finished, reply with no tool call."

    def go_to(self, target: str, distance: float | None = None, side: str = "") -> str | ActionResult:
        if side == "under":
            raise ToolError(NO_CLEARANCE)
        before = self.robot.pose()
        t = self._resolve(target)
        person = isinstance(t, Object) and t.label == "person"
        if side:
            self.robot.go_to(t, distance=distance if distance is not None else (0.8 if person else 0.2), side=side)
        else:
            edge = t.radius + ROBOT_RADIUS if isinstance(t, Object) else 0.0
            self.robot.go_to(t, distance=distance + edge if distance is not None else (0.8 if person else 0.0))
        after = self.robot.pose()
        moved = math.dist((before.x, before.y), (after.x, after.y))
        if not isinstance(t, Object):
            places = {str(k).lower(): v for k, v in getattr(self.robot, "places", {}).items()}
            dest = places.get(str(t).lower())
            if dest is not None and hasattr(dest, "x"):
                left = math.dist((after.x, after.y), (dest.x, dest.y))
                requested = distance if distance is not None else (0.2 if side else 0.0)
                if left > requested + 0.5:
                    return ActionResult(
                        f"Ended {left:.1f} m short of {self._name(t)} (moved {moved:.1f} m).",
                        False, "partial", moved,
                    )
            if moved < 0.1:
                return ActionResult(f"Already at {self._name(t)}: you did not move.", True, "already_there", moved)
            return ActionResult(f"Arrived at {self._name(t)}.", True, "reached", moved)
        expected_gap = distance if distance is not None else (0.8 if person else 0.2 if side else 0.0)
        return self._arrived(t, expected_gap=expected_gap, moved_m=moved)

    def go_to_side(self, target: str, side: str, distance: float | None = None) -> str | ActionResult:
        return self.go_to(target, side=side, distance=distance)

    def stop_near(self, target: str, metres: float) -> str | ActionResult:
        return self.go_to(target, distance=metres)

    def _arrived(self, t: Any, *, expected_gap: float = 0.0, moved_m: float = 0.0) -> ActionResult:
        if isinstance(t, Object):
            try:
                self.robot.face(t)
                fresh = self.robot.obj(t.id)
                if not self.robot.sees_now(fresh):
                    raise NotFound(f"last seen {fresh.last_seen:g} s ago")
            except NotFound as exc:
                self.looked = list(CAMERAS["front"])
                raise ToolError(f"drive finished {geom.dist(self.robot.pose(), t):.1f} m from the last mapped centre "
                                f"of {t.label} {t.id}. Target not confirmed: {exc}. Check the front view") from exc
            if fresh.radius == 0.0:
                return ActionResult(
                    f"Drive finished. Reached {self._name(fresh)} (on the nearest open floor to its centre).",
                    True, "reached", moved_m,
                )
            gap = max(0.0, fresh.distance - fresh.radius - ROBOT_RADIUS)
            if gap <= 0.6:
                return ActionResult(
                    f"Drive finished. Reached {self._name(fresh)}, facing it "
                    f"(gap {gap:.1f} m, {fresh.distance:.1f} m to centre).",
                    True, "reached", moved_m,
                )
            if gap <= expected_gap + 0.6:
                return ActionResult(
                    f"Drive finished. Facing {self._name(fresh)}, estimated gap {gap:.1f} m "
                    f"({fresh.distance:.1f} m to centre).",
                    True, "reached", moved_m,
                )
            return ActionResult(
                f"Ended short of {self._name(fresh)}: facing it with estimated gap {gap:.1f} m "
                f"({fresh.distance:.1f} m to centre).",
                False, "partial", moved_m,
            )
        return ActionResult(f"Arrived at {self._name(t)}.", True, "reached", moved_m)

    def move(self, metres: float, direction: str) -> ActionResult:
        before = self.robot.pose()
        self.robot.move(metres, direction=direction)
        after = self.robot.pose()
        went = math.dist((before.x, before.y), (after.x, after.y))
        if went < 0.8 * metres - 0.1:
            return ActionResult(
                f"Moved only {went:.2f} m of {metres:g} m {direction} (ended short).",
                False, "partial", went,
            )
        shown = f"{metres:g}" if abs(went - metres) <= 0.05 else f"{went:.2f}"
        return ActionResult(f"Moved {shown} m {direction}.", True, "reached", went)

    def turn(self, degrees: float, direction: str) -> ActionResult:
        pose_idx = len(getattr(getattr(self.robot, "trace", None), "poses", ()))
        before = self.robot.pose()
        self.robot.turn(degrees, direction=direction)
        after = self.robot.pose()
        turned = _turn_span(self.robot, pose_idx, before, after, degrees)
        if turned is not None and turned < 0.8 * degrees - 10.0:
            return ActionResult(
                f"Turned only {turned:.0f} of {degrees:g} degrees {direction} (ended short).",
                False, "partial", turned_deg=turned,
            )
        shown = degrees if turned is None or abs(turned - degrees) <= max(5.0, 0.08 * degrees) else round(turned)
        return ActionResult(
            f"Turned {shown:g} degrees {direction}.",
            True, "reached", turned_deg=turned if turned is not None else degrees,
        )

    def drive(self, speed: float, turn_rate: float, seconds: float) -> ActionResult:
        t0 = getattr(self.robot, "t", None)
        before = self.robot.pose()
        self.robot.drive(metres_per_second=speed, degrees_per_second=turn_rate, seconds=seconds)
        t1 = getattr(self.robot, "t", None)
        after = self.robot.pose()
        went = math.dist((before.x, before.y), (after.x, after.y))
        dt = (t1 - t0) if (t0 is not None and t1 is not None) else None
        if dt is not None and dt < 0.8 * seconds - 0.2:
            return ActionResult(f"Drove for {dt:.1f} s of {seconds:g} s (ended short).", False, "partial", went)
        return ActionResult(f"Drove for {seconds:g} s.", True, "reached", went)

    def orbit(self, target: str, laps: float, direction: str, gap: float = 0.5) -> str:
        t = self._resolve(target)
        self.robot.orbit(t, laps=laps, direction="right" if direction == "clockwise" else "left", distance=gap)
        return f"Circled {self._name(t)} {laps:g} times {direction}."

    def follow(self, target: str, seconds: float | None = None) -> ActionResult:
        thing = self._resolve(target, people_only=True)
        if not isinstance(thing, Object):
            raise ToolError(f"follow needs something you can see, and {target!r} is a place")
        t0 = getattr(self.robot, "t", None)
        self.robot.follow(thing, distance=1.0, seconds=seconds)
        t1 = getattr(self.robot, "t", None)
        dt = (t1 - t0) if (t0 is not None and t1 is not None) else None
        if seconds is not None and dt is not None and dt < 0.8 * seconds - 0.5:
            return ActionResult(
                f"Followed {self._name(thing)} for {dt:.1f} s of {seconds:g} s (ended early).",
                False, "partial",
            )
        if seconds is not None:
            return ActionResult(f"Followed for {seconds:g} s.", True, "reached")
        return ActionResult(
            f"Stopped following after {dt:.1f} s." if dt is not None and dt > 0 else "Stopped following.",
            True, "reached",
        )

    def face(self, target: str) -> str:
        t = self._resolve(target, find=False)
        self.robot.face(t)
        return f"Facing {self._name(t)}."

    def keep_facing(self, target: str) -> str:
        t = self._resolve(target, find=False)
        self.robot.face(t, background=True)
        while True:  # until "stop", or words from the user, preempt it
            self.robot.wait(1.0)

    def search(self, what: str, area: str = "here", near: str = "", toward: str = "") -> str:
        _check_visual_name(what)
        near_key = near.strip().lower().replace("_", "-")
        if near_key in BEARINGS and not toward:
            toward, near = near_key, ""
        if near and near_key not in {"here", "everywhere", "room"}:  # start beside a visible/known landmark if found
            try:
                start = self._resolve(near, find=False, fuzzy_place=True)
                self.robot.go_to(start, distance=(start.radius + 0.5) if isinstance(start, Object) else 0.0)
            except (ToolError, NotFound, PathBlocked):
                pass
        try:
            self.found = found = self.robot.search(what, area=area or "here", toward=toward)
        except TypeError:  # a test fake whose search() takes only (what, area)
            self.found = found = self.robot.search(what, area=area or "here")
        if found is None:
            log = getattr(self.robot, "search_log", [])
            poses = getattr(getattr(self.robot, "trace", None), "poses", ())
            trail = [(x, y) for _, x, y, _ in poses[self._task_pose_start:]] or [
                (self.robot.pose().x, self.robot.pose().y)
            ]
            mapped = [
                (str(n), p) for n, p in getattr(self.robot, "places", {}).items()
                if str(n).lower() not in ("start", "where you were before") and hasattr(p, "x")
            ]
            unvisited = [n for n, p in mapped if all(math.dist((p.x, p.y), xy) > 2.2 for xy in trail)]
            hint = f" Areas not visited closely yet: {'; '.join(unvisited[:4])}." if unvisited else ""
            where = "all reachable areas" if (mapped and not unvisited and area in ("here", "everywhere")) else (
                "everywhere I can reach" if area == "everywhere" else f"in {area}" if area != "here" else "here"
            )
            stops = f" ({len(log)} stops)" if log else ""
            return f"Searched {where}{stops}: no {what} found.{hint}"
        crop = getattr(self.robot, "target_crop", lambda _: None)(found.id)
        if crop is not None:
            self.pictures.append((f"Recorded detection crop for {found.label} ID {found.id} (not a live view):", crop))
        others = [o for o in self.robot.where(what, area=area or "here") if o.id != found.id]
        if others:
            return f"Found candidate {found.label} {found.id}. Known matches: {self._choices([found, *others])}"
        marked = getattr(self.robot, "marked_cameras", None)
        self.looked.extend(marked({found.id}) if marked else CAMERAS["around"])
        evidence = "" if self.looked or self.pictures else " No matching image is available to verify this ID."
        return f"Found candidate {found.label} {found.id}, {_where_words(found)}.{evidence}"

    def go_toward(self, direction: str, metres: float) -> ActionResult:
        before = self.robot.pose()
        self.robot.go_to(self.robot.toward(direction, metres))
        after = self.robot.pose()
        went = math.dist((before.x, before.y), (after.x, after.y))
        if went < 0.8 * metres - 0.2:
            return ActionResult(
                f"Drove {went:.1f} m {direction} (ended short: the floor {metres:g} m that way could not be reached).",
                False, "partial", went,
            )
        return ActionResult(f"Drove {went:.1f} m {direction}.", True, "reached", went)

    def look_back(self, seconds: float) -> str:
        if not hasattr(self.robot, "past_views"):
            raise ToolError("this robot keeps no pictures from before")
        past = self.robot.past_views(seconds)
        if not past:
            return "No pictures from before yet."
        self.pictures.extend((f"front camera {ago:.0f} s ago:", rgb) for ago, rgb in past)
        return f"The front camera over the last {seconds:g} s: {len(past)} pictures follow, oldest first."

    def look(self, direction: str, what: str = "") -> str:
        matches = ""
        if what:
            _check_visual_name(what)
            found = list(self.robot.detect(what))
            matches = (" Seen: " + "; ".join(f"{o.label} {o.id} ({_where_words(o)})" for o in found[:6]) + "."
                       if found else f" No {what} detected.")
        self.looked = list(dict.fromkeys([*self.looked, *CAMERAS[direction]]))
        return f"Looking {direction}: the pictures follow.{matches}"

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
        if self._fact_store is not None:
            self._fact_store.remember_fact(fact)
        return "Remembered."

    def recall(self, query: str) -> str:
        lines = recalled_facts(self.memory, query, extra=self.robot.recall(query))
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
