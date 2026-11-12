"""Task families for training data: an instruction in many phrasings, a correct program, and an automatic check.

Each family builds its own random room, picks targets the robot can actually know about, words the instruction one of
several ways (some spoken-style, with fillers), writes the program the planner should learn, and returns a check that
judges a finished run with the suite's own helpers. Programs are teacher-written templates: they teach the API's
idioms (skills first, words for directions, handles from the table); diversity comes from rooms and phrasing.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from robot.eval.suite import bearing_to, end, first_visit, gap, laps, moved, rocks, said, tracked, travelled, turned
from robot.sim.world import Scene, SimRobot, Thing
from robot.train.scenes import CATALOG, random_scene


@dataclass
class Task:
    family: str
    category: str
    scene: Scene
    instruction: str
    say: str
    plan: str
    program: str
    check: Callable[[SimRobot], bool]
    earlier: tuple[str, ...] = ()  # the conversation before this instruction, as the planner is shown it


NL = "\n"  # a line break inside a program text
FAMILIES: list[tuple[str, str, Callable[[random.Random], Task | None]]] = []


def family(category: str) -> Callable:
    def register(fn: Callable[[random.Random], Any]) -> Callable:
        FAMILIES.append((fn.__name__, category, fn))
        return fn
    return register


# ---- words ----------------------------------------------------------------------------------------------------
METRES = {0.3: ("30 centimetres", "0.3 metres"), 0.5: ("half a metre", "0.5 metres", "50 centimetres"),
          0.8: ("80 centimetres", "0.8 metres"), 1.0: ("one metre", "1 metre", "a metre"),
          1.5: ("one and a half metres", "1.5 metres"), 2.0: ("two metres", "2 metres"),
          2.5: ("two and a half metres", "2.5 metres")}
DEGREES = {30: ("30 degrees",), 45: ("45 degrees",), 60: ("60 degrees",), 90: ("90 degrees", "a quarter turn"),
           120: ("120 degrees",), 135: ("135 degrees",), 180: ("180 degrees", "half a turn")}
TIMES = {1: "once", 2: "twice", 3: "three times", 4: "four times"}
NUMBER = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def plural(label: str) -> str:
    return label + ("es" if label.endswith(("x", "s", "sh", "ch")) else "s")
ABSENT = ("elephant", "giraffe", "piano", "bicycle", "guitar", "kangaroo", "skateboard", "telescope")


def spoken(rng: random.Random, text: str) -> str:
    """Speech-to-text gives lower case, dropped full stops, fillers and polite openings; train on those too."""
    r = rng.random()
    low = text[0].lower() + text[1:]
    if r < 0.45:
        return text
    if r < 0.6:
        return text.lower().rstrip(".")
    if r < 0.7:
        return f"Hey Sprocket, {low}"
    if r < 0.8 and not text.endswith("?"):
        return f"Please {low}" if rng.random() < 0.5 else f"{text.rstrip('.')}, please."
    if r < 0.9 and not text.endswith("?"):
        return f"Could you {low.rstrip('.')}?"
    return f"um, {low.rstrip('.')}"


# ---- the room and what the robot knows in it --------------------------------------------------------------------
def objects(scene: Scene) -> list[Thing]:
    return [t for t in scene.things if not t.person and t.label != "door"]


def name_of(scene: Scene, t: Thing) -> str | None:
    """The shortest phrase that names this thing alone: "chair", or "red chair" when there are two."""
    same = [o for o in objects(scene) if o.label == t.label]
    if len(same) == 1:
        return t.label
    if t.color and sum(o.color == t.color for o in same) == 1:
        return f"{t.color} {t.label}"
    return None


def setup(rng: random.Random, **kw: Any) -> tuple[Scene, SimRobot] | None:
    scene = random_scene(rng, **kw)
    return (scene, SimRobot(scene)) if scene else None


def known(probe: SimRobot, *, nameable: bool = True, solid: bool | None = None) -> list[Thing]:
    out = [probe.things[i] for i in probe.memory if not probe.things[i].person and probe.things[i].label != "door"]
    if solid is not None:
        out = [t for t in out if t.solid == solid]
    return [t for t in out if not nameable or name_of(probe.scene, t)]


def visible(probe: SimRobot) -> list[Thing]:
    return [t for t in probe._visible() if not t.person and t.label != "door"]


def user_of(scene: Scene) -> Thing:
    return next(t for t in scene.things if t.user)


def start_of(s: SimRobot) -> tuple[float, float, float]:
    _, x, y, yaw = s.trace.poses[0]
    return x, y, yaw


def along(s: SimRobot) -> tuple[float, float]:
    """Displacement in the start frame: (forward, left)."""
    (x0, y0, a), (x1, y1, _) = start_of(s), end(s)
    dx, dy = x1 - x0, y1 - y0
    return dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)


def near(s: SimRobot, tid: int, within: float = 0.65) -> bool:
    return gap(*end(s)[:2], s.things[tid]) <= within


def task(fam: str, cat: str, scene: Scene, instruction: str, say_text: str, plan: str, program: str,
         check: Callable[[SimRobot], bool]) -> Task:
    return Task(fam, cat, scene, instruction, say_text, plan, program, check)


# ---- relative motion ------------------------------------------------------------------------------------------
@family("relative")
def move_straight(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    back = rng.random() < 0.4
    room = probe.clearance("behind" if back else "ahead")
    choices = [m for m in METRES if m <= room - 0.35]
    if not choices:
        return None
    m = rng.choice(choices)
    words = rng.choice(METRES[m])
    text = rng.choice([f"Back up {words}.", f"Reverse {words}.", f"Go backwards {words}.", f"Move back {words}."]
                      if back else [f"Drive forward {words}.", f"Go forward {words}.", f"Move {words} forward.",
                                    f"Roll ahead {words}."])
    sign = -1 if back else 1
    direction = "backward" if back else "forward"
    return task("move_straight", "relative", scene, spoken(rng, text), "Moving now.",
                f"Drive {direction} {m:g} m in a straight line.", f'robot.move({m:g}, "{direction}")\n',
                lambda s: abs(along(s)[0] - sign * m) <= 0.12 and abs(along(s)[1]) <= 0.15)


@family("relative")
def turn_on_spot(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, _ = got
    if rng.random() < 0.2:
        text = rng.choice(["Turn around.", "Face the other way.", "Turn round.", "Look behind you."])
        return task("turn_on_spot", "relative", scene, spoken(rng, text), "Turning around.",
                    "Turn half a turn on the spot.", 'robot.turn(180, "left")\n',
                    lambda s: abs(abs(turned(s)) - math.pi) < 0.15 and moved(s) < 0.1)
    deg = rng.choice(list(DEGREES))
    side = rng.choice(["left", "right"])
    words = rng.choice(DEGREES[deg])
    text = rng.choice([f"Turn {side} {words}.", f"Turn {words} to the {side}.", f"Rotate {side} by {words}.",
                       f"Turn {side}." if deg == 90 else f"Turn {side} {words}."])
    sign = 1 if side == "left" else -1
    return task("turn_on_spot", "relative", scene, spoken(rng, text), f"Turning {side}.",
                f"Turn {side} {deg} degrees on the spot.", f'robot.turn({deg}, "{side}")\n',
                lambda s: abs(turned(s) - sign * math.radians(deg)) < 0.14 and moved(s) < 0.1)


@family("relative")
def spin_around(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, _ = got
    if rng.random() < 0.5:
        n = rng.choice([1, 2, 3])
        text = rng.choice([f"Spin around {TIMES[n]}.", f"Do {NUMBER[n]} full spins.", f"Turn round in a circle {TIMES[n]}.",
                           "Do a full spin." if n == 1 else f"Spin {TIMES[n]}."])
        return task("spin_around", "relative", scene, spoken(rng, text), "Here I go!",
                    f"Spin on the spot {n} full turns.", f"robot.spin({n})\n",
                    lambda s: n * 2 * math.pi - 0.4 <= abs(turned(s)) <= n * 2 * math.pi + 0.6 and moved(s) < 0.15)
    secs, speed = rng.choice([3, 5, 8, 10]), rng.choice(["slow", "fast", "normal"])
    adverb = {"slow": "slowly ", "fast": "quickly ", "normal": ""}[speed]
    text = rng.choice([f"Spin {adverb}for {secs} seconds.", f"Spin {adverb}in place for {secs} seconds.",
                       f"Turn round and round {adverb}for {secs} seconds."])
    return task("spin_around", "relative", scene, spoken(rng, text), "Spinning!",
                f"Spin {speed} for {secs} seconds.", f'robot.spin(seconds={secs}, speed="{speed}")\n',
                lambda s: s.t >= secs - 0.2 and abs(turned(s)) > 0.5 * secs * 0.5 and moved(s) < 0.15)


@family("relative")
def move_then_turn(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    choices = [m for m in (0.5, 1.0, 1.5) if m <= probe.clearance("ahead") - 0.35]
    if not choices:
        return None
    m, deg, side = rng.choice(choices), rng.choice([45, 90, 180]), rng.choice(["left", "right"])
    text = rng.choice([f"Go forward {rng.choice(METRES[m])}, then turn {side} {rng.choice(DEGREES[deg])}.",
                       f"Drive {rng.choice(METRES[m])} ahead and then turn {rng.choice(DEGREES[deg])} to the {side}."])
    sign = 1 if side == "left" else -1
    return task("move_then_turn", "relative", scene, spoken(rng, text), "On it.",
                f"Move forward {m:g} m, then turn {side} {deg} degrees.",
                f'robot.move({m:g}, "forward")\nrobot.turn({deg}, "{side}")\n',
                lambda s: abs(along(s)[0] - m) <= 0.15 and abs(turned(s) - sign * math.radians(deg)) < 0.2)


@family("relative")
def drive_shape(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, _ = got
    sides, name = rng.choice([(4, "square"), (3, "triangle")])
    m = rng.choice([0.5, 0.8, 1.0])
    side = rng.choice(["left", "right"])
    angle = 360 // sides
    text = rng.choice([f"Drive in a {name} with {rng.choice(METRES[m])} sides.",
                       f"Drive a {name}, {rng.choice(METRES[m])} on each side, and end where you started.",
                       f"Make a {name} turning {side}, sides of {rng.choice(METRES[m])}."])
    return task("drive_shape", "relative", scene, spoken(rng, text), f"Drawing a {name}.",
                f"Repeat {sides} times: move {m:g} m, turn {side} {angle} degrees.",
                f'for _ in range({sides}):\n    robot.move({m:g}, "forward")\n    robot.turn({angle}, "{side}")\n',
                lambda s: moved(s) <= 0.3 and travelled(s) >= sides * m * 0.85 and abs(turned(s)) >= 2 * math.pi - 0.5)


@family("relative")
def rock(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, _ = got
    n = rng.choice([2, 3, 4])
    text = rng.choice([f"Rock left and right {TIMES[n]}.", f"Wiggle {TIMES[n]}.", f"Shake your head {TIMES[n]}.",
                       f"Look left and right {TIMES[n]}."])
    return task("rock", "relative", scene, spoken(rng, text), "Wiggle wiggle!",
                f"Turn a little left then right, {n} times.",
                f'for _ in range({n}):\n    robot.turn(30, "left")\n    robot.turn(30, "right")\n',
                lambda s: rocks(s) >= 2 * n - 1 and moved(s) < 0.1)


# ---- objects --------------------------------------------------------------------------------------------------
def _ref(rng: random.Random, probe: SimRobot, t: Thing) -> str:
    """How a program names a known thing: a table handle when it is in the table (nearest ten), else where()."""
    table = sorted(probe.memory, key=lambda i: math.dist(probe.memory[i][0], (probe.x, probe.y)))[:10]
    if t.id in table and rng.random() < 0.4:
        return f"robot.obj({t.id})"
    return f'robot.where("{name_of(probe.scene, t)}")'


@family("object")
def go_to_thing(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    options = known(probe)
    if not options:
        return None
    t = rng.choice(options)
    n = name_of(scene, t)
    text = rng.choice([f"Go to the {n}.", f"Drive over to the {n}.", f"Head to the {n}.", f"Go and stand by the {n}."])
    return task("go_to_thing", "object", scene, spoken(rng, text), f"Going to the {n}.", f"Drive to the {n}.",
                f'robot.go_to({_ref(rng, probe, t)})\nrobot.say("I\'m at the {n}.")\n', lambda s: near(s, t.id))


@family("object")
def side_of_thing(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    options = [t for t in known(probe, solid=True) if t.radius >= 0.15]
    if not options:
        return None
    t, side = rng.choice(options), rng.choice(["behind", "front", "left", "right"])
    n = name_of(scene, t)
    text = {"behind": rng.choice([f"Go behind the {n}.", f"Hide behind the {n}."]),
            "front": rng.choice([f"Stand in front of the {n}.", f"Go to the front of the {n}."]),
            "left": rng.choice([f"Go to the left side of the {n}.", f"Go to the left of the {n}."]),
            "right": rng.choice([f"Go to the right side of the {n}.", f"Go to the right of the {n}."])}[side]
    x0, y0, _ = probe.x, probe.y, None
    ux, uy = t.x - x0, t.y - y0
    norm = math.hypot(ux, uy) or 1.0
    ux, uy = ux / norm, uy / norm
    axis = {"behind": (ux, uy), "front": (-ux, -uy), "left": (-uy, ux), "right": (uy, -ux)}[side]

    def check(s: SimRobot) -> bool:
        ex, ey = end(s)[0] - t.x, end(s)[1] - t.y
        return ex * axis[0] + ey * axis[1] > 0.5 * t.radius and gap(*end(s)[:2], t) <= 1.0

    return task("side_of_thing", "object", scene, spoken(rng, text), "On my way.",
                f"Find a free spot {side} the {n} as I see it now and drive there.",
                f'robot.go_to(robot.spot({_ref(rng, probe, t)}, "{side}"))\n', check)


@family("object")
def circle_thing(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    options = known(probe, solid=True)
    if not options:
        return None
    t, n_laps = rng.choice(options), rng.choice([1, 2, 3])
    n = name_of(scene, t)
    way = rng.choice(["clockwise", "anticlockwise", "counterclockwise", ""])
    text = rng.choice([f"Circle the {n} {TIMES[n_laps]} {way}.", f"Go around the {n} {TIMES[n_laps]} {way}.",
                       f"Do {NUMBER[n_laps]} laps around the {n} {way}."]).replace("  ", " ").replace(" .", ".")
    direction = "right" if way == "clockwise" else "left"
    sign = -1 if way == "clockwise" else 1

    def check(s: SimRobot) -> bool:
        got_laps = laps(s, s.things[t.id])
        return abs((abs(got_laps) if not way else sign * got_laps) - n_laps) <= 0.3

    return task("circle_thing", "object", scene, spoken(rng, text), f"Circling the {n}.",
                f"Orbit the {n} {TIMES[n_laps]} {way or 'anticlockwise'}.",
                f'robot.orbit({_ref(rng, probe, t)}, laps={n_laps}, direction="{direction}")\n', check)


@family("object")
def round_trip(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    options = known(probe)
    if not options:
        return None
    t = rng.choice(options)
    n = name_of(scene, t)
    text = rng.choice([f"Go to the {n} and come back.", f"Visit the {n}, then come back here.",
                       f"Drive to the {n} and return."])

    def check(s: SimRobot) -> bool:
        return first_visit(s, s.things[t.id]) is not None and math.dist(end(s)[:2], start_of(s)[:2]) <= 0.35

    return task("round_trip", "object", scene, spoken(rng, text), "Back in a moment.",
                f"Go to the {n}, then return to where I started.",
                f'robot.go_to({_ref(rng, probe, t)})\nrobot.go_to("start")\n', check)


@family("object")
def visit_two_then_me(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    options = known(probe)
    if len(options) < 2:
        return None
    a, b = rng.sample(options, 2)
    na, nb = name_of(scene, a), name_of(scene, b)
    text = rng.choice([f"Visit the {na}, then the {nb}, then come back to me.",
                       f"Go to the {na} and the {nb}, in that order, then come to me."])
    user = user_of(scene)

    def check(s: SimRobot) -> bool:
        p, q = first_visit(s, s.things[a.id]), first_visit(s, s.things[b.id])
        return p is not None and q is not None and p < q and gap(*end(s)[:2], s.things[user.id]) <= 0.65

    return task("visit_two_then_me", "object", scene, spoken(rng, text), "Off I go.",
                f"Go to the {na}, then the {nb}, then to the user.",
                f'robot.go_to({_ref(rng, probe, a)})\nrobot.go_to({_ref(rng, probe, b)})\n'
                "robot.go_to(robot.user(), 0.8)\nrobot.face(robot.user())\n", check)


@family("object")
def stand_off(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    m = rng.choice([1.0, 1.5])
    options = [t for t in known(probe) if t.radius + 0.5 <= m]
    if not options:
        return None
    t = rng.choice(options)
    n = name_of(scene, t)
    text = rng.choice([f"Stop {rng.choice(METRES[m])} from the {n}, facing it.",
                       f"Get within {rng.choice(METRES[m])} of the {n} and look at it."])

    def check(s: SimRobot) -> bool:
        x, y, yaw = end(s)
        return abs(math.dist((x, y), (t.x, t.y)) - m) <= 0.3 and bearing_to(x, y, yaw, (t.x, t.y)) < 0.35

    return task("stand_off", "object", scene, spoken(rng, text), f"Heading for the {n}.",
                f"Stop {m:g} m from the {n}, then face it.",
                f"thing = {_ref(rng, probe, t)}\nrobot.go_to(thing, {m:g})\nrobot.face(thing)\n", check)


@family("object")
def between(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    pairs = [(a, b) for a in known(probe, solid=True) for b in known(probe, solid=True)
             if a.id < b.id and 1.5 <= math.dist((a.x, a.y), (b.x, b.y)) <= 4.0
             and not probe._occ((a.x + b.x) / 2, (a.y + b.y) / 2, probe.inflated)]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    na, nb = name_of(scene, a), name_of(scene, b)
    mid = ((a.x + b.x) / 2, (a.y + b.y) / 2)
    text = rng.choice([f"Park between the {na} and the {nb}.", f"Go halfway between the {na} and the {nb}.",
                       f"Stand between the {na} and the {nb}."])
    return task("between", "object", scene, spoken(rng, text), "Finding the spot between them.",
                f"Drive to the point halfway between the {na} and the {nb}.",
                f'robot.go_to(geom.between({_ref(rng, probe, a)}, {_ref(rng, probe, b)}))\n',
                lambda s: math.dist(end(s)[:2], mid) <= 0.5)


# ---- search and report ----------------------------------------------------------------------------------------
@family("search")
def is_there(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    present = rng.random() < 0.6
    if present:
        options = [t for t in objects(scene) if name_of(scene, t) and t.label != "door"]
        t = rng.choice(options) if options else None
        if t is None:
            return None
        n = name_of(scene, t)
    else:
        labels = {t.label for t in objects(scene)}
        n = rng.choice([lab for lab in CATALOG if lab not in labels])
    text = rng.choice([f"Is there a {n} anywhere?", f"Can you find a {n}?", f"Have a look for a {n} and tell me if "
                       "there is one.", f"Do we have a {n} somewhere?"])

    def check(s: SimRobot) -> bool:
        return said(s, "yes", "found") and not said(s, "no ", "couldn't") if present else said(s, "no", "couldn't")

    return task("is_there", "search", scene, spoken(rng, text), "Let me look.",
                f"Search everywhere for a {n} and say whether I found one.",
                f'found = robot.search("{n}", "everywhere")\n'
                f'robot.say("Yes, I found a {n}." if found else "No, I couldn\'t find a {n}.")\n', check)


@family("search")
def search_room_stop(rng: random.Random) -> Task | None:
    got = setup(rng, two_rooms=True)
    if not got:
        return None
    scene, probe = got
    room = rng.choice(list(scene.rooms))
    x0, y0, x1, y1 = scene.rooms[room]
    options = [t for t in objects(scene) if name_of(scene, t) and x0 < t.x < x1 and y0 < t.y < y1]
    if not options:
        return None
    t = rng.choice(options)
    n = name_of(scene, t)
    text = rng.choice([f"Search the {room} for a {n} and stop next to it.", f"Find the {n} in the {room} and go "
                       "to it.", f"Look in the {room} for a {n}; if you find it, stop beside it."])
    return task("search_room_stop", "search", scene, spoken(rng, text), f"Searching the {room}.",
                f"Search the {room} for the {n}; if found, stop next to it.",
                f'thing = robot.search("{n}", "{room}")\nif thing:\n    robot.go_to(thing, 0.3)\n'
                f'    robot.say("I found the {n}.")\nelse:\n    robot.say("I couldn\'t find a {n} in the {room}.")\n',
                lambda s: near(s, t.id, 0.7))


@family("search")
def which_room(rng: random.Random) -> Task | None:
    got = setup(rng, two_rooms=True)
    if not got:
        return None
    scene, probe = got
    options = [t for t in objects(scene) if name_of(scene, t)]
    if not options:
        return None
    t = rng.choice(options)
    n = name_of(scene, t)
    room = probe._room_at(t.x, t.y)
    text = rng.choice([f"Find the {n} and tell me which room it is in.", f"Which room is the {n} in?",
                       f"Where is the {n}? Tell me the room."])
    return task("which_room", "search", scene, spoken(rng, text), "Let me find it.",
                f"Search everywhere for the {n}, go to it and say the room.",
                f'thing = robot.search("{n}", "everywhere")\nif thing:\n    robot.go_to(thing)\n'
                f'    robot.say(f"The {n} is in the {{robot.room()}}.")\nelse:\n'
                f'    robot.say("I couldn\'t find the {n}.")\n', lambda s: said(s, room))


@family("search")
def how_many(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    labels = {}
    for t in objects(scene):
        labels.setdefault(t.label, []).append(t)
    options = [lab for lab, ts in labels.items() if all(t.id in probe.memory for t in ts)]
    if not options:
        return None
    lab = rng.choice(options)
    k, many = len(labels[lab]), plural(lab)
    text = rng.choice([f"How many {many} are there?", f"Count the {many}.", f"How many {many} can you see?"])
    return task("how_many", "search", scene, spoken(rng, text), "Counting.", f"Count the {many} I know of and say it.",
                f'n = len(robot.where("{lab}"))\nrobot.say(f"There are {{n}} {many}.")\n',
                lambda s: said(s, str(k), NUMBER.get(k, str(k))))


@family("search")
def what_colour(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    options = [t for t in known(probe) if t.color and name_of(scene, t) == t.label]
    if not options:
        return None
    t = rng.choice(options)
    text = rng.choice([f"What colour is the {t.label}?", f"Tell me the colour of the {t.label}.",
                       f"What color is the {t.label}?"])
    if rng.random() < 0.5:
        program = (f'thing = robot.where("{t.label}")\nrobot.face(thing)\n'
                   f'robot.say(f"The {t.label} is {{robot.describe(thing, \'What colour is it?\')}}.")\n')
    else:
        program = f'thing = robot.where("{t.label}")[0]\nrobot.say(f"The {t.label} is {{thing.color}}.")\n'
    return task("what_colour", "search", scene, spoken(rng, text), "Let me look.",
                f"Look up the {t.label} and say its colour.", program, lambda s: said(s, t.color))


@family("search")
def look_around(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    labels = [t.label for t in visible(probe)]
    if not labels:
        return None
    text = rng.choice(["Look around and tell me what you see.", "What can you see?", "Describe what's around you."])
    return task("look_around", "search", scene, spoken(rng, text), "Let me look around.",
                "Ask the vision model what is in view and say it.",
                'robot.say("I can see " + robot.describe(question="What do you see?") + ".")\n',
                lambda s: said(s, *labels))


# ---- conditionals ---------------------------------------------------------------------------------------------
@family("conditional")
def if_visible(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    seen = [t for t in visible(probe) if name_of(scene, t)]
    present = bool(seen) and rng.random() < 0.6
    if present:
        t = rng.choice(seen)
        n = name_of(scene, t)
    else:
        labels = {t.label for t in visible(probe)}
        n = rng.choice([lab for lab in CATALOG if lab not in labels])
    text = rng.choice([f"If you can see a {n}, go to it; otherwise tell me.",
                       f"Go to the {n} if you can see one, else say you can't."])

    def check(s: SimRobot) -> bool:
        return near(s, t.id) if present else said(s) and moved(s) < 0.3

    return task("if_visible", "conditional", scene, spoken(rng, text), "Let me see.",
                f"If a {n} is in view now, go to it; otherwise say so.",
                f'seen = robot.detect("{n}")\nif seen:\n    robot.go_to(seen[0])\nelse:\n'
                f'    robot.say("I can\'t see a {n} right now.")\n', check)


@family("conditional")
def closer_of_two(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    pairs = [(a, b) for a in known(probe) for b in known(probe) if a.id < b.id and a.label != b.label
             and abs(math.dist((a.x, a.y), (probe.x, probe.y)) - math.dist((b.x, b.y), (probe.x, probe.y))) > 0.5]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    na, nb = name_of(scene, a), name_of(scene, b)
    closer = a if math.dist((a.x, a.y), (probe.x, probe.y)) < math.dist((b.x, b.y), (probe.x, probe.y)) else b
    text = f"If the {na} is closer than the {nb}, go to the {na}; otherwise go to the {nb}."
    return task("closer_of_two", "conditional", scene, spoken(rng, text), "Comparing distances.",
                f"Go to whichever of the {na} and the {nb} is closer.",
                f'a = robot.where("{na}")[0]\nb = robot.where("{nb}")[0]\n'
                "robot.go_to(a if a.distance < b.distance else b)\n", lambda s: near(s, closer.id))


@family("conditional")
def nearest_not_colour(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    groups: dict[str, list[Thing]] = {}
    for t in known(probe, nameable=False):
        groups.setdefault(t.label, []).append(t)
    options = [(lab, ts) for lab, ts in groups.items() if len(ts) >= 2 and len({t.color for t in ts}) >= 2]
    if not options:
        return None
    lab, ts = rng.choice(options)
    colour = rng.choice(sorted({t.color for t in ts}))
    others = [t for t in ts if t.color != colour]
    target = min(others, key=lambda t: math.dist((t.x, t.y), (probe.x, probe.y)))
    text = rng.choice([f"Go to the nearest {lab} that is not {colour}.", f"Go to the closest {lab} that isn't {colour}."])
    return task("nearest_not_colour", "conditional", scene, spoken(rng, text), f"Looking at the {lab}s.",
                f"Of the {lab}s that are not {colour}, go to the nearest.",
                f'options = [o for o in robot.where("{lab}") if o.color != "{colour}"]\n'
                f'if options:\n    robot.go_to(min(options, key=lambda o: o.distance))\nelse:\n'
                f'    robot.say("Every {lab} I know of is {colour}.")\n', lambda s: near(s, target.id))


@family("conditional")
def each_in_turn(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    groups: dict[str, list[Thing]] = {}
    for t in known(probe, nameable=False):
        groups.setdefault(t.label, []).append(t)
    options = [(lab, ts) for lab, ts in groups.items() if 2 <= len(ts) <= 3]
    if not options:
        return None
    lab, ts = rng.choice(options)
    text = rng.choice([f"Go to each {lab} in turn and tell me its colour.", f"Visit every {lab} and say what colour "
                       "it is."])

    def check(s: SimRobot) -> bool:
        return all(first_visit(s, s.things[t.id]) is not None for t in ts) and all(said(s, t.color) for t in ts)

    return task("each_in_turn", "conditional", scene, spoken(rng, text), f"Visiting the {lab}s.",
                f"Go to every {lab} and say its colour.",
                f'for thing in robot.where("{lab}"):\n    robot.go_to(thing)\n'
                f'    robot.say(f"This {lab} is {{thing.color}}.")\n', check)


# ---- following and reactive -----------------------------------------------------------------------------------
def _walk(rng: random.Random, scene: Scene, start: tuple[float, float], n: int) -> tuple | None:
    """A route for the user through free floor, each leg clear of furniture."""
    w, h = scene.width, scene.height
    things = objects(scene)

    def clear(a: tuple[float, float], b: tuple[float, float]) -> bool:
        for k in range(21):
            p = (a[0] + (b[0] - a[0]) * k / 20, a[1] + (b[1] - a[1]) * k / 20)
            if any(math.dist(p, (t.x, t.y)) < t.radius + 0.45 for t in things):
                return False
            if not (-w / 2 + 0.5 < p[0] < w / 2 - 0.5 and -h / 2 + 0.5 < p[1] < h / 2 - 0.5):
                return False
            if scene.walls and abs(p[0]) < 0.4:
                return False
        return True

    route, here = [], start
    for _ in range(n):
        nxt = next((p for p in ((rng.uniform(-w / 2, w / 2), rng.uniform(-h / 2, h / 2)) for _ in range(80))
                    if 1.2 < math.dist(p, here) < 3.0 and clear(here, p)), None)
        if nxt is None:
            return None
        route.append((round(nxt[0], 2), round(nxt[1], 2)))
        here = nxt
    return tuple(route)


@family("following")
def follow_me(rng: random.Random) -> Task | None:
    scene = random_scene(rng, two_rooms=False)
    if not scene:
        return None
    user = user_of(scene)
    route = _walk(rng, scene, (user.x, user.y), 3)
    if not route:
        return None
    user.route, user.speed, scene.stop_at = route, 0.35, 35.0
    length = sum(math.dist(a, b) for a, b in zip(((user.x, user.y), *route), route, strict=False))
    text = rng.choice(["Follow me.", "Follow me until I say stop.", "Come with me.", "Walk with me.",
                       "Follow me around."])

    def check(s: SimRobot) -> bool:
        return (tracked(s, 6.0, 35.0, lambda x, y, _, u: 0.5 <= math.dist((x, y), u) <= 2.2) >= 0.6
                and travelled(s, 0.0, 35.0) >= 0.4 * length)

    return task("follow_me", "following", scene, spoken(rng, text), "Right behind you!",
                "Follow the user until they say stop.", "robot.follow(robot.user())\n", check)


@family("following")
def follow_and_watch(rng: random.Random) -> Task | None:
    """Two things at once: following runs in the background while the program waits for something else."""
    scene = random_scene(rng, two_rooms=False)
    if not scene:
        return None
    user = user_of(scene)
    route = _walk(rng, scene, (user.x, user.y), 3)
    if not route:
        return None
    user.route, user.speed, scene.stop_at = route, 0.35, 40.0
    animal = rng.choice(["cat", "dog"])
    w, h = scene.width, scene.height
    spot = next(((x, y) for x, y in ((rng.uniform(-w / 2 + 0.6, w / 2 - 0.6), rng.uniform(-h / 2 + 0.6, h / 2 - 0.6))
                                      for _ in range(80))
                 if all(math.dist((x, y), (t.x, t.y)) > t.radius + 0.4 for t in objects(scene))), None)
    if spot is None:
        return None
    appear = rng.choice([12.0, 18.0, 24.0])
    scene.things.append(Thing(animal, round(spot[0], 2), round(spot[1], 2), 0.15, "black", solid=False,
                              appear_at=appear))
    text = rng.choice([f"Follow me and tell me if you see a {animal}.", f"Come with me, and let me know when you spot "
                       f"a {animal}.", f"Follow me, and say something if a {animal} shows up."])

    def check(s: SimRobot) -> bool:
        return said(s, animal) and tracked(s, 6.0, appear, lambda x, y, _, u: math.dist((x, y), u) <= 2.2) >= 0.6

    return task("follow_and_watch", "following", scene, spoken(rng, text), "Right behind you, and watching.",
                f"Follow the user in the background while waiting for a {animal}; say so when it appears.",
                f'robot.follow(robot.user(), background=True)\npet = robot.wait_for("{animal}")\n'
                f'robot.say("I can see a {animal}!")\nrobot.follow(robot.user())\n', check)


@family("following")
def come_here(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, _ = got
    u = user_of(scene)
    text = rng.choice(["Come here.", "Come to me.", "Come over here.", "Come here please.", "Over here!"])

    def check(s: SimRobot) -> bool:
        x, y, yaw = end(s)
        return math.dist((x, y), (u.x, u.y)) <= 1.3 and bearing_to(x, y, yaw, (u.x, u.y)) < 0.4

    return task("come_here", "following", scene, spoken(rng, text), "Coming!", "Drive to the user and face them.",
                "me = robot.user()\nrobot.go_to(me, 1.0)\nrobot.face(me)\n", check)


@family("following")
def keep_facing(rng: random.Random) -> Task | None:
    scene = random_scene(rng, two_rooms=False)
    if not scene:
        return None
    user = user_of(scene)
    route = _walk(rng, scene, (user.x, user.y), 3)
    if not route:
        return None
    user.route, user.speed, scene.stop_at = route, 0.3, 25.0
    text = rng.choice(["Keep facing me while I walk around.", "Watch me as I move.", "Keep looking at me."])
    return task("keep_facing", "following", scene, spoken(rng, text), "Watching you.",
                "Keep turning to face the user until they say stop.",
                "while True:\n    robot.face(robot.user())\n    robot.wait(0.3)\n",
                lambda s: tracked(s, 3.0, 25.0, lambda x, y, yaw, u: bearing_to(x, y, yaw, u) < 0.4) >= 0.7)


@family("following")
def wait_then_go(rng: random.Random) -> Task | None:
    got = setup(rng, two_rooms=False)
    if not got:
        return None
    scene, probe = got
    animal = rng.choice(["cat", "dog"])
    spot = next(((x, y) for x, y in ((rng.uniform(-2.5, 2.5) + probe.x, rng.uniform(-2.5, 2.5) + probe.y)
                                      for _ in range(80))
                 if 1.2 < math.dist((x, y), (probe.x, probe.y)) < 3.5 and not probe._occ(x, y, probe.inflated)
                 and probe._line_of_sight((probe.x, probe.y), (x, y))), None)
    if spot is None:
        return None
    appear = rng.choice([5.0, 8.0, 12.0])
    pet = Thing(animal, round(spot[0], 2), round(spot[1], 2), 0.2, rng.choice(["black", "brown", "white"]),
                solid=False, appear_at=appear)
    scene.things.append(pet)
    text = rng.choice([f"Wait until you see a {animal}, then go to it.", f"Stay here until a {animal} shows up, then go "
                       "say hello.", f"When you see the {animal}, go to it."])
    return task("wait_then_go", "following", scene, spoken(rng, text), f"Waiting for the {animal}.",
                f"Wait until a {animal} is in view, then go to it.",
                f'pet = robot.wait_for("{animal}")\nrobot.go_to(pet)\nrobot.say("Hello, {animal}!")\n',
                lambda s: first_visit(s, next(t for t in s.things.values() if t.label == animal), after=appear)
                is not None)


@family("following")
def until_close(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    d = rng.choice([0.5, 0.8, 1.0])
    room = probe.clearance("ahead")
    if room < d + 0.8:
        return None
    text = rng.choice([f"Drive forward until something is less than {rng.choice(METRES[d])} in front of you.",
                       f"Keep going forward until you are {rng.choice(METRES[d])} from whatever is ahead."])

    def check(s: SimRobot) -> bool:
        return abs(s.clearance("ahead") - d) <= 0.15 and moved(s) >= 0.4 and s.trace.bumps == 0

    return task("until_close", "following", scene, spoken(rng, text), "Driving forward.",
                f"Move forward in small steps while the clearance ahead is more than {d:g} m.",
                f'while robot.clearance("ahead") > {d:g}:\n    robot.move(0.1)\n', check)


@family("following")
def patrol(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    options = known(probe, solid=True)
    if len(options) < 2:
        return None
    a, b = rng.sample(options, 2)
    na, nb = name_of(scene, a), name_of(scene, b)
    n = rng.choice([2, 3])
    text = rng.choice([f"Patrol between the {na} and the {nb} {TIMES[n]}.",
                       f"Go back and forth between the {na} and the {nb} {TIMES[n]}."])

    def check(s: SimRobot) -> bool:
        seq = []
        for _, x, y, _ in s.trace.poses:
            for key, thing in (("a", s.things[a.id]), ("b", s.things[b.id])):
                if gap(x, y, thing) <= 0.75 and (not seq or seq[-1] != key):
                    seq.append(key)
        return len(seq) >= 2 * n - (seq[:1] == ["b"])

    return task("patrol", "following", scene, spoken(rng, text), "Starting my patrol.",
                f"Repeat {n} times: go to the {na}, then the {nb}.",
                f'for _ in range({n}):\n    robot.go_to({_ref(rng, probe, a)})\n    robot.go_to({_ref(rng, probe, b)})\n',
                check)


# ---- memory ---------------------------------------------------------------------------------------------------
FACTS = (("charger", "next to the sofa", "sofa"), ("keys", "on the kitchen table", "table"),
         ("remote", "under the cushion", "cushion"), ("glasses", "by the window", "window"),
         ("wallet", "in the hall", "hall"), ("passport", "in the top drawer", "drawer"),
         ("umbrella", "behind the door", "door"), ("headphones", "on the desk", "desk"))


@family("memory")
def remember_fact(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, _ = got
    thing, where, _ = rng.choice(FACTS)
    text = rng.choice([f"Remember that my {thing} is {where}.", f"Remember: the {thing} is {where}.",
                       f"Note that I left my {thing} {where}."])
    return task("remember_fact", "memory", scene, spoken(rng, text), "Got it.", "Store the fact and confirm.",
                f'robot.remember("The {thing} is {where}.")\nrobot.say("Okay, I\'ll remember that your {thing} is '
                f'{where}.")\n', lambda s: any(thing in n.lower() for n in s.trace.notes))


@family("memory")
def recall_fact(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, _ = got
    (thing, where, key), other = rng.sample(FACTS, 2)
    scene.notes = [f"The {thing} is {where}.", f"The {other[0]} is {other[1]}."]
    rng.shuffle(scene.notes)
    text = rng.choice([f"What did I tell you about my {thing}?", f"Where did I say my {thing} was?",
                       f"Do you remember where my {thing} is?"])
    return task("recall_fact", "memory", scene, spoken(rng, text), "Let me remember.",
                f"Recall notes about the {thing} and say them.",
                f'facts = robot.recall("{thing}")\nrobot.say(facts[0] if facts else "You haven\'t told me about your '
                f'{thing}.")\n', lambda s: said(s, key))


@family("memory")
def back_to_seen(rng: random.Random) -> Task | None:
    scene = random_scene(rng, two_rooms=True)
    if not scene:
        return None
    probe = SimRobot(scene)
    here = probe._room_at(probe.x, probe.y)
    movable = [t for t in objects(scene) if t.label in ("ball", "shoe", "backpack", "box", "basket", "pillow",
                                                         "laptop") and name_of(scene, t)
               and probe._room_at(t.x, t.y) != here and t.id not in probe.memory]
    if not movable:
        return None
    t = rng.choice(movable)
    n = name_of(scene, t)
    x0, y0, x1, y1 = scene.rooms[here]
    spot = next(((x, y) for x, y in ((rng.uniform(x0 + 0.5, x1 - 0.5), rng.uniform(y0 + 0.5, y1 - 0.5))
                                      for _ in range(80)) if not probe._occ(x, y, probe.inflated)
                 and math.dist((x, y), (probe.x, probe.y)) > 1.0), None)
    if spot is None:
        return None
    scene.seen = [(n, (round(spot[0], 2), round(spot[1], 2)))]
    text = rng.choice([f"Go back to where you saw the {n}.", f"Take me to the place where you last saw the {n}."])
    return task("back_to_seen", "memory", scene, spoken(rng, text), "Going back there.",
                f"Drive to where the {n} was last seen.",
                f'seen = robot.where("{n}")\nif seen:\n    robot.go_to(seen[0].position)\n'
                f'    robot.say("This is where I last saw the {n}.")\nelse:\n'
                f'    robot.say("I don\'t remember seeing a {n}.")\n', lambda s: math.dist(end(s)[:2], spot) <= 0.6)


# ---- ambiguous or impossible ----------------------------------------------------------------------------------
@family("ambiguous")
def which_one(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    groups: dict[str, list[Thing]] = {}
    for t in known(probe, nameable=False):
        groups.setdefault(t.label, []).append(t)
    options = [(lab, ts) for lab, ts in groups.items() if len(ts) >= 2 and len({t.color for t in ts}) == len(ts)]
    if not options:
        return None
    lab, ts = rng.choice(options)
    pick = rng.choice(ts)
    scene.replies = [(0.0, rng.choice([f"the {pick.color} one", f"{pick.color}", f"I mean the {pick.color} {lab}"]))]
    text = rng.choice([f"Go to the {lab}.", f"Take me to the {lab}.", f"Drive to the {lab}."])
    return task("which_one", "ambiguous", scene, spoken(rng, text), "Let me check which one.",
                f"There is more than one {lab}: ask which, then go to that one.",
                f'items = robot.where("{lab}")\nif len(items) > 1:\n'
                f'    answer = (robot.ask("I can see more than one {lab}. Which one do you mean?") or "").lower()\n'
                "    items = [o for o in items if o.color and o.color in answer] or items\nrobot.go_to(items[0])\n",
                lambda s: bool(s.trace.listened) and near(s, pick.id))


IMPOSSIBLE = (("Fly up to the ceiling.", "I can't fly, I only have wheels."),
              ("Climb the stairs.", "I can't climb stairs, I only have wheels."),
              ("Pick up the {thing}.", "I can't pick things up, I don't have arms."),
              ("Bring me the {thing}.", "I can't carry things, I don't have arms."),
              ("Jump over the {thing}.", "I can't jump, I only have wheels."),
              ("Open the window.", "I can't open windows, I don't have arms."),
              ("Make me a sandwich.", "I can't make food, I don't have arms."),
              ("Bring me a cup of tea.", "I can't carry drinks, I don't have arms."))


@family("ambiguous")
def impossible(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    text, reply = rng.choice(IMPOSSIBLE)
    options = known(probe)
    if "{thing}" in text:
        if not options:
            return None
        text = text.format(thing=name_of(scene, rng.choice(options)))
    return task("impossible", "ambiguous", scene, spoken(rng, text), "Hmm.", "Explain that I can't do that.",
                f'robot.say("{reply}")\n', lambda s: said(s) and moved(s) < 0.3)


@family("ambiguous")
def unknown_thing(rng: random.Random) -> Task | None:
    got = setup(rng)
    if not got:
        return None
    scene, _ = got
    word = rng.choice(ABSENT)
    text = rng.choice([f"Go to the {word}.", f"Take me to the {word}.", f"Drive to the {word}."])
    return task("unknown_thing", "ambiguous", scene, spoken(rng, text), "Let me look for it.",
                f"Look for a {word}; go to it if found, otherwise say so.",
                f'thing = robot.where("{word}") or robot.search("{word}", "here")\nif thing:\n    robot.go_to(thing)\n'
                f'else:\n    robot.say("I couldn\'t find a {word}.")\n',
                lambda s: said(s, "couldn't", "can't", "no ") and s.trace.bumps == 0)


# ---- the conversation so far ----------------------------------------------------------------------------------
def _after(scene: Scene, program: str, instruction: str) -> tuple[Scene, str] | None:
    """Run a first instruction's program in the room: the room with the robot where it ended, and the "Earlier" line
    the planner would be shown for it (the same readback the agent records)."""
    from dataclasses import replace

    from robot.agent.api import Pose
    from robot.agent.gate import check
    from robot.agent.readback import readback

    gate = check(program, SimRobot(scene))
    if not gate.ok:
        return None
    _, x, y, yaw = gate.trace.poses[-1]
    return (replace(scene, start=Pose(x, y, math.degrees(yaw))),
            f'"{instruction}": {readback(gate.trace, gate.ended_by)} (done)')


@family("conversation")
def the_other_one(rng: random.Random) -> Task | None:
    """Two things of a kind; the robot went to one; "no, the other one" means the one it did not go to."""
    got = setup(rng)
    if not got:
        return None
    scene, probe = got
    kinds: dict[str, list[Thing]] = {}
    for t in known(probe, nameable=False):
        kinds.setdefault(t.label, []).append(t)
    pairs = [k for k in kinds.values() if len(k) >= 2]
    if not pairs:
        return None
    first, other = rng.sample(rng.choice(pairs), 2)
    label = first.label
    asked = rng.choice([f"Go to the {label}.", f"Take me to the {label}.", f"Drive to the {label}."])
    went = f'robot.go_to(robot.obj({first.id}), distance=0.3)' + NL + 'robot.say("Here.")' + NL
    after = _after(scene, went, asked)
    if after is None:
        return None
    scene, line = after
    text = rng.choice(["No, the other one.", f"Not that one, the other {label}.", f"Wrong {label}. The other one.",
                       f"I meant the other {label}.", "The other one, please."])
    program = f'robot.go_to(robot.obj({other.id}), distance=0.3)' + NL + f'robot.say("Here is the other {label}.")' + NL
    t = task("the_other_one", "conversation", scene, spoken(rng, text), f"Going to the other {label}.",
             f"Earlier I went to {label} {first.id}, so go to the other one, {other.id}.", program,
             lambda s: near(s, other.id))
    t.earlier = (line,)
    return t


@family("conversation")
def again(rng: random.Random) -> Task | None:
    """A motion just done; "again" means the same program once more, from where the robot is now."""
    got = setup(rng)
    if not got:
        return None
    scene, _ = got
    asked, program, kind = rng.choice([("Turn left 90 degrees.", 'robot.turn(90, "left")' + NL, "turn"),
                                       ("Turn right 45 degrees.", 'robot.turn(45, "right")' + NL, "turn"),
                                       ("Spin around once.", "robot.spin(1)" + NL, "spin"),
                                       ("Drive forward half a metre.", 'robot.move(0.5, "forward")' + NL, "move")])
    after = _after(scene, program, asked)
    if after is None:
        return None
    scene, line = after
    text = rng.choice(["Do that again.", "Again!", "Once more.", "Same again, please.", "Do it one more time."])
    if kind == "turn":
        deg = 90 if "90" in asked else 45
        sign = 1 if "left" in asked else -1

        def check(s: SimRobot) -> bool:
            return abs(turned(s) - sign * math.radians(deg)) < 0.14 and moved(s) < 0.1
    elif kind == "spin":
        def check(s: SimRobot) -> bool:
            return abs(abs(turned(s)) - 2 * math.pi) < 0.3 and moved(s) < 0.1
    else:
        def check(s: SimRobot) -> bool:
            return abs(along(s)[0] - 0.5) < 0.1
    t = task("again", "conversation", scene, spoken(rng, text), "Doing it again.", f"Repeat what I did: {asked}",
             program, check)
    t.earlier = (line,)
    return t
