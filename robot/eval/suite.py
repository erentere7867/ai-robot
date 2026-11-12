"""The 50-instruction suite (architecture section 13): an instruction, a simulator scene and an automatic check each.

Written before any planner work so it cannot be fitted to the model. The two tasks from the brief ("three circles
around the box behind you" and "search for a yellow box, come back and tell me") are held out of this suite, of
every prompt and of every seed example.

    python -m robot.eval.suite            # runs the hand-written programs in robot/eval/programs.py
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from robot.agent.api import wrap
from robot.sim.world import RADIUS, Scene, SimRobot, Thing

SEAT = ("seat", "something to sit on")


# ---- scenes ------------------------------------------------------------------------------------------------------
def living_room(**changes: object) -> Scene:
    """7 x 5 m. The robot starts in the middle facing +x; the user stands front-left of it."""
    scene = Scene(7.0, 5.0, [
        Thing("chair", 2.2, 1.2, 0.3, "black", names=SEAT),
        Thing("sofa", -1.4, 1.9, 0.55, "grey", names=("couch", *SEAT)),
        Thing("table", 1.6, -1.0, 0.45, "brown"),
        Thing("box", -1.8, -0.3, 0.25, "brown"),
        Thing("ball", 1.2, 1.8, 0.12, "red", names=("toy",)),
        Thing("backpack", -0.9, -1.8, 0.2, "blue", names=("bag",)),
        Thing("plant", 3.0, -2.0, 0.25, "green"),
        Thing("lamp", -3.0, 1.9, 0.2, "white"),
        Thing("person", -1.3, 0.9, 0.25, solid=False, user=True),
    ], {"living": (-3.5, -2.5, 3.5, 2.5)})
    for key, value in changes.items():
        setattr(scene, key, value)
    return scene


def two_rooms(door_open: bool = True, **changes: object) -> Scene:
    """10 x 5 m: a living room and a kitchen joined by one door. The robot starts in the living room facing it."""
    scene = Scene(10.0, 5.0, [
        Thing("door", 0.0, 0.0, 0.45, "white", facts={"open": door_open}, solid=not door_open),
        Thing("sofa", -3.8, 1.8, 0.55, "grey", names=("couch", *SEAT)),
        Thing("tv", -4.6, -1.0, 0.25, "black"),
        Thing("lamp", -1.0, 2.0, 0.2, "white"),
        Thing("fridge", 4.5, 2.0, 0.4, "white"),
        Thing("table", 2.5, -0.8, 0.45, "brown"),
        Thing("mug", 3.8, -2.0, 0.08, "green", names=("cup",)),
        Thing("backpack", 4.3, 0.9, 0.2, "blue", names=("bag",)),
        Thing("person", -3.5, -1.2, 0.25, solid=False, user=True),
    ], {"living": (-5.0, -2.5, 0.0, 2.5), "kitchen": (0.0, -2.5, 5.0, 2.5)},
        walls=[((0.0, -2.5), (0.0, -0.45)), ((0.0, 0.45), (0.0, 2.5))], start=_pose(-2.5, 0.0, 0.0))
    for key, value in changes.items():
        setattr(scene, key, value)
    return scene


def office(**changes: object) -> Scene:
    """6 x 6 m with three boxes, two chairs and a desk with two cups on it."""
    scene = Scene(6.0, 6.0, [
        Thing("box", 1.0, -0.6, 0.25, "blue"),
        Thing("box", 1.5, 1.2, 0.25, "yellow"),
        Thing("box", -1.6, 1.6, 0.35, "brown"),
        Thing("chair", -1.0, -1.6, 0.3, "black", names=SEAT),
        Thing("chair", 1.6, -1.8, 0.3, "black", names=SEAT),
        Thing("desk", 2.3, -0.2, 0.45, "white"),
        Thing("cup", 2.2, -0.4, 0.05, "white", solid=False),
        Thing("cup", 2.4, 0.0, 0.05, "red", solid=False),
        Thing("plant", -2.5, -2.5, 0.25, "green"),
        Thing("person", 0.0, 2.3, 0.25, solid=False, user=True),
    ], {"office": (-3.0, -3.0, 3.0, 3.0)})
    for key, value in changes.items():
        setattr(scene, key, value)
    return scene


def _pose(x: float, y: float, yaw: float):  # noqa: ANN202
    from robot.agent.api import Pose

    return Pose(x, y, yaw)


def with_things(scene: Scene, *extra: Thing) -> Scene:
    scene.things = [*scene.things, *extra]
    return scene


def walking_user(route: tuple, speed: float, **changes: object) -> Scene:
    scene = living_room(**changes)
    user = next(t for t in scene.things if t.user)
    user.route, user.speed = route, speed
    return scene


# ---- what a check can ask about a finished run -------------------------------------------------------------------
def thing(sim: SimRobot, label: str, color: str = "") -> Thing:
    return next(t for t in sim.things.values() if t.label == label and (not color or t.color == color))


def end(sim: SimRobot) -> tuple[float, float, float]:
    _, x, y, yaw = sim.trace.poses[-1]
    return x, y, yaw


def gap(x: float, y: float, t: Thing, when: float | None = None) -> float:
    """Edge-to-edge distance between the robot at (x, y) and a thing."""
    px, py = t.at(when) if when is not None and t.at(when) else (t.x, t.y)
    return math.hypot(px - x, py - y) - t.radius - RADIUS


def ends_near(sim: SimRobot, label: str, within: float = 0.65, color: str = "") -> bool:
    return gap(*end(sim)[:2], thing(sim, label, color)) <= within


def first_visit(sim: SimRobot, t: Thing, within: float = 0.75, after: float = -1.0) -> float | None:
    return next((s for s, x, y, _ in sim.trace.poses if s > after and gap(x, y, t, s) <= within), None)


def moved(sim: SimRobot) -> float:
    (_, x0, y0, _), (x1, y1, _) = sim.trace.poses[0], end(sim)
    return math.hypot(x1 - x0, y1 - y0)


def turned(sim: SimRobot) -> float:
    yaws = [p[3] for p in sim.trace.poses]
    return sum(wrap(b - a) for a, b in zip(yaws, yaws[1:], strict=False))


def laps(sim: SimRobot, t: Thing) -> float:
    angles = [math.atan2(y - t.y, x - t.x) for _, x, y, _ in sim.trace.poses]
    return sum(wrap(b - a) for a, b in zip(angles, angles[1:], strict=False)) / (2 * math.pi)


def said(sim: SimRobot, *words: str) -> bool:
    text = " ".join(t for _, t in sim.trace.said).lower()
    return any(w in text for w in words) if words else bool(text)


def room_at_end(sim: SimRobot) -> str:
    x, y, _ = end(sim)
    return sim._room_at(x, y)


def covered(sim: SimRobot, room: str) -> float:
    x0, y0, x1, y1 = sim.scene.rooms[room]
    i0, j0 = sim._cell(x0, y0)
    i1, j1 = sim._cell(x1, y1)
    free = ~sim.inflated[i0:i1, j0:j1]
    return float((sim.coverage[i0:i1, j0:j1] & free).sum() / max(1, free.sum()))


def bearing_to(x: float, y: float, yaw: float, p: tuple[float, float]) -> float:
    return abs(wrap(math.atan2(p[1] - y, p[0] - x) - yaw))


def share(values: list[bool]) -> float:
    return sum(values) / max(1, len(values))


def travelled(sim: SimRobot, t0: float = 0.0, t1: float = math.inf) -> float:
    """Metres driven between t0 and t1."""
    xy = [(x, y) for s, x, y, _ in sim.trace.poses if t0 <= s <= t1]
    return sum(math.dist(a, b) for a, b in zip(xy, xy[1:], strict=False))


def tracked(sim: SimRobot, t0: float, t1: float, test: Callable[[float, float, float, tuple], bool]) -> float:
    """Share of the poses between t0 and t1 for which test(x, y, yaw, user position) holds."""
    user = next(t for t in sim.things.values() if t.user)
    return share([test(x, y, yaw, user.at(s)) for s, x, y, yaw in sim.trace.poses if t0 <= s <= t1 and user.at(s)])


def rocks(sim: SimRobot) -> int:
    """How often the turning direction changes while the robot is turning."""
    yaws = [p[3] for p in sim.trace.poses]
    signs = [np.sign(wrap(b - a)) for a, b in zip(yaws, yaws[1:], strict=False) if abs(wrap(b - a)) > 1e-4]
    return sum(1 for a, b in zip(signs, signs[1:], strict=False) if a != b)


# ---- the suite ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    id: str
    category: str
    say: str
    scene: Callable[[], Scene]
    check: Callable[[SimRobot], bool]


def _cases() -> list[Case]:
    L, T, O = living_room, two_rooms, office  # noqa: E741, N806 - short scene names keep the table readable
    sofa_table_mid = (0.1, 0.45)
    walk = ((0.2, 0.4), (2.4, 0.2), (2.4, -1.9), (0.0, -2.0))
    loop = ((1.0, 1.0), (1.0, -0.6), (-1.2, -0.8), (-1.3, 0.9), (1.0, 1.0), (1.0, -0.6), (-1.2, -0.8), (-1.3, 0.9))
    visitor = with_things(T(), Thing("person", 2.0, 0.0, 0.25, solid=False, route=((0.0, 0.0), (-1.0, 1.0)),
                                     appear_at=12.0))

    def ball_or_chair(s: SimRobot) -> bool:
        return ends_near(s, "ball")

    def visits_all_boxes(s: SimRobot) -> bool:
        return all(first_visit(s, b) is not None for b in s.things.values() if b.label == "box")

    def patrols(s: SimRobot) -> bool:
        door, sofa = thing(s, "door"), thing(s, "sofa")
        stops, last = [], None
        for t, x, y, _ in s.trace.poses:
            here = "door" if gap(x, y, door, t) <= 1.0 else "sofa" if gap(x, y, sofa, t) <= 0.8 else None
            if here and here != last:
                stops.append(here)
                last = here
        return stops.count("door") >= 3 and stops.count("sofa") >= 3

    def greets_visitor(s: SimRobot) -> bool:
        guest = next(t for t in s.things.values() if t.person and not t.user)
        x, y, _ = end(s)
        return any(t >= 12.0 for t, _ in s.trace.said) and gap(x, y, guest, s.t) <= 1.3

    cases = [
        # relative motion
        Case("R1", "relative", "Back up one metre, then spin around twice.", L,
             lambda s: -1.15 <= end(s)[0] <= -0.85 and abs(end(s)[1]) < 0.15 and abs(turned(s)) >= 4 * math.pi - 0.4),
        Case("R2", "relative", "Drive forward one and a half metres.", L, lambda s: 1.35 <= end(s)[0] <= 1.65),
        Case("R3", "relative", "Turn left 90 degrees.", L,
             lambda s: abs(turned(s) - math.pi / 2) < 0.12 and moved(s) < 0.1),
        Case("R4", "relative", "Turn around.", L, lambda s: abs(abs(turned(s)) - math.pi) < 0.15 and moved(s) < 0.1),
        Case("R5", "relative", "Drive in a square with one metre sides and end where you started.", L,
             lambda s: moved(s) < 0.3 and np.ptp([p[1] for p in s.trace.poses]) > 0.8
             and np.ptp([p[2] for p in s.trace.poses]) > 0.8),
        Case("R6", "relative", "Move forward half a metre, then turn right 45 degrees.", L,
             lambda s: 0.4 <= end(s)[0] <= 0.6 and abs(turned(s) + math.pi / 4) < 0.12),
        Case("R7", "relative", "Spin slowly in place for five seconds.", L,
             lambda s: moved(s) < 0.1 and abs(turned(s)) > 0.3 and s.t >= 4.5),
        Case("R8", "relative", "Rock left and right three times.", L, lambda s: moved(s) < 0.15 and rocks(s) >= 4),
        # object-relative motion
        Case("O1", "object", "Go to the chair.", L, lambda s: ends_near(s, "chair")),
        Case("O2", "object", "Drive a figure of eight around the two chairs.", O,
             lambda s: (lambda a, b: abs(a) >= 0.8 and abs(b) >= 0.8 and a * b < 0)(
                 *(laps(s, c) for c in s.things.values() if c.label == "chair"))),
        Case("O3", "object", "Circle the table twice clockwise.", L, lambda s: laps(s, thing(s, "table")) <= -1.8),
        Case("O4", "object", "Park between the sofa and the table.", lambda: living_room(start=_pose(-2.6, -1.6, 0.0)),
             lambda s: math.dist(end(s)[:2], sofa_table_mid) <= 0.6),
        Case("O5", "object", "Go behind the box.", L,
             lambda s: end(s)[0] < thing(s, "box").x - 0.2 and ends_near(s, "box", 0.8)),
        Case("O6", "object", "Go to the ball and come back.", L,
             lambda s: first_visit(s, thing(s, "ball")) is not None and moved(s) < 0.3),
        Case("O7", "object", "Visit the plant, then the lamp, then come back to me.", L,
             lambda s: (lambda p, q: p is not None and q is not None and p < q)(
                 first_visit(s, thing(s, "plant")), first_visit(s, thing(s, "lamp")))
             and gap(*end(s)[:2], next(t for t in s.things.values() if t.user)) <= 0.6),
        Case("O8", "object", "Stop one metre from the backpack, facing it.", L,
             lambda s: 0.9 <= math.dist(end(s)[:2], (thing(s, "backpack").x, thing(s, "backpack").y)) <= 1.35
             and bearing_to(*end(s), (thing(s, "backpack").x, thing(s, "backpack").y)) < 0.35),
        Case("O9", "object", "Drive once around the whole room, close to the walls.", L,
             lambda s: max(p[1] for p in s.trace.poses) >= 2.4 and min(p[1] for p in s.trace.poses) <= -2.4
             and max(p[2] for p in s.trace.poses) >= 1.4 and min(p[2] for p in s.trace.poses) <= -1.4),
        Case("O10", "object", "Go to the left side of the table.", L,
             lambda s: end(s)[1] > thing(s, "table").y + 0.3 and ends_near(s, "table", 0.8)),
        # search and report
        Case("S1", "search", "Is there a backpack anywhere in here?", T, lambda s: said(s, "yes") and not said(s, "no ")),
        Case("S2", "search", "Find the mug and tell me which room it is in.", T, lambda s: said(s, "kitchen")),
        Case("S3", "search", "How many chairs are in this room?", O, lambda s: said(s, "2", "two")),
        Case("S4", "search", "Is there a cat in the kitchen?", T,
             lambda s: said(s, "no") and covered(s, "kitchen") >= 0.6),
        Case("S5", "search", "Look around and tell me what you see.", L,
             lambda s: sum(said(s, w) for w in ("chair", "sofa", "table", "box", "ball", "lamp", "plant")) >= 3),
        Case("S6", "search", "Search the kitchen for a mug and stop next to it.", T, lambda s: ends_near(s, "mug")),
        Case("S7", "search", "What colour is the box behind you?", L, lambda s: said(s, "brown")),
        Case("S8", "search", "Find something to sit on and go to it.", L,
             lambda s: ends_near(s, "chair") or ends_near(s, "sofa")),
        # conditional and counting
        Case("C1", "conditional", "If the door is open, go through it. Otherwise tell me it is closed.", T,
             lambda s: room_at_end(s) == "kitchen"),
        Case("C2", "conditional", "If the door is open, go through it. Otherwise tell me it is closed.",
             lambda: two_rooms(door_open=False), lambda s: room_at_end(s) == "living" and said(s, "closed")),
        Case("C3", "conditional", "If there are more than two boxes, go to the biggest one.", O,
             lambda s: ends_near(s, "box", color="brown")),
        Case("C4", "conditional", "If you can see a person, turn to face them; otherwise spin once.", L,
             lambda s: bearing_to(*end(s), (-1.3, 0.9)) < 0.35 and abs(turned(s)) < 2 * math.pi - 0.5),
        Case("C5", "conditional", "Go to the nearest box that is not blue.", O,
             lambda s: ends_near(s, "box", color="yellow")),
        Case("C6", "conditional", "Count the cups on the desk and tell me how many there are.", O,
             lambda s: said(s, "2", "two")),
        Case("C7", "conditional", "If the ball is closer than the chair, go to the ball; otherwise go to the chair.", L,
             ball_or_chair),
        Case("C8", "conditional", "Go to each box in turn and tell me its colour.", O,
             lambda s: visits_all_boxes(s) and all(said(s, c) for c in ("blue", "yellow", "brown"))),
        # following and reactive
        Case("F1", "following", "Follow me until I say stop.", lambda: walking_user(walk, 0.4, stop_at=45.0),
             lambda s: tracked(s, 8.0, 45.0, lambda x, y, _, u: 0.5 <= math.dist((x, y), u) <= 2.2) >= 0.6
             and travelled(s, 8.0, 45.0) >= 3.0),
        Case("F2", "following", "Come here.", L,
             lambda s: math.dist(end(s)[:2], (-1.3, 0.9)) <= 1.3 and bearing_to(*end(s), (-1.3, 0.9)) < 0.4),
        Case("F3", "following", "Drive forward until something is less than half a metre in front of you.", L,
             lambda s: 2.75 <= end(s)[0] <= 3.15 and s.trace.bumps == 0),
        Case("F4", "following", "Keep facing me while I walk around.", lambda: walking_user(loop, 0.3, stop_at=30.0),
             lambda s: tracked(s, 3.0, 30.0, lambda x, y, yaw, u: bearing_to(x, y, yaw, u) < 0.4) >= 0.7),
        Case("F5", "following", "Wait until someone comes in through the door, then go and greet them.",
             lambda: visitor, greets_visitor),
        Case("F6", "following", "Patrol between the door and the sofa three times.", T, patrols),
        # memory
        Case("M1", "memory", "Where did you last see my keys?",
             lambda: with_things(two_rooms(seen=[("keys", (3.7, 1.6))]), Thing("keys", 3.7, 1.6, 0.05, "silver",
                                                                               solid=False)),
             lambda s: said(s, "fridge", "kitchen")),
        Case("M2", "memory", "Remember that the charger is next to the sofa.", L,
             lambda s: any("charger" in n.lower() for n in s.trace.notes)),
        Case("M3", "memory", "What did I ask you to remember about the charger?",
             lambda: living_room(notes=["The charger is next to the sofa."]), lambda s: said(s, "sofa")),
        Case("M4", "memory", "Go back to where you saw the red ball.",
             lambda: with_things(two_rooms(seen=[("red ball", (-4.0, -2.0))]), Thing("ball", 3.0, 1.5, 0.12, "red")),
             lambda s: math.dist(end(s)[:2], (-4.0, -2.0)) <= 0.6),
        Case("M5", "memory", "Take me to the backpack you saw earlier.",
             lambda: two_rooms(seen=[("backpack", (4.3, 0.9))]), lambda s: ends_near(s, "backpack", 0.75)),
        # ambiguous or impossible
        Case("A1", "ambiguous", "Go to the box.", lambda: office(replies=[(0.0, "the yellow one")]),
             lambda s: bool(s.trace.listened) and ends_near(s, "box", color="yellow")),
        Case("A2", "ambiguous", "Fly up to the ceiling.", L, lambda s: moved(s) < 0.3 and said(s)),
        Case("A3", "ambiguous", "Go into the kitchen.", lambda: two_rooms(door_open=False),
             lambda s: room_at_end(s) == "living" and said(s) and s.trace.bumps == 0),
        Case("A4", "ambiguous", "Bring me a cup of coffee.", L, lambda s: said(s) and s.trace.bumps == 0),
        Case("A5", "ambiguous", "Go to the elephant.", L, lambda s: said(s) and s.trace.bumps == 0),
    ]
    return cases


CASES = _cases()


def run_case(case: Case, program: str) -> tuple[bool, str, SimRobot]:
    """Gate the program on the case's scene, run it, and check the result."""
    from robot.agent.gate import check
    from robot.agent.runner import Runner

    sim = SimRobot(case.scene())
    gate = check(program, sim)
    if not gate.ok:
        return False, f"gate: {gate}", sim
    result = Runner(sim).run(program)
    if result.status == "error":
        return False, f"run: {result.error} (line {result.line})", sim
    ok = bool(case.check(sim))
    return ok, "passed" if ok else "check failed", sim


def run_planner(tag: str, only: list[str], think: int = 0, check_readback: bool = False, noise: float = 0.0) -> None:
    """Phase 5: the model writes every program. Pass means the check holds with at most two repairs."""
    import json
    import statistics
    from collections import defaultdict
    from pathlib import Path

    from robot.agent.agent import Agent
    from robot.agent.planner import LocalPlanner

    agent = Agent(LocalPlanner(think=think), readback=check_readback)
    if not agent.planner.brain.healthy():
        raise SystemExit("llama-server is not answering on http://127.0.0.1:8080")
    rows, by_category = [], defaultdict(list)
    for case in CASES:
        if only and case.id not in only:
            continue
        scene = case.scene()
        scene.noise = noise
        sim = SimRobot(scene)
        agent.earlier.clear()  # each case is a conversation of its own
        record = agent.do(case.say, sim)
        try:
            checked = bool(case.check(sim))
        except Exception as exc:  # a check that cannot be evaluated counts as a failure
            checked = False
            record.problems.append(f"check: {exc}")
        ok = checked and record.outcome in ("done", "preempted") and record.repairs <= 2
        by_category[case.category].append(ok)
        rows.append({"id": case.id, "category": case.category, "say": case.say, "ok": ok, "outcome": record.outcome,
                     "repairs": record.repairs, "plan_s": record.plan_s, "tokens": record.tokens,
                     "programs": record.programs, "problems": record.problems, "checks": record.checks,
                     "said": [t for _, t in sim.trace.said], "simulated_s": round(sim.t, 1)})
        print(f"[{'ok' if ok else 'XX'}] {case.id:<4} {case.say[:56]:<56} {record.outcome:<9} repairs {record.repairs} "
              f"plan {sum(record.plan_s):5.1f} s  {(record.problems or [''])[-1][:70]}", flush=True)
    passed = sum(r["ok"] for r in rows)
    print(f"\n{tag}: {passed}/{len(rows)} = {100 * passed / max(1, len(rows)):.0f}% pass (target 80%)")
    for category, oks in by_category.items():
        print(f"  {category:<12} {sum(oks)}/{len(oks)}")
    first = [r["plan_s"][0] for r in rows if r["plan_s"]]
    tokens = [t for r in rows for t in r["tokens"]]
    if first:
        print(f"  first plan: median {statistics.median(first):.1f} s on this laptop, median {statistics.median(tokens):.0f} "
              "tokens written")
    out = Path("~/robot-runtime/runs/p5").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"  details: {out / f'{tag}.json'}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", action="store_true", help="let the model write the programs (needs llama-server)")
    parser.add_argument("--tag", default="planner")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--think", type=int, default=0, help="let the model reason first, in up to this many characters")
    parser.add_argument("--readback", action="store_true", help="check each dry run, in words, against the instruction")
    parser.add_argument("--noise", type=float, default=0.0, help="detector noise, 0 (perfect) to 1 (poor)")
    args = parser.parse_args()
    if args.planner:
        run_planner(args.tag, args.only, args.think, args.readback, args.noise)
        return
    from robot.eval.programs import PROGRAMS

    passed = 0
    for case in CASES:
        if case.id in PROGRAMS:
            ok, why, sim = run_case(case, PROGRAMS[case.id])
            passed += ok
            print(f"[{'ok' if ok else 'XX'}] {case.id:<4} {case.say[:58]:<58} {why}  ({sim.t:.0f} s simulated)")
    print(f"{passed}/{len(PROGRAMS)} hand-written programs pass; the suite has {len(CASES)} instructions")


if __name__ == "__main__":
    main()
