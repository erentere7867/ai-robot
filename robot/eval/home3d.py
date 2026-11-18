"""Tasks in real-looking 3D apartments (robot/sim/home3d.py), each with an automatic check against the home's truth.

The checks are worked out from whichever layout a task runs in (the nearest chair, the table closest to the sofa,
whichever of two things is closer), so the same tasks run in any of ReplicaCAD's 91 layouts. Some tasks are two
requests in a row to the same agent (remember, then recall; go, then come back).

    python -m robot.eval.home3d --tag first                          # the tuning layouts
    python -m robot.eval.home3d --tag held --scenes apt_4 v3_sc3_staging_07   # layouts nobody has tuned on
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot.agent.api import wrap
from robot.eval.suite import end, rocks, turned
from robot.sim.home3d import Home3D, _Server

TUNE = ("apt_0", "v3_sc0_staging_00")  # the layouts tasks and harness are tuned on; hold others out
WORDS = {"sofa": ("sofa", "couch"), "bike": ("bike", "bicycle"), "tv": ("tv", "television", "monitor", "screen"),
         "fridge": ("fridge", "refrigerator")}
CLEAR = 400  # pixels (of a 512 x 288 camera) for a thing to be plainly in view, not just detectable
NUMBERS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


@dataclass(frozen=True)
class Task:
    id: str
    says: tuple[str, ...]  # one request, or several in a row to the same agent
    check: Callable[[Home3D, dict], bool]
    needs: tuple[str, ...] = ()  # labels the layout must have, or the task is skipped there


# ---- what the checks read ----------------------------------------------------------------------------------------
def said(sim: Home3D, last_only: bool = False) -> str:
    texts = [t for _, t in sim.trace.said]
    return (texts[-1] if last_only and texts else " ".join(texts)).lower()


def mentions(text: str, label: str) -> bool:
    return any(re.search(rf"\b{re.escape(w)}", text) for w in WORDS.get(label, (label,)))


def things(sim: Home3D, label: str) -> list[Any]:
    return [t for t in sim.things.values() if t.label == label or label == "tv" and t.label in ("tv", "tv stand")]


def gap(xy: tuple[float, float], t: Any) -> float:
    return math.dist(xy, (t.x, t.y)) - t.radius


def ends_near(sim: Home3D, ts: list[Any] | Any, margin: float = 0.75) -> bool:
    x, y, _ = end(sim)
    return any(gap((x, y), t) <= margin for t in (ts if isinstance(ts, list) else [ts]))


def nearest(ts: list[Any], xy: tuple[float, float]) -> Any:
    return min(ts, key=lambda t: math.dist((t.x, t.y), xy))


def visited_in_order(sim: Home3D, targets: list[Any], margin: float = 0.75) -> bool:
    k = 0
    for _, x, y, _ in sim.trace.poses:
        if k < len(targets) and gap((x, y), targets[k]) <= margin:
            k += 1
    return k == len(targets)


def forward(sim: Home3D, start: dict) -> float:
    """Metres moved along the starting heading (+ forward)."""
    x, y, _ = end(sim)
    sx, sy, syaw = start["pose"]
    return (x - sx) * math.cos(syaw) + (y - sy) * math.sin(syaw)


def spun(sim: Home3D) -> float:
    """Radians turned in all, either way: once round to the left and once to the right is still twice round."""
    yaws = [p[3] for p in sim.trace.poses]
    return sum(abs(wrap(b - a)) for a, b in zip(yaws, yaws[1:], strict=False))


def moved(sim: Home3D, start: dict) -> float:
    x, y, _ = end(sim)
    return math.dist((x, y), start["pose"][:2])


def first_named(text: str, a: str, b: str) -> bool:
    """`a` is named, and before `b` if `b` is named at all."""
    def first(lbl: str) -> int:
        hits = [m.start() for w in WORDS.get(lbl, (lbl,)) for m in re.finditer(rf"\b{re.escape(w)}", text)]
        return min(hits, default=-1)
    ia, ib = first(a), first(b)
    return ia >= 0 and (ib < 0 or ia < ib)


def says_number(text: str, n: int) -> bool:
    return bool(re.search(rf"\b({n}|{NUMBERS[n]})\b", text)) if n < len(NUMBERS) else str(n) in text


def says_no(text: str) -> bool:
    return bool(re.search(r"\b(no|not|n't|cannot|unable)\b", text)) and not re.search(r"\byes\b", text)


def says_cannot(text: str) -> bool:
    return bool(re.search(r"\b(can't|cannot|can not|unable|not able|won't be able|isn't possible|not possible)\b",
                          text.replace("’", "'")))


def _closer(sim: Home3D, start: dict, a: str, b: str) -> str:
    da = min(math.dist((t.x, t.y), start["xy"]) for t in things(sim, a))
    db = min(math.dist((t.x, t.y), start["xy"]) for t in things(sim, b))
    return a if da <= db else b


TASKS = [
    # moving relative to itself
    Task("M1", ("Turn left 90 degrees.",),
         lambda s, st: abs(turned(s) - math.pi / 2) < 0.3 and moved(s, st) < 0.2),
    Task("M2", ("Turn around.",), lambda s, st: abs(abs(turned(s)) - math.pi) < 0.35 and moved(s, st) < 0.2),
    Task("M3", ("Drive forward one metre.",), lambda s, st: 0.8 <= forward(s, st) <= 1.2),
    Task("M4", ("Back up half a metre, then turn right 90 degrees.",),
         lambda s, st: -0.65 <= forward(s, st) <= -0.35 and abs(turned(s) + math.pi / 2) < 0.35),
    Task("M5", ("Spin around twice.",), lambda s, st: spun(s) >= 4 * math.pi - 0.5 and moved(s, st) < 0.3),
    Task("M6", ("Rock left and right three times.",), lambda s, st: rocks(s) >= 4 and moved(s, st) < 0.3),
    # going to things
    Task("N1", ("Go to the sofa.",), lambda s, st: ends_near(s, things(s, "sofa")), ("sofa",)),
    Task("N2", ("Go to the fridge.",), lambda s, st: ends_near(s, things(s, "fridge")), ("fridge",)),
    Task("N3", ("Go to the nearest chair.",),
         lambda s, st: ends_near(s, nearest(things(s, "chair"), st["xy"])), ("chair",)),
    Task("N4", ("Go to the table that is closest to the sofa.",),
         lambda s, st: ends_near(s, nearest(things(s, "table"), (things(s, "sofa")[0].x, things(s, "sofa")[0].y))),
         ("table", "sofa")),
    Task("N5", ("Go to the TV.",), lambda s, st: ends_near(s, things(s, "tv") + things(s, "monitor"), 0.9), ("tv",)),
    Task("N6", ("Go to the sofa, then to the fridge, then come back here.",),
         lambda s, st: visited_in_order(s, [things(s, "sofa")[0], things(s, "fridge")[0]]) and moved(s, st) < 0.8,
         ("sofa", "fridge")),
    # seeing and answering
    Task("Q1", ("What do you see in front of you?",),
         lambda s, st: sum(mentions(said(s), lbl) for lbl in st["front"]) >= 2),
    Task("Q2", ("Turn around and tell me what you see.",),
         lambda s, st: abs(turned(s)) >= 2.5 and any(mentions(said(s), lbl) for lbl in st["rear"])),
    Task("Q3", ("How many chairs can you see right now?",),  # from those plainly in view to all the detector found:
         lambda s, st: any(says_number(said(s), n) for n in range(  # a chair behind a table and a bike, legs only,
             sum(st["pixels"].get(k, 0) >= CLEAR for k in st["ids"]["chair"]),  # counts either way
             st["seen"].count("chair") + 1)), ("chair",)),
    Task("Q4", ("Which is closer to you, the sofa or the fridge?",),
         lambda s, st: first_named(said(s, last_only=True), *(lambda c: (c, "fridge" if c == "sofa" else "sofa"))(
             _closer(s, st, "sofa", "fridge"))), ("sofa", "fridge")),
    Task("Q5", ("Is there a bike in the apartment?",), lambda s, st: bool(re.search(r"\byes\b", said(s))), ("bike",)),
    # finding things
    Task("S1", ("Find a plant and go to it.",), lambda s, st: ends_near(s, things(s, "plant")), ("plant",)),
    Task("S2", ("Is there a dog in the apartment?",), lambda s, st: says_no(said(s, last_only=True))),
    Task("S3", ("Go to the elephant.",), lambda s, st: says_no(said(s, last_only=True)) or says_cannot(said(s))),
    # conditions
    Task("C1", ("If you can see a sofa, go to it. Otherwise turn around.",),
         lambda s, st: ends_near(s, things(s, "sofa")) if "sofa" in st["seen"] else abs(abs(turned(s)) - math.pi)
         < 0.35, ("sofa",)),
    Task("C2", ("If the fridge is closer to you than the sofa, go to the fridge. Otherwise go to the sofa.",),
         lambda s, st: ends_near(s, things(s, _closer(s, st, "fridge", "sofa"))), ("sofa", "fridge")),
    # two requests in a row: context and memory
    Task("R1", ("Remember that my keys are on the sofa.", "Where are my keys?"),
         lambda s, st: mentions(said(s, last_only=True), "sofa")),
    Task("R2", ("Go to the fridge.", "Come back."), lambda s, st: moved(s, st) < 0.8, ("fridge",)),
    Task("R3", ("Go to the sofa.", "Now turn around."),
         lambda s, st: ends_near(s, things(s, "sofa"), 1.2) and abs(wrap(end(s)[2] - st["ends"][0][2])) >= 2.5,
         ("sofa",)),
    # what it cannot do
    Task("X1", ("Go upstairs.",), lambda s, st: says_cannot(said(s))),
    Task("X2", ("Fly up to the ceiling.",), lambda s, st: says_cannot(said(s)) and moved(s, st) < 0.3),
]


def start_facts(robot: Home3D) -> dict:
    """What the cameras show at the start (for the checks) and where the robot starts."""
    def labels(cams: tuple[str, ...]) -> list[str]:
        return [robot.things[k].label for cam in cams for k, b in robot._boxes.get(cam, {}).items()
                if b[4] >= 60 and k in robot.things]
    return {"front": set(labels(("front_camera",))), "rear": set(labels(("rear_left_camera", "rear_right_camera"))),
            "seen": [robot.things[k].label for k in sorted(robot._detected)], "xy": (robot.x, robot.y),
            "pixels": dict(robot._seen), "ids": {lbl: [k for k in robot._detected if robot.things[k].label == lbl]
                                                 for lbl in {robot.things[k].label for k in robot._detected}},
            "pose": (robot.x, robot.y, robot.yaw)}


def run(tag: str, only: list[str], scenes: list[str], brain_url: str, cameras: str, think: str) -> list[dict]:
    from robot.agent.agent import Agent
    from robot.mind.brain import Brain

    brain = Brain(brain_url)
    if not brain.healthy():
        raise SystemExit(f"llama-server is not answering on {brain_url}")
    rows = []
    for scene in scenes:
        for other in [s for s in _Server.running if s != scene]:  # one Habitat process at a time
            _Server.running.pop(other).close()
        for task in TASKS:
            if only and task.id not in only:
                continue
            robot = Home3D(scene)
            labels = {t.label for t in robot.things.values()}
            if any(n not in labels and not (n == "tv" and "tv stand" in labels) for n in task.needs):
                continue
            start = start_facts(robot)
            agent = Agent(brain, cameras=cameras, think=think)
            records = []
            for say in task.says:
                records.append(agent.do(say, robot))
                start.setdefault("ends", []).append(end(robot))  # the pose after each request
            try:
                passed = all(r.outcome in ("done", "preempted") for r in records) and bool(task.check(robot, start))
            except Exception as exc:  # noqa: BLE001 - a check that cannot be evaluated is a failure
                passed = False
                records[-1].problems.append(f"check: {exc}")
            rows.append({"scene": scene, "id": task.id, "says": task.says, "passed": passed,
                         "outcomes": [r.outcome for r in records],
                         "steps": [line for r in records for line in [f"## {r.instruction}", *r.lines(thoughts=True)]],
                         "said": [t for _, t in robot.trace.said],
                         "model_s": [s.model_s for r in records for s in r.steps], "simulated_s": round(robot.t, 1)})
            last = rows[-1]["said"][-1] if rows[-1]["said"] else ""
            print(f"[{'ok' if passed else 'XX'}] {scene:<18} {task.id:<3} {' / '.join(task.says)[:58]:<58} "
                  f"| {last[:70]}", flush=True)
    n = sum(r["passed"] for r in rows)
    print(f"\n{tag}: {n}/{len(rows)} passed in {', '.join(scenes)}")
    for scene in scenes:
        mine = [r for r in rows if r["scene"] == scene]
        print(f"  {scene:<20} {sum(r['passed'] for r in mine)}/{len(mine)}")
    turns = [t for r in rows for t in r["model_s"]]
    if turns:
        print(f"  model turn: median {statistics.median(turns):.2f} s")
    out = Path("~/robot-runtime/runs").expanduser() / f"{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
    print(f"  details: {out}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="home3d")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--scenes", nargs="*", default=list(TUNE), help="ReplicaCAD layouts, e.g. apt_0 apt_4")
    parser.add_argument("--brain", default="http://127.0.0.1:8081")
    parser.add_argument("--cameras", choices=("front", "all"), default="front")
    parser.add_argument("--think", choices=("off", "low", "medium", "high"), default="medium",
                        help="how long the model may reason before each turn")
    args = parser.parse_args()
    run(args.tag, args.only, args.scenes, args.brain, args.cameras, args.think)


if __name__ == "__main__":
    main()
