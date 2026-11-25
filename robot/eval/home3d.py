"""Tasks in real-looking 3D apartments on the robot's real software, each with an automatic check against the truth.

The robot runs as it would at home (robot/sim/stack.py): its own map from exploring, Nav2, YOLOE-26 and depth from its
cameras, and the agent. Checks read the truth only: where the body really went (not where the robot thinks it is),
which objects are really there, and what it said. They are worked out from whichever layout a task runs in (the
nearest chair, the table closest to the sofa, whichever of two things is closer), so the same tasks run in any of
ReplicaCAD's layouts. Some tasks are two requests in a row to the same agent (remember, then recall; go, then come
back). Before each task the robot is put back on its dock and has a few seconds to look around, as it would have
had sitting there.

    python -m robot.eval.home3d --tag first                          # the tuning layouts
    python -m robot.eval.home3d --tag held --scenes apt_4 v3_sc3_staging_07   # layouts nobody has tuned on
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot.agent.api import wrap
from robot.eval.suite import end, rocks, turned
from robot.sim.stack import Home

TUNE = ("apt_0", "v3_sc0_staging_00")  # the layouts tasks and harness are tuned on; hold others out
WORDS = {"sofa": ("sofa", "couch"), "bike": ("bike", "bicycle"), "tv": ("tv", "television", "monitor", "screen")}
SEEN = 94  # pixels (of the judge's 640 x 360 picture) for a thing to be in view: a detector might find it
CLEAR = 625  # pixels for a thing to be plainly in view, not just detectable
WATCH = 6.0  # simulated seconds the robot looks around on its dock before a task
STOP_DWELL = 0.5  # simulated seconds of stillness at the end of a return
NUMBERS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


@dataclass(frozen=True)
class Task:
    id: str
    says: tuple[str, ...]  # one request, or several in a row to the same agent
    criterion: Callable[[Referee, dict], bool | str]  # a string explains a failed condition
    needs: tuple[str, ...] = ()  # labels the layout must have, or the task is skipped there
    observe_after: float = 0.0

    def check(self, referee: Referee, start: dict) -> bool:
        return not self.problem(referee, start)

    def problem(self, referee: Referee, start: dict) -> str:
        result = self.criterion(referee, start)
        return result if isinstance(result, str) else "" if result else "task criterion not met"


class Referee:
    """The run as the truth saw it: the home's objects, where the body really went, and what the robot said."""

    def __init__(self, home: Home, robot: Any = None, since: int = 0) -> None:
        from robot.sim.world import Thing, Trace

        self.things = {o["id"]: Thing(o["label"], o["x"], o["y"], o["radius"], solid=False, id=o["id"])
                       for o in home.truth["objects"]}
        t0 = home.path[since][0] if since < len(home.path) else 0.0
        self.trace = Trace(poses=[(t - t0, x, y, yaw) for t, x, y, yaw in home.path[since:]])
        if robot is not None:
            self.trace.said = list(robot.trace.said)


# ---- what the checks read ----------------------------------------------------------------------------------------
def out_dir() -> Path:
    from robot import paths

    return paths.RUNS


def keep_pictures(robot: Any, folder: Path) -> list[tuple[float, str]]:
    """Save camera views and object evidence as numbered JPEGs; the report shows which track produced a crop."""
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    view, kept = robot.camera_view, []

    def save(label: str, frame: Any) -> Any:
        if frame is not None and getattr(frame, "size", 0):
            name = f"{len(kept) + 1:03d}_{label}.jpg"
            cv2.imwrite(str(folder / name), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            kept.append((round(robot.t, 1), name))
        return frame

    robot.camera_view = lambda camera="front_camera", width=512, height=288: save(camera, view(camera, width, height))
    if crop := getattr(robot, "target_crop", None):
        robot.target_crop = lambda i: save(f"candidate_{i}", crop(i))
    return kept


def write_report(run: Path, task: Task, rows: list[dict], records: list, robot: Any, home: Home, mark: int,
                 kept: list[tuple[float, str]], referee: Referee) -> None:
    """The task's report page and the run's index (robot/eval/report.py), with where it ended against the real
    objects the task names."""
    from robot.eval import report

    x, y, _ = end(referee)
    near = "; ".join(f"{gap((x, y), nearest(things(referee, n), (x, y))):.2f} m from the nearest real {n}'s edge"
                     for n in task.needs if things(referee, n))
    try:
        report.write(run / home.scene / task.id, task, rows[-1], records, robot, home, mark, kept,
                     f"ended at ({x:.1f}, {y:.1f}) in the house" + (f": {near}" if near else ""))
        report.index(run, rows)
    except Exception as exc:  # noqa: BLE001 - a page for people to read must not end the run
        print(f"report for {task.id}: {type(exc).__name__}: {exc}", flush=True)


def said(sim: Referee, last_only: bool = False) -> str:
    texts = [t for _, t in sim.trace.said]
    return (texts[-1] if last_only and texts else " ".join(texts)).lower()


def mentions(text: str, label: str) -> bool:
    return any(re.search(rf"\b{re.escape(w)}", text) for w in WORDS.get(label, (label,)))


def things(sim: Referee, label: str) -> list[Any]:
    return [t for t in sim.things.values() if t.label == label or label == "tv" and t.label in ("tv", "tv stand")]


def gap(xy: tuple[float, float], t: Any) -> float:
    return math.dist(xy, (t.x, t.y)) - t.radius


def ends_near(sim: Referee, ts: list[Any] | Any, margin: float = 0.75) -> bool:
    x, y, _ = end(sim)
    return any(gap((x, y), t) <= margin for t in (ts if isinstance(ts, list) else [ts]))


def nearest(ts: list[Any], xy: tuple[float, float]) -> Any:
    return min(ts, key=lambda t: math.dist((t.x, t.y), xy))


def visited_in_order(sim: Referee, targets: list[Any], margin: float = 0.75) -> bool:
    k = 0
    for _, x, y, _ in sim.trace.poses:
        if k < len(targets) and gap((x, y), targets[k]) <= margin:
            k += 1
    return k == len(targets)


def relative_xy(x: float, y: float, origin: tuple[float, float, float]) -> tuple[float, float]:
    sx, sy, yaw = origin
    dx, dy = x - sx, y - sy
    return dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)


def forward(sim: Referee, start: dict) -> float:
    """Metres moved along the starting heading (+ forward)."""
    return relative_xy(*end(sim)[:2], start["pose"])[0]


def straight_drive(sim: Referee, start: dict, metres: float) -> bool | str:
    if not sim.trace.poses:
        return "no ground-truth drive samples"
    along = forward(sim, start)
    if abs(along - metres) > 0.2:
        return f"travelled {along:.2f} m forward; expected {metres:g} m within 0.20 m"
    origin = start["pose"]
    lateral = max(abs(relative_xy(x, y, origin)[1]) for _, x, y, _ in sim.trace.poses)
    if lateral > 0.2:
        return f"strayed {lateral:.2f} m sideways; allowed 0.20 m"
    yaw_error = max(abs(wrap(yaw - origin[2])) for _, _, _, yaw in sim.trace.poses)
    if yaw_error > 0.3:
        return f"heading changed by {math.degrees(yaw_error):.0f}°; allowed {math.degrees(0.3):.0f}°"
    return True


def spun(sim: Referee) -> float:
    """Radians turned in all, either way: once round to the left and once to the right is still twice round."""
    yaws = [p[3] for p in sim.trace.poses]
    return sum(abs(wrap(b - a)) for a, b in zip(yaws, yaws[1:], strict=False))


def moved(sim: Referee, start: dict) -> float:
    x, y, _ = end(sim)
    return math.dist((x, y), start["pose"][:2])


def out_and_back(sim: Referee, start: dict, targets: list[Any], margin: float) -> bool | str:
    if not start.get("ends"):
        return "no ground-truth pose at the end of the outbound request"
    origin, arrival = start["pose"][:2], start["ends"][0][:2]
    if not any(gap(arrival, t) <= margin and gap(origin, t) - gap(arrival, t) >= 0.10 for t in targets):
        return "did not reach the target before the return request"
    poses = sim.trace.poses
    if not poses:
        return "no ground-truth return samples"
    remaining = moved(sim, start)
    if remaining >= 0.8 or math.dist(origin, arrival) - remaining < 0.10:
        return f"did not return to the start after the outbound leg (ended {remaining:.2f} m away)"
    cutoff = poses[-1][0] - STOP_DWELL
    ex, ey, eyaw = end(sim)
    for t, x, y, yaw in reversed(poses):
        if math.dist((x, y), (ex, ey)) > 0.02 or abs(wrap(yaw - eyaw)) > 0.05:
            break
        if t <= cutoff:
            return True
    return f"did not hold still at the finish for {STOP_DWELL:g} simulated seconds"


def first_named(text: str, a: str, b: str) -> bool:
    """`a` is named, and before `b` if `b` is named at all."""
    def first(lbl: str) -> int:
        hits = [m.start() for w in WORDS.get(lbl, (lbl,)) for m in re.finditer(rf"\b{re.escape(w)}", text)]
        return min(hits, default=-1)
    ia, ib = first(a), first(b)
    return ia >= 0 and (ib < 0 or ia < ib)


def says_number(text: str, n: int) -> bool:
    if n == 0 and re.search(r"\b(no|none|not any|don't see any|can't see any)\b", text.replace("’", "'")):
        return True  # "I can see no chairs"
    return bool(re.search(rf"\b({n}|{NUMBERS[n]})\b", text)) if n < len(NUMBERS) else str(n) in text


def says_no(text: str) -> bool:
    low = text.lower().replace("’", "'")
    return bool(re.search(r"\b(no|not|cannot|unable)\b|n't\b", low)) and not re.search(r"\byes\b", low)


def says_cannot(text: str) -> bool:
    return bool(re.search(r"\b(can't|cannot|can not|unable|not able|won't be able|isn't possible|not possible)\b",
                          text.replace("’", "'")))


def confirms_presence(sim: Referee, label: str, place: str) -> bool | str:
    """The final spoken answer must assert presence, not just contain an affirmative word."""
    answers = [text.lower().replace("’", "'").strip() for _, text in sim.trace.said if text.strip()]
    target = rf"(?:{'|'.join(map(re.escape, WORDS.get(label, (label,))))})s?\b"
    negation = re.compile(r"\b(?:no|not|never|cannot|unable|none)\b|n't\b")
    failure = f"final answer did not unambiguously confirm that a {label} is present"
    if not answers:
        return failure
    affirmative = re.match(r"^(?:yes|yeah|yep)\b[\s,.!:-]*", answers[-1])
    answer = answers[-1][affirmative.end():] if affirmative else answers[-1]
    claim = r"(?:there (?:is|are)|there's|i (?:(?:can )?see|found|spotted))"
    noun = rf"(?:(?:a|an|the|one)\s+)?(?:(?!(?:a|an|the|of|for|with)\b)[\w-]+\s+){{0,2}}{target}"
    location = r"(?:here|(?:parked |standing |sitting )?(?:in|inside|by|near|at|on|under|beside|behind|next to)\s+.+)"
    offsite = rf"\b(?:outside|outdoors|elsewhere|(?:another|a different|the other)\s+{re.escape(place)})\b"
    presence = rf"(?:{claim}\s+{noun}(?:\s+{location})?|{noun}\s+(?:is|are)\s+(?:present|{location}))"

    def clauses(text: str) -> list[str]:
        return [part.rstrip(" .!;") for part in re.split(r"(?<=[.!?;])\s+|,?\s+(?:but|however)\s+", text)]

    def denies(clause: str) -> bool:
        if re.match(rf"^no(?:[,!:.]|$|\s+{noun})", clause):
            return True
        assertion = rf"(?:there\b|{noun}\s+(?:is|are)|i\s+(?:(?:can|do|did)(?:n't| not)?)?\s*(?:see|find|spot))"
        return bool(
            mentions(clause, label) and re.search(rf"(?:^|,\s*){assertion}", clause)
            and (negation.search(clause) or re.search(offsite, clause))
        )

    final = clauses(answer)
    if any(denies(clause) for clause in final):
        return failure
    if any(re.fullmatch(presence, clause) and not re.search(offsite, clause) and "?" not in clause for clause in final):
        return True
    # An elliptical answer has no subject to disambiguate it after a denial.
    if any(denies(clause) for text in answers[:-1] for clause in clauses(text)):
        return failure
    if (affirmative and not answer) or re.fullmatch(r"(?:there (?:is|are) one|there's one)[.! ]*", answer):
        return True
    return failure


def between(sim: Referee, a: str, b: str, margin: float = 1.0) -> bool:
    """Ends in the gap between an `a` and the `b` nearest it: within `margin` of both edges, and along the line from
    one to the other rather than off beyond either."""
    x, y, _ = end(sim)
    for ta in things(sim, a):
        tb = nearest(things(sim, b), (ta.x, ta.y))
        dx, dy = tb.x - ta.x, tb.y - ta.y
        along = ((x - ta.x) * dx + (y - ta.y) * dy) / max(1e-6, dx * dx + dy * dy)
        if gap((x, y), ta) <= margin and gap((x, y), tb) <= margin and 0.0 < along < 1.0:
            return True
    return False


def _closer(sim: Referee, start: dict, a: str, b: str) -> str:
    da = min(math.dist((t.x, t.y), start["xy"]) for t in things(sim, a))
    db = min(math.dist((t.x, t.y), start["xy"]) for t in things(sim, b))
    return a if da <= db else b


TASKS = [
    # moving relative to itself
    Task("M1", ("Turn left 90 degrees.",),
         lambda s, st: abs(turned(s) - math.pi / 2) < 0.3 and moved(s, st) < 0.2),
    Task("M2", ("Turn around.",), lambda s, st: abs(abs(turned(s)) - math.pi) < 0.35 and moved(s, st) < 0.2),
    Task("M3", ("Drive forward one metre.",), lambda s, st: straight_drive(s, st, 1.0)),
    Task("M4", ("Back up half a metre, then turn right 90 degrees.",),
         lambda s, st: -0.65 <= forward(s, st) <= -0.35 and abs(turned(s) + math.pi / 2) < 0.35),
    Task("M5", ("Spin around twice.",), lambda s, st: spun(s) >= 4 * math.pi - 0.5 and moved(s, st) < 0.3),
    Task("M6", ("Rock left and right three times.",), lambda s, st: rocks(s) >= 4 and moved(s, st) < 0.3),
    # going to things
    Task("N1", ("Go to the sofa.",), lambda s, st: ends_near(s, things(s, "sofa")), ("sofa",)),
    Task("N3", ("Go to the nearest chair.",),
         lambda s, st: ends_near(s, nearest(things(s, "chair"), st["xy"])), ("chair",)),
    Task("N4", ("Go to the table that is closest to the sofa.",),
         lambda s, st: ends_near(s, nearest(things(s, "table"), (things(s, "sofa")[0].x, things(s, "sofa")[0].y))),
         ("table", "sofa")),
    Task("N5", ("Go to the TV.",), lambda s, st: ends_near(s, things(s, "tv") + things(s, "monitor"), 0.9), ("tv",)),
    Task("N6", ("Go to the sofa, then to the TV, then come back here.",),
         lambda s, st: visited_in_order(s, [things(s, "sofa")[0], things(s, "tv")[0]]) and moved(s, st) < 0.8,
         ("sofa", "tv")),
    # seeing and answering
    Task("P1", ("Go between the sofa and the table.",), lambda s, st: between(s, "sofa", "table"), ("sofa", "table")),
    Task("Q1", ("What do you see in front of you?",),
         lambda s, st: sum(mentions(said(s), lbl) for lbl in st["front"]) >= 2),
    Task("Q2", ("Turn around and tell me what you see.",),
         lambda s, st: abs(turned(s)) >= 2.5 and any(mentions(said(s), lbl) for lbl in st["rear"])),
    Task("Q3", ("How many chairs can you see right now?",),  # from those plainly in view to all the detector found:
         lambda s, st: any(says_number(said(s), n) for n in range(  # a chair behind a table and a bike, legs only,
             sum(st["pixels"].get(k, 0) >= CLEAR for k in st["ids"]["chair"]),  # counts either way
             st["seen"].count("chair") + 1)), ("chair",)),
    Task("Q4", ("Which is closer to you, the sofa or the TV?",),
         lambda s, st: first_named(said(s, last_only=True), *(lambda c: (c, "tv" if c == "sofa" else "sofa"))(
             _closer(s, st, "sofa", "tv"))), ("sofa", "tv")),
    Task("Q5", ("Is there a bike in the apartment?",), lambda s, st: confirms_presence(s, "bike", "apartment"),
         ("bike",)),
    # finding things
    Task("S1", ("Find a plant and go to it.",), lambda s, st: ends_near(s, things(s, "plant")), ("plant",)),
    Task("S2", ("Is there a dog in the apartment?",), lambda s, st: says_no(said(s, last_only=True))),
    Task("S3", ("Go to the elephant.",), lambda s, st: says_no(said(s, last_only=True)) or says_cannot(said(s))),
    # conditions
    Task("C1", ("If you can see a sofa, go to it. Otherwise turn around.",),
         lambda s, st: ends_near(s, things(s, "sofa")) if "sofa" in st["seen"] else abs(abs(turned(s)) - math.pi)
         < 0.35, ("sofa",)),
    Task("C2", ("If the TV is closer to you than the sofa, go to the TV. Otherwise go to the sofa.",),
         lambda s, st: ends_near(s, things(s, _closer(s, st, "tv", "sofa"))), ("sofa", "tv")),
    # two requests in a row: context and memory
    Task("R1", ("Remember that my keys are on the sofa.", "Where are my keys?"),
         lambda s, st: mentions(said(s, last_only=True), "sofa")),
    Task("R2", ("Go to the TV.", "Come back."), lambda s, st: out_and_back(s, st, things(s, "tv"), 0.9),
         ("tv",), observe_after=2 * STOP_DWELL),
    Task("R3", ("Go to the sofa.", "Now turn around."),
         lambda s, st: ends_near(s, things(s, "sofa"), 1.2) and abs(wrap(end(s)[2] - st["ends"][0][2])) >= 2.5,
         ("sofa",)),
    # what it cannot do
    Task("X1", ("Go upstairs.",), lambda s, st: says_cannot(said(s))),
    Task("X2", ("Fly up to the ceiling.",), lambda s, st: says_cannot(said(s)) and moved(s, st) < 0.3),
]


def start_facts(home: Home) -> dict:
    """What the cameras really show from the dock (for the checks), and where the robot really starts."""
    _, x, y, yaw = home.path[-1]
    view = home.world.call(op="visible", pose=(x, y, yaw), cameras=["front_camera", "rear_left_camera",
                                                                    "rear_right_camera"])
    label = {o["id"]: o["label"] for o in home.truth["objects"]}

    def labels(cams: tuple[str, ...]) -> list[str]:
        return [label[k] for cam in cams for k, n in view[cam].items() if n >= SEEN and k in label]

    pixels: dict[int, int] = {}
    for cam in view.values():
        for k, n in cam.items():
            pixels[k] = pixels.get(k, 0) + n
    seen = sorted(k for k, n in pixels.items() if n >= SEEN and k in label)
    return {"front": set(labels(("front_camera",))), "rear": set(labels(("rear_left_camera", "rear_right_camera"))),
            "seen": [label[k] for k in seen], "xy": (x, y), "pose": (x, y, yaw), "pixels": pixels,
            "ids": {lbl: [k for k in seen if label[k] == lbl] for lbl in {label[k] for k in seen}}}


def run(tag: str, only: list[str], scenes: list[str], brain_url: str, cameras: str, think: str,
        rtf: float, review: bool = False, think_when: str = "always", live_port: int = 0) -> list[dict]:
    from robot.agent.agent import Agent
    from robot.mind.brain import Brain
    from robot.perception.detector import Detector

    brain = Brain(brain_url)
    if not brain.healthy():
        raise SystemExit(f"llama-server is not answering on {brain_url}")
    detector = Detector()
    live, holder = None, None
    if live_port:  # watch it run: the live page shows the prompt, each call, the pictures, the map and the cameras
        from robot.interfaces.demo import Holder, attach_3d, refresh, show_cameras
        from robot.interfaces.live import Live

        live, holder = Live(), Holder(None)
        attach_3d(live, holder)
        live.files = out_dir()  # the run's reports, at /runs/<tag>/index.html
        live.serve(live_port)
        show_cameras(live, holder)
        threading.Thread(target=refresh, args=(live, holder), daemon=True).start()
        print(f"live page: http://127.0.0.1:{live_port}  reports: http://127.0.0.1:{live_port}/runs/{tag}/index.html",
              flush=True)
    rows = []
    out = out_dir() / f"{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")
    for scene in scenes:
        home = Home(scene, rtf=rtf)
        if holder is not None:
            holder.home = home
        try:
            labels = {o["label"] for o in home.truth["objects"]}
            for task in TASKS:
                if only and task.id not in only:
                    continue
                if any(n not in labels and not (n == "tv" and "tv stand" in labels) for n in task.needs):
                    continue
                home.reset()
                robot = home.robot(detector, brain)
                kept = keep_pictures(robot, out.parent / tag / scene / task.id)
                try:
                    home.wait(WATCH)  # sitting on its dock, looking around
                    robot._perceive_objects()
                    start = start_facts(home)
                    mark, bumps = len(home.path), home.bumps
                    if holder is not None:
                        holder.sim = robot
                        live.update(scene=f"{scene}, task {task.id}")
                    agent = Agent(brain, cameras=cameras, think=think, review=review, think_when=think_when,
                                  live=live)
                    records = []
                    for say in task.says:
                        records.append(agent.do(say, robot))
                        start.setdefault("ends", []).append(home.path[-1][1:])  # the true pose after each request
                    if task.observe_after:
                        home.wait(task.observe_after)
                    referee = Referee(home, robot, since=mark)
                    problem = ""
                    passed = False
                    try:
                        if all(r.outcome in ("done", "preempted") for r in records):
                            problem = task.problem(referee, start)
                            passed = not problem
                    except Exception as exc:  # noqa: BLE001 - a check that cannot be evaluated is a failure
                        problem = str(exc) or type(exc).__name__
                    if problem:
                        records[-1].problems.append(f"check: {problem}")
                    rows.append({"scene": scene, "id": task.id, "says": task.says, "passed": passed,
                                 "outcomes": [r.outcome for r in records],
                                 "problems": [p for r in records for p in r.problems],  # why, when it failed
                                 "steps": [line for r in records for line in [f"## {r.instruction}",
                                                                              *r.lines(thoughts=True)]],
                                 "said": [t for _, t in robot.trace.said],
                                 "knew": sorted(f"{t.label} {t.id}" for t in robot.things.values()),
                                 "model_s": [s.model_s for r in records for s in r.steps],
                                 "tokens": [s.tokens for r in records for s in r.steps],  # written, thought included
                                 "simulated_s": round(referee.trace.poses[-1][0], 1) if referee.trace.poses else 0,
                                 "bumps": home.bumps - bumps})
                    with out.open("a", encoding="utf-8") as f:  # as each task ends: a crash keeps the rest
                        f.write(json.dumps(rows[-1], default=str) + "\n")
                    write_report(out.parent / tag, task, rows, records, robot, home, mark, kept, referee)
                finally:
                    if holder is not None:
                        holder.sim = None
                    robot.perception.stop()
                    robot.shutdown()
                if live is not None:
                    live.update(history=(live.state.history + [{"instruction": f"{task.id}: {' / '.join(task.says)}",
                                                                "outcome": "passed" if rows[-1]["passed"]
                                                                else "FAILED"}])[-30:])
                last = rows[-1]["said"][-1] if rows[-1]["said"] else ""
                print(f"[{'ok' if rows[-1]['passed'] else 'XX'}] {scene:<18} {task.id:<3} "
                      f"{' / '.join(task.says)[:58]:<58} | {last[:70]}"
                      + (f" | check: {problem}" if problem else ""), flush=True)
        finally:
            if holder is not None:
                holder.home = None
            home.close()
    n = sum(r["passed"] for r in rows)
    print(f"\n{tag}: {n}/{len(rows)} passed in {', '.join(scenes)}")
    for scene in scenes:
        mine = [r for r in rows if r["scene"] == scene]
        print(f"  {scene:<20} {sum(r['passed'] for r in mine)}/{len(mine)}")
    turns = [t for r in rows for t in r["model_s"]]
    if turns:
        print(f"  model turn: median {statistics.median(turns):.2f} s")
    print(f"  bumps: {sum(r['bumps'] for r in rows)}")
    print(f"  details: {out}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="home3d")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--scenes", nargs="*", default=list(TUNE), help="ReplicaCAD layouts, e.g. apt_0 apt_4")
    parser.add_argument("--brain", default="http://127.0.0.1:8081")
    parser.add_argument("--cameras", choices=("front", "all", "none"), default="all",
                        help="pictures with each request; none: the object list only, pictures when it looks")
    parser.add_argument("--think", choices=("off", "low", "medium", "high"), default="low",
                        help="how long the model may reason before each turn")
    parser.add_argument("--rtf", type=float, default=0.33, help="simulated seconds per wall-clock second")
    parser.add_argument("--review", action="store_true", help="ask the model for a second check before finishing")
    parser.add_argument("--think-when", choices=("always", "news"), default="always",
                        help="news: reason only after the request, the user's words or a failed call")
    parser.add_argument("--live", type=int, default=0, metavar="PORT",
                        help="serve the live page on this port to watch the run (0: off)")
    args = parser.parse_args()
    run(args.tag, args.only, args.scenes, args.brain, args.cameras, args.think, args.rtf, args.review,
        args.think_when, args.live)


if __name__ == "__main__":
    main()
