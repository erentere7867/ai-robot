"""The readback: what a program will do, in words, built from the gate's dry run rather than from the model's words.

The gate already runs every program in the 2D world before the robot moves. This turns that run's call log and pose
trace into a few plain sentences: "drive backward 1.0 m; spin left 2 turns". It cannot contradict the code, because
it describes what the code did, so it serves twice: the robot can show or say it before acting, and the planner can
be asked "does this match the instruction?", which is easier than writing the program. The untrained planner's
commonest failure was a valid program with the wrong meaning (turn(-90) for "left"); the readback makes those visible.
"""

from __future__ import annotations

import math
from typing import Any

from robot.agent.api import Object, Pose, side, wrap

QUIET = {"pose", "battery", "user", "room", "obj", "detect", "where", "clearance", "elapsed", "recall", "spot"}


def _turned(poses: list[tuple[float, float, float, float]], t0: float, t1: float) -> float:
    yaws = [p[3] for p in poses if t0 <= p[0] <= t1]
    return math.degrees(sum(wrap(b - a) for a, b in zip(yaws, yaws[1:], strict=False)))


def _rotation(deg: float) -> str:
    return f"{'left' if deg > 0 else 'right'} {abs(deg):.0f}°"


def _thing(target: Any, here: tuple[float, float, float]) -> str:
    """A target in words, as seen from where the robot was when the call started."""
    if isinstance(target, list) and target:
        target = target[0]
    if isinstance(target, Object):
        return f"the {target.color} {target.label}".replace("  ", " ") if target.color else f"the {target.label}"
    if isinstance(target, str):
        return f'"{target}"'
    if isinstance(target, Pose):
        target = (target.x, target.y)
    if isinstance(target, (tuple, list)) and len(target) == 2:
        dx, dy = target[0] - here[0], target[1] - here[1]
        bearing = math.degrees(wrap(math.atan2(dy, dx) - here[2]))
        return f"a spot {math.hypot(dx, dy):.1f} m {side(bearing)}"
    return "something"


def _sentence(name: str, args: dict[str, Any], here: tuple[float, float, float], there: tuple[float, float, float],
              turned: float, seconds: float, result: str) -> str:
    moved = math.dist(here[:2], there[:2])
    if name == "move":
        return f"drive {args.get('direction', 'forward')} {moved:.1f} m"
    if name in ("turn", "face"):
        what = f"turn {_rotation(turned)}" if abs(turned) >= 2 else "stay facing the same way"
        return f"{what} to face {_thing(args.get('target'), here)}" if name == "face" else what
    if name == "spin":
        return f"spin {'left' if turned > 0 else 'right'} {abs(turned) / 360:.1f} turns over {seconds:.0f} s"
    if name == "go_to":
        return f"drive to {_thing(args.get('target'), here)} ({moved:.1f} m away)"
    if name == "drive_path":
        return f"drive along a path of {args.get('points')} points"
    if name == "orbit":
        way = "anticlockwise" if args.get("direction", "left") == "left" else "clockwise"
        return f"circle {_thing(args.get('target'), here)} {abs(turned) / 360:.1f} times {way}"  # laps round it
    if name == "search":
        return f"search {args.get('area', 'here')} for \"{args.get('description')}\" ({result or 'not found'})"
    if name == "follow":
        return f"follow {_thing(args.get('person'), here)} for {seconds:.0f} s"
    if name == "wait":
        return f"wait {seconds:.0f} s"
    if name == "wait_for":
        return f"wait until \"{args.get('description')}\" is in view ({seconds:.0f} s)"
    if name == "save_place":
        return f"remember this spot as \"{args.get('name')}\""
    if name in ("say", "ask"):
        return f"{name} \"{args.get('text') or args.get('question')}\""
    if name in ("check", "describe"):
        return f"ask the vision model \"{args.get('question')}\""
    if name == "listen":
        return "listen for an answer"
    if name == "remember":
        return f"remember \"{args.get('fact')}\""
    return name


def readback(trace: Any, ended_by: str = "") -> str:
    """The dry run in words: one clause per action, repeats merged, then where the robot ends up."""
    poses = trace.poses
    if not poses:
        return "do nothing"
    start, final = poses[0], poses[-1]
    events = [e for e in trace.events if e.name not in QUIET]
    clauses: list[str] = []
    i = 0
    while i < len(events):
        e, j = events[i], i
        while e.name == "drive" and j + 1 < len(events) and events[j + 1].name == "drive":
            j += 1  # a loop of short drive steps reads as one motion
        nxt = events[j + 1] if j + 1 < len(events) else None
        t1 = nxt.t if nxt else final[0]
        there = nxt.pose if nxt else final[1:4]
        turned = _turned(poses, e.t, t1)
        if e.name == "orbit":  # count laps round the target's centre, not the robot's own heading
            target = e.args.get("target")
            target = target[0] if isinstance(target, list) and target else target
            if isinstance(target, (Object, Pose, tuple)):
                cx, cy = target.position if isinstance(target, Object) else (target[0], target[1]) \
                    if isinstance(target, tuple) else (target.x, target.y)
                around = [(p[0], p[1], p[2], math.atan2(p[2] - cy, p[1] - cx)) for p in poses if e.t <= p[0] <= t1]
                turned = _turned(around, e.t, t1)
        if e.name == "drive":
            dx, dy = there[0] - e.pose[0], there[1] - e.pose[1]
            ahead = dx * math.cos(e.pose[2]) + dy * math.sin(e.pose[2])
            text = f"drive {'forward' if ahead >= 0 else 'backward'} {math.hypot(dx, dy):.1f} m"
            if abs(turned) >= 5:
                text += f" while turning {_rotation(turned)}"
            if j > i:
                text += f" in {j - i + 1} short steps"
        else:
            text = _sentence(e.name, e.args, e.pose, there, turned, t1 - e.t, e.result)
        clauses.append(text)
        i = j + 1
    if not clauses:
        clauses.append("do nothing")
    dx, dy = final[1] - start[1], final[2] - start[2]
    net = math.degrees(wrap(final[3] - start[3]))
    if math.hypot(dx, dy) < 0.1:
        end = "end where it started"
    else:
        where = side(math.degrees(wrap(math.atan2(dy, dx) - start[3])))
        end = f"end {math.hypot(dx, dy):.1f} m {'behind' if where == 'behind' else where + ' of'} the start"
    end += f", facing {_rotation(net)} from the start" if abs(net) >= 5 else ", facing the same way"
    tail = " (still going when the 2-minute preview ended)" if ended_by in ("budget", "stop word") else ""
    return "; ".join(clauses) + f"; {end}{tail}."
