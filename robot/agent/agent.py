"""The agent loop: context, plan, gate, run, repair, report (architecture section 7).

A program that fails the gate goes back to the planner with the reason, at most twice. A program that raises while
running goes back with the line, the error and a fresh scene, at most three times in all. A preempted program is never
repaired: "stop" means stop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from robot.agent.context import describe
from robot.agent.gate import check
from robot.agent.planner import FormatError, LocalPlanner, PlanResult
from robot.agent.readback import readback
from robot.agent.runner import Runner

MAX_GATE_RETRIES = 2
MAX_REPAIRS = 3
# What went wrong, in words the user can act on; the robot says it and asks to be told another way, never guessing.
PLAINLY = (("nothing matched", "I couldn't find what you meant"), ("NotFound", "I couldn't find what you meant"),
           ("list index out of range", "I couldn't find what you meant"),
           ("which way", "I wasn't sure which way to go"), ("PathBlocked", "I couldn't find a way there"),
           ("could not finish", "I couldn't find a way there"), ("TimeLimit", "it was taking too long"),
           ("format:", "I couldn't work out how to do it"))


def plainly(problems: list[str]) -> str:
    """The last problem, in the user's words."""
    last = problems[-1] if problems else ""
    return next((words for key, words in PLAINLY if key in last), "I couldn't work out how to do it")
# Python's error text, explained in the API's terms, so a repair knows what to change.
HINTS = (
    ("'list' object has no attribute", "robot.where and robot.detect return lists, best match first: use [0] after "
                                       "checking the list is not empty."),
    ("'Object' object has no attribute", "An Object has id, label, position, distance, bearing, side, radius, color "
                                         "and last_seen."),
    ("'Pose' object has no attribute", "A Pose has x, y and yaw."),
    ("expected a point", "Targets are a point (x, y), a Pose, an Object, or a str naming a place, a room or a thing."),
    ("is not available", "Only robot, geom, math, the four exceptions and basic builtins exist; print does not."),
    ("positional argument", 'Pass unit arguments by name, for example robot.move(metres=1.0, direction="forward").'),
    ("'NoneType' object", "Motion and speech calls (go_to, move, turn, face, orbit, say...) return nothing: keep the "
                          "object you passed in, e.g. chair = robot.where(\"chair\")[0], then robot.go_to(chair)."),
    ("unexpected keyword", "Use the argument names shown in the API."),
)


def explain(problem: str) -> str:
    return " ".join([problem, *(hint for key, hint in HINTS if key in problem)])


@dataclass
class TaskRecord:
    instruction: str
    outcome: str = ""  # done, preempted, failed or refused
    programs: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    plan_s: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    checks: list[tuple[int, bool, str]] = field(default_factory=list)  # program index, the planner's verdict, readback

    @property
    def repairs(self) -> int:
        return max(0, len(self.programs) - 1)


class Agent:
    def __init__(self, planner: LocalPlanner, *, readback: bool = False, live: Any = None,
                 escalate: Any = None) -> None:
        """`escalate`: a stronger planner (the cloud tier, or a larger on-board model) given one try when this one
        has used its repairs; then the robot says plainly what went wrong and asks to be told another way."""
        self.escalate = escalate
        self.planner = planner
        self.readback = readback  # check each program's dry run, in words, against the instruction before it runs
        self.live = live  # robot.interfaces.live.Live, or None

    def _show(self, **changes: Any) -> None:
        if self.live is not None:
            self.live.update(**changes)

    def _ask_planner(self, record: TaskRecord, call: Any) -> PlanResult | None:
        started = time.perf_counter()
        try:
            result = call()
        except FormatError as exc:
            record.problems.append(f"format: {exc}")
            return None
        finally:
            record.plan_s.append(round(time.perf_counter() - started, 2))
        record.programs.append(result.program)
        record.tokens.append(result.reply.write_tokens)
        return result

    def do(self, instruction: str, robot: Any, speak: bool = True) -> TaskRecord:
        record = TaskRecord(instruction)
        if hasattr(robot, "begin"):
            robot.begin()  # "start" now names where this task began
        self._show(status="planning", instruction=instruction, say_first="", plan="", program="", line=0, readback="",
                   problems=[])
        result = self._ask_planner(record, lambda: self.planner.plan(describe(robot.snapshot()), instruction))
        gate_retries, spoke = 0, False
        while result is not None:
            self._show(status="checking", say_first=result.say_first, plan=result.plan, program=result.program, line=0,
                       problems=list(record.problems))
            gate = check(result.program, robot)
            if not gate.ok:
                record.problems.append(f"gate: {gate}")
                if gate_retries >= MAX_GATE_RETRIES or record.repairs >= MAX_REPAIRS:
                    break
                gate_retries += 1
                problem = explain(f"The gate rejected that program. {gate}")
                result = self._ask_planner(record, lambda p=problem: self.planner.repair(p))
                continue
            if self.readback:
                said = readback(gate.trace, gate.ended_by)
                matches, _ = self.planner.judge(instruction, said)
                record.checks.append((len(record.programs) - 1, matches, said))
                if not matches and gate_retries < MAX_GATE_RETRIES and record.repairs < MAX_REPAIRS:
                    gate_retries += 1
                    record.problems.append(f"readback: {said}")
                    problem = (f"Tried in the simulator, your program would: {said} That does not do what was asked: "
                               f"{instruction}")
                    result = self._ask_planner(record, lambda p=problem: self.planner.repair(p))
                    continue
            if not spoke:
                robot.announce(result.say_first)
                spoke = True
            if self.live is not None:
                self._show(status="running", readback=readback(gate.trace, gate.ended_by))
                self.live.show_world(robot)
            run = Runner(robot, on_call=self._on_call(robot)).run(result.program)
            if run.status in ("done", "preempted"):
                record.outcome = run.status
                self._show(status="done" if run.status == "done" else "stopped", line=0)
                if self.live is not None:
                    self.live.show_world(robot)
                return record
            record.problems.append(f"run: line {run.line}: {run.error}")
            if record.repairs >= MAX_REPAIRS:
                break
            problem = explain(f"The program stopped at line {run.line} with {run.error}.")
            result = self._ask_planner(record, lambda p=problem: self.planner.repair(p, describe(robot.snapshot())))
        record.outcome = "failed"
        if self.escalate is not None:
            again = Agent(self.escalate, readback=self.readback, live=self.live).do(instruction, robot, speak=False)
            record.programs += again.programs
            record.problems += again.problems
            if again.outcome != "failed":
                record.outcome = again.outcome
                return record
        self._show(status="failed", problems=list(record.problems))
        if speak:
            try:
                robot.say(f"Sorry, {plainly(record.problems)}. Could you tell me another way?")
            except Exception:  # noqa: BLE001 - a robot that cannot speak still reports on the page
                pass
        return record

    def _on_call(self, robot: Any) -> Any:
        if self.live is None:
            return None

        def seen(name: str, line: int) -> None:
            self.live.update(line=line)
            self.live.show_world(robot)
        return seen
