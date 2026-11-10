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
    def __init__(self, planner: LocalPlanner, *, readback: bool = False) -> None:
        self.planner = planner
        self.readback = readback  # check each program's dry run, in words, against the instruction before it runs

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

    def do(self, instruction: str, robot: Any) -> TaskRecord:
        record = TaskRecord(instruction)
        if hasattr(robot, "begin"):
            robot.begin()  # "start" now names where this task began
        result = self._ask_planner(record, lambda: self.planner.plan(describe(robot.snapshot()), instruction))
        gate_retries, spoke = 0, False
        while result is not None:
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
            run = Runner(robot).run(result.program)
            if run.status in ("done", "preempted"):
                record.outcome = run.status
                return record
            record.problems.append(f"run: line {run.line}: {run.error}")
            if record.repairs >= MAX_REPAIRS:
                break
            problem = explain(f"The program stopped at line {run.line} with {run.error}.")
            result = self._ask_planner(record, lambda p=problem: self.planner.repair(p, describe(robot.snapshot())))
        record.outcome = "failed"
        return record
