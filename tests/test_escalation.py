"""When the on-board planner cannot do a task: a stronger planner gets one try, then the robot says plainly what went

wrong and asks to be told another way; it never guesses.
"""

from __future__ import annotations

from typing import Any

from robot.agent.agent import Agent
from robot.agent.api import Intent
from robot.agent.planner import PlanResult
from robot.eval.suite import living_room
from robot.mind.brain import Reply
from robot.sim.world import SimRobot

WRONG = Intent(action="go_to", target="unicorn", subgoal="find unicorn")  # nothing of the kind in the room: fails
RIGHT = Intent(action="go_to", target="chair", params={"distance": 0.5}, subgoal="approach chair")
DONE = Intent(action="done")


class Canned:
    """A planner that always yields the configured sequence of Intent steps."""

    def __init__(self, *steps: Intent) -> None:
        self.steps = steps or (DONE,)
        self.idx = 0

    def _result(self) -> PlanResult:
        intent = self.steps[min(self.idx, len(self.steps) - 1)]
        self.idx += 1
        return PlanResult(intent=intent, reply=Reply(intent.action), say_first=intent.say or "On my way.")

    def plan(self, context: str, instruction: str, earlier: tuple[str, ...] = ()) -> PlanResult:
        return self._result()

    def repair(self, problem: str, context: str | None = None) -> PlanResult:
        return self._result()


def test_a_stronger_planner_gets_one_try_before_the_robot_gives_up():
    robot = SimRobot(living_room())
    record = Agent(Canned(WRONG), escalate=Canned(RIGHT, DONE)).do("Go to the chair.", robot)
    assert record.outcome == "done"
    assert RIGHT in record.intents and record.intents[-1].action == "done"
    assert not robot.trace.said  # nothing to apologise for


def test_when_nothing_works_the_robot_says_why_and_asks_instead_of_guessing():
    robot = SimRobot(living_room())
    record = Agent(Canned(WRONG)).do("Go to the unicorn.", robot)
    assert record.outcome == "failed"
    (said,) = [text for _, text in robot.trace.said]
    assert said.startswith("Sorry, I couldn't find what you meant") and said.endswith("another way?")
    assert robot.trace.poses[-1][1:3] == robot.trace.poses[0][1:3]  # did not move


def test_the_planner_hears_what_was_done_before_so_the_other_one_has_a_meaning():
    heard: list[list[str]] = []

    class Listening(Canned):
        def plan(self, context: str, instruction: str, earlier: tuple[str, ...] = ()) -> PlanResult:
            heard.append(list(earlier))
            return self._result()

    robot = SimRobot(living_room())
    agent = Agent(Listening(RIGHT, DONE))
    agent.do("Go to the chair.", robot)
    agent.do("No, the other one.", robot)
    assert heard[0] == []
    (line,) = heard[-1]
    chair = robot.where("chair")[0]
    assert line.startswith('"Go to the chair."') and f"chair {chair.id}" in line and line.endswith("(done)")


def test_honest_refusal_when_task_is_impossible():
    class Refusing:
        def plan(self, context: str, instruction: str, earlier: tuple[str, ...] = ()) -> PlanResult:
            intent = Intent(action="refuse", say="I have wheels and cannot climb stairs.")
            return PlanResult(intent=intent, reply=Reply("refuse"), say_first=intent.say, plan="refuse")

        def plan_step(self, *args: Any, **kwargs: Any) -> Intent:
            return Intent(action="refuse", say="I have wheels and cannot climb stairs.")

    robot = SimRobot(living_room())
    agent = Agent(Refusing())
    record = agent.do("Climb the stairs to the second floor.", robot)
    assert record.outcome == "refused"
    assert any("cannot climb" in text for _, text in robot.trace.said)
    assert robot.trace.poses[-1][1:3] == robot.trace.poses[0][1:3]
