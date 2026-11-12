"""When the on-board planner cannot do a task: a stronger planner gets one try, then the robot says plainly what went
wrong and asks to be told another way; it never guesses (robot/agent/agent.py)."""

from robot.agent.agent import Agent
from robot.agent.planner import PlanResult
from robot.eval.suite import living_room
from robot.mind.brain import Reply
from robot.sim.world import SimRobot


class Canned:
    """A planner that always writes the same program."""

    def __init__(self, program: str) -> None:
        self.program = program

    def _result(self) -> PlanResult:
        return PlanResult("On my way.", "go", self.program, Reply(self.program))

    def plan(self, context: str, instruction: str, earlier: tuple = ()) -> PlanResult:
        return self._result()

    def repair(self, problem: str, context: str | None = None) -> PlanResult:
        return self._result()


WRONG = "robot.go_to(robot.where('unicorn')[0])\n"  # nothing of the kind in the room: the dry run fails
RIGHT = "robot.go_to(robot.where('chair')[0], distance=0.5)\n"


def test_a_stronger_planner_gets_one_try_before_the_robot_gives_up():
    robot = SimRobot(living_room())
    record = Agent(Canned(WRONG), escalate=Canned(RIGHT)).do("Go to the chair.", robot)
    assert record.outcome == "done" and record.programs[-1] == RIGHT
    assert not robot.trace.said  # nothing to apologise for


def test_when_nothing_works_the_robot_says_why_and_asks_instead_of_guessing():
    robot = SimRobot(living_room())
    record = Agent(Canned(WRONG)).do("Go to the unicorn.", robot)
    assert record.outcome == "failed"
    (said,) = [text for _, text in robot.trace.said]
    assert said.startswith("Sorry, I couldn't find what you meant") and said.endswith("another way?")
    assert robot.trace.poses[-1][1:3] == robot.trace.poses[0][1:3]  # and it did not move


def test_the_planner_hears_what_was_done_before_so_the_other_one_has_a_meaning():
    heard: list = []

    class Listening(Canned):
        def plan(self, context: str, instruction: str, earlier: tuple = ()) -> PlanResult:
            heard.append(list(earlier))
            return self._result()

    robot = SimRobot(living_room())
    agent = Agent(Listening(RIGHT))
    agent.do("Go to the chair.", robot)
    agent.do("No, the other one.", robot)
    assert heard[0] == []
    (line,) = heard[1]
    chair = robot.where("chair")[0]
    assert line.startswith('"Go to the chair."') and f"chair {chair.id}" in line and line.endswith("(done)")
