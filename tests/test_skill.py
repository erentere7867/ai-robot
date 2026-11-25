from dataclasses import FrozenInstanceError

import pytest

from robot.agent.agent import Agent
from robot.agent.api import PathBlocked, Preempted
from robot.agent.skill import Goal, Skill, SkillMachine
from robot.agent.tools import Tools
from robot.eval.suite import living_room
from robot.sim.world import SimRobot
from tests.test_agent import Scripted, call, say


def test_only_one_immutable_goal_can_run_at_a_time():
    machine = SkillMachine()
    assert (machine.state, machine.goal, machine.busy) == (Skill.IDLE, None, False)
    goal = Goal(Skill.SEARCHING, "keys", "hall", 5.0)
    machine.start(goal)
    assert (machine.state, machine.goal, machine.busy) == (Skill.SEARCHING, goal, True)
    with pytest.raises(RuntimeError, match="already running"):
        machine.start(Goal(Skill.WAITING))
    with pytest.raises(FrozenInstanceError):
        goal.target = "wallet"
    assert machine.goal is goal
    machine.finish(True)
    assert (machine.state, machine.goal, machine.busy) == (Skill.IDLE, None, False)


def test_tools_classify_running_skills_without_changing_schemas():
    tools = Tools(SimRobot(living_room()))
    actual = {tool.name: tool.skill for tool in tools._tools.values()}
    assert actual == {
        **dict.fromkeys(("go_to", "go_toward", "go_to_side", "stop_near", "move", "turn", "drive", "orbit", "face"),
                        Skill.NAVIGATING),
        "search": Skill.SEARCHING,
        "follow": Skill.FOLLOWING,
        "keep_facing": Skill.FOLLOWING,
        "wait": Skill.WAITING,
        "ask": Skill.WAITING,
        **dict.fromkeys(("look", "look_back", "remember", "recall", "todo", "todo_done", "todo_blocked", "say")),
    }
    assert all("skill" not in schema["function"] for schema in tools.schemas())


@pytest.mark.parametrize(("answer", "expected"), [("The black one.", 'The user answered: "The black one."'),
                                                  (None, "No answer.")])
def test_ask_is_waiting_and_its_result_is_unchanged(answer, expected, monkeypatch):
    robot = SimRobot(living_room())
    tools = Tools(robot)

    def ask(question):
        assert question == "Which bag?"
        assert tools.machine.state is Skill.WAITING and tools.machine.busy
        return answer

    monkeypatch.setattr(robot, "ask", ask)
    result = tools.execute("ask", {"question": "Which bag?"})
    assert (result.text, result.ok, result.status) == (expected, True, "ok")
    assert tools.machine.state is Skill.IDLE


def test_validation_does_not_start_a_skill_and_failure_does_not_block_the_next_one(monkeypatch):
    robot = SimRobot(living_room())
    tools = Tools(robot)

    def fail(what, area, *, toward):
        assert tools.machine.goal == Goal(Skill.SEARCHING, "bag", "everywhere")
        raise PathBlocked("no way through")

    monkeypatch.setattr(robot, "search", fail)
    assert not tools.execute("search", {}).ok
    assert tools.machine.state is Skill.IDLE
    assert not tools.execute("search", {"what": " bag ", "area": "everywhere"}).ok
    assert (tools.machine.state, tools.machine.goal, tools.machine.busy) == (Skill.FAILED, None, False)

    def wait(seconds):
        assert tools.machine.goal == Goal(Skill.WAITING, timeout=0.5)

    monkeypatch.setattr(robot, "wait", wait)
    result = tools.execute("wait", {"seconds": "0.5 s"})
    assert (result.text, result.ok, result.status) == ("Waited 0.5 s.", True, "ok")
    assert (tools.machine.state, tools.machine.goal, tools.machine.busy) == (Skill.IDLE, None, False)


def test_preempted_skill_propagates_and_returns_to_idle(monkeypatch):
    robot = SimRobot(living_room())
    tools = Tools(robot)

    def stop(degrees, direction):
        assert tools.machine.goal == Goal(Skill.NAVIGATING, "left")
        raise Preempted

    monkeypatch.setattr(robot, "turn", stop)
    with pytest.raises(Preempted):
        tools.execute("turn", {"degrees": 90, "direction": "left"})
    assert (tools.machine.state, tools.machine.goal, tools.machine.busy) == (Skill.IDLE, None, False)


def test_interject_cancels_a_running_skill_and_preserves_the_correction(monkeypatch):
    robot = SimRobot(living_room())
    cancels = []
    monkeypatch.setattr(robot, "cancel", lambda: cancels.append(agent.machine.state))

    def turn(degrees, direction):
        agent.interject("Actually, stay where you are.")
        raise Preempted

    monkeypatch.setattr(robot, "turn", turn)
    agent = Agent(Scripted(call("turn", degrees=90, direction="left"), say("Staying here.")), cameras="none")
    record = agent.do("Turn left.", robot, speak=False)
    assert cancels == [Skill.NAVIGATING]
    assert record.outcome == "done"
    assert record.steps[0].calls == [
        ("turn", {"degrees": 90, "direction": "left"}, "Interrupted: the user spoke (their words follow).", False),
    ]
    assert agent.machine.state is Skill.IDLE


def test_interjection_during_tool_preparation_does_not_cancel_or_start_motion(monkeypatch):
    robot = SimRobot(living_room())
    agent = Agent(Scripted(call("turn", degrees=90, direction="left"), say("Staying here.")), cameras="none")
    pose = robot.pose
    preparing = False
    cancels = []

    def push(**fields):
        nonlocal preparing
        preparing = bool(fields.get("steps") and fields["steps"][-1].endswith(" ..."))

    def interrupting_pose():
        nonlocal preparing
        if preparing:
            preparing = False
            agent.interject("Actually, stay where you are.")
        return pose()

    monkeypatch.setattr(agent, "_push", push)
    monkeypatch.setattr(robot, "pose", interrupting_pose)
    monkeypatch.setattr(robot, "cancel", lambda: cancels.append(agent.machine.state))
    record = agent.do("Turn left.", robot, speak=False)
    assert not cancels and not any(event.name == "turn" for event in robot.trace.events)
    assert record.outcome == "done"
    assert record.steps[0].calls[0][2] == "Interrupted: the user spoke (their words follow)."
