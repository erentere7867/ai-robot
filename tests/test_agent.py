"""The agent loop with a scripted model: results come back, errors teach, loops are cut, the stop hook checks once."""

from __future__ import annotations

from robot.agent.agent import Agent
from robot.eval.suite import living_room, office
from robot.mind.brain import Reply, ToolCall
from robot.sim.world import SimRobot


class Scripted:
    """Stands in for the model: replies in order, and keeps every conversation it was shown."""

    def __init__(self, *replies: Reply) -> None:
        self.replies, self.seen, self.thinks = list(replies), [], []

    def chat(self, messages, tools, think=0, **_):  # noqa: ANN001, ANN003, ANN201
        self.seen.append([dict(m) for m in messages])
        self.thinks.append(think)
        return self.replies.pop(0)


def call(name: str, **args: object) -> Reply:
    return Reply("", tool_calls=[ToolCall(f"c{name}", name, dict(args))])


def say(text: str) -> Reply:
    return Reply(text)


def tool_results(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "tool"]


def test_a_result_comes_back_and_a_plain_reply_ends_the_task_after_one_check():
    sim = SimRobot(living_room())
    brain = Scripted(call("turn", degrees=90, direction="left"), say("Done!"), say("Turned left."))
    record = Agent(brain).do("Turn left 90 degrees.", sim)
    assert record.outcome == "done"
    assert tool_results(brain.seen[1])[0].startswith("Turned 90 degrees left.\nThis request so far: turned 90 degrees left")
    assert "Before you finish" in brain.seen[2][-1]["content"]  # the stop hook asked once
    assert [t for _, t in sim.trace.said] == ["Turned left."]
    first = brain.seen[0][1]["content"]
    assert any(p.get("type") == "image_url" for p in first)  # the model sees the camera, not only a list


def test_reasoning_only_after_news_skips_the_turns_after_calls_that_worked():
    sim = SimRobot(office())
    brain = Scripted(call("turn", degrees=90, direction="left"), call("go_to", target="box"), say("Which box?"),
                     say("Which box?"))
    Agent(brain, think="medium", think_when="news").do("Turn left, then go to the box.", sim)
    assert brain.thinks == [512, 0, 512, 0]  # the request; after a turn that worked; after an error; the review


def test_with_no_cameras_the_model_gets_the_list_and_pictures_only_when_it_looks():
    sim = SimRobot(living_room())
    brain = Scripted(call("look", direction="front"), say("A sofa."), say("A sofa."))
    Agent(brain, cameras="none").do("What do you see?", sim)
    first = brain.seen[0][1]["content"]
    assert not any(p.get("type") == "image_url" for p in first)
    assert "call look to see" in first[0]["text"] and "Objects" in first[0]["text"]
    assert any(p.get("type") == "image_url" for p in brain.seen[1][-1]["content"])  # after look


def test_an_ambiguous_target_is_explained_and_nothing_moves():
    sim = SimRobot(office())
    start = (sim.x, sim.y)
    brain = Scripted(call("go_to", target="box"), say("Which box?"), say("Which box?"))
    Agent(brain).do("Go to the box.", sim)
    result = tool_results(brain.seen[1])[0]
    assert result.startswith("Error:") and "could be any of" in result and "Use the id" in result
    assert (sim.x, sim.y) == start


def test_the_same_call_again_and_again_is_refused_with_a_note():
    sim = SimRobot(living_room())
    turn = call("turn", degrees=90, direction="left")
    brain = Scripted(turn, turn, turn, say("ok"), say("ok"))
    Agent(brain).do("Turn left.", sim)
    assert tool_results(brain.seen[3])[-1].startswith("Not run: you made this exact call 2 times")


def test_the_next_request_sees_what_the_last_one_did_and_where_it_began():
    sim = SimRobot(living_room())
    brain = Scripted(call("move", metres=1, direction="forward"), say("Done."), say("Moved."),
                     say("I'm back."), say("Back."))
    agent = Agent(brain)
    agent.do("Go forward a metre.", sim)
    agent.do("Come back.", sim)
    assert "Places you can go_to by name: start (where this request began, 1.0 m behind)" \
        in tool_results(brain.seen[1])[0]  # a move's result says where the request began, from the new spot
    second = brain.seen[3]  # the first turn of "Come back." (the first request took three: act, reply, review)
    assert [m["role"] for m in second] == ["system", "user"]  # one user turn: the last request is told, not replayed
    text = " ".join(p.get("text", "") for p in second[1]["content"])
    assert "move(metres=1, direction='forward') -> Moved 1 m forward." in text  # step by step, results included
    assert "where you were before (where the request before began, 1.0 m behind)" in text
    assert text.endswith('The user says: "Come back."')


def test_stop_never_waits_for_the_model():
    brain = Scripted()
    record = Agent(brain).do("stop", SimRobot(living_room()))
    assert record.outcome == "preempted" and brain.seen == []


def test_each_motion_says_what_the_request_has_done_so_far():
    sim = SimRobot(living_room())
    spin = call("turn", degrees=360, direction="left")
    brain = Scripted(spin, spin, say("Done."), say("Spun twice."))
    Agent(brain).do("Spin around twice.", sim)
    second = tool_results(brain.seen[2])[-1]  # what the model reads after its second full turn
    assert "turned 360 degrees left; turned 360 degrees left; 720 degrees of turning in all" in second
    assert "facing the way it began" in second


def test_a_number_written_with_an_exponent_or_a_leading_dot_still_moves_the_robot():
    sim = SimRobot(living_room())
    brain = Scripted(call("turn", degrees="9e1", direction="left"), call("move", metres=".5", direction="forward"),
                     say("Done."), say("Turned and moved."))
    Agent(brain).do("Turn left a quarter turn, then go forward half a metre.", sim)
    results = tool_results(brain.seen[2])
    assert results[0].startswith("Turned 90 degrees left.") and "0.5" in results[1]
