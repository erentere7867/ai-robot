"""The agent loop with a scripted model: results come back, errors teach, loops are cut, the stop hook checks once."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from robot.agent.agent import Agent
from robot.agent.api import NotFound, Pose, Preempted
from robot.agent.tools import Tools
from robot.eval.suite import living_room, office
from robot.mind.brain import Reply, ToolCall
from robot.perception.camera import data_url
from robot.sim.world import Scene, SimRobot, Thing


class Scripted:
    """Stands in for the model: replies in order, and keeps every conversation it was shown."""

    def __init__(self, *replies: Reply) -> None:
        self.replies, self.seen, self.thinks = list(replies), [], []

    def chat(self, messages, tools, think=0, **_):  # noqa: ANN001, ANN003, ANN201
        self.seen.append(deepcopy(messages))  # the live context later trims older images in place
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
    record = Agent(brain, review=True).do("Turn left 90 degrees.", sim)
    assert record.outcome == "done"
    assert tool_results(brain.seen[1])[0].startswith("Turned 90 degrees left.\nThis request so far: turned 90 degrees left")
    assert "Before you finish" in brain.seen[2][-1]["content"]  # the stop hook asked once
    assert [t for _, t in sim.trace.said] == ["Turned left."]
    assert [line for line in record.lines() if line.startswith("reply:")] == ['reply: "Turned left."']
    first = brain.seen[0][1]["content"]
    assert any(p.get("type") == "image_url" for p in first)  # the model sees the camera, not only a list


def test_by_default_a_completed_action_answers_once_and_does_not_repeat_say():
    sim = SimRobot(living_room())
    brain = Scripted(call("turn", degrees=90, direction="left"), say("Turned left."))
    record = Agent(brain).do("Turn left 90 degrees.", sim)
    assert record.outcome == "done" and len(brain.seen) == 2 and brain.thinks == [128, 128]
    assert [t for _, t in sim.trace.said] == ["Turned left."]

    # If the model calls say() and then ends with plain text, speech is still delivered once; a second say() is blocked.
    sim2 = SimRobot(living_room())
    brain2 = Scripted(call("say", text="Already there."), call("say", text="Really there."), say("Done."))
    Agent(brain2).do("Say hi.", sim2)
    assert [t for _, t in sim2.trace.said] == ["Already there."]
    assert tool_results(brain2.seen[2])[-1].startswith("Not run: you already spoke")

    # Open to-do steps still trigger a single review check even when review=False.
    sim3 = SimRobot(living_room())
    brain3 = Scripted(call("todo", steps=["step 1", "step 2"]), say("Done."), say("Finished both."))
    Agent(brain3).do("Do two things.", sim3)
    assert "Open to-do steps: step 1; step 2" in brain3.seen[2][-1]["content"]


def test_a_camera_answer_is_delivered_once_and_only_that_answer_is_remembered():
    sim = SimRobot(living_room())
    brain = Scripted(say("A brown chair."), say("You're welcome."))
    agent = Agent(brain)
    agent.do("What do you see?", sim)
    agent.do("Thanks!", sim)
    assert len(brain.seen) == 2 and [text for _, text in sim.trace.said] == ["A brown chair.", "You're welcome."]
    history = brain.seen[1][1]["content"][0]["text"]
    assert history.count("A brown chair.") == 1


def test_stopping_during_search_keeps_the_interrupted_call_in_the_trace(monkeypatch):
    sim = SimRobot(living_room())

    def stopped(*args, **kwargs):
        raise Preempted("stopped")

    monkeypatch.setattr(sim, "search", stopped)
    record = Agent(Scripted(call("search", what="bag"))).do("Find the bag.", sim)
    assert record.outcome == "preempted"
    assert record.steps[-1].calls == [("search", {"what": "bag"}, "Interrupted before returning a result.", False)]


def test_reasoning_only_after_news_skips_the_turns_after_calls_that_worked():
    sim = SimRobot(office())
    brain = Scripted(call("turn", degrees=90, direction="left"), call("go_to", target="box"), say("Which box?"),
                     say("Which box?"))
    Agent(brain, think="medium", review=True, think_when="news").do("Turn left, then go to the box.", sim)
    assert brain.thinks == [512, 0, 512, 0]  # the request; after a turn that worked; after an error; the review


def test_with_no_cameras_the_model_gets_the_list_and_pictures_only_when_it_looks():
    sim = SimRobot(living_room())
    brain = Scripted(call("look", direction="front"), say("A sofa."), say("A sofa."))
    Agent(brain, cameras="none").do("What do you see?", sim)
    first = brain.seen[0][1]["content"]
    assert not any(p.get("type") == "image_url" for p in first)
    assert "call look to see" in first[0]["text"] and "Objects" in first[0]["text"]
    assert any(p.get("type") == "image_url" for p in brain.seen[1][-1]["content"])  # after look


@pytest.mark.parametrize("marked", [[], ["rear_right_camera"], ["front_camera", "rear_right_camera"]])
def test_an_ambiguous_target_is_explained_and_nothing_moves(marked):
    sim = SimRobot(office())
    sim.marked_cameras = lambda ids: marked if len(ids) >= 2 else []
    start = (sim.x, sim.y)
    brain = Scripted(call("go_to", target="box"), say("Which box?"), say("Which box?"))
    Agent(brain).do("Go to the box.", sim)
    result = tool_results(brain.seen[1])[0]
    if marked:
        assert [p["text"] for p in result if p.get("text", "").endswith("camera:")] == [
            cam.removesuffix("_camera").replace("_", "-") + " camera:" for cam in marked]
        assert sum(p["type"] == "image_url" for p in result) == len(marked)
        result = result[0]["text"]
    assert result.startswith("Error:") and "could be any of" in result and "Use the id" in result
    assert (sim.x, sim.y) == start


def test_the_same_call_again_and_again_is_refused_with_a_note():
    sim = SimRobot(living_room())
    turn = call("turn", degrees=90, direction="left")
    brain = Scripted(turn, turn, turn, say("ok"), say("ok"))
    Agent(brain).do("Turn left.", sim)
    assert tool_results(brain.seen[3])[-1].startswith("Not run: you made this exact call 2 times")


def test_repeated_look_samples_the_requested_camera_again_while_stationary(monkeypatch):
    sim = SimRobot(living_room())
    reads = []

    def frame(camera, width, height):
        reads.append(camera)
        return np.full((20, 20, 3), len(reads), dtype=np.uint8)

    monkeypatch.setattr(sim, "camera_view", frame)
    brain = Scripted(*(call("look", direction=d) for d in ("rear right", "rear_left", "front", "front", "front")),
                     say("Done."))
    Agent(brain, review=False).do("Check the cameras.", sim)
    assert reads[3:] == ["rear_right_camera", "rear_left_camera", "front_camera", "front_camera", "front_camera"]
    urls = [tool_results(turn)[-1][-1]["image_url"]["url"] for turn in brain.seen[1:]]
    assert len(set(urls)) == 5 and (sim.x, sim.y) == (0.0, 0.0)


def test_search_returns_its_crop_before_a_queued_drive(monkeypatch):
    sim = SimRobot(living_room())
    seen, = sim.detect("chair")
    crop = np.full((18, 15, 3), 67, dtype=np.uint8)
    monkeypatch.setattr(sim, "target_crop", lambda i: crop if i == seen.id else None, raising=False)
    monkeypatch.setattr(sim, "marked_cameras", lambda ids: [], raising=False)  # no badge in the latest frames
    both = Reply("", tool_calls=[ToolCall("find", "search", {"what": "chair"}),
                                 ToolCall("drive", "go_to", {"target": str(seen.id)})])
    brain = Scripted(both, say("I see it."))
    Agent(brain, cameras="none", review=False).do("Find the chair.", sim)
    find, drive = tool_results(brain.seen[1])
    assert "candidate" in find[0]["text"] and f"ID {seen.id}" in find[-2]["text"]
    assert find[-1]["image_url"]["url"] == data_url(crop) and "Not run" in drive
    assert not any(e.name == "go_to" for e in sim.trace.events)


def test_cut_off_model_turn_does_not_dispatch_a_partial_drive():
    sim = SimRobot(living_room())
    cut = call("move", metres=1)
    cut.raw = {"choices": [{"finish_reason": "length"}]}
    record = Agent(Scripted(cut)).do("Drive a metre.", sim)
    assert record.outcome == "failed" and "cut off" in sim.trace.said[-1][1]
    assert not any(e.name == "move" for e in sim.trace.events)


def test_a_side_and_an_edge_gap_compose_without_losing_the_target_id():
    sim = SimRobot(living_room())
    chair, = sim.detect("chair")
    tools = Tools(sim)
    before = sim.pose()
    ambiguous, ok = tools.run("go_to", {"target": "right chair"})
    assert not ok and "viewpoint" in ambiguous and sim.pose() == before
    refused, ok = tools.run("go_to_side", {"target": str(chair.id), "side": "under"})
    assert not ok and "clearance" in refused and sim.pose() == before
    endpoint = sim.spot(chair, "behind", distance=0.75)
    _, ok = tools.run("go_to_side", {"target": str(chair.id), "side": "behind", "distance": 0.75})
    assert ok and np.linalg.norm(np.asarray(endpoint) - (sim.x, sim.y)) < 0.2
    dispatch, = [e for e in sim.trace.events if e.name == "go_to"]
    assert dispatch.args["target"].id == chair.id


def test_a_distance_comparison_never_becomes_a_detector_label_or_starts_a_drive():
    sim = SimRobot(office())
    start, words = sim.pose(), sim.vocabulary.copy()
    tools = Tools(sim)
    for name, args in (("go_to", {"target": "the nearest box"}), ("search", {"what": "furthest bag", "near": "start"}),
                       ("wait", {"seconds": 1, "until": "closest person"})):
        result, ok = tools.run(name, args)
        assert not ok and "just the object name" in result
        assert sim.pose() == start and sim.vocabulary == words


@pytest.mark.parametrize("gone", [True, False])
def test_a_remembered_or_lost_id_does_not_confirm_arrival(monkeypatch, gone):
    sim = SimRobot(living_room())
    target, = sim.detect("chair")
    lookup = sim.obj

    def observed(i):
        if not any(e.name == "face" for e in sim.trace.events):
            return lookup(i)
        if gone:
            raise NotFound("no longer in view")
        return replace(target, last_seen=4.0)

    monkeypatch.setattr(sim, "obj", observed)
    tools = Tools(sim)
    result, ok = tools.run("go_to", {"target": str(target.id)})
    assert not ok and "Target not confirmed" in result and tools.looked == ["front_camera"]


def test_the_next_request_sees_what_the_last_one_did_and_where_it_began():
    sim = SimRobot(living_room())
    brain = Scripted(call("move", metres=1, direction="forward"), say("Moved."),
                     call("go_to", target="where you were before"), say("Back."))
    agent = Agent(brain)
    agent.do("Go forward a metre.", sim)
    agent.do("Come back.", sim)
    assert "Places you can go_to by name: start (where this request began, 1.0 m behind)" \
        in tool_results(brain.seen[1])[0]  # a move's result says where the request began, from the new spot
    second = brain.seen[2]  # the first turn of "Come back." (the first request took two: act, reply)
    assert [m["role"] for m in second] == ["system", "user"]  # one user turn: the last request is told, not replayed
    text = " ".join(p.get("text", "") for p in second[1]["content"])
    assert "move(metres=1, direction='forward') -> Moved 1 m forward." in text  # step by step, results included
    assert "ended 1.0 m from where that request began" in text
    assert "where you were before (where the request before began, 1.0 m behind)" in text
    assert text.endswith('The user says: "Come back."')
    # After arriving back at "where you were before", "start" is hidden so it doesn't lure the robot back out.
    returned = tool_results(brain.seen[3])[0]
    assert "where you were before (where the request before began, here)" in returned
    assert "start (where this request began" not in returned


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


@pytest.mark.parametrize(("name", "args", "progress"), [
    ("turn", {"degrees": "90 degrees", "direction": "left"}, "turned 90 degrees left; 90 degrees of turning in all"),
    ("move", {"metres": "0.3 m", "direction": "forward"}, "moved 0.3 m forward"),
    ("turn", {"degrees": 90, "direction": "Left"}, "turned 90 degrees left; 90 degrees of turning in all"),
], ids=["turn-with-unit", "move-with-unit", "mixed-case-direction"])
def test_motion_progress_uses_validated_arguments(name, args, progress):
    tools = Tools(SimRobot(Scene(6, 6, [], {})))
    result, ok = tools.run(name, args)
    assert ok
    assert f"\nThis request so far: {progress}." in result


def test_go_to_a_name_not_yet_seen_searches_for_it_and_then_goes_there():
    sim = SimRobot(Scene(16.0, 6.0, [Thing("ball", 6.0, 0.0, radius=0.15, color="red")], {}, start=Pose(-6.0, 0, 0)))
    ball, = sim.things.values()
    turn_only, ok = Tools(sim).run("face", {"target": "red ball"})  # facing never drives off to look
    assert not ok and "look('around'" in turn_only and not any(e.name == "search" for e in sim.trace.events)
    result, ok = Tools(sim).run("go_to", {"target": "red ball"})  # 12 m away: out of the cameras' range
    assert ok and result.startswith(f"Searched for it: found ball {ball.id}. Drive finished.")
    names = [e.name for e in sim.trace.events]
    assert names.index("search") < names.index("go_to")
    assert math.dist((sim.x, sim.y), (ball.x, ball.y)) < 1.0
    missing, ok = Tools(sim).run("go_to", {"target": "elephant"})
    assert not ok and "searched here and found no 'elephant'" in missing


def test_look_and_go_toward_take_the_same_direction_words_and_look_with_a_camera_that_sees_that_way():
    from robot.agent.tools import CAMERAS
    from robot.perception.rig import RIG
    from robot.sim.world import BEARINGS

    assert set(CAMERAS) - {"around"} == set(BEARINGS)
    for word, bearing in BEARINGS.items():
        for camera in CAMERAS[word]:
            off = (bearing - math.degrees(RIG[camera].yaw) + 180) % 360 - 180
            assert abs(off) <= math.degrees(RIG[camera].hfov) / 2, (word, camera)


def test_a_number_written_with_an_exponent_or_a_leading_dot_still_moves_the_robot():
    sim = SimRobot(living_room())
    brain = Scripted(call("turn", degrees="9e1", direction="left"), call("move", metres=".5", direction="forward"),
                     say("Done."), say("Turned and moved."))
    Agent(brain).do("Turn left a quarter turn, then go forward half a metre.", sim)
    results = tool_results(brain.seen[2])
    assert results[0].startswith("Turned 90 degrees left.") and "0.5" in results[1]
