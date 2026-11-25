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


def _text(content: str | list[dict]) -> str:
    return content[0]["text"] if isinstance(content, list) else content


def test_a_result_comes_back_and_a_plain_reply_ends_the_task_after_one_check():
    sim = SimRobot(living_room())
    brain = Scripted(call("turn", degrees=90, direction="left"), say("Done!"), say("Turned left."))
    record = Agent(brain, review=True).do("Turn left 90 degrees.", sim)
    assert record.outcome == "done"
    moved_result = tool_results(brain.seen[1])[0]
    assert _text(moved_result).startswith("Turned 90 degrees left.\nThis request so far: turned 90 degrees left")
    assert isinstance(moved_result, list) and sum(p.get("type") == "image_url" for p in moved_result) == 1
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


def test_the_same_search_can_run_again_after_repositioning(monkeypatch):
    sim = SimRobot(living_room())
    viewpoints = []

    def miss(what, area, toward):
        viewpoints.append(sim.pose())

    monkeypatch.setattr(sim, "search", miss)
    search = call("search", what="umbrella")
    brain = Scripted(search, call("move", metres=0.5, direction="forward"), search, say("No umbrella found."))
    Agent(brain, cameras="none").do("Look for the umbrella.", sim)
    before, after = viewpoints
    assert after.x > before.x
    assert "Searched here: no umbrella found." in _text(tool_results(brain.seen[3])[-1])


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


@pytest.mark.parametrize("history_count", [2, 4])
def test_correction_keeps_look_back_frames_beside_a_live_view(monkeypatch, history_count):
    sim = SimRobot(living_room())
    frames = [(ago, np.full((18, 15, 3), ago * 20, dtype=np.uint8)) for ago in range(history_count, 0, -1)]
    current = np.full((18, 15, 3), 200, dtype=np.uint8)
    brain = Scripted(call("look_back", seconds=5), say("It moved left."))
    agent = Agent(brain)

    def past_views(seconds):
        agent.interject("Actually, where is it now?")
        return frames

    monkeypatch.setattr(sim, "past_views", past_views, raising=False)
    monkeypatch.setattr(sim, "camera_view", lambda camera, *_: None if camera == "front_camera" else current)
    agent.do("What happened in the last five seconds?", sim, speak=False)
    result, = tool_results(brain.seen[1])
    actual = [(part["text"], image["image_url"]["url"]) for part, image in zip(result[1::2], result[2::2], strict=True)]
    assert actual == [(f"front camera {ago} s ago:", data_url(rgb)) for ago, rgb in frames[-2:]]
    correction = brain.seen[1][-1]
    assert correction["role"] == "user" and "Actually, where is it now?" in _text(correction["content"])
    assert "What you sense now:" in _text(correction["content"])
    assert correction["content"][1:] == [
        {"type": "text", "text": "rear-left camera:"},
        {"type": "image_url", "image_url": {"url": data_url(current)}},
    ]


@pytest.mark.parametrize(("see_after_moves", "pictures"), [("front", 1), (True, 3)])
def test_post_move_pictures_are_seen_before_another_batched_action(see_after_moves, pictures):
    sim = SimRobot(living_room())
    batch = Reply("", tool_calls=[ToolCall("first", "turn", {"degrees": 90, "direction": "left"}),
                                 ToolCall("second", "move", {"metres": 1, "direction": "forward"})])
    brain = Scripted(batch, say("I've turned, but haven't moved forward."))
    Agent(brain, see_after_moves=see_after_moves).do("Turn left and move if the way is clear.", sim)
    turned, skipped = tool_results(brain.seen[1])
    assert sum(p["type"] == "image_url" for p in turned) == pictures
    assert "inspect the returned pictures" in skipped
    assert [e.name for e in sim.trace.events if e.name in {"turn", "move"}] == ["turn"]


def test_unavailable_post_turn_picture_is_reported_without_executing_the_rest_of_the_batch(monkeypatch):
    sim = SimRobot(living_room())
    camera_view = sim.camera_view

    def view(*args):
        if not any(event.name == "turn" for event in sim.trace.events):
            return camera_view(*args)
        return None

    monkeypatch.setattr(sim, "camera_view", view)
    batch = Reply("", tool_calls=[ToolCall("first", "turn", {"degrees": 90, "direction": "left"}),
                                 ToolCall("second", "move", {"metres": 1, "direction": "forward"})])
    brain = Scripted(batch, say("I turned but can't see whether the way is clear."))
    Agent(brain, think_when="news").do("Turn left and move if the way is clear.", sim)
    turned, skipped = tool_results(brain.seen[1])
    assert "Turned 90 degrees left." in turned
    assert "No recent camera pictures are available." in turned
    assert "no recent camera pictures were available" in skipped
    assert brain.thinks[1] == 128
    assert [e.name for e in sim.trace.events if e.name in {"turn", "move"}] == ["turn"]


@pytest.mark.parametrize("answer", ["right", None])
def test_clarification_is_read_before_running_an_action(answer):
    sim = SimRobot(living_room())
    if answer:
        sim.replies = [(0, answer)]
    batch = Reply("", tool_calls=[ToolCall("question", "ask", {"question": "Which way?"}),
                                 ToolCall("guess", "turn", {"degrees": 90, "direction": "left"})])
    followup = [call("turn", degrees=90, direction="right")] if answer else []
    brain = Scripted(batch, *followup, say("I turned right." if answer else "I'm staying put."))
    Agent(brain, think_when="news").do("Turn toward the side I name.", sim)
    result, skipped = tool_results(brain.seen[1])
    assert result == ('The user answered: "right"' if answer else "No answer.")
    assert "read the result of asking the user" in skipped and brain.thinks[1] == 128
    turns = [e for e in sim.trace.events if e.name == "turn"]
    assert len(turns) == bool(answer)
    if answer:
        assert turns[0].args["direction"] == "right"


def test_a_malformed_call_blocks_its_batch_and_can_be_repaired():
    sim = SimRobot(living_room())
    batch = Reply("", tool_calls=[ToolCall("bad", "turn", {"_unparsed": '{"degrees":'}),
                                 ToolCall("skipped", "move", {"metres": 1, "direction": "forward"})])
    brain = Scripted(batch, call("turn", degrees=90, direction="left"), say("Turned left."))
    record = Agent(brain).do("Turn left.", sim)
    assert record.outcome == "done"
    bad, skipped = tool_results(brain.seen[1])
    assert "send the arguments as a JSON object" in bad and "earlier call in this turn failed" in skipped
    assert [e.name for e in sim.trace.events if e.name in {"turn", "move"}] == ["turn"]


def test_empty_model_reply_gets_one_repair_turn():
    sim = SimRobot(living_room())
    brain = Scripted(say(""), say("A brown chair."))
    assert Agent(brain).do("What do you see?", sim).outcome == "done"
    assert "no answer or tool call" in brain.seen[1][-1]["content"]
    assert [text for _, text in sim.trace.said] == ["A brown chair."]

    silent = Scripted(say(""), say(""), say("Should not be reached."))
    assert Agent(silent).do("What do you see?", SimRobot(living_room())).outcome == "failed"
    assert len(silent.seen) == 2


def test_cut_off_model_turn_does_not_dispatch_a_partial_drive():
    sim = SimRobot(living_room())
    cut = call("move", metres=1)
    cut.raw = {"choices": [{"finish_reason": "length"}]}
    record = Agent(Scripted(cut)).do("Drive a metre.", sim)
    assert record.outcome == "failed" and "cut off" in sim.trace.said[-1][1]
    assert not any(e.name == "move" for e in sim.trace.events)


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_correction_supersedes_an_answer_or_cut_off_reply(finish_reason):
    sim = SimRobot(living_room())
    stale = call("turn", degrees=90, direction="left") if finish_reason == "length" else say("Turning left.")
    stale.raw = {"choices": [{"finish_reason": finish_reason}]}

    class Corrected(Scripted):
        def chat(self, messages, tools, think=0, **_):  # noqa: ANN001, ANN003, ANN201
            if not self.seen:
                agent.interject("Actually, stay where you are.")
            return super().chat(messages, tools, think=think)

    brain = Corrected(stale, say("I'll stay here."))
    agent = Agent(brain)
    assert agent.do("Turn left.", sim).outcome == "done"
    correction = brain.seen[1][-1]["content"]
    assert "Actually, stay where you are." in _text(correction)
    assert sum(p["type"] == "image_url" for p in correction) == 3
    assert not any(e.name == "turn" for e in sim.trace.events)
    assert [text for _, text in sim.trace.said] == ["I'll stay here."]


def test_a_side_and_an_edge_gap_compose_without_losing_the_target_id():
    sim = SimRobot(living_room())
    chair, = sim.detect("chair")
    tools = Tools(sim)
    before = sim.pose()
    ambiguous = tools.execute("go_to", {"target": "right chair"})
    assert not ambiguous.ok and "viewpoint" in ambiguous.text and sim.pose() == before
    refused = tools.execute("go_to_side", {"target": str(chair.id), "side": "under"})
    assert not refused.ok and "clearance" in refused.text and sim.pose() == before
    endpoint = sim.spot(chair, "behind", distance=0.75)
    result = tools.execute("go_to_side", {"target": str(chair.id), "side": "behind", "distance": 0.75})
    assert result.ok and np.linalg.norm(np.asarray(endpoint) - (sim.x, sim.y)) < 0.2
    dispatch, = [e for e in sim.trace.events if e.name == "go_to"]
    assert dispatch.args["target"].id == chair.id


def test_a_distance_comparison_never_becomes_a_detector_label_or_starts_a_drive():
    sim = SimRobot(office())
    start, words = sim.pose(), sim.vocabulary.copy()
    tools = Tools(sim)
    for name, args in (("go_to", {"target": "the nearest box"}), ("search", {"what": "furthest bag", "near": "start"}),
                       ("wait", {"seconds": 1, "until": "closest person"})):
        result = tools.execute(name, args)
        assert not result.ok and "just the object name" in result.text
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
    result = tools.execute("go_to", {"target": str(target.id)})
    assert not result.ok and "Target not confirmed" in result.text and tools.looked == ["front_camera"]


def test_the_next_request_sees_what_the_last_one_did_and_where_it_began():
    sim = SimRobot(living_room())
    brain = Scripted(call("move", metres=1, direction="forward"), say("Moved."),
                     call("go_to", target="where you were before"), say("Back."))
    agent = Agent(brain)
    agent.do("Go forward a metre.", sim)
    agent.do("Come back.", sim)
    assert "Places you can go_to by name: start (where this request began, 1.0 m behind)" \
        in _text(tool_results(brain.seen[1])[0])  # a move's result says where the request began, from the new spot
    second = brain.seen[2]  # the first turn of "Come back." (the first request took two: act, reply)
    assert [m["role"] for m in second] == ["system", "user"]  # one user turn: the last request is told, not replayed
    text = " ".join(p.get("text", "") for p in second[1]["content"])
    assert "move(metres=1, direction='forward') -> Moved 1 m forward." in text  # step by step, results included
    assert "ended 1.0 m from where that request began" in text
    assert "where you were before (where the request before began, 1.0 m behind)" in text
    assert text.endswith('The user says: "Come back."')
    # After arriving back at "where you were before", "start" is hidden so it doesn't lure the robot back out.
    returned = _text(tool_results(brain.seen[3])[0])
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
    second = _text(tool_results(brain.seen[2])[-1])  # what the model reads after its second full turn
    assert "turned 360 degrees left; turned 360 degrees left; 720 degrees of turning in all" in second
    assert "facing the way it began" in second


@pytest.mark.parametrize(("name", "args", "progress"), [
    ("turn", {"degrees": "90 degrees", "direction": "left"}, "turned 90 degrees left; 90 degrees of turning in all"),
    ("move", {"metres": "0.3 m", "direction": "forward"}, "moved 0.3 m forward"),
    ("turn", {"degrees": 90, "direction": "Left"}, "turned 90 degrees left; 90 degrees of turning in all"),
], ids=["turn-with-unit", "move-with-unit", "mixed-case-direction"])
def test_motion_progress_uses_validated_arguments(name, args, progress):
    tools = Tools(SimRobot(Scene(6, 6, [], {})))
    result = tools.execute(name, args)
    assert result.ok
    assert f"\nThis request so far: {progress}." in result.text


def test_go_to_a_name_not_yet_seen_searches_for_it_and_then_goes_there():
    sim = SimRobot(Scene(16.0, 6.0, [Thing("ball", 6.0, 0.0, radius=0.15, color="red")], {}, start=Pose(-6.0, 0, 0)))
    ball, = sim.things.values()
    turn_only = Tools(sim).execute("face", {"target": "red ball"})  # facing never drives off to look
    assert not turn_only.ok and "look('around'" in turn_only.text
    assert not any(e.name == "search" for e in sim.trace.events)
    result = Tools(sim).execute("go_to", {"target": "red ball"})  # 12 m away: out of the cameras' range
    assert result.ok and result.text.startswith(f"Searched for it: found ball {ball.id}. Drive finished.")
    names = [e.name for e in sim.trace.events]
    assert names.index("search") < names.index("go_to")
    assert math.dist((sim.x, sim.y), (ball.x, ball.y)) < 1.0
    missing = Tools(sim).execute("go_to", {"target": "elephant"})
    assert not missing.ok and "searched here and found no 'elephant'" in missing.text


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
    assert _text(results[0]).startswith("Turned 90 degrees left.") and "0.5" in _text(results[1])


def test_partial_action_stops_remaining_batched_calls() -> None:
    robot = SimRobot(living_room())
    robot.move = lambda metres, direction="forward": None  # type: ignore[method-assign]
    turns = []
    robot.turn = lambda degrees, direction="left": turns.append((degrees, direction))  # type: ignore[method-assign]
    brain = Scripted(
        Reply("", tool_calls=[
            ToolCall("c1", "move", {"metres": 1.0, "direction": "forward"}),
            ToolCall("c2", "turn", {"degrees": 90, "direction": "left"}),
        ]),
        say("Stopped short."),
    )
    record = Agent(brain).do("Move forward 1 m then turn left.", robot)
    assert record.outcome == "done"
    assert turns == []
    results = tool_results(brain.seen[1])
    assert "ended short" in _text(results[0])
    assert "Not run:" in _text(results[1])


def test_correction_during_thinking_cancels_stale_tool_calls() -> None:
    robot = SimRobot(living_room())
    ran = []
    robot.turn = lambda degrees, direction="left": ran.append((degrees, direction))  # type: ignore[method-assign]
    agent: Agent

    class InterjectingBrain(Scripted):
        def chat(self, messages, tools, think=0, **_):  # noqa: ANN001, ANN003, ANN201
            if not self.seen:
                agent.interject("Wait, turn right instead.")
            return super().chat(messages, tools, think=think)

    brain = InterjectingBrain(
        call("turn", degrees=90, direction="left"),
        call("turn", degrees=90, direction="right"),
        say("Turned right instead."),
    )
    agent = Agent(brain)
    record = agent.do("Turn left 90 degrees.", robot)
    assert record.outcome == "done"
    assert ran == [(90.0, "right")]
    assert "Not run: the user spoke" in _text(tool_results(brain.seen[1])[0])
    assert any("Wait, turn right instead." in _text(m["content"]) for m in brain.seen[1] if m["role"] == "user")


def test_correction_between_batched_calls_cancels_remaining_calls() -> None:
    robot = SimRobot(living_room())
    ran = []
    agent: Agent

    def first_turn(degrees: float, direction: str = "left") -> None:
        ran.append((degrees, direction))
        if len(ran) == 1:
            agent.interject("Actually don't move forward.")

    robot.turn = first_turn  # type: ignore[method-assign]
    robot.move = lambda metres, direction="forward": ran.append(("move", metres))  # type: ignore[method-assign]
    brain = Scripted(
        Reply("", tool_calls=[
            ToolCall("c1", "turn", {"degrees": 90, "direction": "left"}),
            ToolCall("c2", "move", {"metres": 1.0, "direction": "forward"}),
        ]),
        say("Okay, staying put."),
    )
    agent = Agent(brain)
    record = agent.do("Turn left and move forward.", robot)
    assert record.outcome == "done"
    assert ran == [(90.0, "left")]
    assert "Not run: the user spoke" in _text(tool_results(brain.seen[1])[1])


def test_open_todo_steps_mark_task_failed() -> None:
    robot = SimRobot(living_room())
    brain = Scripted(
        call("todo", steps=["go to sofa", "turn around"]),
        call("todo_done", step=1),
        say("Done!"),
        say("Really done!"),  # still ignores the open step after the review prompt
    )
    record = Agent(brain).do("Go to the sofa and turn around.", robot)
    assert record.outcome == "failed"
    assert any("turn around" in p for p in record.problems)


def test_restart_recalls_corrected_fact_on_first_prompt_and_fails_without_memory(tmp_path) -> None:
    from robot.mind.memory import CirculatingMemory

    db = tmp_path / "restart.db"
    mem1 = CirculatingMemory(db)
    robot1 = SimRobot(living_room())
    brain1 = Scripted(
        call("remember", fact="spare keys are on the couch"),
        say("Got it."),
        call("remember", fact="spare keys are on the dining table"),
        say("Updated."),
    )
    agent1 = Agent(brain1, memory=mem1)
    assert agent1.do("Remember that my spare keys are on the couch.", robot1).outcome == "done"
    assert agent1.do("Actually, I moved my spare keys to the dining table.", robot1).outcome == "done"
    mem1.close()

    # Restart: fresh SQLite handle, fresh robot, fresh Agent (no transcript)
    mem2 = CirculatingMemory(db)
    robot2 = SimRobot(living_room())
    brain2 = Scripted(say("Turned."), say("Your backup key is on the dining table."))
    agent2 = Agent(brain2, memory=mem2)
    agent2.do("Turn left 90 degrees.", robot2)
    irrelevant_prompt = _text(brain2.seen[0][1]["content"])
    assert "Remembered facts" not in irrelevant_prompt

    agent2.do("Where did I leave my backup key?", robot2)
    recall_prompt = _text(brain2.seen[1][1]["content"])
    assert "Remembered facts" in recall_prompt
    assert "dining table (most recent)" in recall_prompt
    assert recall_prompt.index("couch") < recall_prompt.index("dining table")
    mem2.close()

    # Ablation: without durable memory after restart, no stored facts reach the fresh agent
    robot3 = SimRobot(living_room())
    brain3 = Scripted(say("I don't know."))
    Agent(brain3, memory=None).do("Where did I leave my backup key?", robot3)
    assert "dining table" not in _text(brain3.seen[0][1]["content"])


def test_home3d_r1r_restart_passes_with_memory_and_fails_without(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    from robot.eval import home3d
    from robot.mind import brain as brain_mod
    from robot.perception import detector as det_mod

    seen_stores: list[object] = []
    shutdowns: list[int] = []

    class FakeHome:
        def __init__(self, scene: str, rtf: float = 1.0) -> None:
            self.scene, self.bumps = scene, 0
            self.truth = {"objects": []}
            self.path = [(0.0, 0.0, 0.0, 0.0)]
            self.world = SimpleNamespace(
                call=lambda **kw: {"front_camera": {}, "rear_left_camera": {}, "rear_right_camera": {}},
            )

        def reset(self) -> None:
            pass

        def wait(self, seconds: float) -> None:
            pass

        def close(self) -> None:
            pass

        def robot(self, detector: object, brain: object, store: object = None) -> SimRobot:
            seen_stores.append(store)
            sim = SimRobot(living_room())
            sim.store = store  # type: ignore[attr-defined]
            sim.perception = SimpleNamespace(stop=lambda: None)  # type: ignore[attr-defined]
            sim._perceive_objects = lambda: None  # type: ignore[attr-defined]
            sim.shutdown = lambda: shutdowns.append(1)  # type: ignore[attr-defined]
            return sim

    recall_prompts: list[str] = []

    class PromptAwareBrain:
        def __init__(self, url: str) -> None:
            self.turn = 0

        def healthy(self) -> bool:
            return True

        def chat(self, messages, tools, think=0, **_):  # noqa: ANN001, ANN003, ANN201
            self.turn += 1
            if self.turn == 1:
                return call("remember", fact="spare keys are on the couch")
            if self.turn == 2:
                return say("Got it.")
            if self.turn == 3:
                return call("remember", fact="spare keys are on the dining table")
            if self.turn == 4:
                return say("Updated.")
            if self.turn == 5:
                return call("turn", degrees=90, direction="left")
            if self.turn == 6:
                return say("Turned left.")
            prompt = _text(messages[1]["content"])
            recall_prompts.append(prompt)
            if "dining table (most recent)" in prompt:
                return say("Your backup key is on the dining table.")
            return say("I don't know where your keys are.")

    monkeypatch.setattr(home3d, "Home", FakeHome)
    monkeypatch.setattr(home3d, "out_dir", lambda: tmp_path)
    monkeypatch.setattr(home3d, "keep_pictures", lambda robot, folder, kept=None: kept if kept is not None else [])
    monkeypatch.setattr(home3d, "write_report", lambda *a, **kw: None)
    monkeypatch.setattr(brain_mod, "Brain", PromptAwareBrain)
    monkeypatch.setattr(det_mod, "Detector", lambda: object())

    with_mem = home3d.run("mem", ["R1R"], ["apt_0"], "http://fake", "front", "off", 1.0, use_memory=True)
    assert len(with_mem) == 1 and with_mem[0]["passed"] is True
    assert len(seen_stores) == 2 and seen_stores[0] is not seen_stores[1]
    assert len(shutdowns) == 2
    assert len(recall_prompts) == 1 and "Recent conversation:" not in recall_prompts[0]

    seen_stores.clear()
    shutdowns.clear()
    recall_prompts.clear()
    without_mem = home3d.run("nomem", ["R1R"], ["apt_0"], "http://fake", "front", "off", 1.0, use_memory=False)
    assert len(without_mem) == 1 and without_mem[0]["passed"] is False
    assert seen_stores == [None, None]
    assert len(shutdowns) == 2
    assert len(recall_prompts) == 1 and "Recent conversation:" not in recall_prompts[0]
