"""The model owns decisions; the runtime only observes, executes, and schedules."""

from __future__ import annotations

import asyncio
import json

import pytest

from robot.audio.service import AudioService
from robot.cognition.backend import InferenceResult, UnavailableBackend
from robot.core.bus import EventBus
from robot.core.clock import RealClock, ReplayClock
from robot.core.config import RobotConfig
from robot.core.events import EventPriority
from robot.main import RobotHarness


class SequenceBackend:
    name = "scripted_decisions"

    def __init__(self, decisions):
        self.decisions = decisions
        self.messages = []
        self.schemas = []

    async def generate(self, messages, *, json_schema=None):
        self.messages.append(messages)
        self.schemas.append(json_schema)
        decision = self.decisions[min(len(self.messages) - 1, len(self.decisions) - 1)]
        return InferenceResult(json.dumps({"say": "", "sound": "none", "emotion": {"name": "neutral"},
                                           "actions": [], "memory": [], **decision}), self.name)


def make_harness(tmp_path, decisions, *, real=False):
    config = RobotConfig.from_yaml()
    config.database_path = tmp_path / "model-loop.db"
    config.record_events = False
    backend = decisions if hasattr(decisions, "generate") else SequenceBackend(decisions)
    harness = RobotHarness(config, backend=backend, clock=RealClock() if real else ReplayClock(),
                           tts_mode="warble", realtime_audio_sync=False)
    harness.face_bridge._send_udp = lambda _: None
    return harness


async def finish(harness):
    await asyncio.wait_for(harness.arbiter.wait_until_done(), 5)
    await harness.bus.drain()


async def wake_for_change(harness):
    harness.sandbox.move_object("red_ball", 900, 160)
    harness.clock.advance(13)
    await harness.autonomy.step()
    assert harness.autonomy._task is not None
    await harness.autonomy._task
    await harness.bus.drain()


@pytest.mark.parametrize("user_text", ["What do you see?", "Find the rubber duck", "Explore the room", "Look happy"])
async def test_completed_model_decision_never_injects_movement_or_continuation(tmp_path, user_text):
    async with make_harness(tmp_path, [{"say": "Here is my current observation.", "status": "completed",
                                      "emotion": {"name": "curious"}}]) as harness:
        response = await harness.submit_message(user_text)
        await finish(harness)
        assert response["status"] == "completed"
        assert len(harness.backend.messages) == 1
        assert harness.executor.completed_count == 0
        assert harness.body.pose().x_mm == 0
        assert harness.body.pose().heading_deg == 0
        assert harness.body.pose().head_tilt_deg == 0
        assert "goal" not in harness.backend.schemas[0]["properties"]
        content = harness.backend.messages[0][-1]["content"]
        assert isinstance(content, str)
        assert "Camera:" not in content
        assert harness.arbiter.active_goal["status"] == "completed"


async def test_model_controls_each_action_and_when_the_loop_ends(tmp_path):
    decisions = [
        {"say": "I'll turn.", "status": "continue", "actions": [{"tool": "spin", "args": {"angle_deg": 90}}]},
        {"say": "Now I see a different view.", "status": "continue",
         "actions": [{"tool": "tilt_head", "args": {"angle_deg": -10}}]},
        {"say": "That's enough.", "status": "completed"},
        {"say": "Must never run", "status": "continue", "actions": [{"tool": "drive", "args": {"distance_mm": 200}}]},
    ]
    async with make_harness(tmp_path, decisions) as harness:
        replies = []
        harness.bus.subscribe("robot.response", replies.append, immediate=True)
        await harness.submit_message("Look around however you think is useful")
        await finish(harness)
        assert len(harness.backend.messages) == 3
        assert [e.payload["say"] for e in replies] == ["I'll turn.", "Now I see a different view.", "That's enough."]
        assert harness.body.pose().heading_deg == pytest.approx(90)
        assert harness.body.pose().head_tilt_deg == pytest.approx(-10)
        assert harness.body.pose().x_mm == 0
        assert "Spun 90" in str(harness.backend.messages[1][-1]["content"])
        assert "heading_deg" in str(harness.backend.messages[1][-1]["content"])
        assert len([turn for turn in harness.context_builder._recent_turns if turn.role == "user"]) == 1


async def test_idle_model_may_choose_to_do_nothing_and_set_its_next_wake(tmp_path):
    async with make_harness(tmp_path, [{"status": "completed", "next_wake_s": 120}, {"status": "completed"}]) as harness:
        await wake_for_change(harness)
        assert len(harness.backend.messages) == 1
        assert harness.executor.completed_count == 0
        assert "Background Event" in str(harness.backend.messages[0][-1]["content"])
        assert "observation_changed" in str(harness.backend.messages[0][-1]["content"])
        assert harness.autonomy.snapshot()["next_decision_in_s"] == pytest.approx(120)
        harness.clock.advance(20)
        await harness.autonomy.step()
        assert len(harness.backend.messages) == 1
        assert not any(t.role == "user" for t in harness.context_builder._recent_turns)
        assert not harness.context_builder._recent_turns
        harness.clock.advance(100)
        await harness.autonomy.step()
        await harness.autonomy._task
        assert len(harness.backend.messages) == 2
        assert "scheduled_wake" in str(harness.backend.messages[1][-1]["content"])
        harness.clock.advance(600)
        await harness.autonomy.step()
        assert len(harness.backend.messages) == 2


async def test_idle_motion_and_speech_are_chosen_by_the_model(tmp_path):
    async with make_harness(tmp_path, [
        {"say": "I want to check the other side.", "status": "continue",
         "actions": [{"tool": "spin", "args": {"angle_deg": -45}}]},
        {"say": "I'll stay here for now.", "status": "completed"},
    ]) as harness:
        replies = []
        harness.bus.subscribe("robot.response", replies.append, immediate=True)
        await wake_for_change(harness)
        assert harness.body.pose().heading_deg == pytest.approx(-45)
        assert len(harness.backend.messages) == 2
        assert [e.payload["say"] for e in replies] == ["I want to check the other side.", "I'll stay here for now."]
        assert all(e.payload["origin"] == "autonomous" for e in replies)
        harness.clock.advance(60)
        await harness.autonomy.step()
        assert len(harness.backend.messages) == 2  # Own motion must not trigger another idle goal.


async def test_startup_and_unchanged_idle_make_no_inference_or_history(tmp_path):
    async with make_harness(tmp_path, [{"say": "Must never be requested", "status": "completed"}]) as harness:
        for _ in range(120):
            harness.clock.advance(5)
            await harness.autonomy.step()
        assert not harness.backend.messages
        assert not harness.context_builder._recent_turns
        assert harness.executor.completed_count == 0
        assert harness.audio.tts_spoken_count == 0


async def test_real_startup_loop_stays_silent(tmp_path):
    harness = make_harness(tmp_path, [{"say": "Must never be requested", "status": "completed"}], real=True)
    harness.config.autonomy.idle_after_s = 0.1
    harness.config.autonomy.observation_interval_s = 0.1
    async with harness:
        await asyncio.sleep(0.35)
        assert not harness.backend.messages
        assert harness.audio.tts_spoken_count == 0


async def test_descriptive_reply_does_not_immediately_restart_as_idle_work(tmp_path):
    async with make_harness(tmp_path, [{"say": "I see the current room.", "status": "completed"}]) as harness:
        await harness.submit_message("what do you see")
        await finish(harness)
        harness.clock.advance(15)
        await harness.autonomy.step()
        assert len(harness.backend.messages) == 1
        assert harness.autonomy._task is None


async def test_user_stop_cancels_whole_model_loop_and_pauses_idle_decisions(tmp_path):
    async with make_harness(tmp_path, [{"status": "continue", "actions": [{"tool": "spin", "args": {"angle_deg": 180}}]}], real=True) as harness:
        entered = asyncio.Event()
        harness.bus.subscribe("action.started", lambda e: entered.set())
        pending = asyncio.create_task(harness.submit_message("look around"))
        await asyncio.wait_for(entered.wait(), 2)
        await harness.submit_message("stop")
        await finish(harness)
        await pending
        assert harness.body.pose().angular_vel_deg_s == 0
        assert harness.autonomy.paused
        assert len(harness.backend.messages) == 1


async def test_unavailable_model_never_runs_a_scripted_substitute(tmp_path):
    async with make_harness(tmp_path, UnavailableBackend()) as harness:
        response = await harness.submit_message("find the rubber duck")
        await finish(harness)
        assert response["status"] == "blocked"
        assert harness.executor.completed_count == 0


async def test_ordered_speech_and_interruption():
    bus = EventBus(ReplayClock())
    audio = AudioService(bus, enable_speaker=False)
    await audio.start()
    gate, started = asyncio.Event(), asyncio.Event()
    spoken = []

    async def speak(text, **kwargs):
        spoken.append(text)
        started.set()
        await gate.wait()

    audio.speak = speak
    try:
        await bus.emit("robot.response", "test", {"say": "First"})
        await started.wait()
        await bus.emit("robot.response", "test", {"say": "Second"})
        assert spoken == ["First"]
        gate.set()
        await audio.wait_until_done()
        assert spoken == ["First", "Second"]
        gate.clear()
        started.clear()
        await bus.emit("robot.response", "test", {"say": "Third"})
        await started.wait()
        await bus.emit("robot.response", "test", {"say": "Cancelled"})
        await bus.emit("action.cancel", "test", priority=EventPriority.CRITICAL)
        assert audio._response_queue.empty()
        assert spoken == ["First", "Second", "Third"]
    finally:
        await audio.stop()
