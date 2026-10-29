"""Failure-oriented tests for safety, bounded resources and lifecycle ownership."""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import threading
from pathlib import Path

import httpx
import numpy as np
import pytest

from robot.actions.registry import CapabilityRegistry
from robot.actions.tools import ToolCallRequest
from robot.audio.asr import ASRResult
from robot.audio.service import AudioService
from robot.audio.tts import SynthesizedSpeech
from robot.audio.vad import VADEvent
from robot.body.hal import BodyCapabilities, BodyCommand, JointSpec
from robot.body.sim_body import SimBody
from robot.cognition.backend import InferenceResult, NullBackend
from robot.cognition.context import ContextBuilder, estimate_tokens, truncate_to_token_budget
from robot.cognition.llamacpp import AutoBackend, LlamaCppBackend
from robot.cognition.validator import ResponseValidator
from robot.core.bus import BusOverloaded, EventBus
from robot.core.clock import RealClock, ReplayClock
from robot.core.config import LLMConfig, RobotConfig
from robot.core.events import Event, EventPriority
from robot.core.recorder import EventRecorder
from robot.main import RobotHarness
from robot.memory.models import MemoryProvenance
from robot.memory.store import MemoryStore
from robot.world.state import WorldState


def make_harness(tmp_path: Path, *, backend=None, real=False, dashboard=False):
    cfg = RobotConfig.from_yaml()
    cfg.database_path = tmp_path / "isolated.db"
    cfg.recordings_dir = tmp_path / "recordings"
    cfg.record_events = False
    cfg.llm.backend = "null"
    harness = RobotHarness(
        cfg,
        clock=RealClock() if real else ReplayClock(),
        backend=backend or NullBackend(),
        enable_dashboard=dashboard,
        dashboard_port=0,
        tts_mode="warble",
        realtime_audio_sync=False,
    )
    harness.face_bridge._send_udp = lambda payload: None
    return harness


class ControlledBackend:
    name = "controlled"

    def __init__(self, actions=None, ignore_cancel=False):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.actions = actions or [{"tool": "drive", "args": {"distance_mm": 300}}]
        self.ignore_cancel = ignore_cancel

    async def generate(self, messages, *, json_schema=None):
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            if not self.ignore_cancel:
                raise
            await self.release.wait()
        return InferenceResult(
            json.dumps(
                {
                    "say": "Moving.",
                    "sound": "none",
                    "emotion": {"name": "neutral"},
                    "actions": self.actions,
                    "memory": [],
                }
            ),
            self.name,
        )


@pytest.mark.asyncio
async def test_stop_invalidates_even_uncancellable_late_inference(tmp_path):
    backend = ControlledBackend(ignore_cancel=True)
    async with make_harness(tmp_path, backend=backend) as harness:
        pending = asyncio.create_task(harness.submit_message("drive"))
        await asyncio.wait_for(backend.entered.wait(), 1)
        stopped = await harness.submit_message("stop")
        assert stopped["backend"] == "reflex_stop"
        backend.release.set()
        await harness.bus.drain()
        assert harness.body.pose().x_mm == 0
        assert (await pending)["backend"] == "cancelled"
        assert not harness.executor._active_tasks


@pytest.mark.asyncio
async def test_cancel_cannot_run_remaining_plan(tmp_path):
    backend = ControlledBackend(
        actions=[{"tool": "drive", "args": {"distance_mm": 300}}, {"tool": "spin", "args": {"angle_deg": 90}}]
    )
    backend.release.set()
    async with make_harness(tmp_path, backend=backend, real=True) as harness:
        turn = asyncio.create_task(harness.arbiter.handle_user_turn("move then spin"))
        for _ in range(100):
            if harness.body.pose().linear_vel_mm_s:
                break
            await asyncio.sleep(0.002)
        await harness.bus.emit("action.cancel", "test", {"reason": "stop"}, priority=EventPriority.CRITICAL)
        await turn
        assert harness.body.pose().x_mm < 300
        assert harness.body.pose().heading_deg == 0
        assert harness.body.pose().linear_vel_mm_s == 0


@pytest.mark.asyncio
async def test_new_input_replaces_old_turn_and_correlates_requests(tmp_path):
    backend = ControlledBackend()
    async with make_harness(tmp_path, backend=backend) as harness:
        first = asyncio.create_task(harness.submit_message("first"))
        await asyncio.wait_for(backend.entered.wait(), 1)
        second = asyncio.create_task(harness.submit_message("second"))
        await asyncio.sleep(0.02)
        backend.release.set()
        assert (await first)["backend"] == "cancelled"
        assert (await second)["backend"] == "controlled"
        assert harness.body.pose().x_mm == 300


@pytest.mark.asyncio
async def test_telemetry_cannot_remove_control_and_critical_bypasses_backpressure():
    bus = EventBus(default_queue_size=2)
    delivered, immediate = [], []
    bus.subscribe("*", lambda event: delivered.append(event.type))
    bus.subscribe("action.cancel", lambda event: immediate.append(event.type), immediate=True)
    await bus.emit("action.cancel", "test", priority=EventPriority.CRITICAL)
    await bus.emit("body.pose", "test", priority=EventPriority.TELEMETRY)
    await bus.emit("body.pose", "test", priority=EventPriority.TELEMETRY)
    assert immediate == ["action.cancel"]
    await bus.start()
    await bus.drain()
    await bus.stop()
    assert delivered == ["action.cancel", "body.pose"]


@pytest.mark.asyncio
async def test_reliable_overload_fails_before_work_is_started():
    bus = EventBus(default_queue_size=1)
    immediate = []
    bus.subscribe("input.user_message", lambda event: None)
    bus.subscribe("input.user_message", lambda event: immediate.append(event.id), immediate=True)
    await bus.emit("input.user_message", "test")
    with pytest.raises(BusOverloaded):
        await bus.emit("input.user_message", "test")
    assert len(immediate) == 1


@pytest.mark.asyncio
async def test_paused_replay_clock_drain_timeout_is_real_time():
    bus = EventBus(ReplayClock())
    wait = asyncio.Event()
    bus.subscribe("blocked", lambda event: wait.wait())
    await bus.start()
    await bus.emit("blocked", "test")
    await asyncio.wait_for(bus.drain(timeout=0.01), 0.2)
    wait.set()
    await bus.stop()


@pytest.mark.parametrize(
    "tool,args",
    [
        ("spin", {"angle_deg": "NaN"}),
        ("spin", {"angle_deg": float("inf")}),
        ("drive", {"distance_mm": "NaN"}),
        ("drive", {"distance_mm": True}),
        ("tilt_head", {"angle_deg": "-Infinity"}),
        ("drive", {}),
        ("spin", {}),
    ],
)
def test_nonfinite_and_missing_motor_args_are_rejected(tool, args):
    assert not CapabilityRegistry(BodyCapabilities()).validate_tool_call(ToolCallRequest(tool, args)).valid


@pytest.mark.asyncio
async def test_cliff_mid_motion_reports_partial_pose():
    body = SimBody(EventBus(), step_dt_s=0.01)
    move = asyncio.create_task(body.execute(BodyCommand("moving", "drive", {"distance_mm": 100})))
    await asyncio.sleep(0.035)
    body.set_simulated_obstacle(cliff=True)
    result = await move
    assert result.status == "reflex_stopped"
    assert 0 < body.pose().x_mm < 100
    for tool, args in (("spin", {"angle_deg": 90}), ("perform_trick", {"trick": "dance"})):
        result = await body.execute(BodyCommand(tool, tool, args))
        assert result.status == "reflex_stopped"
    assert body.pose().heading_deg == 0


@pytest.mark.asyncio
async def test_tricks_obey_custom_joint_limits_and_encoder_precision():
    caps = BodyCapabilities(joints={"head_tilt": JointSpec("head_tilt", -5, 5)})
    body = SimBody(EventBus(ReplayClock()), caps, step_dt_s=0.001)
    await body.execute(BodyCommand("head", "perform_trick", {"trick": "curious_tilt"}))
    assert body.pose().head_tilt_deg == pytest.approx(5)
    await body.execute(BodyCommand("drive", "drive", {"distance_mm": 100}))
    expected = round(100 * caps.encoder_ticks_per_rev / (math.pi * caps.wheel_diameter_mm))
    assert body.sensors().wheel_ticks_left == expected


@pytest.mark.asyncio
async def test_idle_processing_without_dashboard(tmp_path):
    harness = make_harness(tmp_path, real=True)
    harness.fsm.follow_up_window_s = 0.01
    harness.fsm.idle_sleepy_timeout_s = 0.04
    async with harness:
        await harness.arbiter.handle_user_turn("My cat's name is Mochi.")
        for _ in range(50):
            if harness.consolidator.total_passes:
                break
            await asyncio.sleep(0.02)
        assert harness.consolidator.total_passes == 1
        assert harness.memory.get_active("cat's name") is not None


@pytest.mark.asyncio
async def test_partial_startup_failure_closes_all_resources(tmp_path):
    class Broken:
        name = "broken"

        async def start(self):
            raise RuntimeError("injected")

        async def stop(self):
            pass

    harness = make_harness(tmp_path)
    harness.supervisor.register(Broken())
    with pytest.raises(RuntimeError, match="injected"):
        await harness.start()
    assert not harness.bus._running
    assert not harness.audio._running
    assert harness._closed
    await harness.stop()


def test_broad_forget_blocks_resurrection_and_explicit_remember_restores(tmp_path):
    store = MemoryStore(tmp_path / "mem.db")
    try:
        store.remember("My cat's name is Mochi")
        assert store.forget("my cat")[0] == 1
        assert store.remember("My cat's name is Mochi", provenance=MemoryProvenance.MODEL_PROPOSED) is None
        assert store.remember("My cat's name is Luna") is not None
        assert store.get_active("cat's name").content.endswith("Luna")
        with pytest.raises(ValueError):
            store.forget("")
    finally:
        store.close()


def test_retrieval_never_pads_unrelated_results(tmp_path):
    store = MemoryStore(tmp_path / "mem.db")
    try:
        store.remember("My favorite color is green")
        assert store.search("volcano geology") == []
    finally:
        store.close()


def test_context_and_retained_turns_are_bounded_for_multibyte_input():
    registry = CapabilityRegistry(BodyCapabilities())
    context = ContextBuilder(Path("config/SOUL.md"), registry)
    for _ in range(100):
        context.add_turn("user", "世界" * 5000)
    messages, usage = context.build_messages("世界" * 20000, memories=[], world_state=WorldState(EventBus()))
    assert usage["total_tokens"] + usage["response_reserve_tokens"] <= context.budget.total_tokens
    assert sum(estimate_tokens(message["content"]) + 6 for message in messages) == usage["total_tokens"]
    assert len(context._recent_turns) <= context.budget.max_recent_turns
    assert "Safety overrides personality" in messages[0]["content"]
    assert truncate_to_token_budget("hello", 0) == ""


def test_spatial_observations_are_bounded_and_expire_on_monotonic_clock():
    clock = ReplayClock()
    world = WorldState(EventBus(clock))
    for i in range(1000):
        world.record_ego_observation(str(i), "object", ttl_s=1)
    assert len(world._observations) == 128
    clock.advance(2)
    clock.set_time(clock.now() - 3600)
    assert world.active_observations() == []


@pytest.mark.asyncio
async def test_dashboard_stop_validation_and_rejected_action(tmp_path):
    async with make_harness(tmp_path, dashboard=True) as harness:
        async with httpx.AsyncClient(base_url=harness.dashboard.url) as client:
            stopped = await client.post("/api/message", json={"text": "stop"})
            assert stopped.json()["backend"] == "reflex_stop"
            rejected = await client.post("/api/action", json={"tool": "raw_gpio_write"})
            assert rejected.json()["ok"] is False
            assert (await client.post("/api/message", json=[])).status_code == 400
            assert (
                await client.post(
                    "/api/message", json={"text": "hi"}, headers={"Origin": "https://attacker.invalid"}
                )
            ).status_code == 403
            assert (
                await client.post(
                    "/api/message", content=b"x" * 17000, headers={"Content-Type": "application/json"}
                )
            ).status_code == 400
            assert (await client.post("/api/memory/forget", json={"topic": ""})).status_code == 400


@pytest.mark.asyncio
async def test_auto_backend_backoff_prevents_repeated_failed_requests():
    class Failing:
        calls = 0

        async def generate(self, *args, **kwargs):
            self.calls += 1
            raise OSError("offline")

    primary = Failing()
    backend = AutoBackend(primary)
    for _ in range(5):
        result = await backend.generate([])
        assert result.backend_name == "inference_unavailable"
    assert primary.calls == 1


def test_pi_profile_and_strict_response_validation():
    cfg = RobotConfig.from_yaml("config/robot.pi5.yaml")
    assert cfg.max_queue_size == 64
    assert cfg.record_events is False
    assert cfg.face_fps == 30
    registry = CapabilityRegistry(BodyCapabilities())
    response = ResponseValidator(registry).validate(
        {
            "say": "hello",
            "emotion": {"name": "happy", "intensity": "NaN"},
            "actions": [{"tool": "stop", "args": {}}, {"tool": "spin", "args": {"angle_deg": 90}}],
        }
    )
    assert math.isfinite(response.emotion.intensity)
    assert [action.tool for action in response.actions] == ["stop"]


@pytest.mark.asyncio
async def test_recorded_replay_uses_original_outputs_without_mutating_memory(tmp_path):
    from scripts.replay import replay_session

    path = tmp_path / "session.jsonl"
    response = {"say": "Original model output", "actions": [], "memory": []}
    path.write_text(
        Event("input.user_message", "test", {"text": "forget my cat"}).to_json()
        + "\n"
        + Event("robot.response", "test", response).to_json()
        + "\n",
        encoding="utf-8",
    )
    assert await replay_session(path) == [response]


def test_recording_ids_are_unique_and_close_flushes(tmp_path):
    one, two = EventRecorder(tmp_path), EventRecorder(tmp_path)
    assert one.file_path != two.file_path
    one.open()
    one.record(Event("test", "test", {"v": 1}))
    one.close()
    assert len(EventRecorder.load_events(one.file_path)) == 1
    one.record(Event("after_close", "test"))
    assert len(EventRecorder.load_events(one.file_path)) == 1


@pytest.mark.asyncio
async def test_speech_production_is_owned_and_stays_stopped_during_synthesis():
    bus = EventBus()
    audio = AudioService(bus, enable_speaker=False, tts_mode="warble", realtime_sync=True)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def synthesize(text):
        calls.append(text)
        entered.set()
        release.wait(2)
        return SynthesizedSpeech(text, np.zeros(2400, dtype=np.float32), 24000, [0.5] * 6)

    audio.tts.synthesize = synthesize
    await bus.start()
    await audio.start()
    try:
        await bus.emit("robot.response", "test", {"say": "first"})
        assert await asyncio.to_thread(entered.wait, 1)
        for i in range(20):
            await bus.emit("robot.response", "test", {"say": str(i)})
        await bus.emit("action.cancel", "test", priority=EventPriority.CRITICAL)
        release.set()
        await bus.drain()
        assert len(calls) == 1
        assert not audio.is_speaking
        assert audio._utterance_task is None or audio._utterance_task.done()
    finally:
        release.set()
        await audio.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_transcription_never_blocks_vad_and_pending_audio_is_bounded():
    bus = EventBus()
    audio = AudioService(bus, enable_speaker=False, tts_mode="warble")
    entered, release = threading.Event(), threading.Event()

    def transcribe(samples, rate):
        entered.set()
        release.wait(2)
        return ASRResult("test", 1.0, 0.0, "fake")

    audio.asr.transcribe_samples = transcribe
    audio.vad.process_frame = lambda *args, **kwargs: [
        VADEvent("speech_end", 0.1, np.zeros(480, dtype=np.float32))
    ]
    await bus.start()
    await audio.start()
    try:
        await asyncio.wait_for(audio.process_mic_frame(np.zeros(480)), 0.2)
        assert await asyncio.to_thread(entered.wait, 1)
        for _ in range(20):
            await asyncio.wait_for(audio.process_mic_frame(np.zeros(480)), 0.2)
        assert audio._asr_queue.qsize() == 1
        assert not audio._asr_future.done()
    finally:
        release.set()
        await audio.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_follow_up_window_starts_after_actual_speech(tmp_path):
    backend = ControlledBackend(actions=[{"tool": "set_gaze", "args": {"target": "center"}}])
    backend.release.set()
    harness = make_harness(tmp_path, backend=backend, real=True)
    harness.audio.realtime_sync = True
    harness.audio.tts.synthesize = lambda text: SynthesizedSpeech(
        text, np.zeros(12000, dtype=np.float32), 24000, [0.5] * 30
    )
    harness.fsm.follow_up_window_s = 0.05
    async with harness:
        await harness.submit_message("hello")
        await asyncio.sleep(0.1)
        assert str(harness.fsm.state) == "speaking"
        await harness.audio.wait_until_done()
        await harness.bus.drain()
        assert harness.fsm.in_follow_up_window()


def test_memory_update_failure_rolls_back_supersession(tmp_path):
    store = MemoryStore(tmp_path / "atomic.db")
    try:
        store.remember("My cat's name is Mochi")
        store._conn.execute(
            "CREATE TRIGGER fail_insert BEFORE INSERT ON memories BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.remember("My cat's name is Luna")
        assert store.get_active("cat's name").content.endswith("Mochi")
        assert len(store.search("Mochi")) == 1
    finally:
        store.close()


@pytest.mark.asyncio
async def test_failing_critical_observer_cannot_prevent_other_safety_handlers():
    bus = EventBus(default_queue_size=1)
    stopped = []

    def failing(event):
        raise RuntimeError("broken observer")

    bus.subscribe("action.cancel", failing, immediate=True)
    bus.subscribe("action.cancel", lambda event: stopped.append(True), immediate=True)
    await bus.emit("action.cancel", "test", priority=EventPriority.CRITICAL)
    assert stopped == [True]


def test_recording_rotation_is_streamed_and_session_size_is_capped(tmp_path):
    recorder = EventRecorder(tmp_path)
    recorder.max_file_bytes = 1
    recorder.max_parts = 2
    recorder.open()
    for i in range(10):
        recorder.record(Event("test", "test", {"i": i}))
    recorder.close()
    assert len(list(tmp_path.glob("*.jsonl"))) == 2
    assert len(EventRecorder.load_events(recorder.file_path)) == 2
    assert recorder.dropped_count == 8


@pytest.mark.asyncio
async def test_long_input_cannot_apply_partial_memory_operations(tmp_path):
    async with make_harness(tmp_path) as harness:
        with pytest.raises(ValueError, match="context budget"):
            await harness.submit_message("Remember my name is " + "a" * 3000)
        assert harness.memory.list_active() == []


def test_corrupt_deeply_nested_model_json_never_escapes_validator():
    validator = ResponseValidator(CapabilityRegistry(BodyCapabilities()))
    response = validator.validate('{"x":' + "[" * 4000 + "0" + "]" * 4000 + "}")
    assert response.fallback_used
    assert response.actions == []


@pytest.mark.asyncio
async def test_request_timeout_invalidates_late_action(tmp_path):
    backend = ControlledBackend(ignore_cancel=True)
    async with make_harness(tmp_path, backend=backend) as harness:
        with pytest.raises(TimeoutError):
            await harness.submit_message("drive", timeout_s=0.02)
        backend.release.set()
        await harness.bus.drain()
        assert harness.body.pose().x_mm == 0
        assert harness._requests == {}


@pytest.mark.asyncio
async def test_event_match_cache_is_bounded_and_invalidated():
    bus = EventBus()
    received = []
    for i in range(200):
        await bus.emit(str(i), "test")
    assert len(bus._match_cache) == 64
    sub = bus.subscribe("cached", lambda event: received.append(True), immediate=True)
    await bus.emit("cached", "test")
    bus.unsubscribe(sub)
    await bus.emit("cached", "test")
    assert received == [True]


@pytest.mark.asyncio
async def test_slow_motion_cannot_create_unbounded_execution_deadline(tmp_path):
    async with make_harness(tmp_path) as harness:
        assert harness.executor._timeout_for("drive", {"distance_mm": 2000, "speed_mm_s": 1e-9}) <= 30


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized", [False, True])
async def test_inference_transport_is_bounded_without_loading_a_model(oversized):
    def handler(request):
        payload = json.loads(request.content)
        assert payload["stream"] is False
        if oversized:
            return httpx.Response(200, content=b"x" * 70000)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"say":"hello"}'}}]})

    client = LlamaCppBackend(LLMConfig(auto_start_server=False))
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1")
    try:
        if oversized:
            with pytest.raises(ValueError, match="64 KiB"):
                await client.generate([])
        else:
            assert json.loads((await client.generate([])).raw_text)["say"] == "hello"
    finally:
        await client.close()
