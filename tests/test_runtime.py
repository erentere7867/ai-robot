"""The running robot as a whole: cancellation, request ownership, bounded resources, the dashboard."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from robot_fakes import ScriptedBackend

from robot.audio.service import AudioService
from robot.audio.tts import SynthesizedSpeech
from robot.audio.vad import VoiceActivityDetector
from robot.body.hal import load_capabilities
from robot.cognition.backend import ChatResult, NullBackend, ToolCall
from robot.cognition.llamacpp import LlamaCppBackend
from robot.cognition.prompts import PromptBuilder, estimate_tokens, truncate_to_tokens
from robot.cognition.textcalls import recover_text_tool_calls
from robot.core.bus import EventBus
from robot.core.clock import RealClock, ReplayClock
from robot.core.config import PROJECT_ROOT, AgentConfig, RobotConfig
from robot.core.events import Event, EventPriority
from robot.core.recorder import EventRecorder
from robot.main import RobotHarness
from robot.skills.base import SkillError, _coerce
from robot.world.spatial_map import TrackedObject


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
    harness.sandbox.remove_object("person")
    return harness


class ControlledBackend:
    """Blocks the first decision of each turn until released, then proposes `calls`; answers after results."""

    name = "controlled"

    def __init__(self, calls=None, ignore_cancel=False):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = calls or [("move", {"direction": "forward", "distance_cm": 30})]
        self.ignore_cancel = ignore_cancel

    async def chat(self, messages, *, tools=None, max_tokens=None, tool_choice="auto"):
        if messages and messages[-1].get("role") == "tool":
            return ChatResult("Done.", backend_name=self.name)
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            if not self.ignore_cancel:
                raise
            await self.release.wait()
        return ChatResult("", [ToolCall(f"c{i}", name, args) for i, (name, args) in enumerate(self.calls)],
                          backend_name=self.name)


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
        assert harness.fast.behavior is None


@pytest.mark.asyncio
async def test_cancel_cannot_run_remaining_plan(tmp_path):
    backend = ControlledBackend(
        calls=[("move", {"direction": "forward", "distance_cm": 30}), ("turn", {"direction": "left"})]
    )
    backend.release.set()
    async with make_harness(tmp_path, backend=backend, real=True) as harness:
        turn = asyncio.create_task(harness.arbiter.handle_user_turn("move then spin"))
        for _ in range(100):
            if harness.body.pose().linear_vel_mm_s:
                break
            await asyncio.sleep(0.002)
        await harness.bus.emit("action.cancel", "test", {"reason": "user_stop"}, priority=EventPriority.CRITICAL)
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
        assert harness.body.pose().x_mm == pytest.approx(300, abs=15)


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


def test_a_non_finite_speed_command_is_no_command():
    caps = load_capabilities()
    assert caps.clamp_velocity(float("nan"), float("inf")) == (0.0, 0.0)
    assert caps.clamp_velocity(9999.0, -9999.0) == (caps.max_linear_speed_mm_s, -caps.max_angular_speed_deg_s)


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
async def test_failing_critical_observer_cannot_prevent_other_safety_handlers():
    bus = EventBus(default_queue_size=1)
    stopped = []

    def failing(event):
        raise RuntimeError("broken observer")

    bus.subscribe("action.cancel", failing, immediate=True)
    bus.subscribe("action.cancel", lambda event: stopped.append(True), immediate=True)
    await bus.emit("action.cancel", "test", priority=EventPriority.CRITICAL)
    assert stopped == [True]


def test_corrupt_deeply_nested_model_json_never_escapes_parsers():
    nested = '{"x":' + "[" * 4000 + "0" + "]" * 4000 + "}"
    calls = LlamaCppBackend.parse_tool_calls([{"id": "a", "function": {"name": "move", "arguments": nested}}])
    assert calls[0].name == "move" and (calls[0].invalid_arguments or "x" in calls[0].arguments)
    assert recover_text_tool_calls(nested, {"move"}) == []
    assert recover_text_tool_calls("move(" * 2000 + ")" * 2000, {"move"}) == []


def test_context_and_retained_turns_are_bounded_for_multibyte_input():
    prompts = PromptBuilder(Path("config/SOUL.md"))
    for _ in range(100):
        prompts.add_exchange("世界" * 5000, "世界" * 5000)
    messages = prompts.build("hello", prompts.situation(world="Known objects: none yet.", core_facts=[],
                                                        memories=[], recent=[]), background=False)
    history = messages[1:-1]
    assert len(prompts.history) <= prompts.budget.max_recent_turns
    assert sum(estimate_tokens(m["content"]) + 6 for m in history) <= prompts.budget.recent_turns_tokens
    assert "Safety always wins" in messages[0]["content"]
    assert truncate_to_tokens("hello", 0) == ""


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


def test_ventuno_profile_extends_the_base_and_model_numbers_are_validated():
    cfg = RobotConfig.from_yaml("config/robot.ventuno.yaml")
    assert cfg.body_backend == "ventuno" and cfg.perception.loop_hz == RobotConfig.from_yaml().perception.loop_hz
    # Model-supplied numbers are validated before any skill runs: NaN never reaches a motor.
    with pytest.raises(SkillError):
        _coerce("degrees", {"type": "number"}, "NaN")
    assert _coerce("degrees", {"type": "number", "minimum": 5, "maximum": 360}, 9999) == 360


@pytest.mark.asyncio
async def test_malformed_model_output_never_crashes(tmp_path: Path) -> None:
    cfg = RobotConfig.from_yaml()
    cfg.database_path = tmp_path / "mem.db"
    cfg.record_events = False
    backend = ScriptedBackend([
        "NOT JSON AT ALL {{{",
        ChatResult("", [ToolCall("a", "raw_gpio_write", {"pin": 17}),
                        ToolCall("b", "move", {}, invalid_arguments=True)]),
        "I could not do that.",
        ChatResult("", [ToolCall("c", "turn", {"direction": "sideways", "degrees": "lots"})]),
        "",
    ])
    harness = RobotHarness(cfg, clock=ReplayClock(), backend=backend)
    await harness.start()
    try:
        r1 = await harness.arbiter.handle_user_turn("Hello?")
        assert r1.status == "completed" and r1.say == "NOT JSON AT ALL {{{"

        r2 = await harness.arbiter.handle_user_turn("Try bad tools")
        assert [entry["ok"] for entry in r2.tool_log] == [False]  # unknown tool rejected, the rest skipped
        assert "no tool named" in r2.tool_log[0]["message"]

        r3 = await harness.arbiter.handle_user_turn("Bad arguments")
        assert r3.tool_log[0]["ok"] is False
        assert harness.body.pose().x_mm == 0 and harness.body.pose().heading_deg == 0
    finally:
        await harness.stop()


async def test_subscriber_survives_handler_awaiting_a_cancelled_future():
    bus = EventBus()
    seen = []

    async def handler(event):
        seen.append(event.payload["n"])
        if event.payload["n"] == 0:
            future = asyncio.get_running_loop().create_future()
            future.cancel()
            await future

    bus.subscribe("work", handler, name="worker")
    await bus.start()
    try:
        await bus.emit("work", "test", {"n": 0})
        await bus.emit("work", "test", {"n": 1})
        await bus.drain()
        assert seen == [0, 1]
        assert bus.stats()["subscribers"]["worker"]["errors"] == 1
    finally:
        await bus.stop()


def test_config_tolerates_empty_sections_and_rejects_ambiguous_values(tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text(
        'robot:\n  wake_words: "sprocket"\nruntime:\nagent:\nautonomy:\naudio:\n'
        'bus:\n  record_events: false\nllm:\n  backend: "null"\n',
        encoding="utf-8",
    )
    cfg = RobotConfig.from_yaml(path, project_root=PROJECT_ROOT)
    assert cfg.wake_words == ["sprocket"]
    assert cfg.record_events is False
    assert cfg.agent.max_steps == AgentConfig().max_steps and cfg.autonomy.enabled is True

    path.write_text('llm:\n  auto_start_server: "false"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="auto_start_server"):
        RobotConfig.from_yaml(path, project_root=PROJECT_ROOT)
    path.write_text("agent: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="agent"):
        RobotConfig.from_yaml(path, project_root=PROJECT_ROOT)


def test_an_unknown_setting_is_an_error_not_silently_ignored(tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text("perception:\n  lidarr: sim\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lidarr"):
        RobotConfig.from_yaml(path, project_root=PROJECT_ROOT)


def test_the_real_robot_runs_without_the_simulator():
    """MuJoCo is only for the laptop: nothing the robot imports may need it."""
    code = ("import sys; sys.modules['mujoco'] = None\n"
            "import robot.main, robot.hardware.platform, robot.hardware.ventuno_body\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-2000:]


async def test_a_stop_with_any_configured_wake_word_halts_before_the_model(tmp_path):
    cfg = RobotConfig.from_yaml()
    cfg.database_path, cfg.spatial_map_path = tmp_path / "m.db", tmp_path / "map.json"
    cfg.record_events, cfg.llm.backend, cfg.perception.fast_detector = False, "null", "off"
    cfg.wake_words = ["jarvis"]
    async with RobotHarness(cfg, clock=ReplayClock(), backend=NullBackend(), tts_mode="warble") as harness:
        harness.face_bridge._send_udp = lambda payload: None
        assert (await harness.submit_message("jarvis, stop!"))["backend"] == "reflex_stop"


def test_vad_utterance_segmentation_and_barge_in() -> None:
    """Verify VAD pre-roll buffering, speech start/end segmentation, and echo-aware barge-in."""
    vad = VoiceActivityDetector(
        sample_rate=16000,
        frame_ms=30,
        start_rms_threshold=0.02,
        barge_in_rms_threshold=0.06,
        min_speech_frames=3,
        silence_hangover_frames=5,
        pre_roll_frames=6,
    )
    frame_len = vad.frame_samples
    silence = np.zeros(frame_len, dtype=np.float32)
    speech_frame = (0.12 * np.sin(np.linspace(0, 20, frame_len))).astype(np.float32)

    events = []
    for _ in range(4):
        events.extend(vad.process_frame(silence))
    for _ in range(6):
        events.extend(vad.process_frame(speech_frame))
    for _ in range(7):
        events.extend(vad.process_frame(silence))

    kinds = [e.kind for e in events]
    assert "speech_start" in kinds
    assert "speech_end" in kinds
    end_ev = next(e for e in events if e.kind == "speech_end")
    assert end_ev.audio_samples is not None
    # Pre-roll + 6 speech frames + 5 hangover frames = at least 14 frames
    assert len(end_ev.audio_samples) >= 14 * frame_len

    # Now test barge-in while robot_is_speaking=True
    barge_events = []
    for _ in range(4):
        barge_events.extend(vad.process_frame(speech_frame, robot_is_speaking=True))
    assert any(e.kind == "barge_in" for e in barge_events)


async def test_stop_interrupts_real_playback_and_discards_pending_sentence():
    bus = EventBus()
    audio = AudioService(bus, enable_speaker=False, realtime_sync=True)
    audio.tts.synthesize = lambda text: SynthesizedSpeech(text, np.zeros(24000, dtype=np.float32), 24000, [0.5] * 60)
    spoken = []
    started = asyncio.Event()

    def on_started(event):
        spoken.append(event.payload["text"])
        started.set()

    bus.subscribe("audio.speech_started", on_started, immediate=True)
    await bus.start()
    await audio.start()
    try:
        await bus.emit("robot.response", "test", {"say": "First"})
        await asyncio.wait_for(started.wait(), 1)
        await bus.emit("robot.response", "test", {"say": "Must not play"})
        await asyncio.wait_for(bus.emit("action.cancel", "test", {"reason": "user_stop_command"},
                                       priority=EventPriority.CRITICAL), 1)
        assert spoken == ["First"]
        assert not audio.is_speaking
        assert audio._response_queue.empty()
        assert audio._speak_task is None
    finally:
        await audio.stop()
        await bus.stop()


# ---- the dashboard's views of the robot's own map and reasoning ----


def _dash_config(tmp_path: Path) -> RobotConfig:
    cfg = RobotConfig.from_yaml()
    cfg.database_path = tmp_path / "memory.sqlite3"
    cfg.spatial_map_path = tmp_path / "spatial_map.json"
    cfg.recordings_dir = tmp_path / "recordings"
    cfg.record_events = False
    cfg.llm.backend = "null"
    cfg.autonomy.enabled = False
    cfg.perception.fast_detector = "off"  # this test covers the VLM detection path
    return cfg


async def _get_json(dash, path: str, query: dict[str, list[str]] | None = None):
    status, content_type, data = await dash._route_request("GET", path, query or {}, {})
    assert status == "200 OK", (path, status, data[:200])
    assert content_type == "application/json"
    return json.loads(data)


@pytest.fixture
async def dash_harness(tmp_path: Path):
    h = RobotHarness(_dash_config(tmp_path), clock=ReplayClock(), enable_dashboard=True, dashboard_port=0)
    await h.start()
    try:
        yield h
    finally:
        await h.stop()


async def test_map_endpoint_shows_the_robots_own_map_and_optional_sim_truth(dash_harness) -> None:
    harness = dash_harness
    dash = harness.dashboard
    smap = harness.spatial_map
    now = harness.clock.monotonic()
    smap.objects["box_1"] = TrackedObject("box_1", "brown cardboard box", 950.0, 420.0, 120.0, now, now)
    smap.objects["ball_1"] = TrackedObject("ball_1", "red ball", 680.0, 160.0, 50.0, now, now, stale=True)
    smap.grid.add_obstacle_point(1200.0, -300.0, weight=2.0)
    smap.grid.mark_coverage(harness.body.pose(), harness.camera.model)

    data = await _get_json(dash, "/api/map")
    assert data["robot"] == {"x_mm": 0.0, "y_mm": 0.0, "heading_deg": 0.0, "head_tilt_deg": 0.0}
    objects = {o["id"]: o for o in data["objects"]}
    assert objects["box_1"]["label"] == "brown cardboard box"
    assert objects["box_1"]["x_mm"] == 950.0 and objects["box_1"]["radius_mm"] == 120.0
    assert objects["box_1"]["distance_mm"] == round((950.0**2 + 420.0**2) ** 0.5)
    assert objects["box_1"]["age_s"] == 0.0 and objects["ball_1"]["stale"] is True
    assert data["places"]["start"] == [0, 0]
    assert any(abs(x - 1200) <= 50 and abs(y + 300) <= 50 for x, y in data["occupied_cells"])
    grid = data["coverage_grid"]
    assert len(grid["bits"]) == grid["n"] ** 2 and set(grid["bits"]) == {"0", "1"}
    assert grid["origin_mm"] == -data["half_extent_mm"] and grid["cell_mm"] == 2 * data["cell_mm"]
    # The cell 1 m straight ahead was looked at; the one 1 m behind was not.
    i_ahead = int((1000.0 - grid["origin_mm"]) // grid["cell_mm"])
    i_behind = int((-1000.0 - grid["origin_mm"]) // grid["cell_mm"])
    j_center = int((0.0 - grid["origin_mm"]) // grid["cell_mm"])
    assert grid["bits"][i_ahead * grid["n"] + j_center] == "1"
    assert grid["bits"][i_behind * grid["n"] + j_center] == "0"
    assert 0.0 < data["coverage"] < 1.0
    assert data["camera"]["hfov_deg"] > 30
    assert "box_1" in data["prompt_view"]
    assert "sim_truth" not in data  # ground truth only on request

    truth = (await _get_json(dash, "/api/map", {"truth": ["1"]}))["sim_truth"]
    assert truth["room_half_extent_mm"] == harness.sandbox.room_half_extent_mm
    sim = {o["id"]: o for o in truth["objects"]}
    assert (sim["cardboard_box"]["x_mm"], sim["cardboard_box"]["y_mm"]) == (950.0, 420.0)
    assert {"label", "shape", "width_mm", "depth_mm", "color"} <= set(sim["cardboard_box"])
    assert truth["robot"]["x_mm"] == 0.0


async def test_thinking_stream_records_a_null_backend_turn(dash_harness) -> None:
    harness = dash_harness
    dash = harness.dashboard
    empty = await _get_json(dash, "/api/thinking")
    assert empty["entries"] == [] and empty["view"] is None and empty["goal"] is None
    assert empty["busy"] is False and empty["backend"] == "null_offline"
    status, _, _ = await dash._route_request("GET", "/api/thinking/frame", {}, {})
    assert status == "404 Not Found"

    reply = json.loads((await dash._route_request("POST", "/api/message", {}, {"text": "hello robot"}))[2])
    assert reply["status"] == "completed" and reply["final"] is True

    data = await _get_json(dash, "/api/thinking")
    kinds = [e["type"] for e in data["entries"]]
    assert kinds[0] == "turn.started"
    assert "cognition.model_call" in kinds and kinds[-1] == "robot.response"
    turn_ids = {e["turn"] for e in data["entries"]}
    assert len(turn_ids) == 1 and None not in turn_ids
    header = data["entries"][0]
    assert header["origin"] == "user" and header["objective"] == "hello robot" and header["trigger"] is None
    call = next(e for e in data["entries"] if e["type"] == "cognition.model_call")
    assert call["step"] == 1 and call["tool_calls"] == [] and "language model" in call["content"]
    final = data["entries"][-1]
    assert final["status"] == "completed" and final["steps"] == 1 and final["origin"] == "user"
    assert final["say"] == reply["say"] and final["backend"] == "null_offline"
    assert data["goal"]["objective"] == "hello robot" and data["goal"]["status"] == "completed"
    assert [e["seq"] for e in data["entries"]] == sorted(e["seq"] for e in data["entries"])

    # Incremental polling: nothing new since the latest sequence number.
    latest = data["latest_seq"]
    again = await _get_json(dash, "/api/thinking", {"since": [str(latest)]})
    assert again["entries"] == [] and again["reset"] is False
    # A page left open across a server restart asks for a future sequence number: send everything.
    restarted = await _get_json(dash, "/api/thinking", {"since": [str(latest + 1000)]})
    assert restarted["reset"] is True and len(restarted["entries"]) == len(data["entries"])
