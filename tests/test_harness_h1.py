"""Automated test suite for Milestone H1 (Text Brain on a Simulated Body)."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot.behavior.fsm import InteractionState
from robot.body.hal import BodyCapabilities, BodyCommand
from robot.body.sim_body import SimBody
from robot.cognition.backend import InferenceResult
from robot.core.bus import EventBus
from robot.core.clock import ReplayClock
from robot.core.config import RobotConfig
from robot.core.events import Event, EventPriority
from robot.core.recorder import EventRecorder, EventReplayer
from robot.main import RobotHarness
from robot.memory.models import MemoryProvenance


class FaultyBackend:
    """Fault-injection backend returning malformed/broken model outputs."""

    name = "faulty_backend"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self._idx = 0

    async def is_available(self) -> bool:
        return True

    async def generate(self, messages, *, json_schema=None) -> InferenceResult:
        out = self.outputs[min(self._idx, len(self.outputs) - 1)]
        self._idx += 1
        return InferenceResult(raw_text=out, backend_name=self.name, latency_ms=2.0)


@pytest.mark.asyncio
async def test_event_bus_priorities_and_drop_policy() -> None:
    clock = ReplayClock()
    bus = EventBus(clock=clock, default_queue_size=3)
    received: list[str] = []

    bus.subscribe("sensor.*", lambda e: received.append(f"{e.type}:{e.payload['v']}"), max_queue_size=2)
    # Publish 4 telemetry events before bus starts -> oldest 2 should be dropped cleanly
    for i in range(4):
        await bus.emit("sensor.imu", source="test", payload={"v": i}, priority=EventPriority.TELEMETRY)

    await bus.start()
    await bus.drain()
    await bus.stop()

    assert received == ["sensor.imu:2", "sensor.imu:3"]
    assert bus.dropped_count == 2


@pytest.mark.asyncio
async def test_validator_and_head_tilt_limit_clamping(tmp_path: Path) -> None:
    cfg = RobotConfig.from_yaml()
    cfg.database_path = tmp_path / "mem.db"
    cfg.record_events = False
    cfg.llm.backend = "null"

    harness = RobotHarness(cfg, clock=ReplayClock())
    await harness.start()
    try:
        # "Tilt your head up 200 degrees" -> clamped to +35° physical limit and explained
        resp = await harness.arbiter.handle_user_turn("Tilt your head up 200 degrees")
        assert harness.body.pose().head_tilt_deg == 35.0
        assert any("clamped" in w.lower() or "35" in w for w in resp.warnings)
        assert "35" in resp.say or "limit" in resp.say.lower() or "clamped" in resp.say.lower()
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_malformed_model_output_never_crashes(tmp_path: Path) -> None:
    cfg = RobotConfig.from_yaml()
    cfg.database_path = tmp_path / "mem.db"
    cfg.record_events = False

    bad_outputs = [
        "NOT JSON AT ALL {{{",
        '{"say": "hi", "sound": "explosion_noise", "emotion": {"name": "rage", "intensity": 99}, '
        '"actions": [{"tool": "raw_gpio_write", "args": {"pin": 17}}], "memory": "bad"}',
        "",
    ]
    harness = RobotHarness(cfg, clock=ReplayClock(), backend=FaultyBackend(bad_outputs))
    await harness.start()
    try:
        r1 = await harness.arbiter.handle_user_turn("Hello?")
        assert r1.fallback_used is True
        assert r1.emotion.name == "confused"

        r2 = await harness.arbiter.handle_user_turn("Try bad tools and emotions")
        assert r2.fallback_used is False
        assert r2.sound == "none"
        assert r2.emotion.name == "neutral"
        assert r2.emotion.intensity == 1.0
        assert len(r2.actions) == 0
        assert len(r2.rejected_actions) == 1
        assert r2.rejected_actions[0]["tool"] == "raw_gpio_write"

        r3 = await harness.arbiter.handle_user_turn("Empty output test")
        assert r3.fallback_used is True
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_memory_persistence_restart_and_forget_tombstone(tmp_path: Path) -> None:
    db_file = tmp_path / "persistent_memory.db"
    clock = ReplayClock()

    # Session 1: Remember favorite color is green
    cfg1 = RobotConfig.from_yaml()
    cfg1.database_path = db_file
    cfg1.record_events = False
    cfg1.llm.backend = "null"

    h1 = RobotHarness(cfg1, clock=clock)
    await h1.start()
    await h1.arbiter.handle_user_turn("Remember my favorite color is green")
    await h1.stop()

    # Session 2 (after restart): Recall favorite color, then forget it
    clock.advance(10.0)
    cfg2 = RobotConfig.from_yaml()
    cfg2.database_path = db_file
    cfg2.record_events = False
    cfg2.llm.backend = "null"

    h2 = RobotHarness(cfg2, clock=clock)
    await h2.start()
    recall_resp = await h2.arbiter.handle_user_turn("What's my favorite color?")
    assert "green" in recall_resp.say.lower()

    # Now forget it
    await h2.arbiter.handle_user_turn("Forget my favorite color")
    assert len(h2.memory.list_active()) == 0
    assert h2.memory.is_tombstoned("favorite color")

    # Verify model-proposed inference cannot silently re-add a tombstoned topic
    blocked = h2.memory.remember(
        "User's favorite color is green",
        subject="favorite color",
        provenance=MemoryProvenance.MODEL_PROPOSED,
    )
    assert blocked is None
    await h2.stop()

    # Session 3 (after second restart): Verify forgotten memory does not come back
    clock.advance(10.0)
    h3 = RobotHarness(cfg2, clock=clock)
    await h3.start()
    post_forget_resp = await h3.arbiter.handle_user_turn("What's my favorite color?")
    assert "green" not in post_forget_resp.say.lower()
    await h3.stop()


@pytest.mark.asyncio
async def test_spatial_short_term_memory_what_was_behind_me(tmp_path: Path) -> None:
    clock = ReplayClock()
    cfg = RobotConfig.from_yaml()
    cfg.database_path = tmp_path / "spatial.db"
    cfg.record_events = False
    cfg.llm.backend = "null"

    harness = RobotHarness(cfg, clock=clock)
    await harness.start()
    try:
        # 1. Observe a blue backpack 900mm directly in front (bearing 0°)
        harness.world.record_ego_observation(
            "obs_1",
            "blue backpack",
            distance_mm=900.0,
            relative_bearing_deg=0.0,
            ttl_s=60.0,
        )
        obs_initial = harness.world.active_observations()
        assert obs_initial[0]["direction"] == "front"

        # 2. Robot spins 180 degrees
        await harness.arbiter.handle_user_turn("Spin 180 degrees")
        assert abs(abs(harness.body.pose().heading_deg) - 180.0) < 1e-3

        # 3. Because the observation is anchored in the world frame, it is now BEHIND the robot!
        obs_after_spin = harness.world.active_observations()
        assert obs_after_spin[0]["direction"] == "behind"

        resp = await harness.arbiter.handle_user_turn("What was behind you?")
        assert "behind" in resp.say.lower()
        assert "blue backpack" in resp.say.lower()
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_sim_body_obstacle_and_cliff_reflexes() -> None:
    clock = ReplayClock()
    bus = EventBus(clock=clock)
    body = SimBody(bus, BodyCapabilities())
    reflex_events: list[Event] = []
    bus.subscribe("reflex.triggered", lambda e: reflex_events.append(e))
    await bus.start()
    await body.start()
    try:
        # Inject close front obstacle (80mm <= 120mm threshold)
        body.set_simulated_obstacle(front_tof_mm=80.0)
        res = await body.execute(
            BodyCommand(id="c1", tool="drive", args={"direction": "forward", "distance_mm": 300.0})
        )
        await bus.drain()
        assert res.ok is False
        assert res.status == "reflex_stopped"
        assert len(reflex_events) == 1
        assert reflex_events[0].payload["reflex"] == "obstacle_stop"
    finally:
        await body.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_fsm_follow_up_emotion_decay_and_idle_sleepy(tmp_path: Path) -> None:
    clock = ReplayClock()
    cfg = RobotConfig.from_yaml()
    cfg.database_path = tmp_path / "fsm.db"
    cfg.record_events = False
    cfg.llm.backend = "null"

    harness = RobotHarness(cfg, clock=clock)
    await harness.start()
    try:
        resp = await harness.arbiter.handle_user_turn("Do a dance!")
        assert resp.emotion.name == "excited"
        assert harness.fsm.state == InteractionState.FOLLOW_UP
        assert harness.fsm.in_follow_up_window() is True

        # Advance past follow-up window (8s) -> IDLE, emotion decays toward neutral
        clock.advance(10.0)
        await harness.arbiter.step_idle_timers()
        assert harness.fsm.state == InteractionState.IDLE
        assert harness.emotion.snapshot().emotion == "neutral"

        # Idle expression changes do not inject a physical action without a model decision
        clock.advance(40.0)
        ev = await harness.arbiter.step_idle_timers()
        await harness.bus.drain()
        assert ev == "became_sleepy"
        assert harness.emotion.snapshot().emotion == "sleepy"
        assert harness.body.pose().head_tilt_deg == 5.0
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_deterministic_event_recording_and_replay(tmp_path: Path) -> None:
    clock1 = ReplayClock()
    cfg1 = RobotConfig.from_yaml()
    cfg1.database_path = tmp_path / "rec1.db"
    cfg1.recordings_dir = tmp_path / "recordings"
    cfg1.record_events = True
    cfg1.llm.backend = "null"

    h1 = RobotHarness(cfg1, clock=clock1)
    console1 = h1.bus
    await h1.start()
    assert h1.recorder is not None
    rec_path = h1.recorder.file_path

    await console1.emit(
        "input.user_message", source="console", payload={"text": "Do a dance", "direct": True}
    )
    await console1.drain()
    clock1.advance(2.0)
    await console1.emit(
        "input.user_message",
        source="console",
        payload={"text": "Tilt your head up 20 degrees", "direct": True},
    )
    await console1.drain()
    pose_run1 = h1.body.pose().to_dict()
    await h1.stop()

    # Now replay the recorded user messages into a fresh harness and verify identical final pose
    recorded_events = EventRecorder.load_events(rec_path)
    user_events = [e for e in recorded_events if e.type == "input.user_message"]
    assert len(user_events) == 2

    clock2 = ReplayClock()
    cfg2 = RobotConfig.from_yaml()
    cfg2.database_path = tmp_path / "rec2.db"
    cfg2.record_events = False
    cfg2.llm.backend = "null"

    h2 = RobotHarness(cfg2, clock=clock2)
    await h2.start()
    replayer = EventReplayer(h2.bus, clock2)
    await replayer.replay(user_events)
    pose_run2 = h2.body.pose().to_dict()
    await h2.stop()

    assert pose_run1 == pose_run2
