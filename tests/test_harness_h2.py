"""Automated test suite for Milestone H2 (Face Process, Web Dashboard, Idle Memory Consolidation)."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from robot.core.clock import ReplayClock
from robot.core.config import RobotConfig
from robot.face.animator import FaceAnimator
from robot.face.renderer import PygameFaceRenderer
from robot.main import RobotHarness


def test_face_animator_interpolation_blink_and_themes() -> None:
    anim = FaceAnimator(blink_period_s=1.0)
    anim.set_state(activity="speaking", emotion="curious", intensity=0.9, gaze_target="left", theme="emerald")

    frame = anim.step(0.2)
    assert frame.theme == "green"
    assert frame.color_rgb == (46, 235, 138)
    # Curious has asymmetric eyes (left wider than right) and speaking opens mouth
    for _ in range(10):
        frame = anim.step(0.05)
    assert frame.left_openness > frame.right_openness
    assert frame.gaze_x < -0.2
    assert frame.mouth_open > 0.1

    # Trigger blink and verify eye closure during blink envelope
    anim.trigger_blink()
    blink_frame = anim.step(0.08)
    assert blink_frame.blink_amount > 0.5
    assert blink_frame.left_openness < 0.5


def test_pygame_face_renderer_headless_snapshot(tmp_path: Path) -> None:
    anim = FaceAnimator()
    renderer = PygameFaceRenderer(800, 480, headless=True)
    renderer.init_surface()

    for emo in ("neutral", "happy", "curious", "surprised", "sleepy"):
        anim.set_state(emotion=emo, intensity=0.85, activity="speaking" if emo == "happy" else "idle")
        params = anim.step(0.1)
        surf = renderer.render(params)
        assert surf.get_size() == (800, 480)

    snap_file = renderer.save_snapshot(tmp_path / "face_test.bmp")
    assert snap_file.exists()
    assert snap_file.stat().st_size > 10_000


@pytest.mark.asyncio
async def test_idle_memory_consolidation_add_update_noop_and_tombstone(tmp_path: Path) -> None:
    clock = ReplayClock()
    cfg = RobotConfig.from_yaml()
    cfg.database_path = tmp_path / "consolidate.db"
    cfg.record_events = False
    cfg.llm.backend = "null"

    harness = RobotHarness(cfg, clock=clock)
    await harness.start()
    try:
        # 1. Implicit statement during conversation -> queued for idle consolidation
        await harness.arbiter.handle_user_turn("My cat's name is Mochi and I prefer green eye color.")
        assert harness.consolidator.pending_count == 1

        # Advance past follow-up window (8s) so robot enters IDLE and runs consolidation automatically
        clock.advance(10.0)
        await harness.arbiter.step_idle_timers()
        await harness.bus.drain()

        rep1 = harness.consolidator.last_report
        assert rep1.added == 2
        active = {m.subject: m.content for m in harness.memory.list_active()}
        assert "cat's name" in active
        assert "Mochi" in active["cat's name"]
        # Eye color memory also automatically updates the face theme to green!
        assert harness.face_bridge.current_frame.theme == "green"

        # 2. Duplicate statement -> NOOP
        harness.consolidator.enqueue_turn("My cat's name is Mochi.", "Meow!")
        rep2 = await harness.consolidator.consolidate()
        assert rep2.noop == 1
        assert rep2.added == 0

        # 3. Updated fact on same subject -> UPDATE (supersedes Mochi)
        harness.consolidator.enqueue_turn("My cat's name is Pumpkin.", "Ooh, Pumpkin!")
        rep3 = await harness.consolidator.consolidate()
        assert rep3.updated == 1
        cat_mems = [m for m in harness.memory.list_active() if m.subject == "cat's name"]
        assert len(cat_mems) == 1
        assert "Pumpkin" in cat_mems[0].content
        assert cat_mems[0].supersedes is not None

        # 4. User forgets cat's name -> Tombstone blocks future implicit consolidation
        harness.memory.forget("my cat's name")
        harness.consolidator.enqueue_turn("My cat's name is Luna.", "Cute!")
        rep4 = await harness.consolidator.consolidate()
        assert rep4.blocked_by_tombstone == 1
        assert all(m.subject != "cat's name" for m in harness.memory.list_active())
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_web_dashboard_server_endpoints(tmp_path: Path) -> None:
    clock = ReplayClock()
    cfg = RobotConfig.from_yaml()
    cfg.database_path = tmp_path / "dash.db"
    cfg.record_events = False
    cfg.llm.backend = "null"

    # Use port 0 so OS assigns an ephemeral free port
    harness = RobotHarness(cfg, clock=clock, enable_dashboard=True, dashboard_port=0)
    await harness.start()
    try:
        assert harness.dashboard is not None
        base_url = harness.dashboard.url
        async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
            # 1. HTML index
            r_html = await client.get("/")
            assert r_html.status_code == 200
            assert "SPROCKET // EMBODIED AI HARNESS DASHBOARD" in r_html.text

            # 2. Send chat message via REST API
            r_msg = await client.post("/api/message", json={"text": "Do a dance!"})
            assert r_msg.status_code == 200
            msg_data = r_msg.json()
            assert msg_data["say"] and msg_data["final"] is True  # null backend talks, never moves

            # 3. Change face expression and theme via REST API
            r_expr = await client.post(
                "/api/expression",
                json={"emotion": "curious", "intensity": 0.9, "gaze_target": "right", "theme": "amber"},
            )
            assert r_expr.status_code == 200
            expr_data = r_expr.json()
            assert expr_data["theme"] == "amber"
            assert expr_data["emotion"] == "curious"

            # 4. Inject spatial observation & check /api/state
            r_obs = await client.post(
                "/api/observe",
                json={"label": "red ball", "distance_mm": 650, "relative_bearing_deg": 0},
            )
            assert r_obs.status_code == 200
            assert r_obs.json()["observations"][0]["label"] == "red ball"

            # 5. Check full state, capability registry, and event stream
            r_state = await client.get("/api/state")
            assert r_state.status_code == 200
            state = r_state.json()
            assert state["face"]["theme"] == "amber"
            assert len(state["recent_conversations"]) == 1
            assert "capabilities" in state
            assert "sad" in state["capabilities"]["allowed_emotions"]
            assert "angry" in state["capabilities"]["allowed_emotions"]

            # 6. A model-chosen expression drives face, sound and body through the express skill.
            from robot.cognition.backend import ScriptedBackend

            harness.agent.backend = ScriptedBackend([
                {"call": "express", "args": {"emotion": "sad", "gesture": "sleepy_droop"}},
                "Here is my sad face.",
            ])
            r_sad = await client.post("/api/message", json={"text": "try making a sad face"})
            assert r_sad.status_code == 200
            sad_data = r_sad.json()
            assert sad_data["emotion"]["name"] == "sad"
            assert sad_data["sound"] == "sad_whimper"
            assert harness.body.pose().head_tilt_deg == -18.0

            r_events = await client.get("/api/events")
            assert r_events.status_code == 200
            assert len(r_events.json()["events"]) > 0
    finally:
        await harness.stop()
