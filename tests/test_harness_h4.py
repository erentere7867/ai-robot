"""Milestone H4 integration tests: 3D Simulation Sandbox, Head Camera VL Grounding, and Phonon Q8 TTS."""

from __future__ import annotations

import json
from pathlib import Path

from robot.actions.tools import ToolCallRequest
from robot.audio.tts import RobotTTSEngine
from robot.body.hal import BodyPose
from robot.cognition.backend import InferenceResult
from robot.core.clock import ReplayClock
from robot.core.config import RobotConfig
from robot.interfaces.dashboard import DashboardServer
from robot.main import RobotHarness
from robot.world.sandbox import SimulationSandbox


def _make_config(tmp_path: Path):
    cfg = RobotConfig.from_yaml()
    cfg.database_path = tmp_path / "memory.sqlite3"
    cfg.recordings_dir = str(tmp_path / "recordings")
    cfg.record_events = False
    cfg.llm.backend = "null"
    return cfg


def test_simulation_sandbox_3d_camera_and_object_grounding() -> None:
    sandbox = SimulationSandbox()
    pose0 = BodyPose(x_mm=0.0, y_mm=0.0, heading_deg=0.0, head_tilt_deg=0.0)

    ext0 = sandbox.compute_camera_extrinsics(pose0)
    assert ext0["x_mm"] == 67.0
    assert ext0["y_mm"] == 0.0
    assert ext0["z_mm"] == 286.0

    frame0 = sandbox.capture_camera_view(pose0)
    assert frame0.width == 320
    assert frame0.height == 240
    assert frame0.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert frame0.data_url.startswith("data:image/png;base64,")

    visible_ids = {d.id for d in frame0.visible_objects}
    assert "red_ball" in visible_ids
    assert "blue_cube" in visible_ids
    assert "green_mug" in visible_ids
    # Charging dock is behind the robot at x = -680 mm
    assert "charging_dock" not in visible_ids

    red_ball_det0 = next(d for d in frame0.visible_objects if d.id == "red_ball")
    assert 0.0 < red_ball_det0.u_norm < 0.5  # Red ball is at +y (front-left) -> left half of image
    assert 0.0 < red_ball_det0.v_norm < 1.0

    # Tilt head down -8 degrees: object moves up toward center of camera frame (smaller v_norm)
    pose_tilted = BodyPose(x_mm=0.0, y_mm=0.0, heading_deg=0.0, head_tilt_deg=-8.0)
    frame_tilted = sandbox.capture_camera_view(pose_tilted)
    red_ball_tilted = next(d for d in frame_tilted.visible_objects if d.id == "red_ball")
    assert red_ball_tilted.v_norm < red_ball_det0.v_norm

    # Spin 180 degrees: charging dock is now in front and visible in camera
    pose_back = BodyPose(x_mm=0.0, y_mm=0.0, heading_deg=180.0, head_tilt_deg=0.0)
    frame_back = sandbox.capture_camera_view(pose_back)
    back_ids = {d.id for d in frame_back.visible_objects}
    assert "charging_dock" in back_ids


async def test_sandbox_raycast_tof_and_cliff_sensor_integration(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    clock = ReplayClock()
    harness = RobotHarness(cfg, clock=clock)
    await harness.start()
    try:
        # Place the cardboard box directly in front of the robot bumper at x=310mm, y=0mm
        harness.sandbox.move_object("cardboard_box", 310.0, 0.0)
        tof_mm = harness.body.sensors().front_tof_mm
        assert tof_mm < 150.0

        # Attempting to drive forward 400mm should trigger the hardware ToF obstacle reflex
        res = await harness.executor.execute_call(
            ToolCallRequest("drive", {"direction": "forward", "distance_mm": 400.0})
        )
        assert res.status == "reflex_stopped"
        assert "obstacle" in res.message.lower()
    finally:
        await harness.stop()


async def test_vl_multimodal_grounding_and_spatial_navigation(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    clock = ReplayClock()
    class Decisions:
        name = "test_decisions"

        def __init__(self):
            self.calls = 0

        async def generate(self, messages, *, json_schema=None):
            self.calls += 1
            return InferenceResult(json.dumps({
                "say": "I see a Red Ball and a Blue Cube." if self.calls == 1 else "Approaching.",
                "sound": "none", "emotion": {"name": "neutral"}, "memory": [], "status": "completed",
                "actions": [] if self.calls == 1 else [{"tool": "drive", "args": {"distance_mm": 100}}],
            }), self.name)

    harness = RobotHarness(cfg, clock=clock, backend=Decisions())
    await harness.start()
    try:
        # 1. Visual query: "What do you see in front of you?"
        resp1 = await harness.submit_message("What do you see in front of you?")
        assert "Red Ball" in resp1["say"] or "Blue Cube" in resp1["say"]

        # 2. Spatial navigation query: "Go to the red ball"
        ball = harness.sandbox.get_object("red_ball")
        assert ball is not None
        initial_dist = (ball.x_mm**2 + ball.y_mm**2) ** 0.5

        resp2 = await harness.submit_message("Go to the red ball")
        await harness.bus.drain()
        pose_after = harness.body.pose()
        final_dist = ((ball.x_mm - pose_after.x_mm) ** 2 + (ball.y_mm - pose_after.y_mm) ** 2) ** 0.5

        assert final_dist < initial_dist
        assert resp2.get("camera_verification") is not None
        assert "Camera:" in resp2["camera_verification"]["after_summary"]
    finally:
        await harness.stop()


def test_phonon_q8_gguf_tts_synthesis_and_mouth_envelope() -> None:
    tts = RobotTTSEngine(mode="phonon_q8")
    speech = tts.synthesize("Hi! I can see the red ball and the blue cube in front of me!")
    assert speech.sample_rate == 24000
    assert speech.duration_s > 0.5
    assert len(speech.mouth_envelope) > 20
    assert max(speech.mouth_envelope) > 0.2
    assert speech.backend_name == "phonon_q8_gguf"


async def test_dashboard_sandbox_api_endpoints(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    clock = ReplayClock()
    harness = RobotHarness(cfg, clock=clock)
    await harness.start()
    dash = DashboardServer(harness, port=0)
    await dash.start()
    try:
        status, _, state_bytes = await dash._route_request("GET", "/api/state", {}, {})
        assert status == "200 OK"
        st = json.loads(state_bytes.decode("utf-8"))
        assert "sandbox" in st
        assert st["sandbox"]["camera"]["data_url"].startswith("data:image/png;base64,")
        assert len(st["sandbox"]["objects"]) >= 6

        # Move object via API
        status, _, move_bytes = await dash._route_request(
            "POST",
            "/api/sandbox",
            {},
            {"op": "move", "id": "red_ball", "x_mm": 500.0, "y_mm": -100.0},
        )
        assert status == "200 OK"
        moved_state = json.loads(move_bytes.decode("utf-8"))
        rb = next(o for o in moved_state["objects"] if o["id"] == "red_ball")
        assert rb["x_mm"] == 500.0
        assert rb["y_mm"] == -100.0

        # Add object via API
        status, _, add_bytes = await dash._route_request(
            "POST",
            "/api/sandbox",
            {},
            {"op": "add", "label": "Purple Ball", "shape": "sphere", "x_mm": 600.0, "y_mm": 120.0},
        )
        assert status == "200 OK"
        added_state = json.loads(add_bytes.decode("utf-8"))
        assert any(o["label"] == "Purple Ball" for o in added_state["objects"])
    finally:
        await dash.stop()
        await harness.stop()
