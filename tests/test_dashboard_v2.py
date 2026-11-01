"""Dashboard "why" views: spatial map, thinking stream, drives & autonomy (plus the legacy endpoints)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from robot.cognition.backend import ScriptedBackend
from robot.core.clock import ReplayClock
from robot.core.config import RobotConfig
from robot.main import RobotHarness
from robot.world.spatial_map import TrackedObject


def _config(tmp_path: Path) -> RobotConfig:
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
async def harness(tmp_path: Path):
    h = RobotHarness(_config(tmp_path), clock=ReplayClock(), enable_dashboard=True, dashboard_port=0)
    await h.start()
    try:
        yield h
    finally:
        await h.stop()


async def test_map_endpoint_shows_the_robots_own_map_and_optional_sim_truth(harness) -> None:
    dash = harness.dashboard
    smap = harness.spatial_map
    now = harness.clock.monotonic()
    smap.objects["box_1"] = TrackedObject("box_1", "brown cardboard box", 950.0, 420.0, 120.0, now, now)
    smap.objects["ball_1"] = TrackedObject("ball_1", "red ball", 680.0, 160.0, 50.0, now, now, stale=True)
    smap.add_obstacle_point(1200.0, -300.0, weight=2.0)
    smap.mark_coverage(harness.body.pose(), harness.camera.model)

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


async def test_thinking_stream_records_a_null_backend_turn(harness) -> None:
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


async def test_thinking_pairs_the_camera_frame_with_its_detections(tmp_path: Path) -> None:
    backend = ScriptedBackend([
        {"call": "observe", "args": {}},
        '[{"label": "brown cardboard box", "bbox": [560, 380, 760, 640]}]',  # the detector's answer
        "I see a brown box.",
    ])
    harness = RobotHarness(_config(tmp_path), clock=ReplayClock(), backend=backend,
                           enable_dashboard=True, dashboard_port=0)
    await harness.start()
    try:
        dash = harness.dashboard
        reply = await harness.submit_message("what do you see?")
        assert reply["say"] == "I see a brown box."

        data = await _get_json(dash, "/api/thinking")
        kinds = [e["type"] for e in data["entries"]]
        assert kinds.index("skill.started") < kinds.index("perception.view") < kinds.index("skill.completed")
        first_call = next(e for e in data["entries"] if e["type"] == "cognition.model_call")
        assert first_call["tool_calls"] == [{"name": "observe", "args": {}}]
        view_entry = next(e for e in data["entries"] if e["type"] == "perception.view")
        assert view_entry["detections"] == [{"label": "brown cardboard box", "bbox": [0.56, 0.38, 0.76, 0.64]}]
        assert view_entry["has_image"] is True and view_entry["version"] == 1
        assert view_entry["pose"]["x_mm"] == 0.0
        done = next(e for e in data["entries"] if e["type"] == "skill.completed")
        assert done["tool"] == "observe" and done["ok"] is True and done["status"] == "completed"
        assert data["view"]["version"] == 1 and data["view"]["detections"] == view_entry["detections"]
        assert data["kept_frames"] == [1]
        assert data["entries"][-1]["type"] == "robot.response" and data["entries"][-1]["steps"] == 2

        async with httpx.AsyncClient(base_url=dash.url, timeout=5.0) as client:
            latest = await client.get("/api/thinking/frame")
            assert latest.status_code == 200
            assert latest.headers["content-type"] == "image/png"
            assert latest.content.startswith(b"\x89PNG\r\n\x1a\n")
            assert (await client.get("/api/thinking/frame", params={"v": 1})).content == latest.content
            assert (await client.get("/api/thinking/frame", params={"v": 99})).status_code == 404
            assert (await client.get("/api/thinking", params={"since": "x"})).status_code == 400
    finally:
        await harness.stop()


async def test_autonomy_endpoint_reports_drives_battery_and_follows_the_controls(harness) -> None:
    dash = harness.dashboard
    data = await _get_json(dash, "/api/autonomy")
    assert data["enabled"] is False and data["paused"] is False and data["quiet"] is False
    assert {"reason", "last_trigger", "next_decision_in_s", "minutes_since_human", "unexplored_fraction"} <= set(data)
    assert set(data["drives"]) == {"curiosity", "social", "play", "energy"}
    assert {"level", "threshold", "rate_per_min"} <= set(data["drives"]["curiosity"])
    assert 0.0 < data["body"]["battery_pct"] <= 100.0 and data["body"]["charging"] is False
    assert data["busy"] is False

    async with httpx.AsyncClient(base_url=dash.url, timeout=5.0) as client:
        r = await client.post("/api/autonomy", json={"enabled": True, "paused": True, "quiet": True})
        assert r.status_code == 200 and r.json()["paused"] is True
        assert (await client.post("/api/autonomy", json={"enabled": "yes"})).status_code == 400
        after = (await client.get("/api/autonomy")).json()
        assert after["enabled"] is True and after["paused"] is True and after["quiet"] is True
        assert after["reason"] == "paused by user"
        state = (await client.get("/api/state")).json()
        assert state["autonomy"]["paused"] is True and "drives" in state["autonomy"]
        r = await client.post("/api/autonomy", json={"paused": False, "quiet": False})
        assert r.json()["paused"] is False and r.json()["quiet"] is False


async def test_page_and_legacy_endpoints_still_work(harness) -> None:
    dash = harness.dashboard
    async with httpx.AsyncClient(base_url=dash.url, timeout=5.0) as client:
        page = await client.get("/")
        assert page.status_code == 200
        assert "SPROCKET // EMBODIED AI HARNESS DASHBOARD" in page.text
        for element_id in ("mapCanvas", "truthToggle", "thinkLog", "viewCanvas", "driveBars", "batteryBox"):
            assert f'id="{element_id}"' in page.text
        assert "camera_verification" not in page.text  # removed with the old cognition stack
        assert "runAction('turn'" not in page.text  # "turn" is not a body primitive; "spin" is

        reply = (await client.post("/api/message", json={"text": "hi there"})).json()
        assert reply["status"] == "completed"
        expr = (await client.post("/api/expression", json={"emotion": "curious", "theme": "amber"})).json()
        assert expr["theme"] == "amber" and expr["emotion"] == "curious"
        obs = (await client.post("/api/observe", json={"label": "red ball", "distance_mm": 650})).json()
        assert obs["observations"][0]["label"] == "red ball"
        spin = (await client.post("/api/action", json={"tool": "spin", "args": {"angle_deg": 30}})).json()
        assert spin["ok"] is True

        state = (await client.get("/api/state")).json()
        assert state["face"]["theme"] == "amber"
        assert state["sandbox"]["camera"]["data_url"].startswith("data:image/png;base64,")
        assert len(state["sandbox"]["objects"]) >= 6
        assert state["goal"]["objective"] == "hi there" and state["goal"]["status"] == "completed"
        turn = state["recent_conversations"][-1]
        assert turn["user"] == "hi there" and turn["status"] == "completed" and turn["steps"] == 1
        assert "camera_verification" not in turn
        assert "sad" in state["capabilities"]["allowed_emotions"]

        moved = (await client.post("/api/sandbox", json={"op": "move", "id": "red_ball", "x_mm": 500.0,
                                                          "y_mm": -100.0})).json()
        ball = next(o for o in moved["objects"] if o["id"] == "red_ball")
        assert (ball["x_mm"], ball["y_mm"]) == (500.0, -100.0)
        added = (await client.post("/api/sandbox", json={"op": "add", "label": "Purple Ball", "shape": "sphere",
                                                          "x_mm": 600.0, "y_mm": 120.0})).json()
        assert any(o["label"] == "Purple Ball" for o in added["objects"])
        assert len((await client.get("/api/events")).json()["events"]) > 0
        assert (await client.get("/api/nope")).status_code == 404
