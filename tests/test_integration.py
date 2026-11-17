"""Integration tests for Sprocket v2 Live UI, Demo runner, Eval suite, SFT Dataset Builder, and pure-Python ROS checks."""

from __future__ import annotations

import ast
import inspect
import json
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from robot.agent.agent import Agent
from robot.agent.api import Intent
from robot.agent.planner import SYSTEM_PROMPT, parse_intent
from robot.eval.suite import CASES, run_case, run_planner
from robot.interfaces import demo
from robot.interfaces.live import Live, State
from robot.ros import model
from robot.sim.world import SimRobot, Thing
from robot.train.build import (
    ALLOWED_ACTIONS,
    ALLOWED_CHIMES,
    ALLOWED_POSTURES,
    SCENARIOS,
    build_examples,
    too_close,
)
from robot.train.build import (
    main as build_main,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_live_state_and_http_ui() -> None:
    state = State()
    for field_name in (
        "subgoal",
        "action",
        "posture",
        "chime",
        "intents",
        "scratchpad",
        "scene",
        "status",
        "instruction",
        "problems",
        "said",
        "pose",
        "poses",
        "trail",
        "objects",
        "rooms",
        "scan",
        "view",
    ):
        assert hasattr(state, field_name)
    for dead_field in ("readback", "program", "line", "say_first", "plan"):
        assert not hasattr(state, dead_field)

    live = Live()
    sim = SimRobot(next(c for c in CASES if c.id == "O1").scene())
    bystander_id = sim.scene.things[-1].id + 100
    sim.things[bystander_id] = Thing("person", -2.5, -1.5, radius=0.3, user=False, id=bystander_id)
    sim.memory[bystander_id] = ((-2.5, -1.5), 0.0)
    live.show_world(sim, include_view=True)
    live.update(
        scene="living room",
        scenes=["living room", "office"],
        status="running",
        instruction="Go to the chair.",
        subgoal="approach chair",
        action="go_to",
        posture="curious",
        chime="listen",
        intents=[Intent(action="go_to", target="chair", params={"distance": 0.5}, subgoal="approach chair")],
        scratchpad="goal: Go to the chair.",
    )
    snap = json.loads(live.snapshot())
    assert snap["subgoal"] == "approach chair"
    assert snap["posture"] == "curious"
    assert snap["chime"] == "listen"
    assert len(snap["intents"]) == 1
    assert snap["intents"][0]["action"] == "go_to"
    assert snap["view"].startswith("data:image/png;base64,")
    assert "readback" not in snap
    bystanders = [o for o in snap["objects"] if o["id"] == bystander_id]
    assert bystanders and bystanders[0]["person"] is True and bystanders[0]["user"] is False
    users = [o for o in snap["objects"] if o["user"] is True]
    assert all(u["person"] for u in users)
    assert set(snap["room_bounds"]) == set(sim.scene.rooms)

    server = live.serve(port=0)
    try:
        assert server.daemon_threads is True
        assert server.allow_reuse_address is True
        assert server.request_queue_size == 64

        port = server.server_address[1]
        base = f"http://127.0.0.1:{port}"
        r_state = httpx.get(f"{base}/state", timeout=2.0, trust_env=False)
        assert r_state.status_code == 200
        payload = r_state.json()
        for key in ("subgoal", "posture", "chime", "intents", "scratchpad", "pose", "poses", "trail", "objects"):
            assert key in payload
        assert "readback" not in payload

        r_html = httpx.get(f"{base}/", timeout=2.0, trust_env=False)
        assert r_html.status_code == 200
        html = r_html.text
        for token in (
            "Sprocket v2",
            "Living Mind",
            "Cognitive Synthesis (Perception · Memory · Intent)",
            "room_bounds",
            "subgoal",
            "posture",
            "chime",
            "intents",
            "scratchpad",
            "pollLoop",
            "AbortController",
            "camsKey",
            "dataset.loading",
            "isUser?'you'",
        ):
            assert token in html
        assert "t.person?'you'" not in html
        assert "setInterval" not in html
        for dead in ("readback", "say_first", "program"):
            assert dead not in html.lower()

        prev_cam = live.camera
        try:
            live.camera = None
            assert httpx.get(f"{base}/camera/front_camera.png", timeout=2.0, trust_env=False).status_code == 204
            live.camera = lambda _name: b""
            assert httpx.get(f"{base}/camera/front_camera.png", timeout=2.0, trust_env=False).status_code == 204
            live.camera = lambda name: b"PNGDATA" if name == "front_camera" else b""
            r_cam = httpx.get(f"{base}/camera/front_camera.png?t=1", timeout=2.0, trust_env=False)
            assert r_cam.status_code == 200
            assert r_cam.content == b"PNGDATA"
        finally:
            live.camera = prev_cam

        r_post = httpx.post(f"{base}/instruct", json={"text": "come back"}, timeout=2.0, trust_env=False)
        assert r_post.status_code == 200
        assert r_post.json() == {"ok": True}
        assert live.inbox.get(timeout=1.0) == ("instruct", "come back")
    finally:
        server.shutdown()
        server.server_close()


def test_no_deleted_v1_module_refs() -> None:
    deleted_modules = (
        "robot.agent.gate",
        "robot.agent.readback",
        "robot.agent.runner",
        "robot.eval.programs",
        "robot.train.families",
    )
    deleted_symbols = ("COME_BACK", "check_readback", "api_card")
    owned = (
        "robot/interfaces/live.py",
        "robot/interfaces/demo.py",
        "robot/eval/suite.py",
        "robot/train/build.py",
    )
    for rel in owned:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(text, filename=rel)
        assert isinstance(tree, ast.Module)
        for mod in deleted_modules:
            assert mod not in text, f"{rel} still references {mod}"
        for sym in deleted_symbols:
            assert sym not in text, f"{rel} still references {sym}"


def test_demo_run_case_and_drive_come_back() -> None:
    class MockChairPlanner:
        def __init__(self) -> None:
            self.step = 0

        def plan_step(self, *args: Any, **kwargs: Any) -> Intent:
            self.step += 1
            if self.step == 1:
                return Intent(
                    say="Heading to the chair.",
                    chime="listen",
                    posture="curious",
                    action="go_to",
                    target="chair",
                    params={"distance": 0.4},
                    subgoal="approach chair",
                )
            return Intent(
                say="Arrived at the chair.",
                chime="happy",
                posture="happy",
                action="done",
                subgoal="completed",
            )

    live = Live()
    record, sim = demo.run_case(live, "O1", planner=MockChairPlanner(), realtime=False)
    assert record is not None
    assert record.outcome == "done"
    assert live.state.status == "done"
    assert live.state.posture == "happy"
    assert live.state.chime == "happy"
    assert len(live.state.intents) == 2
    assert any(c.id == "O1" and c.check(sim) for c in CASES)

    # Unreachable brain server fails cleanly instead of falling back to deleted script runner
    live_fail = Live()
    fail_rec, _ = demo.run_case(live_fail, "O1", brain_url="http://127.0.0.1:19999", realtime=False)
    assert fail_rec is None
    assert live_fail.state.status == "failed"
    assert live_fail.state.problems

    # drive() handles "come back" directly via agent.do("come back", holder.sim)
    live_drive = Live()
    holder = demo.Holder(SimRobot(demo.make_scene("living room"), realtime=False))
    agent = Agent(MockChairPlanner(), live=live_drive)
    stop_drive = threading.Event()
    t = threading.Thread(target=demo.drive, args=(live_drive, agent, holder), kwargs={"stop_event": stop_drive},
                         daemon=True)
    t.start()
    try:
        live_drive.inbox.put(("instruct", "come back"))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not live_drive.state.history:
            time.sleep(0.05)
        assert live_drive.state.history
        assert live_drive.state.history[-1]["instruction"] == "come back"
        assert live_drive.state.history[-1]["outcome"] == "done"

        live_drive.inbox.put(("stop", ""))
        stop_deadline = time.monotonic() + 2.0
        while time.monotonic() < stop_deadline and live_drive.state.status != "stopped":
            time.sleep(0.05)
        assert live_drive.state.status == "stopped"
    finally:
        stop_drive.set()
        t.join(timeout=2.0)


def test_demo_drive_exception_marks_failed() -> None:
    class FailingPlanner:
        def plan_step(self, *args: Any, **kwargs: Any) -> Intent:
            raise RuntimeError("boom")

    class FailingAgent(Agent):
        def do(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    live_err = Live()
    holder = demo.Holder(SimRobot(demo.make_scene("living room"), realtime=False))
    agent = FailingAgent(FailingPlanner(), live=live_err)
    stop_drive = threading.Event()
    t = threading.Thread(target=demo.drive, args=(live_err, agent, holder), kwargs={"stop_event": stop_drive},
                         daemon=True)
    t.start()
    try:
        live_err.inbox.put(("instruct", "trigger failure"))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and live_err.state.status != "failed":
            time.sleep(0.05)
        assert live_err.state.status == "failed"
        assert live_err.state.problems == ["boom"]
    finally:
        stop_drive.set()
        t.join(timeout=2.0)


def test_observing_planner_completed_steps_and_pop_last_intent() -> None:
    live = Live()
    sim = SimRobot(demo.make_scene("living room"), realtime=False)
    seen_completed: list[tuple[str, ...]] = []

    class RepeatingWrappedPlanner:
        def plan_step(
            self,
            context: str,
            instruction: str,
            image_url: str | None = None,
            scratchpad: str = "",
            earlier: tuple[str, ...] | list[str] = (),
            completed_steps: tuple[str, ...] | list[str] = (),
        ) -> Intent:
            seen_completed.append(tuple(completed_steps))
            return Intent(
                say="Heading to the chair." if len(seen_completed) == 1 else "Repeating chair step!",
                chime="listen",
                posture="curious",
                action="go_to",
                target="chair",
                params={"distance": 0.5},
                subgoal="approach chair",
            )

    wrapped = demo._wrap_planner(RepeatingWrappedPlanner(), live)
    rec = Agent(wrapped, live=live).do("go to the chair", sim)
    assert rec.outcome == "done"
    assert len(seen_completed) == 2
    assert seen_completed[0] == ()
    assert len(seen_completed[1]) == 1 and "go_to" in seen_completed[1][0]
    assert len(rec.intents) == 1
    assert len(live.state.intents) == 1


def test_eval_suite_cases_and_run_case(tmp_path: Path) -> None:
    assert len(CASES) == 50
    for case in CASES:
        sim = SimRobot(case.scene())
        assert sim.pose() is not None

    class MockO1Planner:
        def __init__(self) -> None:
            self.calls = 0

        def plan_step(self, *args: Any, **kwargs: Any) -> Intent:
            self.calls += 1
            if self.calls == 1:
                return Intent(
                    say="Going to chair.",
                    chime="listen",
                    posture="curious",
                    action="go_to",
                    target="chair",
                    params={"distance": 0.4},
                    subgoal="go to chair",
                )
            return Intent(say="Done.", chime="happy", posture="happy", action="done", subgoal="arrived")

    o1 = next(c for c in CASES if c.id == "O1")
    passed, outcome, record, sim = run_case(o1, MockO1Planner())
    assert passed is True
    assert outcome == "done"
    assert len(record.intents) == 2
    assert o1.check(sim)

    class MockRefusePlanner:
        def plan_step(self, *args: Any, **kwargs: Any) -> Intent:
            return Intent(
                say="I cannot fly because I am a wheeled robot.",
                chime="uh_oh",
                posture="uh_oh",
                action="refuse",
                subgoal="refuse impossible flight",
            )

    a2 = next(c for c in CASES if c.id == "A2")
    passed_a2, outcome_a2, record_a2, _ = run_case(a2, MockRefusePlanner())
    assert passed_a2 is True
    assert outcome_a2 == "refused"
    assert len(record_a2.intents) == 1

    rows = run_planner("smoke", ["O1"], planner=MockO1Planner(), out_dir=tmp_path)
    assert len(rows) == 1
    jsonl_path = tmp_path / "smoke.jsonl"
    assert jsonl_path.exists()
    logged = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    assert logged["passed"] is True
    assert logged["outcome"] == "done"
    assert logged["steps"] == 2
    assert len(logged["intents"]) == 2
    assert "problems" in logged and "plan_s" in logged


def test_train_build_examples_and_cli(tmp_path: Path, monkeypatch: Any) -> None:
    examples = build_examples(20, seed=13)
    assert len(examples) == 20
    seen_scenarios = {ex["scenario"] for ex in examples}
    assert seen_scenarios == set(SCENARIOS)

    for ex in examples:
        assert not too_close(ex["instruction"])
        msgs = ex["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert msgs[0]["content"] == SYSTEM_PROMPT
        assert "Scene State:" in msgs[1]["content"]
        assert f"Instruction: {ex['instruction']}" in msgs[1]["content"]
        parsed = parse_intent(msgs[2]["content"])
        assert parsed.action in ALLOWED_ACTIONS
        assert parsed.posture in ALLOWED_POSTURES
        assert parsed.chime in ALLOWED_CHIMES
        assert isinstance(parsed.params, dict)
        assert parsed.subgoal

    out_dir = tmp_path / "sft"
    monkeypatch.setattr(
        sys,
        "argv",
        ["robot.train.build", "--count", "20", "--seed", "5", "--out", str(out_dir)],
    )
    build_main()
    train_lines = (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    valid_lines = (out_dir / "valid.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train_lines) + len(valid_lines) == 20
    for raw in train_lines + valid_lines:
        row = json.loads(raw)
        parse_intent(row["messages"][2]["content"])


def test_ros_model_and_pure_python_ast() -> None:
    assert len(model.CAMERAS) == 3
    assert set(model.CAMERAS) == {"front_camera", "rear_left_camera", "rear_right_camera"}
    urdf_xml = model.urdf()
    for cam in ("front_camera", "rear_left_camera", "rear_right_camera"):
        assert cam in urdf_xml
    lower_xml = urdf_xml.lower()
    for forbidden in ("lidar", "laser", 'type="ray"', "gpu_lidar"):
        assert forbidden not in lower_xml

    for rel in ("robot/ros/robot.py", "robot/ros/sensors_check.py"):
        src = (REPO_ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(src, filename=rel)
        assert isinstance(tree, ast.Module)


def test_ros_tof_urdf_and_scan_publisher() -> None:
    from robot.ros.perceive import SyntheticScanPublisher

    urdf_xml = model.urdf()
    for token in (
        "front_tof_link",
        "rear_tof_link",
        "front_tof_optical_frame",
        "rear_tof_optical_frame",
        "<topic>front_tof</topic>",
        "<topic>rear_tof</topic>",
        "<width>8</width><height>8</height>",
    ):
        assert token in urdf_xml

    assert hasattr(model, "TOF_BRIDGES")
    assert any("front_tof" in b for b in model.TOF_BRIDGES)
    assert any("rear_tof" in b for b in model.TOF_BRIDGES)

    class _Stamp:
        sec = 1
        nanosec = 0

    pub = SyntheticScanPublisher()
    tof_8x8 = np.full((8, 8), 1.4, dtype=np.float32)
    assert hasattr(pub, "ingest_tof_depth")
    hazard = pub.ingest_tof_depth("front", tof_8x8)
    assert hazard is not None and hazard.kind in ("clear", "traversable", "cliff", "obstacle")
    assert "front" in pub.tof_grids and pub.tof_grids["front"].shape == (8, 8)

    raw_depth = np.full((720, 1280), 0.7, dtype=np.float32)
    scan = pub.ingest_depth("front_camera", _Stamp(), raw_depth)
    assert scan is not None
    ranges, angle_min, step = scan
    assert len(ranges) == 360
    assert angle_min < 0.0 and step > 0.0


def test_detector_pose_keypoints_and_mono_depth(monkeypatch: Any) -> None:
    class _FakeArray:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        def tolist(self) -> list[Any]:
            return self._arr.tolist()

        def detach(self) -> _FakeArray:
            return self

        def cpu(self) -> _FakeArray:
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

    class _FakeBoxes:
        def __init__(self, xyxy: list[list[float]], cls_ids: list[int], conf: list[float]) -> None:
            self.xyxy = _FakeArray(np.asarray(xyxy, dtype=np.float32))
            self.cls = _FakeArray(np.asarray(cls_ids, dtype=np.float32))
            self.conf = _FakeArray(np.asarray(conf, dtype=np.float32))

        def __len__(self) -> int:
            return len(self.xyxy._arr)

    class _FakeKeypoints:
        def __init__(self, data: np.ndarray) -> None:
            self.data = _FakeArray(np.asarray(data, dtype=np.float32))

    class _FakeDepth:
        def __init__(self, data: np.ndarray) -> None:
            self.data = _FakeArray(np.asarray(data, dtype=np.float32))

    class _FakeYOLOE:
        def __init__(self, weights: str) -> None:
            self.weights = weights
            self.classes: list[str] = []

        def set_classes(self, classes: list[str]) -> None:
            self.classes = list(classes)

        def predict(self, img: np.ndarray, **kwargs: Any) -> list[Any]:
            res = types.SimpleNamespace(
                names={0: "person", 1: "chair"},
                boxes=_FakeBoxes([[100.0, 50.0, 200.0, 350.0], [300.0, 200.0, 400.0, 380.0]], [0, 1], [0.92, 0.80]),
                masks=None,
            )
            return [res]

    expected_kpts = np.full((17, 3), 0.75, dtype=np.float32)
    expected_kpts[:, 0] = 150.0
    expected_kpts[:, 1] = np.linspace(80.0, 320.0, 17, dtype=np.float32)

    class _FakeYOLO:
        def __init__(self, weights: str) -> None:
            self.weights = weights

        def predict(self, img: np.ndarray, **kwargs: Any) -> list[Any]:
            if "pose" in self.weights:
                other_kpts = np.zeros((17, 3), dtype=np.float32)
                return [
                    types.SimpleNamespace(
                        boxes=_FakeBoxes([[0.0, 0.0, 20.0, 20.0], [104.0, 54.0, 198.0, 346.0]], [0, 0], [0.5, 0.95]),
                        keypoints=_FakeKeypoints(np.stack([other_kpts, expected_kpts], axis=0)),
                    )
                ]
            return [types.SimpleNamespace(depth=_FakeDepth(np.full((240, 320), 1.85, dtype=np.float32)))]

    fake_torch = types.ModuleType("torch")
    fake_torch.set_num_threads = lambda n: None  # type: ignore[attr-defined]
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLOE = _FakeYOLOE  # type: ignore[attr-defined]
    fake_ultralytics.YOLO = _FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    from robot.perception.detector import Detector, Seen
    from robot.ros import run3d

    s_plain = Seen("chair", 0.7, (0.0, 0.0, 10.0, 10.0))
    assert s_plain.keypoints is None
    s_with_kpts = Seen("person", 0.9, (0.0, 0.0, 10.0, 20.0), keypoints=expected_kpts)
    assert s_with_kpts.keypoints is not None and s_with_kpts.keypoints.shape == (17, 3)

    det = Detector(weights="yoloe-26s-seg.pt", threads=2, mono_depth=True)
    assert det.mono_depth is True
    assert det._pose_model is None
    assert det._depth_model is None

    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    seen = det(rgb)
    assert [s.label for s in seen] == ["person", "chair"]
    assert seen[0].keypoints is not None
    assert seen[0].keypoints.shape == (17, 3)
    assert seen[0].keypoints.dtype == np.float32
    assert np.allclose(seen[0].keypoints, expected_kpts)
    assert seen[1].keypoints is None

    depth_map = det.estimate_depth(rgb)
    assert depth_map is not None
    assert depth_map.shape == (480, 640)
    assert depth_map.dtype == np.float32
    assert np.allclose(depth_map, 1.85)

    det_no_pose = Detector(weights="yoloe-26s-seg.pt", pose_weights="", depth_weights="")
    seen_no_pose = det_no_pose(rgb)
    assert seen_no_pose[0].keypoints is None
    assert det_no_pose.estimate_depth(rgb) is None

    assert "mono_depth" in inspect.signature(run3d.start).parameters
    assert "mono_depth" in inspect.signature(run3d.run_cases).parameters
    assert "--mono-depth" in (REPO_ROOT / "robot/ros/run3d.py").read_text(encoding="utf-8")


def test_decision_scoring_live_wiring() -> None:
    from robot.agent.api import Pose
    from robot.body.controller import BodyController
    from robot.mind.brain import Brain
    from robot.perception.camera import data_url
    from robot.sim.world import Object as WorldObj

    b = Brain("http://127.0.0.1:18081", decision_url="http://127.0.0.1:18082")
    assert str(b.http.base_url).rstrip("/") == "http://127.0.0.1:18081"
    assert str(b.decision_http.base_url).rstrip("/") == "http://127.0.0.1:18082"
    b_same = Brain("http://127.0.0.1:18081")
    assert b_same.decision_http is b_same.http

    # Live step_pursuit incorporates visual_probs when steering candidate arcs
    ctrl = BodyController()
    _, w_straight, _, _ = ctrl.step_pursuit(
        (0.0, 0.0, 0.0), [(2.0, 0.0)], 0, visual_probs={"A": 0.0, "D": 1.0, "G": 0.0}
    )
    _, w_biased, _, _ = ctrl.step_pursuit(
        (0.0, 0.0, 0.0), [(2.0, 0.0)], 0, visual_probs={"A": 1.0, "D": 0.0, "G": 0.0}
    )
    assert abs(w_biased) > abs(w_straight)

    # SimRobot.search ranks multiple candidate frontiers with brain.score_frontiers
    scene = demo.make_scene("living room")

    class _RecordingBrain:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[str, ...]]] = []

        def score_frontiers(self, image_url: str, target: str, labels: tuple[str, ...] = ("A", "B", "C")) -> dict[str, float]:
            assert image_url.startswith("data:image/png;base64,")
            self.calls.append((target, labels))
            out = {lbl: 0.05 for lbl in labels}
            out[labels[-1]] = 0.90
            return out

    class _FrontierSim(SimRobot):
        def __init__(self) -> None:
            super().__init__(scene, realtime=False)
            self.memory.clear()
            self.brain = _RecordingBrain()
            self.visited_order: list[tuple[float, float]] = []

        def render_view(self, width: int = 320, height: int = 240) -> np.ndarray:
            return np.zeros((24, 24, 3), dtype=np.uint8)

        def _unseen(self, region: str = "room") -> list[Pose]:
            return [Pose(0.5, 0.0, 0.0), Pose(1.0, 0.0, 0.0), Pose(1.5, 0.0, 0.0)]

        def _go(self, target: Any, distance: float) -> None:
            pt = (target.x, target.y) if isinstance(target, Pose) else target.position
            self.visited_order.append(pt)
            if pt == (1.5, 0.0):
                thing = next(iter(self.things.values()))
                self._found_override = WorldObj(thing.id, "keys", (thing.x, thing.y), 1.0, 0.0, "front", 0.3, thing.color, 0.0)

        def _found(self, description: str, since: float) -> WorldObj | None:
            return getattr(self, "_found_override", None)

    _ = data_url(np.zeros((8, 8, 3), dtype=np.uint8))
    sim = _FrontierSim()
    found = sim.search("keys", area="here")
    assert found is not None
    assert sim.brain.calls
    assert sim.visited_order[0] == (1.5, 0.0)


def test_living_mind_synthesis_and_scratchpad() -> None:
    live = Live()
    sim = SimRobot(demo.make_scene("living room"), realtime=False)
    assert sim.snapshot()["objects"]

    class _ExplainPlanner:
        def plan_step(
            self,
            context: str,
            instruction: str,
            image_url: str | None = None,
            scratchpad: str = "",
            earlier: tuple[str, ...] | list[str] = (),
            completed_steps: tuple[str, ...] | list[str] = (),
        ) -> Intent:
            return Intent(
                say="I see furniture around me.",
                chime="curious",
                posture="curious",
                action="done",
                target=None,
                params={},
                subgoal="describe surroundings",
            )

    wrapped = demo._wrap_planner(_ExplainPlanner(), live)
    agent = Agent(wrapped, live=live)
    rec = agent.do("what do you see?", sim)
    assert rec.outcome == "done"

    pad = demo._format_scratchpad(agent, sim)
    assert "focus: what do you see?" in pad
    assert "aware of:" in pad
    assert "dialogue memory: 1 recent turn" in pad
    assert not pad.startswith("goal:")

    agent.short_term.scratchpad.blockers.append("sofa blocking aisle")
    pad_with_blocker = demo._format_scratchpad(agent, sim)
    assert "blockers: sofa blocking aisle" in pad_with_blocker

