"""Integration tests for Sprocket v2 Live UI, Demo runner, Eval suite, SFT Dataset Builder, and pure-Python ROS checks."""

from __future__ import annotations

import ast
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from robot.agent.agent import Agent
from robot.agent.api import Intent
from robot.agent.planner import SYSTEM_PROMPT, parse_intent
from robot.eval.suite import CASES, run_case, run_planner
from robot.interfaces import demo
from robot.interfaces.live import Live, State
from robot.ros import model
from robot.sim.world import SimRobot
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

    server = live.serve(port=0)
    try:
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
        for token in ("Sprocket v2", "subgoal", "posture", "chime", "intents", "scratchpad"):
            assert token in html
        for dead in ("readback", "say_first", "program"):
            assert dead not in html.lower()

        r_post = httpx.post(f"{base}/instruct", json={"text": "come back"}, timeout=2.0, trust_env=False)
        assert r_post.status_code == 200
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
