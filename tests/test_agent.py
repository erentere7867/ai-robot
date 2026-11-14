"""Tests for Sprocket v2: Living Body Engine, Decision Scorer & Closed-Loop Agent."""

from __future__ import annotations

import importlib
import inspect
import json
import math
from dataclasses import dataclass
from typing import Any

from robot.agent.agent import Agent
from robot.agent.api import Intent, Pose
from robot.agent.instant import COME_BACK_INTENT, instant
from robot.agent.planner import LocalPlanner, PlanResult, parse_intent
from robot.body.controller import BodyController
from robot.interfaces.live import Live
from robot.mind.brain import Brain, active_vision_action
from robot.mind.memory import Memory, ShortTermMemory
from robot.sim.world import Scene, SimRobot, Thing


@dataclass
class StepHazard:
    kind: str
    delta_z: float
    distance: float
    torque_boost: bool = False


def _make_sim(things: list[Thing] | None = None) -> SimRobot:
    scene = Scene(
        width=10.0,
        height=10.0,
        things=things or [Thing("table", 2.0, 0.0, radius=0.4, color="brown")],
        rooms={"room": (-5.0, -5.0, 5.0, 5.0)},
        start=Pose(0.0, 0.0, 0.0),
    )
    return SimRobot(scene)


def test_body_controller_trajectory_rollout_and_obstacle_avoidance():
    ctrl = BodyController()
    pose = (0.0, 0.0, 0.0)
    goal = (3.0, 0.0)

    # 1. In free space, rolls out 7 smooth candidate paths
    paths = ctrl.rollout_trajectories(pose, goal, occupied_fn=lambda x, y: False)
    assert len(paths) == 7
    labels = [p.label for p in paths]
    assert set(labels) == {"A", "B", "C", "D", "E", "F", "G"}
    # Best trajectory toward (3.0, 0.0) should be straight ahead (w=0)
    best = paths[0]
    assert abs(best.w) < 0.1
    assert best.clearance > 0.5
    assert best.progress_score > 0.0

    # 2. Obstacle blocking straight path
    def occupied_ahead(x: float, y: float) -> bool:
        return math.hypot(x - 0.8, y - 0.0) < 0.25

    blocked_paths = ctrl.rollout_trajectories(pose, goal, occupied_fn=occupied_ahead)
    best_evading = blocked_paths[0]
    # The evading trajectory must curve around the obstacle (non-zero w)
    assert abs(best_evading.w) > 0.2
    assert best_evading.total_score > -100.0


def test_trajectory_visual_scores_reranking():
    ctrl = BodyController()
    paths = ctrl.rollout_trajectories((0.0, 0.0, 0.0), (3.0, 0.0), occupied_fn=lambda x, y: False)
    # Pick a curving path like 'B' and give it high d1-2B visual confidence
    visual_probs = {"B": 0.98, "A": 0.05}
    reranked = ctrl.apply_visual_scores(paths, visual_probs, weight=5.0)
    assert reranked[0].label == "B"
    assert reranked[0].visual_prob == 0.98


def test_nonstop_waypoint_spline_pursuit():
    ctrl = BodyController()
    waypoints = [(1.0, 0.0), (2.0, 1.0), (3.0, 1.0)]

    # Robot approaching intermediate waypoint 0 within blend_r (0.28m)
    pose = (0.85, 0.0, 0.0)
    v, w, next_idx, arrived = ctrl.step_pursuit(pose, waypoints, 0, blend_r=0.28, v_cruise=0.4)
    # Advances idx to 1 without stopping
    assert next_idx == 1
    assert not arrived
    assert v > 0.1  # Robot keeps moving forward non-stop!

    # Robot at final waypoint within 0.08m
    final_pose = (2.98, 1.0, 0.0)
    v_end, w_end, idx_end, arrived_end = ctrl.step_pursuit(final_pose, waypoints, 2, blend_r=0.28, v_cruise=0.4)
    assert arrived_end
    assert v_end == 0.0


def test_predictive_social_hallway_yielding():
    ctrl = BodyController()
    pose = (0.0, 0.0, 0.0)
    goal = (4.0, 0.0)
    # Person approaching head-on down hallway at 2.0m moving at -0.7 m/s toward robot
    person = [(2.0, 0.0, -0.7, 0.0)]

    paths = ctrl.rollout_trajectories(pose, goal, occupied_fn=lambda x, y: False, people=person)
    best = paths[0]
    # Best path must yield to robot's right side (negative w in standard robot coordinate frame)
    assert best.w < 0.0
    assert best.social_score > 0.0


def test_procedural_aliveness():
    ctrl = BodyController()
    pose = (0.0, 0.0, 0.0)

    # 1. Listening posture: gentle rock + wake-word DOA turn reflex
    v, w = ctrl.aliveness_velocity("listening", t_in_posture=0.1, pose=pose, speaker_doa=0.5)
    assert abs(v) > 0.0
    assert abs(w - min(1.8, 2.5 * 0.5)) < 1e-4

    # 2. Listening polite yield when person steps inside 0.45m
    v_close, _ = ctrl.aliveness_velocity(
        "listening", t_in_posture=0.1, pose=pose, nearest_person_ahead=0.35
    )
    assert v_close == -0.15

    # 3. Happy posture: double-shimmy for first 0.6s
    _, w_happy = ctrl.aliveness_velocity("happy", t_in_posture=0.2, pose=pose)
    assert abs(w_happy) > 0.1
    _, w_happy_done = ctrl.aliveness_velocity("happy", t_in_posture=0.7, pose=pose)
    assert w_happy_done == 0.0

    # 4. Curious posture: sinusoidal scanning turn
    v_cur, w_cur = ctrl.aliveness_velocity("curious", t_in_posture=0.25, pose=pose)
    assert v_cur == 0.0
    assert abs(w_cur) > 0.1


def test_proprioception_snag_step_and_cliff_reflex():
    ctrl = BodyController()

    # 1. Anti-cable snag: high drive command (0.2 m/s) with near-zero progress (< 0.015 m/s) for >= 0.15 s
    ctrl.check_proprioception(v_cmd=0.2, w_cmd=0.0, v_actual=0.0, w_actual=0.0, dt=0.05)
    ctrl.check_proprioception(v_cmd=0.2, w_cmd=0.0, v_actual=0.0, w_actual=0.0, dt=0.05)
    status, v, w, torque = ctrl.check_proprioception(
        v_cmd=0.2, w_cmd=0.0, v_actual=0.0, w_actual=0.0, dt=0.05
    )
    assert status == "snag_recovery"
    assert v == -0.15

    # 2. Traversable step with torque boost (1.5 cm step-up)
    step_hazard = StepHazard(kind="traversable", delta_z=0.015, distance=0.2, torque_boost=True)
    status, v, w, torque = ctrl.check_proprioception(
        v_cmd=0.3, w_cmd=0.0, v_actual=0.28, w_actual=0.0, dt=0.05, hazard=step_hazard
    )
    assert status == "torque_boost"
    assert torque == 1.35
    assert v == 0.3

    # 3. Cliff stop (6 cm drop)
    cliff_hazard = StepHazard(kind="cliff", delta_z=-0.06, distance=0.18)
    status, v, w, torque = ctrl.check_proprioception(
        v_cmd=0.3, w_cmd=0.0, v_actual=0.28, w_actual=0.0, dt=0.05, hazard=cliff_hazard
    )
    assert status == "cliff_stop"
    assert v == -0.15


def test_active_vision_maneuvers_and_decision_thresholds():
    ctrl = BodyController()

    # 1. plan_peek finds perpendicular vantage points
    peek_point = ctrl.plan_peek(
        robot_pose=(0.0, 0.0, 0.0),
        target_xy=(2.0, 0.0),
        occupied_fn=lambda x, y: False,
        offset=0.35,
    )
    assert peek_point is not None
    assert abs(abs(peek_point[1]) - 0.35) < 0.01

    # 2. plan_look_under computes standoff around 1.25m from furniture anchor
    look_point = ctrl.plan_look_under(
        robot_pose=(0.0, 0.0, 0.0),
        anchor_xy=(3.0, 0.0),
        anchor_radius=0.5,
        occupied_fn=lambda x, y: False,
        standoff=1.25,
    )
    assert look_point is not None
    dist_to_anchor = math.hypot(look_point[0] - 3.0, look_point[1] - 0.0)
    assert abs(dist_to_anchor - (1.25 + 0.5 * 0.3)) < 0.05

    # 3. Active vision action calibration
    assert active_vision_action(0.85) == "confirm"
    assert active_vision_action(0.50) == "inspect"
    assert active_vision_action(0.20) == "reject"
    assert Brain.active_vision_action(0.75) == "inspect"


def test_closed_loop_agent_step_execution():
    sim = _make_sim()

    class MockStepPlanner:
        def __init__(self) -> None:
            self.steps = [
                Intent(
                    say="Going to the table.",
                    chime="listen",
                    posture="curious",
                    action="go_to",
                    target="table",
                    params={"distance": 0.5},
                    subgoal="approach table",
                ),
                Intent(
                    say="Arrived at the table.",
                    chime="happy",
                    posture="happy",
                    action="done",
                    subgoal="completed",
                ),
            ]
            self.idx = 0
            self.calls: list[dict[str, Any]] = []

        def plan_step(self, *args: Any, **kwargs: Any) -> Intent:
            self.calls.append({"args": args, "kwargs": kwargs})
            intent = self.steps[self.idx]
            self.idx = min(len(self.steps) - 1, self.idx + 1)
            return intent

    planner = MockStepPlanner()
    agent = Agent(planner)
    record = agent.do("Go to the table.", sim)
    assert record.outcome == "done"
    assert len(record.intents) == 2
    assert not hasattr(record, "programs")
    assert not hasattr(record, "checks")
    assert any("Going to the table." in text for _, text in sim.trace.said)
    assert ("happy",) in [(p,) for _, p in sim.trace.postures] or len(sim.trace.postures) > 0

    # Scratchpad, visual buffer, and short-term dialogue turn wiring
    assert isinstance(agent.short_term, ShortTermMemory)
    assert agent.short_term.scratchpad.goal == "Go to the table."
    assert agent.short_term.scratchpad.steps == 2
    assert "room" in agent.short_term.scratchpad.visited_rooms
    assert len(agent.short_term.scratchpad.checked_spots) >= 2
    assert len(agent.short_term.visual_buffer.recent(seconds=5.0)) >= 1
    assert len(agent.short_term.dialogue) == 1
    assert agent.short_term.dialogue[0][0] == "Go to the table."
    assert "(done)" in agent.short_term.dialogue[0][1]

    # Pure-mock non-multimodal planner receives image_url=None and scratchpad.summary()
    assert planner.calls[0]["kwargs"]["image_url"] is None
    assert "Goal: 'Go to the table.'" in planner.calls[0]["kwargs"]["scratchpad"]
    assert "Rooms: [room]" in planner.calls[1]["kwargs"]["scratchpad"]


def test_agent_multimodal_live_memory_and_clean_interfaces(tmp_path):
    sim = _make_sim()
    live = Live()
    mem = Memory(tmp_path / "agent.db")

    class MultimodalMockPlanner:
        multimodal = True

        def __init__(self) -> None:
            self.image_urls: list[str | None] = []

        def plan_step(self, *args: Any, **kwargs: Any) -> Intent:
            self.image_urls.append(kwargs.get("image_url"))
            return Intent(
                say="All set.",
                chime="happy",
                posture="happy",
                action="done",
                subgoal="verify scene",
            )

    planner = MultimodalMockPlanner()
    agent = Agent(planner, memory=mem, live=live)
    assert agent.short_term is mem.short_term

    record = agent.do("Check the room.", sim)
    assert record.outcome == "done"
    assert planner.image_urls[0] is not None
    assert planner.image_urls[0].startswith("data:image/png;base64,")

    assert live.state.status == "done"
    assert live.state.subgoal == "verify scene"
    assert live.state.action == "done"
    assert live.state.posture == "happy"
    assert live.state.chime == "happy"
    assert "All set." in live.state.said
    assert json.loads(live.snapshot())["status"] == "done"
    assert len(mem.short_term.dialogue) == 1

    # Legacy fallbacks and aliases are completely removed
    instant_mod = importlib.import_module("robot.agent.instant")
    assert "readback" not in inspect.signature(Agent).parameters
    assert not hasattr(PlanResult, "program")
    assert not hasattr(LocalPlanner, "judge")
    assert not hasattr(instant_mod, "COME_BACK")
    assert isinstance(COME_BACK_INTENT, Intent)


def test_instant_commands():
    sim = _make_sim()
    agent = Agent(LocalPlanner())

    # Stop command acts instantly without calling model
    record_stop = agent.do("stop now please", sim)
    assert record_stop.outcome == "preempted"

    # Wait command acts instantly
    record_wait = agent.do("wait", sim)
    assert record_wait.outcome == "done"

    # instant helper
    assert instant("Sprocket, stop!") == "stop"
    assert instant("wait a moment") == "wait"
    assert instant("come back here") == "come back"
    assert instant("go to the kitchen") is None


def test_intent_parsing_and_format():
    text = (
        "say: On my way to the desk.\n"
        "chime: wake\n"
        "posture: curious\n"
        "action: go_to\n"
        "target: 4\n"
        "params: {\"distance\": 0.3}\n"
        "subgoal: reach the desk\n"
    )
    intent = parse_intent(text)
    assert intent.say == "On my way to the desk."
    assert intent.chime == "wake"
    assert intent.posture == "curious"
    assert intent.action == "go_to"
    assert intent.target == 4
    assert intent.params == {"distance": 0.3}
    assert intent.subgoal == "reach the desk"
