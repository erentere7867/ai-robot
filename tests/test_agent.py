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

    noisy = (
        'say: "Hey there!"<|im_end|>\n'
        "chime: neutral\n"
        "posture: alert\n"
        "action: greet\n"
        "target: none\n"
        "params: {}\n"
        'subgoal: "say hello" ```extra\n'
    )
    cleaned = parse_intent(noisy)
    assert cleaned.say == "Hey there!"
    assert cleaned.chime == ""
    assert cleaned.posture == "curious"
    assert cleaned.action == "done"
    assert cleaned.target is None
    assert cleaned.subgoal == "say hello"

    assert parse_intent("action: none\n").action == "done"
    assert parse_intent("target: 10  table\n").target == 10
    assert parse_intent("target: #7\n").target == 7
    assert parse_intent("target: id=4 (desk)\n").target == 4
    assert parse_intent("target: person(user)\n").target == "user"
    assert parse_intent("target: desk(table)\n").target == "desk"
    assert parse_intent("target: sofa(couch)\n").target == "sofa"
    assert parse_intent("target: (none)\n").target is None
    assert parse_intent("target: none (unknown)\n").target is None
    assert parse_intent("target: living room\n").target == "living room"


def test_context_user_highlight_planner_prompt_and_closed_loop_guards():
    from robot.agent.api import Object
    from robot.agent.context import describe
    from robot.agent.planner import GRAMMAR
    from robot.mind.brain import Reply

    # GRAMMAR forbids raw quotes/angle tags/backticks in say/subgoal and allows valid postures/chimes/actions
    assert 'chime ::=' in GRAMMAR and '"none"' in GRAMMAR
    assert 'action ::=' in GRAMMAR and '"refuse"' in GRAMMAR and '"orbit"' in GRAMMAR

    # describe() highlights Pose, User, and inline desk(table) / sofa(couch) aliases
    snap = {
        "room": "living",
        "rooms": ["living", "kitchen"],
        "pose": Pose(1.0, 2.0, 45.0),
        "user": 1,
        "objects": [
            Object(1, "person", (3.0, 2.0), 2.0, 0.0, "front", 0.3, "", 0.0),
            Object(4, "desk", (4.0, 2.0), 3.0, 0.0, "front", 0.5, "brown", 0.0),
            Object(5, "sofa", (2.5, 3.0), 3.5, 25.0, "front-left", 0.7, "gray", 0.0),
        ],
    }
    ctx = describe(snap)
    assert "Pose: x=1.00, y=2.00, yaw=45°" in ctx
    assert "User: id=1 (person(user)) at 2.0 m front (+0°); use target: user or target: 1" in ctx
    assert "desk(table)" in ctx
    assert "sofa(couch)" in ctx

    # LocalPlanner defaults to multimodal=False, puts Instruction before Completed Steps,
    # and omits Blockers: [none] scratchpad clutter
    captured: list[list[dict[str, Any]]] = []

    class FakeBrain:
        def complete(self, messages: list[dict[str, Any]], grammar: str, max_tokens: int, think: int) -> Reply:
            captured.append(messages)
            return Reply(
                text="say: All clear!\nchime: wake\nposture: happy\naction: none\ntarget: none\nparams: {}\nsubgoal: greet user"
            )

    lp = LocalPlanner(brain=FakeBrain())  # type: ignore[arg-type]
    assert lp.multimodal is False
    greet = lp.plan_step(
        ctx,
        "hey",
        scratchpad="Goal: 'hey' | Steps: 1 | Blockers: [none]",
        completed_steps=["go_to target=desk#4"],
    )
    assert greet.action == "done"
    prompt_txt = captured[0][0]["content"]
    assert "Working Memory Scratchpad:" not in prompt_txt
    assert "Instruction: hey\nCompleted Steps in Current Task:\n- go_to target=desk#4\nNext Intent:" in prompt_txt

    lp_blocker = lp.plan_step(ctx, "retry", scratchpad="Goal: 'retry' | Blockers: [closed door]")
    assert lp_blocker.action == "done"
    assert "Working Memory Scratchpad:" in captured[1][0]["content"]

    # Agent._observe omits Scratchpad line from context_text (passed separately via scratchpad kwarg)
    sim_greet = _make_sim([Thing("desk", 2.0, 0.0, radius=0.4, color="brown")])
    start_pose = sim_greet.trace.poses[0][1:3]
    rec_greet = Agent(lp).do("hey", sim_greet)
    assert rec_greet.outcome == "done"
    assert len(rec_greet.intents) == 1
    assert sim_greet.trace.poses[-1][1:3] == start_pose
    assert "Scratchpad:" not in captured[2][0]["content"]

    # Agent stops repeated go_to loops BEFORE speaking redundant announcements & clamps echoed range distances
    from robot.agent.planner import SYSTEM_PROMPT

    assert "Instruction: go to the mug" in SYSTEM_PROMPT
    assert "target: mug" in SYSTEM_PROMPT
    assert "target: 4" not in SYSTEM_PROMPT

    class LoopingTablePlanner:
        def __init__(self) -> None:
            self.seen_completed: list[tuple[str, ...]] = []

        def plan_step(
            self,
            context: str,
            instruction: str,
            image_url: str | None = None,
            scratchpad: str = "",
            earlier: tuple[str, ...] | list[str] = (),
            completed_steps: tuple[str, ...] | list[str] = (),
        ) -> Intent:
            self.seen_completed.append(tuple(completed_steps))
            step_n = len(self.seen_completed)
            # Step 1: emits move(target="table") with echoed distance=2.5 (clamped to 0.5).
            # Step 2: repeats go_to with a NEW say string (suppressed before robot.say fires).
            return Intent(
                say="Heading to the table." if step_n == 1 else "Driving to desk again!",
                chime="listen",
                posture="curious",
                action="move" if step_n == 1 else "go_to",
                target="table",
                params={"distance": 2.5 if step_n == 1 else 0.5},
                subgoal="go to table",
            )

    sim_desk = _make_sim([Thing("desk", 2.5, 0.0, radius=0.4, color="brown")])
    desk_id = next(o.id for o in sim_desk.snapshot()["objects"] if o.label == "desk")
    loop_planner = LoopingTablePlanner()
    rec_loop = Agent(loop_planner).do("go to the table", sim_desk)
    assert rec_loop.outcome == "done"
    assert len(rec_loop.intents) == 1
    assert len(loop_planner.seen_completed[1]) == 1
    assert loop_planner.seen_completed[1][0] == f'Step 1: go_to target=desk#{desk_id} params={{"distance": 0.5}} (completed)'
    assert sim_desk.trace.poses[-1][1] > 1.3  # clamped to 0.5m standoff, drove up to desk instead of stopping 2.5m away
    said_texts = [text for _, text in sim_desk.trace.said]
    assert said_texts == ["Heading to the table."]

    # User go_to distances > 1.0 are clamped to 1.0
    sim_u_clamp = _make_sim([Thing("person", 3.0, 0.0, radius=0.3, user=True)])
    agent_u = Agent(lp)
    _, _, _, _, u_params = agent_u._prepare_intent(
        sim_u_clamp, Intent(action="go_to", target="user", params={"distance": 2.4})
    )
    assert u_params["distance"] == 1.0

    # search_frontier on an already-known object redirects cleanly to go_to(..., distance=0.5)
    sim_ball = _make_sim([Thing("ball", 2.2, 0.0, radius=0.2, color="red")])
    ball_id = next(o.id for o in sim_ball.snapshot()["objects"] if o.label == "ball")

    class SearchKnownObjectPlanner:
        def __init__(self) -> None:
            self.calls = 0

        def plan_step(self, *args: Any, **kwargs: Any) -> Intent:
            self.calls += 1
            if self.calls == 1:
                return Intent(action="search_frontier", target=ball_id, params={}, subgoal="find red ball")
            return Intent(action="done", subgoal="found ball")

    agent_search = Agent(SearchKnownObjectPlanner())
    rec_search_known = agent_search.do("find the red ball", sim_ball)
    assert rec_search_known.outcome == "done"
    assert sim_ball.pose().x > 1.0
    assert agent_search.short_term.scratchpad.completed_steps == [
        f'Step 1: go_to target=ball#{ball_id} params={{"distance": 0.5}} (completed)'
    ]

    # Mixed target parsing ("10  table", "#2", "clockwise" orbit) + strict-signature planner compatibility
    sim_targets = _make_sim([Thing("table", 2.0, 0.0, radius=0.4), Thing("chair", 0.0, 2.0, radius=0.3)])
    table_id = next(o.id for o in sim_targets.snapshot()["objects"] if o.label == "table")
    recorded_orbits: list[dict[str, Any]] = []
    orig_orbit = sim_targets.orbit

    def spy_orbit(target: Any, **kw: Any) -> Any:
        recorded_orbits.append({"target": target, **kw})
        return orig_orbit(target, **kw)

    sim_targets.orbit = spy_orbit  # type: ignore[method-assign]

    class StrictSignaturePlanner:
        def __init__(self) -> None:
            self.calls = 0

        def plan_step(
            self,
            context: str,
            instruction: str,
            image_url: str | None = None,
            scratchpad: str = "",
            earlier: tuple[str, ...] | list[str] = (),
        ) -> Intent:
            self.calls += 1
            if self.calls == 1:
                return Intent(action="go_to", target=f"{table_id}  table", params={"distance": 0.5}, subgoal="table")
            if self.calls == 2:
                return Intent(
                    action="orbit",
                    target=f"#{table_id}",
                    params={"turns": 0.5, "direction": "clockwise", "radius": 0.6},
                    subgoal="orbit",
                )
            return Intent(action="done", subgoal="done")

    rec_strict = Agent(StrictSignaturePlanner()).do("go to and orbit table", sim_targets)
    assert rec_strict.outcome == "done"
    assert len(recorded_orbits) == 1
    assert recorded_orbits[0]["direction"] == "right"
    assert recorded_orbits[0]["laps"] == 0.5
    assert recorded_orbits[0]["distance"] == 0.6
    assert getattr(recorded_orbits[0]["target"], "id", None) == table_id

    # Missing required target on go_to fails cleanly with NotFound (triggers escalation/failure rather than crashing)
    class NullTargetPlanner:
        def plan_step(self, *args: Any, **kwargs: Any) -> Intent:
            return Intent(action="go_to", target=None, subgoal="nowhere")

    rec_null = Agent(NullTargetPlanner()).do("go nowhere", _make_sim())
    assert rec_null.outcome == "failed"
    assert any("NotFound" in p for p in rec_null.problems)

    # Instant "come here" executes without planner and brings robot to user
    user_scene = Scene(
        width=10.0,
        height=10.0,
        things=[Thing("person", 2.0, 0.0, radius=0.3, user=True)],
        rooms={"room": (-5.0, -5.0, 5.0, 5.0)},
        start=Pose(0.0, 0.0, 0.0),
    )
    sim_user = SimRobot(user_scene)
    rec_come = Agent(NullTargetPlanner()).do("come here", sim_user)
    assert rec_come.outcome == "done"
    assert sim_user.pose().x > 0.5

    # SFT training examples match LocalPlanner.plan_step() ordering and Step N format
    from robot.train.build import build_examples

    ex_done = build_examples(1, seed=7, scenarios=("done",))[0]
    u_msg = ex_done["messages"][1]["content"]
    assert "Scratchpad:" not in u_msg
    assert f"Instruction: {ex_done['instruction']}\nCompleted Steps in Current Task:\n- Step 1: go_to " in u_msg
    assert "(completed)\nNext Intent:" in u_msg

    # Parenthetical alias stripping in Agent._resolve_target ("desk(table)" -> nearest desk Object + "id:<n>")
    sim_alias = _make_sim(
        [
            Thing("desk", 2.4, 0.0, radius=0.4, color="brown"),
            Thing("sofa", 0.0, 2.4, radius=0.5, color="gray"),
            Thing("person", -1.5, 0.0, radius=0.3, user=True),
        ]
    )
    alias_desk_id = next(o.id for o in sim_alias.snapshot()["objects"] if o.label == "desk")
    alias_sofa_id = next(o.id for o in sim_alias.snapshot()["objects"] if o.label == "sofa")
    exec_desk, key_desk, obj_desk = agent_u._resolve_target(sim_alias, "desk(table)")
    assert key_desk == f"id:{alias_desk_id}"
    assert getattr(exec_desk, "id", None) == alias_desk_id
    assert obj_desk is exec_desk

    exec_sofa, key_sofa, obj_sofa = agent_u._resolve_target(sim_alias, "sofa(couch)")
    assert key_sofa == f"id:{alias_sofa_id}"
    assert getattr(exec_sofa, "id", None) == alias_sofa_id
    assert obj_sofa is exec_sofa

    exec_user, key_user, _ = agent_u._resolve_target(sim_alias, "person(user)")
    assert key_user == "user"
    assert getattr(exec_user, "label", "") == "person"

    assert agent_u._resolve_target(sim_alias, "(none)") == (None, "none", None)


def test_single_turn_completion_earlier_reply_trace_sync_and_decide_whitespace(monkeypatch):
    from robot.interfaces.demo import _wrap_planner
    from robot.mind.brain import Reply

    # 1. Brain.decide ignores whitespace tokens and preserves 1-char and multi-char choices
    brain = Brain("http://127.0.0.1:18081")

    class _FakeResp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return self._payload

    top_logprobs = [
        {"token": "\n", "logprob": -0.01},
        {"token": "   ", "logprob": -0.05},
        {"token": "\t", "logprob": -0.10},
        {"token": "no", "logprob": -1.50},
    ]
    monkeypatch.setattr(
        brain.decision_http,
        "post",
        lambda *a, **kw: _FakeResp({"choices": [{"logprobs": {"content": [{"top_logprobs": top_logprobs}]}}]}),
    )
    probs = brain.decide([{"role": "user", "content": "test"}], choices=("yes", "no"))
    assert probs["no"] == 1.0
    assert probs["yes"] == 0.0

    # Pure whitespace top_logprobs falls back to uniform rather than favoring choices[0]
    monkeypatch.setattr(
        brain.decision_http,
        "post",
        lambda *a, **kw: _FakeResp({"choices": [{"logprobs": {"content": [{"top_logprobs": [{"token": "\n", "logprob": -0.1}]}]}}]}),
    )
    assert brain.decide([{"role": "user", "content": "test"}], choices=("yes", "no")) == {"yes": 0.5, "no": 0.5}

    # Single-letter choices ("A", "B", "C") still match cleanly
    monkeypatch.setattr(
        brain.decision_http,
        "post",
        lambda *a, **kw: _FakeResp({"choices": [{"logprobs": {"content": [{"top_logprobs": [{"token": "\n", "logprob": -0.01}, {"token": " B", "logprob": -0.2}]}]}}]}),
    )
    abc = brain.decide([{"role": "user", "content": "path?"}], choices=("A", "B", "C"))
    assert abc["B"] == 1.0

    # 2. LocalPlanner single-turn physical completion avoids 2nd LLM round-trip on simple commands
    class SingleOrMultiBrain:
        def __init__(self, replies: list[str]) -> None:
            self.replies = replies
            self.calls = 0

        def complete(self, messages: list[dict[str, Any]], grammar: str, max_tokens: int, think: int) -> Reply:
            idx = min(self.calls, len(self.replies) - 1)
            self.calls += 1
            return Reply(text=self.replies[idx])

    sim_single = _make_sim([Thing("desk", 2.0, 0.0, radius=0.4, color="brown")])
    live = Live()
    single_brain = SingleOrMultiBrain([
        "say: On it, heading to the desk.\nchime: listen\nposture: curious\naction: go_to\ntarget: desk\nparams: {\"distance\": 0.5}\nsubgoal: drive to desk",
        "say: Should not be called!\nchime: happy\nposture: happy\naction: done\ntarget: none\nparams: {}\nsubgoal: done",
    ])
    wrapped = _wrap_planner(LocalPlanner(brain=single_brain), live)  # type: ignore[arg-type]
    agent = Agent(wrapped, live=live)
    rec_single = agent.do("go to the desk", sim_single)
    assert rec_single.outcome == "done"
    assert single_brain.calls == 1
    assert len(rec_single.intents) == 1
    assert sim_single.pose().x > 0.8
    assert live.state.said == ["On it, heading to the desk."]
    assert live.state.said_total == len(sim_single.trace.said) == 1
    assert 'said "On it, heading to the desk."' in agent.earlier[-1]
    assert agent.earlier[-1].endswith("(done)")

    # Second task stays synced on said_total and Earlier carries previous turn reply cleanly
    multi_brain = SingleOrMultiBrain([
        "say: Driving to desk first.\nchime: listen\nposture: curious\naction: go_to\ntarget: desk\nparams: {\"distance\": 0.5}\nsubgoal: drive to desk",
        "say: Spinning twice now.\nchime: curious\nposture: happy\naction: spin\ntarget: none\nparams: {\"turns\": 1, \"direction\": \"left\"}\nsubgoal: spin",
        "say: All wrapped up!\nchime: happy\nposture: happy\naction: done\ntarget: none\nparams: {}\nsubgoal: completed",
    ])
    wrapped_multi = _wrap_planner(LocalPlanner(brain=multi_brain), live)  # type: ignore[arg-type]
    agent_multi = Agent(wrapped_multi, live=live)
    rec_multi = agent_multi.do("go to the desk, then spin around", sim_single)
    assert rec_multi.outcome == "done"
    assert multi_brain.calls == 3
    assert len(rec_multi.intents) == 3
    assert live.state.said_total == len(sim_single.trace.said) == 4
    assert live.state.said == [t for _, t in sim_single.trace.said][-6:]


def test_planner_sampling_retry_greeting_guard_and_subgoal_grounding(monkeypatch):
    from robot.agent.planner import GRAMMAR
    from robot.mind.brain import Reply

    # 1. GRAMMAR forbids :, ;, {, } in say-char and subgoal-char without banning 'H' or 'O' in say-first
    say_first_line = next(line for line in GRAMMAR.splitlines() if line.startswith("say-first ::="))
    say_char_line = next(line for line in GRAMMAR.splitlines() if line.startswith("say-char ::="))
    subgoal_char_line = next(line for line in GRAMMAR.splitlines() if line.startswith("subgoal-char ::="))
    assert say_first_line == "say-first ::= [a-zA-Z0-9]"
    for ch in (":", ";", "{", "}"):
        assert ch in say_char_line
        assert ch in subgoal_char_line

    # 2. parse_intent strips inline field leaks from say while keeping legitimate words like "action figure"
    leaked = (
        "say: Hey! What's up?; chime: wake; posture: happy; action: done; target: none; params: {}. subgoal: repeat\n"
        "chime: wake\n"
        "posture: happy\n"
        "action: done\n"
        "target: none\n"
        "params: {}\n"
        "subgoal: greet user"
    )
    assert parse_intent(leaked).say == "Hey! What's up?"

    legit = parse_intent(
        "say: Picking up the action figure now.\nchime: listen\nposture: curious\naction: done\ntarget: none\nparams: {}\nsubgoal: answer"
    )
    assert legit.say == "Picking up the action figure now."

    # 3. LocalPlanner passes temperature=0.35, presence_penalty=0.5 with earlier and retries at 0.65 / 0.8 on repeat
    class _SamplingBrain:
        def __init__(self, responses: list[str]) -> None:
            self.responses = responses
            self.calls: list[dict[str, Any]] = []

        def complete(
            self,
            messages: list[dict[str, Any]],
            rules: str,
            max_tokens: int,
            think: int = 0,
            temperature: float = 0.0,
            presence_penalty: float = 0.0,
        ) -> Reply:
            idx = min(len(self.calls), len(self.responses) - 1)
            self.calls.append({"rules": rules, "temperature": temperature, "presence_penalty": presence_penalty})
            return Reply(text=self.responses[idx])

    dup_brain = _SamplingBrain([
        "say: Hey! What's up?\nchime: wake\nposture: happy\naction: done\ntarget: none\nparams: {}\nsubgoal: greet user",
        "say: Yo! How's it going?\nchime: wake\nposture: happy\naction: done\ntarget: none\nparams: {}\nsubgoal: greet user",
    ])
    lp_dup = LocalPlanner(brain=dup_brain)  # type: ignore[arg-type]
    out_greet = lp_dup.plan_step("Robot: in room living.", "hey", earlier=['"hey" -> done (said "Hey! What\'s up?")'])
    assert out_greet.say == "Yo! How's it going?"
    assert len(dup_brain.calls) == 2
    assert dup_brain.calls[0]["rules"] == GRAMMAR
    assert dup_brain.calls[0]["temperature"] == 0.35 and dup_brain.calls[0]["presence_penalty"] == 0.5
    assert dup_brain.calls[1]["temperature"] == 0.65 and dup_brain.calls[1]["presence_penalty"] == 0.8

    # Greeting safety guard normalizes hallucinated targetless navigation on "hey" to done
    bad_greet_brain = _SamplingBrain([
        "say: On it, heading over.\nchime: listen\nposture: curious\naction: go_to\ntarget: none\nparams: {}\nsubgoal: approach target"
    ])
    out_guarded = LocalPlanner(brain=bad_greet_brain).plan_step("Robot: in room living.", "hey")  # type: ignore[arg-type]
    assert out_guarded.action == "done"
    assert out_guarded.target is None
    assert out_guarded.params == {}
    assert out_guarded.subgoal == "greet user"

    # 4. LocalPlanner.plan_step grounds generic/example-echoed subgoals on "go to the table"
    ctx = "Known objects:\nid  label          color   dist    bearing  side\n6   desk(table)    brown    2.0 m   +0°   ahead"
    for bad_sg in ("approach target", "drive to mug"):
        spy = _SamplingBrain([
            "say: Heading over.\nchime: listen\nposture: curious\naction: go_to\ntarget: 6\nparams: {\"distance\": 0.5}\n"
            f"subgoal: {bad_sg}"
        ])
        out_nav = LocalPlanner(brain=spy).plan_step(ctx, "go to the table")  # type: ignore[arg-type]
        assert out_nav.subgoal == "go to table"
        assert spy.calls[0]["temperature"] == 0.0 and spy.calls[0]["presence_penalty"] == 0.0

    # 5. Brain.complete sends temperature and conditional presence_penalty in request JSON
    sent_payloads: list[dict[str, Any]] = []

    class _MockHTTPResp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "action: done"}}], "timings": {}}

    real_brain = Brain("http://127.0.0.1:18081")
    monkeypatch.setattr(
        real_brain.http,
        "post",
        lambda path, *, json: sent_payloads.append(json) or _MockHTTPResp(),
    )
    real_brain.complete([{"role": "user", "content": "hi"}], GRAMMAR, 64)
    real_brain.complete([{"role": "user", "content": "hi"}], GRAMMAR, 64, temperature=0.35, presence_penalty=0.5)
    assert sent_payloads[0]["temperature"] == 0.0
    assert "presence_penalty" not in sent_payloads[0]
    assert sent_payloads[1]["temperature"] == 0.35
    assert sent_payloads[1]["presence_penalty"] == 0.5


def test_living_mind_surroundings_and_visual_qa_guards() -> None:
    from robot.agent.api import Object
    from robot.agent.context import describe, synthesize_perception
    from robot.agent.planner import Reply, _is_conversational_or_visual_query

    assert synthesize_perception([]) == "(nothing detected yet)"
    objs = [
        Object(
            id=2,
            label="chair",
            position=(0.0, 1.6),
            distance=1.6,
            bearing=90.0,
            side="left",
            radius=0.35,
            color="blue",
            last_seen=0.0,
        ),
        Object(
            id=7,
            label="desk",
            position=(0.9, 0.0),
            distance=0.9,
            bearing=4.0,
            side="ahead",
            radius=0.5,
            color="grey",
            last_seen=0.0,
        ),
        Object(
            id=1,
            label="person",
            position=(-1.8, -0.8),
            distance=2.1,
            bearing=-155.0,
            side="behind-right",
            radius=0.25,
            color="",
            last_seen=0.0,
        ),
    ]
    summary = synthesize_perception(objs, user=1)
    assert summary == "ahead: grey desk(table) (0.9 m); sides: blue chair (1.6 m); behind: you (person) (2.1 m)"

    ctx = describe(
        {
            "pose": Pose(0.0, 0.0, 0.0),
            "room": "living",
            "rooms": ["living"],
            "battery": 0.92,
            "objects": objs,
            "user": 1,
        }
    )
    assert f"Surroundings: {summary}" in ctx
    assert ctx.index("Surroundings:") < ctx.index("Known objects:")

    class _MovingBrain:
        def __init__(self, reply_text: str) -> None:
            self.reply_text = reply_text

        def complete(self, messages: Any, grammar: str, max_tokens: int, think: bool = False) -> Reply:
            return Reply(text=self.reply_text)

    rogue = _MovingBrain(
        "say: On it, heading over!\n"
        "chime: listen\n"
        "posture: curious\n"
        "action: go_to\n"
        "target: 7\n"
        'params: {"distance": 0.5}\n'
        "subgoal: approach desk"
    )
    lp_rogue = LocalPlanner(brain=rogue)  # type: ignore[arg-type]

    for q, expected_words in (
        ("what do you see", ("grey desk", "blue chair", "you")),
        ("what's around you", ("grey desk", "blue chair")),
        ("where is the desk", ("grey desk", "ahead")),
        ("what's behind you", ("behind", "you")),
    ):
        intent = lp_rogue.plan_step(ctx, q)
        assert intent.action == "done"
        assert intent.target is None
        assert intent.subgoal == "describe surroundings"
        low_say = intent.say.lower()
        assert any(w in low_say for w in expected_words), f"{q!r} -> {intent.say!r}"
        assert "on it" not in low_say

    obedient = _MovingBrain(
        "say: I see a grey desk ahead and a blue chair to the side.\n"
        "chime: curious\n"
        "posture: curious\n"
        "action: done\n"
        "target: none\n"
        "params: {}\n"
        "subgoal: answer"
    )
    obedient_intent = LocalPlanner(brain=obedient).plan_step(ctx, "what do you see?")  # type: ignore[arg-type]
    assert obedient_intent.action == "done"
    assert obedient_intent.target is None
    assert obedient_intent.subgoal == "describe surroundings"
    assert "grey desk" in obedient_intent.say.lower()

    assert _is_conversational_or_visual_query("what do you see?") is True
    assert _is_conversational_or_visual_query("go to the desk") is False
    assert _is_conversational_or_visual_query("turn left") is False
