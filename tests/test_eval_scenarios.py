"""Capability-eval graders (pure, synthetic ground truth) and one end-to-end runner smoke test."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from robot.eval.scenarios import (
    SCENARIO_LIST,
    SCENARIOS,
    CameraModel,
    Grade,
    ScenarioRun,
    SceneSetup,
    TurnRecord,
    family_of,
    footprint_overlaps,
    footprint_radius_mm,
    grade_behind_box,
    grade_between,
    grade_come_back,
    grade_describe,
    grade_find_hidden,
    grade_go_to_ball,
    grade_go_to_dock,
    grade_greet,
    grade_look_around_report,
    grade_move_forward,
    mentioned_families,
    plan_layout,
    reflex_obstacle_id,
    select_scenarios,
    visible_object_ids,
)

ROOT = Path(__file__).resolve().parents[1]


def pose(x=0.0, y=0.0, h=0.0, tilt=0.0):
    return {"x_mm": x, "y_mm": y, "heading_deg": h, "head_tilt_deg": tilt}


def obj(obj_id, label, shape, x, y, size=100.0, height=None, width=None, depth=None):
    return {
        "id": obj_id, "label": label, "shape": shape, "x_mm": x, "y_mm": y, "z_mm": 0.0,
        "width_mm": width or size, "depth_mm": depth or size, "height_mm": height or size,
    }


def scene(**moves):
    objects = [
        obj("red_ball", "Red Ball", "sphere", 680, 160, 96),
        obj("blue_cube", "Blue Cube", "box", 620, -210, 110),
        obj("yellow_duck", "Yellow Toy Duck", "sphere", 470, 340, 90, height=96),
        obj("green_mug", "Green Mug", "cylinder", 840, -30, 86, height=116),
        obj("cardboard_box", "Cardboard Box", "box", 1120, 280, 220, height=190),
        obj("charging_dock", "Charging Dock", "box", -680, 0, width=180, depth=220, height=130),
    ]
    for o in objects:
        if o["id"] in moves:
            o["x_mm"], o["y_mm"] = moves[o["id"]]
    return objects


def turn(text="", say="", before=None, after=None, trajectory=(), reflexes=(), restarted=False):
    return TurnRecord(
        utterance=text, say=say, pose_before=before or pose(), pose_after=after or before or pose(),
        trajectory=list(trajectory), reflexes=list(reflexes), restarted_before=restarted,
    )


def run(*turns, objects=None, name="test"):
    objects = objects if objects is not None else scene()
    return ScenarioRun(name, 0, list(turns), objects, objects)


# --------------------------------------------------------------------------- text and geometry


def test_family_synonyms_plurals_and_no_partial_words():
    assert mentioned_families("I see two boxes, a sphere and a cup") == {"box", "ball", "mug"}
    assert mentioned_families("a cardboard carton near the charger") == {"box", "dock"}
    assert mentioned_families("Blocks! A duck by the docking station") == {"cube", "duck", "dock"}
    assert mentioned_families("something is blocking me while boxing") == set()
    assert [family_of(o) for o in scene()] == ["ball", "cube", "duck", "mug", "box", "dock"]
    assert family_of(obj("cone", "Orange Cone", "cylinder", 0, 0)) is None


def test_camera_ground_truth_frustum_range_and_occlusion():
    objects = [obj("ahead", "Red Ball", "sphere", 800, 0), obj("behind", "Blue Cube", "box", -800, 0),
               obj("far", "Green Mug", "cylinder", 2900, 0), obj("left", "Yellow Duck", "sphere", 0, 900)]
    assert visible_object_ids(pose(), objects) == {"ahead"}
    assert visible_object_ids(pose(h=90), objects) == {"left"}
    assert visible_object_ids(pose(h=180), objects) == {"behind"}
    hidden = [obj("small", "Red Ball", "sphere", 1200, 0, 90), obj("wall", "Cardboard Box", "box", 600, 0, 260)]
    assert visible_object_ids(pose(), hidden) == {"wall"}


def test_camera_ground_truth_matches_simulator_detector():
    import random

    from robot.body.hal import BodyPose
    from robot.world.sandbox import SimulationSandbox

    sandbox = SimulationSandbox(None)
    if not hasattr(sandbox, "detect_objects"):
        pytest.skip("sandbox has no geometric detector to compare against")
    camera = CameraModel.from_sandbox(sandbox)
    rng = random.Random(7)
    agree = 0
    for _ in range(500):
        p = BodyPose(x_mm=rng.uniform(-1300, 1300), y_mm=rng.uniform(-1300, 1300),
                     heading_deg=rng.uniform(-180, 180), head_tilt_deg=rng.uniform(-25, 35))
        truth = {d.id for d in sandbox.detect_objects(p)}
        agree += truth == visible_object_ids(vars(p), sandbox.list_objects(), camera)
    assert agree >= 490


def test_footprint_contact_and_reflex_attribution():
    box = obj("cardboard_box", "Cardboard Box", "box", 900, 400, 220, height=190)
    assert footprint_overlaps(pose(900, 400), box)
    assert footprint_overlaps(pose(700, 400), box)  # front bumper 141 mm ahead reaches the box face
    assert not footprint_overlaps(pose(500, 400), box)
    ball = obj("red_ball", "Red Ball", "sphere", 0, 300, 96)
    assert footprint_overlaps(pose(0, 200, 90), ball)
    assert not footprint_overlaps(pose(0, 0, 90), ball)
    objects = [box, obj("red_ball", "Red Ball", "sphere", 0, -800, 96)]
    assert reflex_obstacle_id(pose(600, 400, 0), objects) == "cardboard_box"
    assert reflex_obstacle_id(pose(600, 400, 180), objects) is None


def test_layout_is_deterministic_non_overlapping_and_inside_room():
    objects = scene()
    setup = SceneSetup(overrides={"yellow_duck": (-800.0, 350.0)}, remove=("green_mug",))
    first = plan_layout(objects, 3, setup)
    assert first == plan_layout(objects, 3, setup)
    assert first != plan_layout(objects, 4, setup)
    assert "green_mug" not in first and len(first) == 5
    assert math.dist(first["yellow_duck"], (-800.0, 350.0)) <= 100.0
    radii = {o["id"]: footprint_radius_mm(o) for o in objects}
    for seed in range(50):
        plan = plan_layout(objects, seed, setup)
        for a, (ax, ay) in plan.items():
            assert abs(ax) + radii[a] <= 1500 and abs(ay) + radii[a] <= 1500
            assert math.hypot(ax, ay) >= radii[a] + 180
            for b, (bx, by) in plan.items():
                if a < b:
                    assert math.hypot(ax - bx, ay - by) >= radii[a] + radii[b] + 60 - 0.2


def test_select_scenarios():
    assert [s.name for s in select_scenarios("all")] == [s.name for s in SCENARIO_LIST]
    assert [s.name for s in select_scenarios("greet, describe,greet")] == ["greet", "describe"]
    with pytest.raises(ValueError, match="nope"):
        select_scenarios("greet,nope")


# --------------------------------------------------------------------------- graders


def test_motionless_silent_robot_passes_nothing():
    for scenario in SCENARIO_LIST:
        objects = [o for o in scene(**scenario.setup.overrides) if o["id"] not in scenario.setup.remove]
        turns = [turn(text, restarted=i in scenario.restart_before) for i, text in enumerate(scenario.utterances)]
        grade = scenario.grade(run(*turns, objects=objects, name=scenario.name))
        assert isinstance(grade, Grade) and grade.reason
        assert not grade.passed, scenario.name


def test_greet():
    assert grade_greet(run(turn("hi!", "Hello!", pose(), pose(10, 0, 5, tilt=20)))).passed
    assert not grade_greet(run(turn("hi!", "Hello!", pose(), pose(100, 0)))).passed
    assert not grade_greet(run(turn("hi!", "Hello!", pose(), pose(0, 0, 45)))).passed
    assert not grade_greet(run(turn("hi!", "  ", pose(), pose()))).passed


def test_turn_graders_use_signed_wrapped_heading():
    turn_around, look_left = SCENARIOS["turn_around"].grade, SCENARIOS["look_left"].grade
    assert turn_around(run(turn(before=pose(h=10), after=pose(h=-170)))).passed
    assert turn_around(run(turn(before=pose(h=0), after=pose(h=-160)))).passed
    assert not turn_around(run(turn(before=pose(h=0), after=pose(h=140)))).passed
    assert look_left(run(turn(before=pose(h=170), after=pose(h=-100)))).passed
    assert not look_left(run(turn(before=pose(h=0), after=pose(h=-90)))).passed


def test_move_forward_in_start_frame():
    assert grade_move_forward(run(turn(before=pose(), after=pose(300, 20, 3)))).passed
    assert grade_move_forward(run(turn(before=pose(0, 0, 90), after=pose(0, 250, 90)))).passed
    assert not grade_move_forward(run(turn(before=pose(), after=pose(-300, 0)))).passed
    assert not grade_move_forward(run(turn(before=pose(), after=pose(900, 0)))).passed
    assert not grade_move_forward(run(turn(before=pose(), after=pose(300, 0, 30)))).passed


def test_go_to_ball_needs_distance_and_facing():
    assert grade_go_to_ball(run(turn(after=pose(330, 160, 0)))).passed
    assert not grade_go_to_ball(run(turn(after=pose(330, 160, 180)))).passed
    assert not grade_go_to_ball(run(turn(after=pose(100, 160, 0)))).passed


def test_describe_counts_only_families_in_view():
    assert grade_describe(run(turn(say="I see a red ball and a green mug."))).passed
    only_one = grade_describe(run(turn(say="A ball!")))
    assert not only_one.passed and only_one.details["correct"] == ["ball"]
    behind = grade_describe(run(turn(say="I see the charging dock and a duck.")))
    assert not behind.passed and "dock" in behind.details["unseen_mentions"]


def test_look_around_report_needs_four_real_families():
    assert grade_look_around_report(run(turn(say="A ball, a cube, a mug and the dock."))).passed
    report = grade_look_around_report(run(turn(say="A ball, a cube and a mug."), objects=scene()))
    assert not report.passed
    no_duck = [o for o in scene() if o["id"] != "yellow_duck"]
    hallucinated = grade_look_around_report(run(turn(say="A ball, a duck, a cube and a mug."), objects=no_duck))
    assert not hallucinated.passed and hallucinated.details["hallucinated"] == ["duck"]


def test_behind_box_geometry_and_collisions():
    objects = scene(cardboard_box=(900, 400))
    look = turn("what do you see?", "a box")
    path = [pose(500, -100, 0), pose(1000, -50, 45), pose(1250, 200, 90)]
    good = run(look, turn("go behind that box", before=pose(), after=pose(1250, 560, 180), trajectory=path),
               objects=objects)
    assert grade_behind_box(good).passed
    short = run(look, turn(before=pose(), after=pose(500, 200)), objects=objects)
    assert not grade_behind_box(short).passed
    through = run(look, turn(before=pose(), after=pose(1250, 560), trajectory=[pose(900, 400)]), objects=objects)
    assert grade_behind_box(through).details["contacts"] == 1 and not grade_behind_box(through).passed
    reflex = {"reflex": "obstacle_stop", "pose": pose(650, 400, 0)}
    bumped = run(look, turn(before=pose(), after=pose(1250, 560), reflexes=[reflex]), objects=objects)
    assert not grade_behind_box(bumped).passed


def test_between_dock_and_come_back():
    assert grade_between(run(turn(after=pose(640, -40)))).passed
    assert not grade_between(run(turn(after=pose(200, -40)))).passed
    assert grade_go_to_dock(run(turn(after=pose(-300, 0, 180)))).passed
    assert not grade_go_to_dock(run(turn(after=pose(200, 0)))).passed
    went = run(turn(before=pose(), after=pose(400, 100)), turn(before=pose(400, 100), after=pose(100, 50, 180)))
    assert grade_come_back(went).passed
    assert not grade_come_back(run(turn(), turn())).passed


def test_find_hidden_facing_or_seen_and_located():
    objects = scene(yellow_duck=(-800, 350))
    bearing = math.degrees(math.atan2(350, -800))
    assert grade_find_hidden(run(turn(after=pose(h=bearing + 30)), objects=objects)).passed
    claim = "The yellow duck is behind me, near the dock."
    assert not grade_find_hidden(run(turn(say=claim, after=pose()), objects=objects)).passed
    looked = [pose(h=bearing)]
    assert grade_find_hidden(run(turn(say=claim, after=pose(), trajectory=looked), objects=objects)).passed
    assert not grade_find_hidden(run(turn(say="Found it!", after=pose(), trajectory=looked), objects=objects)).passed


def test_memory_recall_requires_restart_and_answer():
    grade = SCENARIOS["memory_name"].grade
    assert grade(run(turn("my name is Deniz", "Nice to meet you"), turn("what's my name?", "You're Deniz!",
                                                                          restarted=True))).passed
    assert not grade(run(turn(), turn(say="You're Deniz!"))).passed
    assert not grade(run(turn(), turn(say="I don't know", restarted=True))).passed
    pref = SCENARIOS["memory_pref"]
    assert "green_mug" in pref.setup.remove
    assert pref.grade(run(turn(), turn(say="Green, of course.", restarted=True))).passed


# --------------------------------------------------------------------------- end to end


def test_runner_end_to_end_with_null_backend(tmp_path):
    out = tmp_path / "eval.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_scenarios.py"), "--backend", "null",
         "--scenarios", "greet,memory_name", "--timeout", "60", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "scenario" in proc.stdout and "memory_name" in proc.stdout and "Passed" in proc.stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["backend"] == "null" and report["finished_at"]
    assert report["summary"]["total"] == 2
    assert [r["scenario"] for r in report["results"]] == ["greet", "memory_name"]
    for result in report["results"]:
        assert isinstance(result["passed"], bool) and result["reason"]
        assert {o["id"] for o in result["objects_start"]} >= {"red_ball", "cardboard_box"}
        for entry in result["transcript"]:
            assert {"utterance", "say", "actions", "pose_before", "pose_after"} <= entry.keys()
    memory = report["results"][1]["transcript"]
    assert [t["utterance"] for t in memory] == ["my name is Deniz", "what's my name?"]
    assert [t["restarted_before"] for t in memory] == [False, True]
