"""Exercise the 3D referee from synthetic ground truth, without a model or Habitat."""

import math
from types import SimpleNamespace

import pytest

from robot.eval.home3d import TASKS, Referee
from robot.sim.world import Trace

TASK = {task.id: task for task in TASKS}
TV = {"id": 1, "label": "tv", "x": 4.0, "y": 0.0, "radius": 0.4}


def referee(poses=(), answers=(), objects=()):
    home = SimpleNamespace(path=poses, truth={"objects": objects})
    robot = SimpleNamespace(trace=Trace(said=list(enumerate(answers))))
    return Referee(home, robot)


def _rect(i, label, x, y, half_x, half_y):
    return {"id": i, "label": label, "x": x, "y": y, "radius": max(half_x, half_y),
            "footprint": ((x - half_x, y - half_y), (x + half_x, y - half_y),
                          (x + half_x, y + half_y), (x - half_x, y + half_y))}


def test_nearest_table_task_requires_ending_near_the_table_nearest_the_start():
    objects = [{"id": i, "label": "table", "x": x, "y": 0.0, "radius": 0.4}
               for i, x in enumerate((2.0, 4.0), 1)]
    for x, passed in [(0.0, False), (1.5, True), (3.5, False)]:
        sim = referee([(0.0, 0.0, 0.0, 0.0), (5.0, x, 0.0, 0.0)], objects=objects)
        assert (not TASK["T1"].problem(sim, {"xy": (0.0, 0.0)})) is passed


def test_nearest_table_task_uses_the_footprint_instead_of_its_longest_half_side():
    table = _rect(1, "table", 0.0, 0.0, 0.2, 1.0)
    for (x, y), passed in [((1.1, 0.0), False), ((0.9, 0.0), True), ((1.0, 1.3), False), ((0.0, 0.0), True)]:
        sim = referee([(0.0, 2.0, 0.0, 0.0), (5.0, x, y, 0.0)], objects=[table])
        assert (not TASK["T1"].problem(sim, {"xy": (2.0, 0.0)})) is passed


@pytest.mark.parametrize(("task_id", "label"), [("T1", "table"), ("N3", "chair")])
def test_nearest_object_tasks_choose_the_closest_edge_not_the_closest_centre(task_id, label):
    objects = [_rect(1, label, 0.0, 1.0, 0.1, 0.1), _rect(2, label, 1.5, 0.0, 1.3, 0.2)]
    for (x, y), passed in [((0.0, 1.0), False), ((1.0, 0.0), True)]:
        sim = referee([(0.0, 0.0, 0.0, 0.0), (5.0, x, y, 0.0)], objects=objects)
        assert (not TASK[task_id].problem(sim, {"xy": (0.0, 0.0)})) is passed


@pytest.mark.parametrize("task_id", ["N4", "P1"])
def test_nearest_table_to_sofa_uses_both_footprints(task_id):
    objects = [_rect(1, "sofa", 0.0, 0.0, 2.0, 0.2), _rect(2, "table", 0.0, 1.5, 0.2, 0.2),
               _rect(3, "table", 2.5, 0.0, 0.2, 0.2)]
    for (x, y), passed in [((0.0, 0.8), False), ((2.2, 0.0), True)]:
        sim = referee([(0.0, 0.0, 0.0, 0.0), (5.0, x, y, 0.0)], objects=objects)
        assert (not TASK[task_id].problem(sim, {"xy": (0.0, 0.0)})) is passed


@pytest.mark.parametrize("task_id", ["Q4", "C2"])
def test_sofa_vs_tv_uses_footprints_not_centres(task_id):
    objects = [_rect(1, "sofa", 2.3, 0.0, 1.9, 0.2), _rect(2, "tv stand", 0.0, 1.0, 0.2, 0.1)]
    for label, (x, y), passed in [("TV", (0.0, 1.0), False), ("sofa", (2.3, 0.0), True)]:
        sim = referee([(0.0, 0.0, 0.0, 0.0), (5.0, x, y, 0.0)], [f"The {label} is closer."], objects)
        assert (not TASK[task_id].problem(sim, {"xy": (0.0, 0.0)})) is passed


def test_closer_tv_task_accepts_arrival_at_tv_when_stand_is_closest_at_start():
    objects = [_rect(1, "tv stand", 1.0, 0.0, 0.1, 0.1), _rect(2, "tv", 2.0, 0.0, 0.1, 0.1),
               _rect(3, "sofa", 4.0, 0.0, 0.1, 0.1)]
    for x, passed in [(1.0, True), (2.0, True), (4.0, False)]:
        sim = referee([(0.0, 0.0, 0.0, 0.0), (5.0, x, 0.0, 0.0)], objects=objects)
        assert (not TASK["C2"].problem(sim, {"xy": (0.0, 0.0)})) is passed


@pytest.mark.parametrize(("path", "arrival", "problem"), [
    ([(0, 0, 0, 0), (1, 0, 0, 0)], (0, 0, 0), "before the return request"),
    ([(0, 0, 0, 0), (2, 3, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0)], (0, 0, 0), "before the return request"),
    ([(0, 0, 0, 0), (2, 3, 0, 0), (4, 3, 0, 0)], (3, 0, 0), "did not return"),
    ([(0, 0, 0, 0), (2, 3, 0, 0), (3.9, 0.2, 0, 0), (4, 0, 0, 0)], (3, 0, 0), "did not hold still"),
    ([(0, 0, 0, 0), (2, 3, 0, 0), (4, 1, 1, 1), (6, 0.1, 0, 0.5), (7, 0.1, 0, 0.5)], (3, 0, 0), ""),
])
def test_come_back_requires_arrival_before_return_and_a_still_finish(path, arrival, problem):
    sim = referee(path, objects=[TV])
    start = {"pose": (0, 0, 0), "ends": [arrival, path[-1][1:]]}
    actual = TASK["R2"].problem(sim, start)
    assert problem in actual if problem else not actual


@pytest.mark.parametrize(("outbound_x", "passed"), [(1.8, False), (3.8, True), (5.8, True)])
def test_tv_approach_and_return_accept_only_tv_or_stand(outbound_x, passed):
    monitor = {"id": 2, "label": "monitor", "x": 2.0, "y": 0.0, "radius": 0.3}
    stand = {"id": 3, "label": "tv stand", "x": 6.0, "y": 0.0, "radius": 0.3}
    path = [(0, 0, 0, 0), (2, outbound_x, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0)]
    objects = [TV, monitor, stand]
    start = {"pose": path[0][1:], "ends": [path[1][1:], path[-1][1:]]}
    assert (not TASK["N5"].problem(referee(path[:2], objects=objects), start)) is passed
    assert (not TASK["R2"].problem(referee(path, objects=objects), start)) is passed


@pytest.mark.parametrize(("path", "problem"), [
    ([(1, -5, 1, math.pi / 2)], "5.00 m sideways"),
    ([(1, -1, 0.5, math.pi / 2), (2, 0, 1, math.pi / 2)], "1.00 m sideways"),
    ([(1, 0, 0.5, 0), (2, 0, 1, math.pi / 2)], "heading changed"),
    ([(1, 0, 0.6, math.pi / 2)], "expected 1 m"),
    ([(1, 0, 1.1, math.pi / 2), (2, 0, 1, math.pi / 2)], "backtracked"),
    ([(1, 0, 1.03, math.pi / 2), (2, 0, 1.015, math.pi / 2), (3, 0, 1, math.pi / 2)], "backtracked"),
    ([(1, 0, -0.1, math.pi / 2), (2, 0, 1, math.pi / 2)], "backtracked"),
    ([(1, 0, 0.5, math.pi / 2), (2, 0, 0.49, math.pi / 2), (3, 0, 1, math.pi / 2)], ""),
    ([(1, -0.05, 0.5, math.pi / 2 + 0.1), (2, -0.07, 1.1, math.pi / 2)], ""),
])
def test_forward_uses_the_starting_heading_and_the_whole_path(path, problem):
    sim = referee([(0, 0, 0, math.pi / 2), *path])
    start = {"pose": (0, 0, math.pi / 2)}
    actual = TASK["M3"].problem(sim, start)
    assert problem in actual if problem else not actual


@pytest.mark.parametrize(("answers", "passed"), [
    (["No, there is no bike.", "Yes, I can keep looking."], False),
    (["No, there is no bike.", "Yes."], False),
    (["No bike here.", "Yes."], False),
    (["Yes.", "Actually, there isn't a bicycle here."], False),
    (["Yes, there is a bike. No, that's not a bike."], False),
    (["I heard someone say yes."], False),
    (["Yes, I found a chair."], False),
    (["There is a bike rack."], False),
    (["I can see a picture of bicycles."], False),
    (["There is a bike outside."], False),
    (["There is a bike in another apartment."], False),
    (["There is a bike. There is no bicycle."], False),
    (["There are zero bikes in the apartment."], False),
    (["There is a bike. There are 0 bicycles."], False),
    (["There is a bike. The bicycle is outside."], False),
    (["There is a bike, but not in this apartment."], False),
    (["I don't think there's a bike in the apartment.", "Yes."], False),
    (["Yes."], True),
    (["Yes, there is one."], True),
    (["Yes, there is one in the living room."], True),
    (["Yes, there is one in another apartment."], False),
    (["No bike here.", "Yes, there is one in the living room."], False),
    (["I can see a blue bicycle by the desk."], True),
    (["There is a bike parked inside the apartment."], True),
    (["Yes, there is a bike, parked by the desk."], True),
    (["Yes, the bike is in the living room."], True),
    (["A bike is present."], True),
    (["Yes, a bike is here."], True),
    (["Yes, there is a bike. I don't know whose it is."], True),
    (["There is a bike, but I don't know whose bike it is."], True),
    (["There is a bike, but not by the desk."], True),
    (["No, there is no bike.", "There’s a bicycle by the desk."], True),
])
def test_bike_answer_scores_the_claim_not_a_yes_token(answers, passed):
    sim = referee(answers=answers)
    assert TASK["Q5"].problem(sim, {}) == (
        "" if passed else "final answer did not unambiguously confirm that a bike is present"
    )
