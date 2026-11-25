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
    assert TASK["R2"].check(sim, start) is (not problem)
    assert problem in TASK["R2"].problem(sim, start)


@pytest.mark.parametrize(("path", "problem"), [
    ([(1, -5, 1, math.pi / 2)], "5.00 m sideways"),
    ([(1, -1, 0.5, math.pi / 2), (2, 0, 1, math.pi / 2)], "1.00 m sideways"),
    ([(1, 0, 0.5, 0), (2, 0, 1, math.pi / 2)], "heading changed"),
    ([(1, 0, 0.6, math.pi / 2)], "expected 1 m"),
    ([(1, -0.05, 0.5, math.pi / 2 + 0.1), (2, -0.07, 1.1, math.pi / 2)], ""),
])
def test_forward_uses_the_starting_heading_and_the_whole_path(path, problem):
    sim = referee([(0, 0, 0, math.pi / 2), *path])
    start = {"pose": (0, 0, math.pi / 2)}
    assert TASK["M3"].check(sim, start) is (not problem)
    assert problem in TASK["M3"].problem(sim, start)


@pytest.mark.parametrize(("answers", "passed"), [
    (["No, there is no bike.", "Yes, I can keep looking."], False),
    (["No, there is no bike.", "Yes."], False),
    (["Yes.", "Actually, there isn't a bicycle here."], False),
    (["Yes, there is a bike. No, that's not a bike."], False),
    (["I heard someone say yes."], False),
    (["Yes, I found a chair."], False),
    (["There is a bike rack."], False),
    (["I can see a picture of bicycles."], False),
    (["There is a bike outside."], False),
    (["There is a bike in another apartment."], False),
    (["Yes."], True),
    (["Yes, there is one."], True),
    (["I can see a blue bicycle by the desk."], True),
    (["There is a bike parked inside the apartment."], True),
    (["No, there is no bike.", "There’s a bicycle by the desk."], True),
])
def test_bike_answer_scores_the_claim_not_a_yes_token(answers, passed):
    sim = referee(answers=answers)
    assert TASK["Q5"].check(sim, {}) is passed
    assert TASK["Q5"].problem(sim, {}) == (
        "" if passed else "final answer did not unambiguously confirm that a bike is present"
    )
