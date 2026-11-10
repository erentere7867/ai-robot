"""Phase 4 acceptance (sim): hand-written programs pass, the gate rejects 20 of 20 red-team programs, and a stop leaves
the robot stationary within 300 ms."""

import collections
import threading
import time

import pytest

from robot.agent.api import Preempted, api_card
from robot.agent.gate import check
from robot.agent.runner import Runner
from robot.eval.programs import PROGRAMS
from robot.eval.redteam import RED_TEAM
from robot.eval.suite import CASES, living_room, run_case, two_rooms, with_things
from robot.eval.wrong import WRONG
from robot.sim.world import SimRobot, Thing

BY_ID = {c.id: c for c in CASES}


def test_the_suite_has_fifty_instructions_in_seven_categories():
    counts = collections.Counter(c.category for c in CASES)
    assert len(CASES) == 50 and len({c.id for c in CASES}) == 50
    assert counts == {"relative": 8, "object": 10, "search": 8, "conditional": 8, "following": 6, "memory": 5,
                      "ambiguous": 5}


def test_the_api_card_fits_its_token_budget():
    assert len(api_card()) <= 3800  # about 950 tokens, cached after the first plan


@pytest.mark.parametrize("case_id", sorted(PROGRAMS))
def test_hand_written_programs_pass(case_id):
    ok, why, _ = run_case(BY_ID[case_id], PROGRAMS[case_id])
    assert ok, why


@pytest.mark.parametrize("name", sorted(RED_TEAM))
def test_the_gate_rejects_red_team_programs(name):
    result = check(RED_TEAM[name], SimRobot(living_room()))
    assert not result.ok, f"{name} passed the gate"


CIRCLES = """\
behind = [o for o in robot.where("box") if o.side.startswith("behind")]
box = min(behind, key=lambda o: o.distance) if behind else robot.search("box")
if box:
    robot.orbit(box, laps=3)
    robot.say("Done. Three circles around the box.")
else:
    robot.say("I can't see a box behind me.")
"""

SEARCH = """\
box = robot.search("yellow box", area="here")
if box:
    where = robot.describe(box, question="In a few words, where is this box?")
    robot.go_to(robot.user(), distance=1.0)
    robot.face(robot.user())
    robot.say("I found a yellow box, " + where)
else:
    robot.say("I searched the room and did not find a yellow box.")
"""


def test_no_check_passes_an_idle_robot():
    """A check an idle robot passes measures luck, not the planner."""
    def idle(case):
        sim = SimRobot(case.scene())
        try:
            sim.wait(60.0)
        except Preempted:  # the user said "stop"
            pass
        return case.check(sim)

    assert [c.id for c in CASES if idle(c)] == []


def test_the_specs_worked_examples_run():
    """Fable's two programs from section 11 (the brief's tasks are held out of the suite, not of tests)."""
    sim = SimRobot(living_room())
    assert check(CIRCLES, sim).ok
    assert Runner(sim).run(CIRCLES).ok
    from robot.eval.suite import laps, thing

    assert abs(laps(sim, thing(sim, "box"))) >= 2.8
    scene = with_things(two_rooms(), Thing("box", 3.5, -1.6, 0.25, "brown"), Thing("box", 1.2, 1.8, 0.25, "yellow"))
    sim = SimRobot(scene)
    sim.x = 1.0  # start in the kitchen doorway, so "this room" is the kitchen
    assert check(SEARCH, sim).ok
    assert Runner(sim).run(SEARCH).ok
    assert any("yellow box" in text for _, text in sim.trace.said)


def test_open_ended_programs_pass_the_dry_run():
    follow = ("while True:\n    me = robot.user()\n    robot.face(me)\n"
              "    robot.drive(metres_per_second=0.0, seconds=0.1)\n")
    assert check(follow, SimRobot(living_room())).ok


def test_a_stop_ends_motion_within_300_ms():
    sim = SimRobot(living_room(), realtime=True)
    runner = Runner(sim)
    results = []
    worker = threading.Thread(target=lambda: results.append(runner.run("robot.go_to((3.0, -0.2))")))
    worker.start()
    time.sleep(1.5)
    assert sim.v_now > 0.0, "the robot should be driving"
    stopped = time.perf_counter()
    runner.stop()
    worker.join(2.0)
    assert time.perf_counter() - stopped <= 0.3
    assert results and results[0].status == "preempted" and sim.v_now == 0.0


@pytest.mark.parametrize("case_id", sorted(WRONG))
def test_plausible_but_wrong_programs_fail(case_id):
    """A check must reject a close-but-wrong program too, or training against the checks learns to exploit them."""
    for program in WRONG[case_id]:
        ok, why, _ = run_case(BY_ID[case_id], program)
        assert not ok, f"{case_id} accepted: {program!r}"


@pytest.mark.parametrize(("case_id", "program"), [
    ("R3", 'robot.turn(90, "left")\n'),  # what the untrained planner writes: positional, and right
    ("R3", 'robot.turn("left", 90)\n'),  # swapped: the types say which is which
    ("R1", 'robot.move(1, "backward")\nrobot.spin(2)\n'),
    ("O6", 'robot.go_to(robot.where("ball"))\nrobot.go_to("start")\n'),  # "start" is where the task began
    ("S6", 'mug = robot.search("mug", "the kitchen")\nrobot.go_to(mug, 0.3)\n'),
])
def test_the_forms_people_and_models_write_are_accepted(case_id, program):
    ok, why, _ = run_case(BY_ID[case_id], program)
    assert ok, why


def test_a_count_of_turns_that_means_degrees_is_refused():
    result = check("robot.spin(turns=180)\n", SimRobot(living_room()))
    assert not result.ok and "full turns" in result.message


def test_turn_left_alone_is_a_quarter_turn_and_a_bare_angle_asks_which_way():
    ok, why, _ = run_case(BY_ID["R3"], 'robot.turn("left")\n')
    assert ok, why
    result = check("robot.turn(90)\n", SimRobot(living_room()))
    assert not result.ok and "which way" in result.message
