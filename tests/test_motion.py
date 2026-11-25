import math
from concurrent.futures import Future
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from robot.agent.api import PathBlocked, Pose
from robot.eval.suite import two_rooms
from robot.ros.motion import COMMAND_AGE_NS, NAV, MotionGate, NavGoal
from robot.sim.habitat_ros import Hardware
from robot.sim.world import Scene, SimRobot, Thing


def test_sim_drive_path_reaches_the_corner_before_turning():
    robot = SimRobot(Scene(10, 10, [], {}, start=Pose(0, 0, 180)))
    robot.drive_path([(0, 0), (1, 0), (1, 1)])
    path = [(x, y) for _, x, y, _ in robot.trace.poses]
    corner = path.index(pytest.approx((1, 0)))
    assert all(y == pytest.approx(0) for _, y in path[:corner])
    assert all(x == pytest.approx(1) for x, _ in path[corner:])
    assert (robot.x, robot.y) == pytest.approx((1, 1), abs=0.08)
    assert robot.trace.bumps == 0


def test_sim_drive_path_stops_before_contact_between_waypoints():
    obstacle = Thing("ball", 0.5, 0.075, radius=0.027)
    robot = SimRobot(Scene(10, 10, [obstacle], {}, start=Pose(0, 0, 0)))
    with pytest.raises(PathBlocked, match="obstructed"):
        robot.drive_path([(0, 0), (1, 0)])
    assert robot.x < obstacle.x and robot.trace.bumps == 0


def test_sim_orbit_continues_past_a_held_waypoint_without_contact():
    desk = Thing("desk", 0, 0, radius=0.5)
    robot = SimRobot(Scene(10, 10, [desk], {}, start=Pose(1.5, 0, 90)))
    robot.orbit("desk", distance=0.2)
    bearings = [math.atan2(y, x) for _, x, y, _ in robot.trace.poses]
    travelled = sum(math.remainder(b - a, math.tau) for a, b in zip(bearings, bearings[1:], strict=False))
    assert travelled == pytest.approx(math.tau, abs=0.1)
    assert robot.trace.bumps == 0


def test_sim_drive_path_attempts_short_final_leg_after_a_held_waypoint():
    desk = Thing("desk", 0, 0, radius=0.5)
    robot = SimRobot(Scene(10, 10, [desk], {}, start=Pose(1.5, 0, 90)))
    final = (0.85, 0.3)
    robot.drive_path([(0.85, 0), final])
    assert math.dist((robot.x, robot.y), final) <= 0.08
    assert robot.trace.bumps == 0


@pytest.mark.parametrize("side", [-1, 1])
def test_body_drives_through_doorway_from_off_center_without_contact(side):
    robot = SimRobot(two_rooms(start=Pose(-0.22, side * 0.36, 0)))
    robot.go_to((0.75, side * 0.2))
    assert robot.x > 0 and robot.trace.bumps == 0


@pytest.fixture
def motion():
    clock = SimpleNamespace(now=1_000_000_000)
    selected, wheels = [], []
    gate = MotionGate(lambda: clock.now, lambda *v: selected.append(v), lambda v, w, _: wheels.append((v, w)),
                      0.35, 1.8)
    return gate, clock, selected, wheels


def test_nav_and_direct_are_exclusive_and_a_late_reply_cannot_restart_the_base(motion):
    gate, clock, selected, wheels = motion
    gate.arm(NAV)
    armed = clock.now
    clock.now += 20_000_000
    gate.submit("direct", 0.2, 0.0)
    gate.submit(NAV, 0.2, 0.0, armed)
    assert not selected
    gate.submit(NAV, 0.6, 2.0, clock.now)
    assert selected[-1] == (0.35, 1.8, clock.now)
    gate.checked(0.12, 0.1, clock.now)
    assert wheels[-1] == (0.12, 0.1)
    clock.now += 20_000_000
    gate.submit(NAV, 0.2, 0.0, clock.now)
    late = selected[-1]
    gate.stop()
    gate.checked(*late)
    gate.arm("direct")
    clock.now += 20_000_000
    gate.submit("direct", -0.4, 0.0)
    gate.submit(NAV, 0.3, 0.0, clock.now)
    gate.checked(*late)
    assert wheels[-1] == (0.0, 0.0)
    assert selected[-1] == (-0.25, 0.0, clock.now)
    gate.checked(*selected[-1])
    assert wheels[-1] == (-0.25, 0.0)


def test_silent_collision_monitor_expires_last_movement_and_cannot_replay_it(motion):
    gate, clock, selected, wheels = motion
    gate.arm("direct")
    gate.submit("direct", 0.2, 0.0)
    gate.checked(*selected[-1])
    assert wheels[-1] == (0.2, 0.0)
    clock.now += 20_000_000
    gate.submit("direct", 0.2, 0.0)  # input is still flowing
    clock.now += COMMAND_AGE_NS + 1
    gate.expire()
    gate.checked(*selected[-1])
    gate.expire()
    assert wheels == [(0.0, 0.0), (0.2, 0.0), (0.0, 0.0)]


def test_unmatched_duplicate_and_out_of_order_replies_never_reach_wheels(motion):
    gate, clock, selected, wheels = motion
    gate.arm("direct")
    gate.submit("direct", 0.2, 0.0)
    clock.now += 20_000_000
    gate.submit("direct", 0.3, 0.0)
    gate.checked(0.9, 0.0, clock.now - 1)
    assert wheels[-1] == (0.0, 0.0)
    gate.checked(*selected[-1])
    gate.checked(*selected[0])
    gate.checked(0.4, 0.0, clock.now)
    assert wheels == [(0.0, 0.0), (0.3, 0.0)]


def test_collision_monitor_does_not_extend_the_age_of_a_nav2_command(motion):
    gate, clock, selected, wheels = motion
    gate.arm(NAV)
    stamp = clock.now + 1
    clock.now = stamp + COMMAND_AGE_NS - 10
    gate.submit(NAV, 0.2, 0.0, stamp)
    assert selected == [(0.2, 0.0, stamp)]
    clock.now += 20
    gate.checked(*selected[-1])
    assert wheels == [(0.0, 0.0)]


def test_wheel_output_keeps_nav2_source_stamp():
    clock = SimpleNamespace(now=1_000_000_000)
    select, output = Mock(), Mock()
    gate = MotionGate(lambda: clock.now, select, output, 0.35, 1.8)
    gate.arm(NAV)
    stamp = clock.now + 20_000_000
    clock.now += 60_000_000
    gate.submit(NAV, 0.2, 0.0, stamp)
    select.assert_called_once_with(0.2, 0.0, stamp)
    clock.now += 60_000_000
    gate.checked(0.1, 0.0, stamp)
    output.assert_called_with(0.1, 0.0, stamp)


@pytest.mark.parametrize("fault", ["invalid request", "invalid reply", "clock reset"])
def test_fault_closes_gate(motion, fault):
    gate, clock, selected, wheels = motion
    gate.arm("direct")
    gate.submit("direct", 0.2, 0.0)
    gate.checked(*selected[-1])
    clock.now += 20_000_000
    if fault == "invalid request":
        gate.submit("direct", math.nan, 0.0)
    elif fault == "invalid reply":
        gate.submit("direct", 0.2, 0.0)
        gate.checked(0.2, math.inf, clock.now)
    else:
        clock.now = 0
        gate.expire()
        clock.now = 2_000_000_000
        assert not gate.arm("direct")
    assert gate.owner is None and wheels[-1] == (0.0, 0.0)


def test_cancel_reply_alone_does_not_prove_nav2_is_terminal():
    sent, result = Future(), Future()
    handle = SimpleNamespace(accepted=True, cancel_goal_async=Mock(), get_result_async=lambda: result)
    goal = NavGoal(sent, RLock())
    goal.cancel()
    assert not goal.terminal()
    sent.set_result(handle)
    goal.cancel()
    handle.cancel_goal_async.assert_called_once()
    assert not goal.terminal()
    result.set_result(SimpleNamespace(status=5))
    assert goal.terminal()


def test_habitat_wheel_watchdog_uses_sender_time_not_delivery_time():
    hardware = Hardware.__new__(Hardware)
    hardware.t, hardware.cmd = 2.0, (0, 0.0, 0.0)

    def command(sec, nanosec, v):
        return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec)),
                               twist=SimpleNamespace(linear=SimpleNamespace(x=v), angular=SimpleNamespace(z=0.0)))

    hardware._command(command(1, 100_000_000, 0.25))
    assert hardware._want() == (0.0, 0.0)
    hardware._command(command(1, 900_000_000, 0.25))
    assert hardware._want() == (0.25, 0.0)
    hardware.t = 1.9 + COMMAND_AGE_NS / 1e9
    assert hardware._want() == (0.25, 0.0)
    hardware.t += 1e-9
    assert hardware._want() == (0.0, 0.0)
    hardware.t = 2.0
    hardware._command(command(2, 0, 0.0))
    hardware._command(command(1, 990_000_000, 0.25))
    hardware._command(command(2, 20_000_000, 0.25))
    assert hardware._want() == (0.0, 0.0)
