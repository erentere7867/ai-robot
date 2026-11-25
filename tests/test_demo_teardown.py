"""Exercise the demo's real room-swap callback without ROS or Habitat."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from robot.agent.api import SafetyStop
from robot.interfaces import demo, voice
from robot.interfaces.live import Live
from robot.ros.real import RealRobot


class ApartmentRobot(RealRobot):
    def __init__(self):
        self.perception = SimpleNamespace(stop=Mock())
        self.shutdown = Mock()


def round_trips(monkeypatch, *, fail=False, unconfirmed_stop=False):
    robots = []
    home = SimpleNamespace(reset=Mock(), close=Mock())

    def robot(*args):
        result = ApartmentRobot()
        robots.append(result)
        return result

    home.robot = robot
    stack = ModuleType("robot.sim.stack")
    stack.Home = Mock(return_value=home)
    detector = ModuleType("robot.perception.detector")
    detector.Detector = Mock()
    monkeypatch.setitem(sys.modules, "robot.sim.stack", stack)
    monkeypatch.setitem(sys.modules, "robot.perception.detector", detector)
    monkeypatch.setattr(demo, "HOMES", {})
    monkeypatch.setattr("sys.argv", ["demo"])
    live = Live()
    monkeypatch.setattr(demo, "Live", lambda: live)
    monkeypatch.setattr(live, "serve", lambda port, host: None)
    monkeypatch.setattr(voice, "attach", lambda live: None)
    monkeypatch.setattr(demo, "Brain", lambda url: SimpleNamespace(healthy=lambda: True))
    monkeypatch.setattr(demo, "Agent", lambda *args, **kwargs: SimpleNamespace(forget=Mock()))
    monkeypatch.setattr(demo, "show_cameras", lambda *args: None)
    monkeypatch.setattr(demo.threading.Thread, "start", lambda self: None)
    counts = []

    def swap(live, agent, holder, new_scene):
        counts.append(len(robots))
        if fail:
            new_scene("living room")
            previous = holder.sim
            home.reset.side_effect = RuntimeError("load failed")
            with pytest.raises(RuntimeError, match="load failed"):
                new_scene("3D apartment")
            assert holder.sim is previous
        elif unconfirmed_stop:
            robots[0].shutdown.side_effect = SafetyStop("restart Nav2")
            for scene in ("3D apartment", "living room"):
                with pytest.raises(RuntimeError, match="Restart the demo and Nav2"):
                    new_scene(scene)
                assert holder.sim is holder.home is None
            home.reset.assert_called_once_with()
            agent.forget.assert_not_called()
        else:
            for _ in range(2):
                new_scene("living room")
                new_scene("3D apartment")
            assert holder.sim is robots[-1]
        counts.append(sum(not r.shutdown.called for r in robots))

    monkeypatch.setattr(demo, "drive", swap)
    demo.main()
    home.close.assert_not_called()
    return robots, counts


def test_round_trips_tear_down_outgoing_robots_but_not_shared_home(monkeypatch):
    robots, counts = round_trips(monkeypatch)
    assert counts == [1, 1]
    for robot in robots[:-1]:
        robot.perception.stop.assert_called_once_with()
        robot.shutdown.assert_called_once_with()
    robots[-1].perception.stop.assert_not_called()
    robots[-1].shutdown.assert_not_called()


def test_failed_swap_keeps_current_robot_alive(monkeypatch):
    _, counts = round_trips(monkeypatch, fail=True)
    assert counts == [1, 0]


def test_scene_swap_requires_restart_if_previous_goal_is_unconfirmed(monkeypatch):
    robots, counts = round_trips(monkeypatch, unconfirmed_stop=True)
    assert counts == [1, 0]
    robots[0].shutdown.assert_called_once_with()


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as patch:
        _, counts = round_trips(patch)
    print(f"live 3D robots: initial={counts[0]}, after two round trips={counts[1]}")
