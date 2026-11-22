import math
import random
from unittest.mock import Mock

import pytest

from robot.sim import stack
from robot.sim.habitat_ros import DT, GYRO_NOISE, GyroZero


@pytest.mark.parametrize("bias", [-0.0042, 0.006])
def test_parked_gyro_drift_and_a_real_turn_after_a_long_wait(bias):
    rng, zero = random.Random(17), GyroZero()

    def reading(turn, stationary):
        return zero.correct(turn + bias + rng.gauss(0, GYRO_NOISE), stationary)

    heading = sum(reading(0.0, True) * DT for _ in range(round(220 / DT)))
    assert abs(heading) < math.radians(2)
    angle = sum(reading(0.1, False) * DT for _ in range(round(10 / DT)))
    assert abs(angle - 1.0) < 0.03  # a sustained, slow turn must not become a new zero


@pytest.mark.parametrize("failure", [None, "hardware", "software"])
def test_hardware_log_is_closed(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(stack, "RUNS", tmp_path)
    monkeypatch.setattr(stack, "MAPS", tmp_path)
    (tmp_path / "apt_0.yaml").touch()
    monkeypatch.setattr(stack.bringup, "clear", Mock())
    monkeypatch.setattr(stack, "start", Mock(return_value=(Mock(), 1234)))
    monkeypatch.setattr(stack, "Habitat", Mock())
    monkeypatch.setattr(stack.Home, "_judge_node", lambda home: setattr(home, "node", Mock()))
    software = Mock(side_effect=RuntimeError("software") if failure == "software" else None)
    monkeypatch.setattr(stack.bringup, "start", software)
    logs = []

    def popen(*args, **kwargs):
        log = kwargs["stdout"]
        logs.append(log)
        assert not log.closed
        log.write("hardware started\n")
        if failure == "hardware":
            raise RuntimeError("hardware")
        return Mock()

    monkeypatch.setattr(stack.subprocess, "Popen", Mock(side_effect=popen))
    try:
        if failure is None:
            home = stack.Home()
            home.close()
        else:
            with pytest.raises(RuntimeError, match=failure):
                stack.Home()
        assert len(logs) == 1
        assert logs[0].closed
        assert (tmp_path / "home_apt_0" / "hardware.log").read_text() == "hardware started\n"
    finally:
        for log in logs:
            log.close()
