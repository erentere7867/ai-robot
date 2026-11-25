"""Exercise the real sketch with a simulated clock, sensor bus, and motor driver."""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def ventuno_mcu(tmp_path_factory):
    build = tmp_path_factory.mktemp("ventuno_mcu")
    (build / "zephyr").mkdir()
    for header in (
        "Arduino_RouterBridge.h",
        "Wire.h",
        "Servo.h",
        "SparkFun_VL53L5CX_Library.h",
        "Adafruit_MPU6050.h",
        "zephyr/kernel.h",
    ):
        (build / header).touch()
    executable = build / "ventuno_mcu_test"
    subprocess.run(
        ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(build),
         str(Path(__file__).with_name("ventuno_mcu_test.cpp")), "-o", str(executable)],
        check=True,
    )
    return executable


@pytest.mark.parametrize("scenario", [
    "e-stop", "left bumper", "right bumper", "invalid distance", "invalid status", "missing targets",
    "insufficient coverage", "read failure", "command timeout", "stalled control",
])
def test_ventuno_motor_cutoff(ventuno_mcu, scenario):
    subprocess.run([str(ventuno_mcu), scenario], check=True)
