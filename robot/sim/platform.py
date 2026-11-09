"""The simulated robot: a kinematic or MuJoCo body in the sandbox room, with simulated senses."""

from __future__ import annotations

from typing import Any

from robot.audio.doa import MicArray
from robot.body.hal import BodyCapabilities
from robot.core.config import RobotConfig
from robot.perception.camera import CameraRig
from robot.perception.geometry import CameraModel
from robot.platform import Platform, load_detector, locate
from robot.sim.sandbox import SimulationSandbox
from robot.sim.sensors import SimCamera, SimLidar, SimVoice
from robot.sim.sim_body import SimBody
from robot.world.localization import Localizer


def build_sim(config: RobotConfig, bus: Any, clock: Any, capabilities: BodyCapabilities,
              rig: tuple[CameraModel, ...], localizer: Localizer | None) -> Platform:
    step_dt_s = 1.0 / config.sim_tick_hz
    if config.body_backend == "physics":
        from robot.sim.physics_body import PhysicsBody

        raw = PhysicsBody(bus, capabilities, step_dt_s=step_dt_s)
    else:
        raw = SimBody(bus, capabilities, step_dt_s=step_dt_s)
    sandbox = SimulationSandbox(bus, rig=rig, capabilities=capabilities)
    sandbox.bind(raw)
    body = locate(raw, localizer)
    lidar = None
    if config.perception.lidar != "off":
        lidar = SimLidar(sandbox, body, clock, mount_x_mm=capabilities.lidar_x_mm, height_mm=capabilities.lidar_z_mm)
    return Platform(
        body=body,
        camera=CameraRig([SimCamera(sandbox, body, clock, model) for model in rig]),
        lidar=lidar,
        detector=load_detector(config),
        # A typed message is the simulated person speaking: their voice reaches the simulated microphones from
        # where they stand, and the robot finds them the way it would on hardware.
        voice=SimVoice(sandbox, body, MicArray.from_body_config(config.body_config_path)),
        sandbox=sandbox,
    )
