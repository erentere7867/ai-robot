"""The real robot: the VENTUNO Q's MCU, its cameras and the lidar."""

from __future__ import annotations

from typing import Any

from robot.body.hal import BodyCapabilities
from robot.core.config import RobotConfig
from robot.hardware.sensors import Ld19Lidar, RealCamera
from robot.hardware.ventuno_body import VentunoBody
from robot.perception.camera import CameraRig
from robot.perception.geometry import CameraModel
from robot.platform import Platform, load_detector, locate
from robot.world.localization import Localizer


def build_hardware(config: RobotConfig, bus: Any, clock: Any, capabilities: BodyCapabilities,
                   rig: tuple[CameraModel, ...], localizer: Localizer | None) -> Platform:
    body = locate(VentunoBody(bus, capabilities), localizer)
    lidar = None
    if config.perception.lidar != "off":
        lidar = Ld19Lidar(body, clock, port=config.perception.lidar, mount_x_mm=capabilities.lidar_x_mm)
    return Platform(body=body, camera=CameraRig([RealCamera(body, clock, model, source=config.camera) for model in rig]),
                    lidar=lidar, detector=load_detector(config))
