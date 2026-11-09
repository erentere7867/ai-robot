"""The robot the harness runs on: a body and its senses, simulated or real, assembled in one place.

Everything else takes a `Platform` and does not know which one it got. The real robot never imports the
simulator (and so never needs MuJoCo), and the simulator never imports the hardware drivers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot.body.hal import BodyCapabilities, BodyHAL
from robot.core.config import PROJECT_ROOT, RobotConfig
from robot.perception.camera import CameraRig
from robot.perception.detection import FrameDetector
from robot.perception.geometry import CameraModel
from robot.perception.lidar import Lidar
from robot.world.localization import LocalizedBody, Localizer

logger = logging.getLogger(__name__)


@dataclass
class Platform:
    body: BodyHAL
    camera: CameraRig
    lidar: Lidar | None
    detector: FrameDetector | None
    voice: Any = None  # the simulated person's voice; a real robot hears through its microphones
    sandbox: Any = None  # the simulated room; None on the real robot

    def close(self) -> None:
        if self.sandbox is not None:
            self.sandbox.unbind()
        self.camera.close()
        if self.lidar is not None:
            self.lidar.close()


def locate(body: BodyHAL, localizer: Localizer | None) -> BodyHAL:
    """The body as everything else should see it: in the map frame when localization is on."""
    return LocalizedBody(body, localizer) if localizer is not None else body


def load_detector(config: RobotConfig) -> FrameDetector | None:
    """The trained detector: a freshly trained one in the runtime folder wins, else the one in the repository."""
    cfg = config.perception
    if cfg.fast_detector == "off":
        return None
    from robot.perception.fast_detector import NcnnDetector

    for models_dir in (config.runtime_dir / "models", PROJECT_ROOT / "models"):
        path = Path(models_dir) / "fast_detector_ncnn_model"
        if path.exists():
            return NcnnDetector(path, conf=cfg.confidence, gpu=cfg.detector_gpu, threads=cfg.detector_threads)
    logger.info("No trained fast detector (run scripts/train_detector.py); perception uses the VLM only")
    return None


def build_platform(config: RobotConfig, bus: Any, clock: Any, capabilities: BodyCapabilities,
                   rig: tuple[CameraModel, ...], localizer: Localizer | None) -> Platform:
    if config.body_backend == "ventuno":
        from robot.hardware.platform import build_hardware as build
    else:
        from robot.sim.platform import build_sim as build
    return build(config, bus, clock, capabilities, rig, localizer)
