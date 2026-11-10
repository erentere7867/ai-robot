"""The robot's dimensions from config/body.yaml, and its pose and sensor state."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from robot.core.config import PROJECT_ROOT, load_yaml_file


class Reflex(StrEnum):
    """Why the body itself stopped or limited a motion."""

    ESTOP = "estop_active"
    CLIFF = "cliff_detected"
    PICKED_UP = "picked_up"
    BUMPER = "bumper"
    OBSTACLE_AHEAD = "obstacle_stop"

    @property
    def halts(self) -> bool:
        """Stops every motion, not only the one heading into the obstacle."""
        return self is not Reflex.OBSTACLE_AHEAD


@dataclass(frozen=True)
class JointSpec:
    name: str
    min_deg: float
    max_deg: float
    neutral_deg: float
    max_speed_deg_s: float


@dataclass(frozen=True)
class BodyCapabilities:
    wheel_diameter_mm: float
    track_width_mm: float
    length_mm: float
    width_mm: float
    max_linear_speed_mm_s: float
    max_angular_speed_deg_s: float
    min_front_tof_stop_mm: float
    default_front_tof_mm: float
    encoder_ticks_per_rev: int
    lidar_x_mm: float
    lidar_z_mm: float
    head_tilt: JointSpec

    @property
    def half_length_mm(self) -> float:
        return 0.5 * self.length_mm

    @property
    def half_width_mm(self) -> float:
        return 0.5 * self.width_mm

    @property
    def front_range_x_mm(self) -> float:
        """Where the front range sensor sits: on the front edge."""
        return self.half_length_mm

    @property
    def radius_mm(self) -> float:
        """Circumscribed radius of the footprint."""
        return math.hypot(self.half_length_mm, self.half_width_mm)

    def clamp_velocity(self, v: float, w: float) -> tuple[float, float]:
        """Inside the drive's limits; a non-finite command is no command."""
        v_max, w_max = self.max_linear_speed_mm_s, self.max_angular_speed_deg_s
        return (max(-v_max, min(v_max, float(v))) if math.isfinite(v) else 0.0,
                max(-w_max, min(w_max, float(w))) if math.isfinite(w) else 0.0)

    def clamp_head(self, deg: float) -> float:
        return max(self.head_tilt.min_deg, min(self.head_tilt.max_deg, float(deg)))

    @classmethod
    def from_yaml(cls, path: Path | str) -> BodyCapabilities:
        raw = load_yaml_file(path)
        dims, drive, sensors = raw["dimensions_mm"], raw["drive"], raw["sensors"]
        tilt = raw["joints"]["head_tilt"]
        return cls(
            wheel_diameter_mm=float(dims["wheel_diameter"]),
            track_width_mm=float(dims["track_width"]),
            length_mm=float(dims["length"]),
            width_mm=float(dims["width"]),
            max_linear_speed_mm_s=float(drive["max_linear_speed_mm_s"]),
            max_angular_speed_deg_s=float(drive["max_angular_speed_deg_s"]),
            min_front_tof_stop_mm=float(sensors["front_tof"]["min_stop_mm"]),
            default_front_tof_mm=float(sensors["front_tof"]["default_reading_mm"]),
            encoder_ticks_per_rev=int(sensors["wheel_encoders"]["ticks_per_rev"]),
            lidar_x_mm=float(sensors["lidar"]["x_offset_mm"]),
            lidar_z_mm=float(sensors["lidar"]["z_height_mm"]),
            head_tilt=JointSpec("head_tilt", float(tilt["min_deg"]), float(tilt["max_deg"]),
                                float(tilt["neutral_deg"]), float(tilt["max_speed_deg_s"])),
        )


@functools.cache
def load_capabilities(path: Path | None = None) -> BodyCapabilities:
    return BodyCapabilities.from_yaml(path or PROJECT_ROOT / "config" / "body.yaml")


@dataclass
class BodyPose:
    x_mm: float = 0.0
    y_mm: float = 0.0
    heading_deg: float = 0.0
    head_tilt_deg: float = 0.0
    linear_vel_mm_s: float = 0.0
    angular_vel_deg_s: float = 0.0

    def copy(self) -> BodyPose:
        return replace(self)


@dataclass
class SensorReadings:
    front_tof_mm: float = 1500.0
    cliff_triggered: bool = False
    estop_active: bool = False
    picked_up: bool = False
    wheel_ticks_left: int = 0
    wheel_ticks_right: int = 0
    imu_yaw_deg: float = 0.0
    imu_pitch_deg: float = 0.0
    battery_pct: float = 100.0
    charging: bool = False

    def copy(self) -> SensorReadings:
        return replace(self)
