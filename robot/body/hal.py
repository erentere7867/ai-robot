"""What the rest of the robot knows about a body: its dimensions, its state, and how to command it.

Every motion is a speed command (`set_velocity`, refreshed at the control rate) plus a head angle; closed-loop
behaviours (robot/control/behaviors.py) turn goals into those. Simulated and real bodies implement the same protocol.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from robot.core.config import PROJECT_ROOT, load_yaml_file


class Reflex(StrEnum):
    """Why the body itself stopped or limited a motion."""

    ESTOP = "estop_active"
    CLIFF = "cliff_detected"
    PICKED_UP = "picked_up"
    BUMPER = "bumper"
    OBSTACLE_AHEAD = "obstacle_stop"
    OBSTACLE_BEHIND = "obstacle_behind"

    @property
    def halts(self) -> bool:
        """Stops every motion, not only the one heading into the obstacle."""
        return self in (Reflex.ESTOP, Reflex.CLIFF, Reflex.PICKED_UP, Reflex.BUMPER)

    @property
    def reason(self) -> str:
        return _REFLEX_REASONS[self]

    @property
    def words(self) -> str:
        return _REFLEX_WORDS[self]


_REFLEX_REASONS = {
    Reflex.ESTOP: "Emergency stop is active", Reflex.CLIFF: "Cliff sensor triggered",
    Reflex.PICKED_UP: "Robot was picked up", Reflex.BUMPER: "Bumped into something",
    Reflex.OBSTACLE_AHEAD: "Front obstacle or invalid range reading", Reflex.OBSTACLE_BEHIND: "Obstacle behind",
}
_REFLEX_WORDS = {
    Reflex.ESTOP: "the emergency stop is on", Reflex.CLIFF: "there is a drop ahead",
    Reflex.PICKED_UP: "someone picked me up", Reflex.BUMPER: "Bumper pressed: I bumped into something",
    Reflex.OBSTACLE_AHEAD: "something is right in front of me", Reflex.OBSTACLE_BEHIND: "something is right behind me",
}


@dataclass(frozen=True)
class JointSpec:
    name: str
    min_deg: float
    max_deg: float
    neutral_deg: float
    max_speed_deg_s: float


@dataclass(frozen=True)
class GestureStep:
    seconds: float
    v: float = 0.0  # mm/s
    w: float = 0.0  # deg/s
    head_deg: float | None = None


@dataclass(frozen=True)
class GestureSpec:
    name: str
    description: str
    steps: tuple[GestureStep, ...]


@dataclass(frozen=True)
class BodyCapabilities:
    """The robot as described by config/body.yaml."""

    robot_id: str
    description: str
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
    emotion_sounds: dict[str, str] = field(default_factory=dict)  # every emotion and the sound that goes with it
    gestures: dict[str, GestureSpec] = field(default_factory=dict)

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
            robot_id=str(raw["robot_id"]),
            description=str(raw["description"]),
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
            emotion_sounds={str(k): str(v) for k, v in raw["emotions"].items()},
            gestures={name: GestureSpec(name, str(g["description"]), tuple(
                GestureStep(float(s["s"]), float(s.get("v", 0.0)), float(s.get("w", 0.0)),
                            float(s["head"]) if "head" in s else None) for s in g["steps"]))
                      for name, g in raw["gestures"].items()},
        )


@functools.cache
def load_capabilities(path: Path | None = None) -> BodyCapabilities:
    return BodyCapabilities.from_yaml(path or PROJECT_ROOT / "config" / "body.yaml")


@dataclass
class BodyPose:
    """Where the robot is; simulated or estimated, and in the map frame once localization corrects it."""

    x_mm: float = 0.0
    y_mm: float = 0.0
    heading_deg: float = 0.0
    head_tilt_deg: float = 0.0
    linear_vel_mm_s: float = 0.0
    angular_vel_deg_s: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "x_mm": round(self.x_mm, 1),
            "y_mm": round(self.y_mm, 1),
            "heading_deg": round(self.heading_deg, 1),
            "head_tilt_deg": round(self.head_tilt_deg, 1),
            "linear_vel_mm_s": round(self.linear_vel_mm_s, 1),
            "angular_vel_deg_s": round(self.angular_vel_deg_s, 1),
        }

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "front_tof_mm": round(self.front_tof_mm, 1),
            "cliff_triggered": self.cliff_triggered,
            "estop_active": self.estop_active,
            "picked_up": self.picked_up,
            "wheel_ticks_left": self.wheel_ticks_left,
            "wheel_ticks_right": self.wheel_ticks_right,
            "imu_yaw_deg": round(self.imu_yaw_deg, 1),
            "imu_pitch_deg": round(self.imu_pitch_deg, 1),
            "battery_pct": round(self.battery_pct, 1),
            "charging": self.charging,
        }

    def copy(self) -> SensorReadings:
        return replace(self)


class BodyHAL(Protocol):
    name: str

    def capabilities(self) -> BodyCapabilities: ...

    def pose(self) -> BodyPose: ...

    def sensors(self) -> SensorReadings: ...

    async def set_velocity(self, linear_mm_s: float, angular_deg_s: float) -> Reflex | None:
        """Drive at this speed until the next command (the watchdog stops it if none comes)."""

    def set_head(self, tilt_deg: float) -> None: ...

    async def update_motion(self) -> Reflex | None:
        """Bring the body up to now with no new command; the reflex limiting motion, if any."""

    async def halt(self) -> None: ...

    def set_estop(self, active: bool) -> None: ...

    def reset_pose(self, pose: BodyPose | None = None) -> None: ...

    async def publish_state(self) -> None:
        """Emit `body.pose` now."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def health(self) -> Any: ...
