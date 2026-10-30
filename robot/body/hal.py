"""Body hardware abstraction layer, capabilities, pose, and command types."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from robot.core.config import load_yaml_file


@dataclass(frozen=True)
class JointSpec:
    name: str
    min_deg: float
    max_deg: float
    neutral_deg: float = 0.0
    max_speed_deg_s: float = 90.0

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v) for v in (self.min_deg, self.max_deg, self.neutral_deg, self.max_speed_deg_s)
        ):
            raise ValueError("Joint limits must be finite")
        if not self.min_deg <= self.neutral_deg <= self.max_deg or self.max_speed_deg_s <= 0:
            raise ValueError("Invalid joint range or speed")


@dataclass(frozen=True)
class TrickSpec:
    name: str
    description: str
    duration_s: float = 1.0


@dataclass(frozen=True)
class BodyCapabilities:
    """Declarative capability specification loaded from body.yaml."""

    robot_id: str = "companion_r1"
    description: str = "Mobile 4-wheel skid-steer companion robot"
    wheel_diameter_mm: float = 120.0
    track_width_mm: float = 208.0
    max_linear_speed_mm_s: float = 400.0
    max_angular_speed_deg_s: float = 120.0
    max_single_move_mm: float = 2000.0
    max_single_spin_deg: float = 360.0
    default_linear_speed_mm_s: float = 200.0
    default_angular_speed_deg_s: float = 90.0
    min_front_tof_stop_mm: float = 120.0
    default_front_tof_mm: float = 1500.0
    encoder_ticks_per_rev: int = 720
    joints: dict[str, JointSpec] = field(
        default_factory=lambda: {
            "head_tilt": JointSpec(
                "head_tilt", min_deg=-25.0, max_deg=35.0, neutral_deg=0.0, max_speed_deg_s=90.0
            )
        }
    )
    allowed_sounds: list[str] = field(
        default_factory=lambda: [
            "none",
            "happy_trill",
            "curious_chirp",
            "soft_beep",
            "sleepy_yawn",
            "startle_pip",
            "ack_chirp",
            "sad_whimper",
            "angry_grumble",
            "giggle",
            "purr",
        ]
    )
    allowed_emotions: list[str] = field(
        default_factory=lambda: [
            "neutral",
            "happy",
            "excited",
            "curious",
            "sad",
            "angry",
            "grumpy",
            "confused",
            "surprised",
            "sleepy",
            "concerned",
            "proud",
            "shy",
            "scared",
            "love",
        ]
    )
    tricks: dict[str, TrickSpec] = field(
        default_factory=lambda: {
            "nod": TrickSpec("nod", "Nod head up and down in agreement", 1.0),
            "shake": TrickSpec("shake", "Small left-right body shimmy to say no or show frustration", 1.0),
            "curious_tilt": TrickSpec("curious_tilt", "Tilt head up with a curious posture", 0.8),
            "wiggle": TrickSpec("wiggle", "Playful in-place left-right wiggle", 1.4),
            "dance": TrickSpec("dance", "Happy spin-and-nod dance routine", 2.5),
            "spin": TrickSpec("spin", "Full 360-degree playful spin in place", 2.2),
            "sleepy_droop": TrickSpec("sleepy_droop", "Droop head down gently when sleepy or sad", 1.2),
            "perk_up": TrickSpec("perk_up", "Perk head up attentively when greeted or excited", 0.7),
            "stomp": TrickSpec("stomp", "Short forward-back stomp when angry or grumpy", 1.0),
        }
    )

    def __post_init__(self) -> None:
        for key in (
            "wheel_diameter_mm",
            "track_width_mm",
            "max_linear_speed_mm_s",
            "max_angular_speed_deg_s",
            "max_single_move_mm",
            "max_single_spin_deg",
            "default_linear_speed_mm_s",
            "default_angular_speed_deg_s",
            "min_front_tof_stop_mm",
            "default_front_tof_mm",
            "encoder_ticks_per_rev",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"Body capability {key} must be finite and positive")
        if (
            self.default_linear_speed_mm_s > self.max_linear_speed_mm_s
            or self.default_angular_speed_deg_s > self.max_angular_speed_deg_s
        ):
            raise ValueError("Default speed exceeds physical maximum")
        if "head_tilt" not in self.joints:
            raise ValueError("head_tilt joint is required")

    @classmethod
    def from_yaml(cls, path: Path | str) -> BodyCapabilities:
        raw = load_yaml_file(path)
        dims = raw.get("dimensions_mm", {})
        drive = raw.get("drive", {})
        sensors = raw.get("sensors", {})
        tof = sensors.get("front_tof", {})
        enc = sensors.get("wheel_encoders", {})

        joints: dict[str, JointSpec] = {}
        for jname, jraw in (raw.get("joints") or {}).items():
            joints[jname] = JointSpec(
                name=jname,
                min_deg=float(jraw.get("min_deg", -25.0)),
                max_deg=float(jraw.get("max_deg", 35.0)),
                neutral_deg=float(jraw.get("neutral_deg", 0.0)),
                max_speed_deg_s=float(jraw.get("max_speed_deg_s", 90.0)),
            )

        tricks: dict[str, TrickSpec] = {}
        for tname, traw in (raw.get("tricks") or {}).items():
            tricks[tname] = TrickSpec(
                name=tname,
                description=str(traw.get("description", tname)),
                duration_s=float(traw.get("duration_s", 1.0)),
            )

        sounds = list((raw.get("sounds") or {}).get("allowed") or [])
        emotions = list((raw.get("emotions") or {}).get("allowed") or [])

        return cls(
            robot_id=str(raw.get("robot_id", "companion_r1")),
            description=str(raw.get("description", "Mobile 4-wheel skid-steer companion robot")),
            wheel_diameter_mm=float(dims.get("wheel_diameter", 120.0)),
            track_width_mm=float(dims.get("track_width", 208.0)),
            max_linear_speed_mm_s=float(drive.get("max_linear_speed_mm_s", 400.0)),
            max_angular_speed_deg_s=float(drive.get("max_angular_speed_deg_s", 120.0)),
            max_single_move_mm=float(drive.get("max_single_move_mm", 2000.0)),
            max_single_spin_deg=float(drive.get("max_single_spin_deg", 360.0)),
            default_linear_speed_mm_s=float(drive.get("default_linear_speed_mm_s", 200.0)),
            default_angular_speed_deg_s=float(drive.get("default_angular_speed_deg_s", 90.0)),
            min_front_tof_stop_mm=float(tof.get("min_stop_mm", 120.0)),
            default_front_tof_mm=float(tof.get("default_reading_mm", 1500.0)),
            encoder_ticks_per_rev=int(enc.get("ticks_per_rev", 720)),
            joints=joints or cls().joints,
            allowed_sounds=sounds or cls().allowed_sounds,
            allowed_emotions=emotions or cls().allowed_emotions,
            tricks=tricks or cls().tricks,
        )


@dataclass
class BodyPose:
    """Instantaneous kinematic pose of the robot in the local odom frame."""

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


@dataclass
class SensorReadings:
    """Simulated or physical low-level sensor state."""

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
        }


@dataclass(frozen=True)
class BodyCommand:
    """Validated physical action command dispatched to the BodyHAL."""

    id: str
    tool: str
    args: dict[str, Any]
    timeout_s: float = 5.0


@dataclass(frozen=True)
class CommandResult:
    """Execution outcome returned by BodyHAL."""

    command_id: str
    tool: str
    ok: bool
    status: str  # "completed" | "clamped" | "cancelled" | "reflex_stopped" | "failed"
    message: str
    pose: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


class BodyHAL(Protocol):
    """Hardware Abstraction Layer interface for real and simulated bodies."""

    def capabilities(self) -> BodyCapabilities: ...

    def pose(self) -> BodyPose: ...

    def sensors(self) -> SensorReadings: ...

    async def execute(self, cmd: BodyCommand) -> CommandResult: ...

    async def cancel(self, command_id: str | None = None) -> None: ...

    def set_estop(self, active: bool) -> None: ...
