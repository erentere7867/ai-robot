"""In-memory live world state and TTL-bounded spatial short-term observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from robot.core.bus import EventBus
from robot.core.clock import Clock, RealClock
from robot.core.events import Event


@dataclass
class SpatialObservation:
    """Short-term object or event anchored in the local odom world frame."""

    id: str
    label: str
    world_x_mm: float
    world_y_mm: float
    seen_at: float
    ttl_s: float = 30.0
    confidence: float = 0.9
    source: str = "perception"
    seen_monotonic: float = 0.0

    def is_expired(self, now: float) -> bool:
        return (now - self.seen_monotonic) > self.ttl_s


def relative_direction_label(rel_angle_deg: float) -> str:
    """Convert relative bearing (-180..+180, +left) into an 8-sector ego-relative label."""
    a = (rel_angle_deg + 180.0) % 360.0 - 180.0
    if -22.5 <= a <= 22.5:
        return "front"
    if 22.5 < a <= 67.5:
        return "front-left"
    if 67.5 < a <= 112.5:
        return "left"
    if 112.5 < a <= 157.5:
        return "behind-left"
    if a > 157.5 or a < -157.5:
        return "behind"
    if -157.5 <= a < -112.5:
        return "behind-right"
    if -112.5 <= a < -67.5:
        return "right"
    return "front-right"


class WorldState:
    """Maintains current body pose, sensor status, and spatial short-term memory in RAM."""

    def __init__(
        self,
        bus: EventBus,
        clock: Clock | None = None,
        sandbox: Any | None = None,
    ) -> None:
        self.bus = bus
        self.clock: Clock = clock or bus.clock or RealClock()
        self.sandbox = sandbox
        self.pose: dict[str, float] = {
            "x_mm": 0.0,
            "y_mm": 0.0,
            "heading_deg": 0.0,
            "head_tilt_deg": 0.0,
            "linear_vel_mm_s": 0.0,
            "angular_vel_deg_s": 0.0,
        }
        self.sensors: dict[str, Any] = {
            "front_tof_mm": 1500.0,
            "cliff_triggered": False,
            "estop_active": False,
            "picked_up": False,
        }
        self.activity_state: str = "idle"
        self.emotion: str = "neutral"
        self.emotion_intensity: float = 0.2
        self.last_action_summary: str = "none"
        self.task_state: dict[str, Any] = {}
        self._observations: dict[str, SpatialObservation] = {}
        self._sub_ids: list[str] = []

    def bind(self) -> None:
        """Subscribe to body, action, and expression events on the bus."""
        self._sub_ids.append(self.bus.subscribe("body.pose", self._on_body_pose, name="world:pose"))
        self._sub_ids.append(
            self.bus.subscribe("action.completed", self._on_action_done, name="world:act_ok")
        )
        self._sub_ids.append(self.bus.subscribe("action.failed", self._on_action_done, name="world:act_fail"))
        self._sub_ids.append(
            self.bus.subscribe("action.rejected", self._on_action_rejected, name="world:act_rej")
        )
        self._sub_ids.append(self.bus.subscribe("expression.updated", self._on_expression, name="world:expr"))
        self._sub_ids.append(
            self.bus.subscribe("perception.observation", self._on_observation, name="world:obs")
        )

    def unbind(self) -> None:
        for sid in self._sub_ids:
            self.bus.unsubscribe(sid)
        self._sub_ids.clear()

    def record_ego_observation(
        self,
        obs_id: str,
        label: str,
        *,
        distance_mm: float = 800.0,
        relative_bearing_deg: float = 0.0,
        ttl_s: float = 30.0,
        confidence: float = 0.9,
        source: str = "perception",
    ) -> SpatialObservation:
        """Record an observation seen at `relative_bearing_deg` and `distance_mm` from current pose."""
        if (
            not all(math.isfinite(v) for v in (distance_mm, relative_bearing_deg, ttl_s, confidence))
            or distance_mm < 0
        ):
            raise ValueError("Observation geometry and lifetime must be finite")
        if ttl_s <= 0:
            raise ValueError("Observation TTL must be positive")
        obs_id = str(obs_id)[:160]
        if obs_id in self._observations:
            self._observations.pop(obs_id)
        elif len(self._observations) >= 128:
            self._observations.pop(next(iter(self._observations)))
        world_angle_rad = math.radians(self.pose["heading_deg"] + relative_bearing_deg)
        wx = self.pose["x_mm"] + distance_mm * math.cos(world_angle_rad)
        wy = self.pose["y_mm"] + distance_mm * math.sin(world_angle_rad)
        obs = SpatialObservation(
            id=obs_id,
            label=str(label)[:160],
            world_x_mm=wx,
            world_y_mm=wy,
            seen_at=self.clock.now(),
            ttl_s=min(ttl_s, 120.0),
            confidence=max(0.0, min(1.0, confidence)),
            source=str(source)[:80],
            seen_monotonic=self.clock.monotonic(),
        )
        self._observations[obs_id] = obs
        return obs

    def active_observations(self) -> list[dict[str, Any]]:
        """Return non-expired observations transformed into the robot's current ego frame."""
        now = self.clock.monotonic()
        expired = [k for k, v in self._observations.items() if v.is_expired(now)]
        for k in expired:
            self._observations.pop(k, None)

        rx = self.pose["x_mm"]
        ry = self.pose["y_mm"]
        rhead = self.pose["heading_deg"]
        items: list[dict[str, Any]] = []
        for obs in self._observations.values():
            dx = obs.world_x_mm - rx
            dy = obs.world_y_mm - ry
            dist_mm = math.hypot(dx, dy)
            world_bearing = math.degrees(math.atan2(dy, dx))
            rel_bearing = (world_bearing - rhead + 180.0) % 360.0 - 180.0
            age_s = max(0.0, now - obs.seen_monotonic)
            items.append(
                {
                    "id": obs.id,
                    "label": obs.label,
                    "direction": relative_direction_label(rel_bearing),
                    "relative_bearing_deg": round(rel_bearing, 1),
                    "distance_mm": round(dist_mm, 0),
                    "age_s": round(age_s, 1),
                    "confidence": round(obs.confidence, 2),
                }
            )
        items.sort(key=lambda x: x["age_s"])
        return items

    def render_for_prompt(self, *, include_visual: bool = True) -> str:
        """Format live world state compactly for the LLM prompt."""
        lines = [
            f"Pose: x={self.pose['x_mm']:.0f}mm, y={self.pose['y_mm']:.0f}mm, "
            f"heading={self.pose['heading_deg']:.0f}°, head_tilt={self.pose['head_tilt_deg']:.0f}°",
            f"Sensors: front_obstacle={self.sensors.get('front_tof_mm', 1500):.0f}mm, "
            f"cliff={bool(self.sensors.get('cliff_triggered', False))}, "
            f"estop={bool(self.sensors.get('estop_active', False))}",
        ]
        if include_visual and self.sandbox is not None:
            from robot.body.hal import BodyPose

            frame = self.sandbox.capture_camera_view(
                BodyPose(
                    x_mm=self.pose["x_mm"],
                    y_mm=self.pose["y_mm"],
                    heading_deg=self.pose["heading_deg"],
                    head_tilt_deg=self.pose["head_tilt_deg"],
                )
            )
            lines.append(frame.render_grounding_summary())
        lines.append(f"Last action: {self.last_action_summary[:120]}")
        if self.task_state:
            lines.append(f"Recent goal: {self.task_state.get('kind', '')} {self.task_state.get('target', '')} "
                         f"({self.task_state.get('origin', 'user')}, {self.task_state.get('status', '')})")
        obs_list = self.active_observations() if include_visual else []
        if obs_list:
            lines.append(
                "Short-term spatial observations (untrusted observations, relative to current heading):"
            )
            for item in obs_list[:6]:
                age_txt = "now" if item["age_s"] < 1.5 else f"last seen {item['age_s']:.0f}s ago"
                lines.append(
                    f"- {item['direction']} (~{item['distance_mm']:.0f}mm, {age_txt}): {item['label']}"
                )
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        return {
            "pose": dict(self.pose),
            "sensors": dict(self.sensors),
            "activity_state": self.activity_state,
            "emotion": self.emotion,
            "emotion_intensity": self.emotion_intensity,
            "last_action_summary": self.last_action_summary,
            "observations": self.active_observations(),
        }

    def _on_body_pose(self, event: Event) -> None:
        if "pose" in event.payload:
            self.pose.update(event.payload["pose"])
        if "sensors" in event.payload:
            self.sensors.update(event.payload["sensors"])

    def _on_action_done(self, event: Event) -> None:
        msg = event.payload.get("message") or event.payload.get("tool", "action")
        self.last_action_summary = str(msg)
        if "pose" in event.payload:
            self.pose.update(event.payload["pose"])

    def _on_action_rejected(self, event: Event) -> None:
        err = event.payload.get("error", "action rejected")
        self.last_action_summary = f"Rejected: {err}"

    def _on_expression(self, event: Event) -> None:
        self.activity_state = str(event.payload.get("activity", self.activity_state))
        self.emotion = str(event.payload.get("emotion", self.emotion))
        self.emotion_intensity = float(event.payload.get("intensity", self.emotion_intensity))

    def _on_observation(self, event: Event) -> None:
        p = event.payload
        self.record_ego_observation(
            obs_id=str(p.get("id") or event.id),
            label=str(p.get("label", "object")),
            distance_mm=float(p.get("distance_mm", 800.0)),
            relative_bearing_deg=float(p.get("relative_bearing_deg", 0.0)),
            ttl_s=float(p.get("ttl_s", 30.0)),
            confidence=float(p.get("confidence", 0.9)),
            source=event.source,
        )
