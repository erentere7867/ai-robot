"""Motion progress, gestures and simulated proprioception."""

from __future__ import annotations

import math
from typing import Any


class Progress:
    """Credit new ground or a waypoint advance, not repeated travel along the same stretch."""

    def __init__(self, here: tuple[float, float], now: float, radius: float, timeout: float) -> None:
        self.visited, self.since = [here], now
        self.radius, self.timeout = radius, timeout

    def stalled(self, here: tuple[float, float], now: float, *, advanced: bool = False) -> bool:
        if all(math.dist(here, point) >= self.radius for point in reversed(self.visited)):
            self.visited.append(here)
            advanced = True
        if advanced:
            self.since = now
        return now - self.since > self.timeout


class BodyController:
    """Simulated proprioception and procedural aliveness."""

    def __init__(self) -> None:
        self._snag_timer: float = 0.0

    def aliveness_velocity(
        self,
        posture: str,
        t_in_posture: float,
        pose: tuple[float, float, float],
    ) -> tuple[float, float]:
        """Calculates procedural aliveness velocity and turn reflexes based on posture and elapsed time."""
        if posture == "listening":
            return 0.04 * math.sin(2.0 * math.pi * 1.2 * t_in_posture) * math.exp(-0.3 * t_in_posture), 0.0

        if posture == "curious":
            return 0.0, 0.35 * math.sin(2.0 * math.pi * 0.6 * t_in_posture)

        if posture == "happy":
            w = 0.9 * math.sin(2.0 * math.pi * 3.0 * t_in_posture) if t_in_posture < 0.6 else 0.0
            return 0.0, w

        if posture == "uh_oh":
            v = -0.15 if t_in_posture < 0.6 else 0.0
            return v, 0.0

        if posture == "yielding":
            v = -0.15 if t_in_posture < 1.0 else 0.0
            return v, 0.0

        if posture == "neutral":
            return 0.0, 0.0

        return 0.0, 0.0

    def check_proprioception(
        self,
        v_cmd: float,
        w_cmd: float,
        v_actual: float,
        w_actual: float,
        dt: float,
        pitch_rad: float = 0.0,
        hazard: Any = None,
    ) -> tuple[str, float, float, float]:
        """Monitors motor commands, actual progress, pitch, and hazards to catch cliffs, cable snags and steps."""
        if hazard is not None and getattr(hazard, "kind", "") == "cliff":
            self._snag_timer = 0.0
            return ("cliff_stop", -0.15, 0.0, 1.0)

        if abs(v_cmd) > 0.08 and abs(v_actual) < 0.015 and abs(pitch_rad) < 0.12:
            self._snag_timer += dt
            if self._snag_timer >= 0.15:
                return ("snag_recovery", -0.15, 0.0, 1.0)
        else:
            self._snag_timer = 0.0

        if (
            hazard is not None
            and getattr(hazard, "kind", "") == "traversable"
            and getattr(hazard, "torque_boost", False)
        ):
            return ("torque_boost", v_cmd, w_cmd, 1.35)

        return ("normal", v_cmd, w_cmd, 1.0)
