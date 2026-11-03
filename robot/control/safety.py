"""Collision monitor: the last check between a behaviour's speed command and the motors.

Behaviours plan around what the map knows; this layer only looks at what the sensors see right now
(the newest lidar scan and the front range reading) and limits the command so the robot can always stop
in time, whatever a behaviour or the model asked for. The same idea as Nav2's collision monitor:

* travel zone: the footprint swept ahead in the direction of travel (forward or backward). The allowed
  speed is the one that still stops before the nearest point in it: v <= sqrt(2 a (d - margin)), with
  the controller's reaction time included. Points closer than the margin stop the robot.
* turning: an in-place turn sweeps the circumscribed circle; a point inside it (but outside the
  footprint, which would be the robot itself) slows the turn.

The microcontroller enforces its own envelope from the range sensor; this one adds the lidar (sides and
behind, which the range sensor cannot see) and runs at the control rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Limit:
    v: float
    w: float
    reason: str | None = None  # "obstacle_stop" / "obstacle_behind" when forward / backward travel is blocked
    clearance_mm: float = math.inf


class CollisionMonitor:
    def __init__(self, *, half_length_mm: float = 141.0, half_width_mm: float = 114.0, margin_mm: float = 60.0,
                 side_margin_mm: float = 40.0, decel_mm_s2: float = 700.0, reaction_s: float = 0.08,
                 lookahead_mm: float = 900.0, lidar_x_mm: float = -105.0, range_x_mm: float = 141.0) -> None:
        self.hl, self.hw = half_length_mm, half_width_mm
        self.margin, self.side = margin_mm, side_margin_mm
        self.decel, self.reaction = decel_mm_s2, reaction_s
        self.lookahead = lookahead_mm
        self.lidar_x, self.range_x = lidar_x_mm, range_x_mm
        self.radius = math.hypot(half_length_mm, half_width_mm)

    def points(self, scan: object | None, range_mm: float) -> np.ndarray:
        """Obstacle points in the robot frame (x forward, y left), from the lidar and the range sensor."""
        parts = []
        if scan is not None:
            r = np.asarray(scan.ranges_mm, dtype=np.float64)
            ok = np.isfinite(r)
            a = np.radians(np.asarray(scan.angles_deg, dtype=np.float64)[ok])
            parts.append(np.stack([self.lidar_x + r[ok] * np.cos(a), r[ok] * np.sin(a)], axis=1))
        if math.isfinite(range_mm) and range_mm < 2000.0:
            parts.append(np.array([[self.range_x + range_mm, 0.0]]))
        return np.concatenate(parts) if parts else np.zeros((0, 2))

    def allowed_speed(self, clearance_mm: float) -> float:
        """Fastest speed that still stops `margin` short of an obstacle `clearance` away (reaction included)."""
        d = clearance_mm - self.margin
        if d <= 0:
            return 0.0
        a, t = self.decel, self.reaction
        # v t + v^2 / (2a) = d  ->  v = a (-t + sqrt(t^2 + 2 d / a))
        return a * (-t + math.sqrt(t * t + 2.0 * d / a))

    def limit(self, v: float, w: float, pts: np.ndarray) -> Limit:
        if pts.size == 0:
            return Limit(v, w)
        x, y = pts[:, 0], pts[:, 1]
        inside = (np.abs(x) <= self.hl) & (np.abs(y) <= self.hw)  # the robot's own body (mast, cables)
        x, y = x[~inside], y[~inside]
        clearance = math.inf
        backward = v < 0
        if v != 0.0:
            sign = 1.0 if v > 0 else -1.0
            ahead = sign * x - self.hl  # distance past the front (or rear) edge
            lane = (np.abs(y) <= self.hw + self.side) & (ahead > -1.0) & (ahead < self.lookahead)
            if lane.any():
                clearance = float(ahead[lane].min())
                cap = self.allowed_speed(clearance)
                if cap < abs(v):
                    v = sign * cap
        if w != 0.0:
            near = np.hypot(x, y) < self.radius + self.side
            if near.any():
                w *= 0.35  # something brushes the turning circle: turn slowly
        blocked = clearance < self.margin + 1.0
        reason = ("obstacle_behind" if backward else "obstacle_stop") if blocked else None
        return Limit(v, w, reason, clearance)
