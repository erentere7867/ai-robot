"""A 2D 360-degree lidar's output: walls, furniture and legs in every direction, ten times a second.

The camera finds and names things in front; the lidar knows where solid things are all around, which is
what navigation needs. Scans are fused into the occupancy grid by the fast loop, tagged with the pose at capture.
Sources: robot/sim/sensors.py and robot/hardware/sensors.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class Scan:
    pose: Any  # BodyPose when the scan was taken
    angles_deg: np.ndarray  # relative to the robot's heading, + = left
    ranges_mm: np.ndarray  # inf = no return within range
    stamp: float
    mount_x_mm: float  # where the lidar sits along the robot's x axis (behind the centre on the mast)

    def world_points(self) -> np.ndarray:
        """The returns as world points (N, 2), from the lidar's mount."""
        ranges = np.asarray(self.ranges_mm, dtype=np.float64)
        ok = np.isfinite(ranges)
        if not ok.any():
            return np.zeros((0, 2))
        h = math.radians(self.pose.heading_deg)
        ox, oy = self.pose.x_mm + self.mount_x_mm * math.cos(h), self.pose.y_mm + self.mount_x_mm * math.sin(h)
        th = h + np.radians(np.asarray(self.angles_deg, dtype=np.float64)[ok])
        return np.stack([ox + ranges[ok] * np.cos(th), oy + ranges[ok] * np.sin(th)], axis=1)


class Lidar(Protocol):
    def latest(self) -> Scan | None: ...

    def close(self) -> None: ...
