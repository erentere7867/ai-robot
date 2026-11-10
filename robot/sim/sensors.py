"""The simulated robot's senses: cameras rendered in MuJoCo and a raycast lidar.

Each sees the world from the body's true pose and tags its data with the estimated one, as on hardware.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from robot.perception.camera import Frame
from robot.perception.geometry import CameraModel
from robot.perception.lidar import Scan


class SimCamera:
    """One of the robot's cameras rendered in MuJoCo from the body's *true* pose (what a real lens would see).

    Frames are tagged with the *estimated* pose, as on hardware where only odometry is known.
    """

    def __init__(self, sandbox: Any, body: Any, clock: Any, model: CameraModel) -> None:
        self.sandbox, self.body, self.clock, self.model = sandbox, body, clock, model

    def capture(self) -> Frame:
        pose = self.body.pose()
        rgb = self.sandbox.render_rgb(self.model.name, self.body.true_pose())
        return Frame(pose, self.model, self.clock.monotonic(), rgb)

    def close(self) -> None:
        pass


class SimLidar:
    """Raycasts the simulated room: walls plus every object taller than the lidar's mounting height."""

    def __init__(self, sandbox: Any, body: Any, clock: Any, *, mount_x_mm: float, height_mm: float, beams: int = 360,
                 max_range_mm: float = 8000.0, rate_hz: float = 10.0, noise_mm: float = 8.0) -> None:
        self.sandbox, self.body, self.clock = sandbox, body, clock
        self.angles = np.linspace(-180.0, 180.0, beams, endpoint=False)
        self.max_range = max_range_mm
        self.height = height_mm
        self.period = 1.0 / rate_hz
        self.noise = noise_mm
        self.mount_x = mount_x_mm
        self._rng = np.random.default_rng(0)
        self._last: Scan | None = None

    def latest(self) -> Scan | None:
        now = self.clock.monotonic()
        if self._last is not None and now - self._last.stamp < self.period:
            return self._last
        ranges = self._cast(self.body.true_pose())
        self._last = Scan(self.body.pose(), self.angles, ranges, now, self.mount_x)
        return self._last

    def close(self) -> None:
        pass

    def _cast(self, pose: Any) -> np.ndarray:
        th = np.radians(pose.heading_deg + self.angles)
        dx, dy = np.cos(th), np.sin(th)
        yaw = math.radians(pose.heading_deg)
        ox, oy = pose.x_mm + self.mount_x * math.cos(yaw), pose.y_mm + self.mount_x * math.sin(yaw)
        half = self.sandbox.room_half_extent_mm
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx > 0, (half - ox) / dx, np.where(dx < 0, (-half - ox) / dx, np.inf))
            ty = np.where(dy > 0, (half - oy) / dy, np.where(dy < 0, (-half - oy) / dy, np.inf))
        best = np.minimum(tx, ty)
        for x, y, r in self.sandbox.lidar_circles(self.height):  # a person is two legs at this height
            cx, cy = x - ox, y - oy
            along = cx * dx + cy * dy
            perp2 = (cx * cx + cy * cy) - along * along
            hit = (along > 0) & (perp2 <= r * r)
            t = along - np.sqrt(np.maximum(0.0, r * r - perp2))
            best = np.where(hit & (t > 0) & (t < best), t, best)
        best = best + self._rng.normal(0.0, self.noise, best.shape)
        return np.where(best < self.max_range, best, np.inf)
