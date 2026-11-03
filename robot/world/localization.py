"""Lidar localization: keep the robot's pose honest against the map it is building.

Wheel odometry drifts (skid steer slips in every turn, the gyro wanders), and everything the robot knows
is placed with that pose, so the map smears and "come back to where you started" ends somewhere else.
Each new scan is matched against the occupancy grid (correlative scan matching: try small shifts and
turns of the scan, keep the one whose hit points land on the most occupied cells) and the difference is
folded into a correction from the odometry frame to the map frame. The body itself is untouched; every
consumer reads the corrected pose through `LocalizedBody`.

Guards: no correction until the map has structure where the scan points land (an empty room cannot
correct anything), only a fraction of each match is applied (one noisy match cannot jerk the pose), and
a match must beat "no change" by a margin.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from robot.body.hal import BodyPose
from robot.world.spatial_map import CELL_MM


class Localizer:
    def __init__(self, world: Any, *, search_mm: float = 80.0, step_mm: float = 20.0, search_deg: float = 3.0,
                 step_deg: float = 1.0, gain: float = 0.5, min_support: float = 0.35, margin: float = 1.04,
                 mount_x_mm: float = -105.0, max_points: int = 180, max_range_mm: float = 4000.0) -> None:
        self.world = world
        self.shifts = np.arange(-search_mm, search_mm + 1e-6, step_mm)
        self.turns = np.arange(-search_deg, search_deg + 1e-6, step_deg)
        self.gain, self.min_support, self.margin = gain, min_support, margin
        self.mount_x = mount_x_mm
        self.max_points, self.max_range = max_points, max_range_mm
        self._lock = threading.Lock()
        self.tx = self.ty = self.th = 0.0  # map <- odometry: rotate by th (deg) about the origin, then shift
        self.matches = self.applied = 0
        self.last_support = 0.0

    # ---- frames ---------------------------------------------------------------------------------
    def to_map(self, pose: BodyPose) -> BodyPose:
        with self._lock:
            tx, ty, th = self.tx, self.ty, self.th
        if tx == 0.0 and ty == 0.0 and th == 0.0:
            return pose
        c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
        return BodyPose(c * pose.x_mm - s * pose.y_mm + tx, s * pose.x_mm + c * pose.y_mm + ty,
                        (pose.heading_deg + th + 180.0) % 360.0 - 180.0, pose.head_tilt_deg,
                        pose.linear_vel_mm_s, pose.angular_vel_deg_s)

    def snapshot(self) -> dict[str, float]:
        return {"correction_mm": [round(self.tx, 1), round(self.ty, 1)], "correction_deg": round(self.th, 2),
                "matches": self.matches, "applied": self.applied, "support": round(self.last_support, 2)}

    # ---- matching -------------------------------------------------------------------------------
    def correct(self, pose: BodyPose, angles_deg: np.ndarray, ranges_mm: np.ndarray) -> BodyPose:
        """Match a scan taken at `pose` (map frame) and return the corrected pose (and keep the correction)."""
        r = np.asarray(ranges_mm, dtype=np.float64)
        a = np.radians(np.asarray(angles_deg, dtype=np.float64))
        ok = np.isfinite(r) & (r < self.max_range) & (r > 150.0)
        if ok.sum() < 30:
            return pose
        r, a = r[ok], a[ok]
        if len(r) > self.max_points:
            keep = np.linspace(0, len(r) - 1, self.max_points).astype(int)
            r, a = r[keep], a[keep]
        # Scan points in the robot frame (the lidar sits behind the centre).
        px, py = self.mount_x + r * np.cos(a), r * np.sin(a)
        occ = np.clip(self.world.occupancy, 0.0, 3.0)
        n, half = self.world.n, self.world.half
        best = (-1.0, 0.0, 0.0, 0.0)
        zero = 0.0
        for dth in self.turns:
            yaw = math.radians(pose.heading_deg + dth)
            c, s = math.cos(yaw), math.sin(yaw)
            wx = pose.x_mm + c * px - s * py
            wy = pose.y_mm + s * px + c * py
            gx = wx[None, None, :] + self.shifts[:, None, None]
            gy = wy[None, None, :] + self.shifts[None, :, None]
            i = np.floor((gx + half) / CELL_MM).astype(np.int64)
            j = np.floor((gy + half) / CELL_MM).astype(np.int64)
            inside = (i >= 0) & (i < n) & (j >= 0) & (j < n)
            vals = np.where(inside, occ[np.clip(i, 0, n - 1), np.clip(j, 0, n - 1)], 0.0)
            score = vals.sum(axis=2)  # (shifts x, shifts y)
            k = np.unravel_index(int(np.argmax(score)), score.shape)
            if score[k] > best[0]:
                best = (float(score[k]), float(self.shifts[k[0]]), float(self.shifts[k[1]]), float(dth))
            if dth == 0.0:
                z = len(self.shifts) // 2
                zero = float(score[z, z])
        self.matches += 1
        support = best[0] / (3.0 * len(r))  # share of points on solidly occupied cells
        self.last_support = support
        if support < self.min_support or best[0] <= zero * self.margin or (best[1], best[2], best[3]) == (0, 0, 0):
            return pose
        g = self.gain
        dx, dy, dth = g * best[1], g * best[2], g * best[3]
        self._compose(dx, dy, dth, pose)
        self.applied += 1
        return BodyPose(pose.x_mm + dx, pose.y_mm + dy, (pose.heading_deg + dth + 180.0) % 360.0 - 180.0,
                        pose.head_tilt_deg, pose.linear_vel_mm_s, pose.angular_vel_deg_s)

    def _compose(self, dx: float, dy: float, dth: float, pose: BodyPose) -> None:
        """Fold a correction found at `pose` (rotate by dth about the robot, then shift) into map <- odom."""
        with self._lock:
            c, s = math.cos(math.radians(dth)), math.sin(math.radians(dth))
            # new_map = Rot(dth) about pose, then shift: apply to the existing transform's translation
            ox, oy = self.tx - pose.x_mm, self.ty - pose.y_mm
            self.tx = c * ox - s * oy + pose.x_mm + dx
            self.ty = s * ox + c * oy + pose.y_mm + dy
            self.th = (self.th + dth + 180.0) % 360.0 - 180.0


class LocalizedBody:
    """The body as everything else sees it: same commands, pose in the map frame (corrected odometry)."""

    def __init__(self, body: Any, localizer: Localizer) -> None:
        object.__setattr__(self, "_body", body)
        object.__setattr__(self, "localizer", localizer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._body, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._body, name, value)

    @property
    def raw(self) -> Any:
        return self._body

    def pose(self) -> BodyPose:
        return self.localizer.to_map(self._body.pose())
