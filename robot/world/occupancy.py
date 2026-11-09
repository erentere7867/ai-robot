"""The occupancy grid: where solid things are (lidar and range sensor) and which floor the cameras have seen."""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from robot.body.hal import BodyPose
from robot.core.angles import wrap_deg_array
from robot.perception.geometry import CameraModel

CELL_MM = 50.0


class OccupancyGrid:
    def __init__(self, half_extent_mm: float = 3000.0) -> None:
        self.lock = threading.Lock()
        self.half = float(half_extent_mm)
        self.n = int(math.ceil(2 * self.half / CELL_MM))
        self.occupancy = np.zeros((self.n, self.n), dtype=np.float32)  # log-odds-like evidence
        self.seen = np.zeros((self.n, self.n), dtype=bool)  # floor cells a camera has looked at
        self._centers = -self.half + (np.arange(self.n) + 0.5) * CELL_MM
        self._xs = np.broadcast_to(self._centers[:, None], (self.n, self.n))
        self._ys = np.broadcast_to(self._centers[None, :], (self.n, self.n))

    def occupied_mask(self) -> np.ndarray:
        return self.occupancy > 0.6

    def occupied_points(self) -> np.ndarray:
        i, j = np.nonzero(self.occupied_mask())
        return np.stack([self._centers[i], self._centers[j]], axis=1)

    def coverage(self) -> float:
        return float(self.seen.mean())

    def _cell(self, x_mm: float, y_mm: float) -> tuple[int, int] | None:
        i, j = int((x_mm + self.half) // CELL_MM), int((y_mm + self.half) // CELL_MM)
        return (i, j) if 0 <= i < self.n and 0 <= j < self.n else None

    def _cells(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Flat grid indices of points inside the map."""
        i = np.floor((xs + self.half) / CELL_MM).astype(np.int64)
        j = np.floor((ys + self.half) / CELL_MM).astype(np.int64)
        inside = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        return i[inside] * self.n + j[inside]

    def clear_hits(self) -> None:
        """After the room is rearranged, old hits would be ghost obstacles."""
        with self.lock:
            self.occupancy[:] = 0.0

    def add_obstacle_point(self, x_mm: float, y_mm: float, *, weight: float = 1.5) -> None:
        cell = self._cell(x_mm, y_mm)
        if cell is not None:
            with self.lock:
                self.occupancy[cell] = min(3.0, self.occupancy[cell] + weight)

    def update_range(self, pose: BodyPose, range_mm: float, *, offset_mm: float, max_range_mm: float = 2400.0) -> None:
        """One forward range reading: free space along the beam, an obstacle at its end."""
        if not math.isfinite(range_mm) or range_mm <= 0:
            return
        yaw = math.radians(pose.heading_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        ox, oy = pose.x_mm + offset_mm * c, pose.y_mm + offset_mm * s
        hit = range_mm < max_range_mm
        length = min(range_mm, max_range_mm)
        last = None
        with self.lock:
            for k in range(int(length // (CELL_MM * 0.7))):
                d = k * CELL_MM * 0.7
                cell = self._cell(ox + d * c, oy + d * s)
                if cell is None:
                    break
                if cell != last and d < length - CELL_MM:
                    self.occupancy[cell] = max(-2.0, self.occupancy[cell] - 0.35)
                last = cell
        if hit:
            self.add_obstacle_point(ox + range_mm * c, oy + range_mm * s, weight=0.9)

    def update_scan(self, pose: BodyPose, scan: Any, *, max_range_mm: float = 4000.0, hit_weight: float = 0.7,
                    free_weight: float = 0.25) -> None:
        """A 360-degree scan taken at `pose`: free space along the beams, an obstacle where each one ends.

        Vectorised, ~1 ms: free space is traced on every second beam (neighbours cover the same cells at
        this grid size), hits on all of them.
        """
        ranges = np.asarray(scan.ranges_mm, dtype=np.float64)
        th = np.radians(pose.heading_deg + np.asarray(scan.angles_deg, dtype=np.float64))
        yaw = math.radians(pose.heading_deg)
        ox, oy = pose.x_mm + scan.mount_x_mm * math.cos(yaw), pose.y_mm + scan.mount_x_mm * math.sin(yaw)
        occ = self.occupancy.reshape(-1)
        stride = 2 if len(ranges) >= 180 else 1
        r_free, th_free = ranges[::stride], th[::stride]
        length = np.minimum(np.where(np.isfinite(r_free), r_free, max_range_mm), max_range_mm)
        d = np.arange(0.0, float(length.max(initial=0.0)), CELL_MM)
        free = d[None, :] < (length[:, None] - CELL_MM)
        cells = self._cells((ox + d[None, :] * np.cos(th_free)[:, None])[free],
                            (oy + d[None, :] * np.sin(th_free)[:, None])[free])
        with self.lock:
            mask = np.zeros(occ.size, dtype=bool)
            mask[cells] = True
            occ[mask] = np.maximum(-2.0, occ[mask] - free_weight)
            hit = np.isfinite(ranges) & (ranges < max_range_mm)
            if hit.any():
                mask[:] = False
                mask[self._cells(ox + ranges[hit] * np.cos(th[hit]), oy + ranges[hit] * np.sin(th[hit]))] = True
                occ[mask] = np.minimum(3.0, occ[mask] + hit_weight)

    def mark_coverage(self, pose: BodyPose, model: CameraModel, *, range_mm: float = 2200.0) -> None:
        cx, cy, _, yaw, _ = model.extrinsics(pose)
        dx, dy = self._xs - cx, self._ys - cy
        dist = np.hypot(dx, dy)
        ang = wrap_deg_array(np.degrees(np.arctan2(dy, dx) - yaw))
        visible = (dist <= range_mm) & (dist >= 250.0) & (np.abs(ang) <= 0.5 * model.hfov_deg - 2)
        with self.lock:
            self.seen |= visible

    def unseen_fraction_near(self, x_mm: float, y_mm: float, radius_mm: float = 1500.0) -> float:
        mask = (self._xs - x_mm) ** 2 + (self._ys - y_mm) ** 2 <= radius_mm ** 2
        free = mask & ~self.occupied_mask()
        total = int(free.sum())
        return float((free & ~self.seen).sum()) / total if total else 0.0
