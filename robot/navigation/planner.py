"""Grid path planning (A* + line-of-sight smoothing) and camera coverage tracking for exploration."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircleObstacle:
    """Obstacle footprint approximated by a circle in the floor plane (mm)."""

    x_mm: float
    y_mm: float
    radius_mm: float
    label: str = ""


@dataclass
class PlannedPath:
    ok: bool
    waypoints: list[tuple[float, float]]
    length_mm: float = 0.0
    message: str = ""


class GridPlanner:
    """8-connected A* over an inflated occupancy grid of a rectangular room.

    The robot footprint is handled by inflating every obstacle (and the walls) by
    `robot_radius_mm + clearance_mm`, so the planner can treat the robot as a point.
    """

    def __init__(
        self,
        *,
        room_half_extent_mm: float = 1500.0,
        resolution_mm: float = 40.0,
        robot_radius_mm: float = 180.0,
        clearance_mm: float = 20.0,
    ) -> None:
        self.half = float(room_half_extent_mm)
        self.res = float(resolution_mm)
        self.robot_radius_mm = float(robot_radius_mm)
        self.clearance_mm = float(clearance_mm)
        self.n = int(math.ceil(2.0 * self.half / self.res))

    @property
    def inflation_mm(self) -> float:
        return self.robot_radius_mm + self.clearance_mm

    def to_cell(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        i = int((x_mm + self.half) / self.res)
        j = int((y_mm + self.half) / self.res)
        return (max(0, min(self.n - 1, i)), max(0, min(self.n - 1, j)))

    def to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (-self.half + (cell[0] + 0.5) * self.res, -self.half + (cell[1] + 0.5) * self.res)

    def build_grid(
        self, obstacles: list[CircleObstacle], *, extra_inflation_mm: float = 0.0,
        points: np.ndarray | None = None, point_radius_mm: float = 25.0,
    ) -> np.ndarray:
        """Boolean grid where True = blocked for the robot centre.

        `points` are (N, 2) obstacle points in mm (e.g. range-sensor hits), each a small disc.
        """
        infl = self.inflation_mm + extra_inflation_mm
        centers = -self.half + (np.arange(self.n) + 0.5) * self.res
        xs = centers[:, None]
        ys = centers[None, :]
        wall = self.half - infl
        grid = (np.abs(xs) > wall) | (np.abs(ys) > wall)
        grid = np.broadcast_to(grid, (self.n, self.n)).copy()
        for ob in obstacles:
            r = ob.radius_mm + infl
            grid |= (xs - ob.x_mm) ** 2 + (ys - ob.y_mm) ** 2 <= r * r
        if points is not None and len(points):
            r2 = (point_radius_mm + infl) ** 2
            for px, py in points:
                grid |= (xs - px) ** 2 + (ys - py) ** 2 <= r2
        return grid

    def is_free(self, grid: np.ndarray, x_mm: float, y_mm: float) -> bool:
        if abs(x_mm) >= self.half or abs(y_mm) >= self.half:
            return False
        i, j = self.to_cell(x_mm, y_mm)
        return not bool(grid[i, j])

    def nearest_free(self, grid: np.ndarray, x_mm: float, y_mm: float, max_r_mm: float = 600.0) -> tuple[float, float] | None:
        ci, cj = self.to_cell(x_mm, y_mm)
        if not grid[ci, cj]:
            return self.to_world((ci, cj))
        max_r = int(max_r_mm / self.res)
        best: tuple[float, tuple[int, int]] | None = None
        for r in range(1, max_r + 1):
            for di in range(-r, r + 1):
                for dj in (-r, r) if abs(di) != r else range(-r, r + 1):
                    i, j = ci + di, cj + dj
                    if 0 <= i < self.n and 0 <= j < self.n and not grid[i, j]:
                        d = di * di + dj * dj
                        if best is None or d < best[0]:
                            best = (d, (i, j))
            if best is not None:
                return self.to_world(best[1])
        return None

    def _line_free(
        self, grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], *, allow_escape: bool = False
    ) -> bool:
        """Bresenham-style check; `allow_escape` tolerates the initial blocked run when leaving an inflated zone."""
        steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1) * 2
        escaping = allow_escape
        for k in range(steps + 1):
            t = k / steps
            i = int(round(a[0] + (b[0] - a[0]) * t))
            j = int(round(a[1] + (b[1] - a[1]) * t))
            if grid[i, j]:
                if not escaping:
                    return False
            else:
                escaping = False
        return True

    def line_is_free(self, grid, start, goal, *, allow_escape=False) -> bool:
        """Check a segment from actual odometry, rather than rounded cell centres."""
        steps = max(1, int(math.ceil(math.hypot(goal[0] - start[0], goal[1] - start[1]) / (self.res / 4))))
        escaping = allow_escape
        for k in range(steps + 1):
            t = k / steps
            x, y = start[0] + (goal[0] - start[0]) * t, start[1] + (goal[1] - start[1]) * t
            if not self.is_free(grid, x, y):
                if not escaping:
                    return False
            else:
                escaping = False
        return True

    def plan(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        obstacles: list[CircleObstacle],
        *,
        points: np.ndarray | None = None,
        goal_tolerance_mm: float = 500.0,
    ) -> PlannedPath:
        """Plan a smoothed collision-free path from `start` to `goal` (both in mm)."""
        if any(not math.isfinite(v) or abs(v) >= self.half for v in (*start, *goal)):
            return PlannedPath(False, [], message="Route endpoint is outside the mapped area.")
        grid = self.build_grid(obstacles, points=points)
        goal_free = self.nearest_free(grid, *goal, max_r_mm=goal_tolerance_mm)
        if goal_free is None:
            return PlannedPath(False, [], message="Goal is blocked by obstacles or walls.")
        s = self.to_cell(*start)
        g = self.to_cell(*goal_free)
        # The robot may currently sit inside an inflated zone (e.g. right after approaching
        # something); allow leaving occupied cells but never entering new ones.
        start_blocked = bool(grid[s])
        cells = self._astar(grid, s, g, allow_start_escape=start_blocked)
        if cells is None:
            return PlannedPath(False, [], message="No collision-free route found.")
        # String-pull smoothing
        smoothed = [cells[0]]
        idx = 0
        while idx < len(cells) - 1:
            nxt = len(cells) - 1
            while nxt > idx + 1 and not self.line_is_free(
                grid, start if idx == 0 else self.to_world(cells[idx]), self.to_world(cells[nxt]),
                allow_escape=(idx == 0 and start_blocked)
            ):
                nxt -= 1
            smoothed.append(cells[nxt])
            idx = nxt
        pts = [self.to_world(c) for c in smoothed[1:]]
        if not pts or math.hypot(pts[-1][0] - goal_free[0], pts[-1][1] - goal_free[1]) > 1.0:
            pts.append(goal_free)
        else:
            pts[-1] = goal_free
        length = 0.0
        prev = start
        for p in pts:
            length += math.hypot(p[0] - prev[0], p[1] - prev[1])
            prev = p
        return PlannedPath(True, pts, length_mm=length, message="ok")

    def _astar(
        self, grid: np.ndarray, s: tuple[int, int], g: tuple[int, int], *, allow_start_escape: bool
    ) -> list[tuple[int, int]] | None:
        if s == g:
            return [s, g]
        n = self.n
        open_heap: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, s)]
        came: dict[tuple[int, int], tuple[int, int]] = {}
        cost = {s: 0.0}
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)] + [
            (di, dj, math.sqrt(2.0)) for di in (-1, 1) for dj in (-1, 1)
        ]
        expansions = 0
        while open_heap:
            _, c_cost, cur = heapq.heappop(open_heap)
            if cur == g:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]
            if c_cost > cost.get(cur, math.inf):
                continue
            expansions += 1
            if expansions > 60000:
                return None
            cur_blocked = bool(grid[cur])
            for di, dj, step in moves:
                ni, nj = cur[0] + di, cur[1] + dj
                if not (0 <= ni < n and 0 <= nj < n):
                    continue
                if grid[ni, nj] and not (allow_start_escape and cur_blocked):
                    continue
                if di and dj and (grid[cur[0] + di, cur[1]] or grid[cur[0], cur[1] + dj]):
                    continue
                nc = c_cost + step
                nxt = (ni, nj)
                if nc < cost.get(nxt, math.inf):
                    cost[nxt] = nc
                    came[nxt] = cur
                    h = math.hypot(g[0] - ni, g[1] - nj)
                    heapq.heappush(open_heap, (nc + h, nc, nxt))
        return None


class CoverageMap:
    """Tracks which floor cells the head camera has already looked at (for frontier-style exploration)."""

    def __init__(self, *, room_half_extent_mm: float = 1500.0, resolution_mm: float = 100.0) -> None:
        self.half = float(room_half_extent_mm)
        self.res = float(resolution_mm)
        self.n = int(math.ceil(2.0 * self.half / self.res))
        self.seen = np.zeros((self.n, self.n), dtype=bool)
        centers = -self.half + (np.arange(self.n) + 0.5) * self.res
        self._xs = np.broadcast_to(centers[:, None], (self.n, self.n))
        self._ys = np.broadcast_to(centers[None, :], (self.n, self.n))

    def reset(self) -> None:
        self.seen[:] = False

    def mark_view(
        self, cam_x_mm: float, cam_y_mm: float, heading_deg: float, *, hfov_deg: float, range_mm: float
    ) -> None:
        dx = self._xs - cam_x_mm
        dy = self._ys - cam_y_mm
        dist = np.hypot(dx, dy)
        ang = np.degrees(np.arctan2(dy, dx)) - heading_deg
        ang = (ang + 180.0) % 360.0 - 180.0
        # Floor closer than ~250 mm is below the camera's view cone
        self.seen |= (dist <= range_mm) & (dist >= 250.0) & (np.abs(ang) <= 0.5 * hfov_deg)

    @property
    def fraction_seen(self) -> float:
        return float(self.seen.mean())

    def unseen_near(self, x_mm: float, y_mm: float, radius_mm: float) -> int:
        mask = (self._xs - x_mm) ** 2 + (self._ys - y_mm) ** 2 <= radius_mm * radius_mm
        return int(np.count_nonzero(mask & ~self.seen))
