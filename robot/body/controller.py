"""Living Body Engine: CandidatePath and BodyController."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from robot.agent.api import wrap


@dataclass
class CandidatePath:
    label: str
    v: float
    w: float
    points: list[tuple[float, float]]
    clearance: float
    progress_score: float
    social_score: float
    visual_prob: float = 0.5
    total_score: float = 0.0


def _clear(a: tuple[float, float], b: tuple[float, float], occupied_fn: Callable[[float, float], bool]) -> bool:
    """No occupied point on the straight line from a to b (checked every 5 cm)."""
    n = max(1, int(math.dist(a, b) / 0.05))
    return not any(occupied_fn(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1))


class BodyController:
    """30-50 Hz Living Body Engine for driverless-car trajectory rollout, predictive social yielding,

    active vision maneuvers, non-stop waypoint pursuit, proprioception and procedural aliveness.
    """

    def __init__(self) -> None:
        self._snag_timer: float = 0.0
        self.labels: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")
        self.curvatures: tuple[float, ...] = (-1.5, -0.9, -0.4, 0.0, 0.4, 0.9, 1.5)
        self._turning_to: tuple[float, float] | None = None  # the waypoint it is turning on the spot to face

    def social_yield_score(
        self,
        points: list[tuple[float, float]],
        people: list[tuple[float, float, float, float]] | tuple,
        pose: tuple[float, float, float],
        dt: float = 0.1,
    ) -> float:
        """Projects each person (px, py, vx, vy) up to 2.0 s ahead. Penalizes bubble intrusions (<0.65m),

        rewards lateral right yielding, and biases orientation to yield along the right wall.
        """
        score = 0.0
        rx, ry, ryaw = pose
        for person in people:
            px, py, vx, vy = person
            rel_x = px - rx
            rel_y = py - ry
            dist_p = math.hypot(rel_x, rel_y)
            if dist_p > 2.5:
                continue
            dot = rel_x * vx + rel_y * vy
            if dot < -0.05:  # Moving toward the robot
                for t in (0.5, 1.0, 1.5, 2.0):
                    pred_x = px + vx * t
                    pred_y = py + vy * t
                    step_idx = min(len(points) - 1, max(0, int(round(t / dt)) - 1))
                    pt = points[step_idx]
                    dist_bubble = math.hypot(pt[0] - pred_x, pt[1] - pred_y)
                    if dist_bubble < 0.65:
                        score -= 3.0 * (0.65 - dist_bubble)

                for pt in points:
                    dx = pt[0] - rx
                    dy = pt[1] - ry
                    lateral_right = dx * math.sin(ryaw) - dy * math.cos(ryaw)
                    if lateral_right > 0.02:
                        score += 0.8 * lateral_right
                    elif lateral_right < -0.02:
                        score -= 0.5 * abs(lateral_right)

                if len(points) >= 2:
                    end_dx = points[-1][0] - rx
                    end_dy = points[-1][1] - ry
                    end_lat = end_dx * math.sin(ryaw) - end_dy * math.cos(ryaw)
                    if end_lat > 0.05:
                        score += 0.5
        return score

    def rollout_trajectories(
        self,
        pose: tuple[float, float, float],
        goal: tuple[float, float],
        occupied_fn: Callable[[float, float], bool],
        people: list[tuple[float, float, float, float]] | tuple = (),
        v_max: float = 0.45,
        horizon_s: float = 1.5,
        dt: float = 0.1,
    ) -> list[CandidatePath]:
        """Rolls out 7 smooth constant-curvature candidate arcs over horizon_s, evaluates clearance,

        progress toward goal, predictive social yielding, and returns candidate paths sorted best-first.
        """
        rx, ry, ryaw = pose
        d0 = math.hypot(goal[0] - rx, goal[1] - ry)
        candidates: list[CandidatePath] = []
        num_steps = max(1, int(round(horizon_s / dt)))

        for label, w in zip(self.labels, self.curvatures, strict=False):
            v = v_max / (1.0 + 0.5 * abs(w))
            points: list[tuple[float, float]] = []
            blocked = False
            clearance = horizon_s * v

            for step in range(1, num_steps + 1):
                t = step * dt
                if abs(w) < 1e-4:
                    nx = rx + v * t * math.cos(ryaw)
                    ny = ry + v * t * math.sin(ryaw)
                else:
                    r = v / w
                    nyaw = ryaw + w * t
                    nx = rx + r * (math.sin(nyaw) - math.sin(ryaw))
                    ny = ry - r * (math.cos(nyaw) - math.cos(ryaw))
                points.append((nx, ny))
                if not blocked and occupied_fn(nx, ny):
                    blocked = True
                    clearance = (step - 1) * dt * v

            d_end = math.hypot(goal[0] - points[-1][0], goal[1] - points[-1][1])
            progress_dist = d0 - d_end
            end_yaw = ryaw + w * horizon_s
            goal_yaw = math.atan2(goal[1] - points[-1][1], goal[0] - points[-1][0])
            heading_align = math.cos(goal_yaw - end_yaw)
            progress_score = progress_dist + 0.3 * (heading_align + 1.0)
            social_score = self.social_yield_score(points, people, pose)

            if blocked:
                total_score = -500.0 + clearance + progress_score
            else:
                total_score = 2.0 * progress_score + 1.2 * clearance + 1.5 * social_score - 0.05 * abs(w)

            candidates.append(
                CandidatePath(
                    label=label,
                    v=round(v, 3),
                    w=round(w, 3),
                    points=points,
                    clearance=round(clearance, 3),
                    progress_score=round(progress_score, 3),
                    social_score=round(social_score, 3),
                    visual_prob=0.5,
                    total_score=round(total_score, 3),
                )
            )

        candidates.sort(key=lambda c: c.total_score, reverse=True)
        return candidates

    def step_pursuit(
        self,
        pose: tuple[float, float, float],
        waypoints: list[tuple[float, float]],
        idx: int,
        blend_r: float = 0.28,
        v_cruise: float = 0.4,
        occupied_fn: Callable[[float, float], bool] = lambda x, y: False,
        people: list[tuple[float, float, float, float]] | tuple = (),
    ) -> tuple[float, float, int, bool]:
        """Non-stop waypoint spline pursuit: advances idx when within blend_r of intermediate waypoints

        so the robot never stops (v > 0) until the final waypoint. Returns (v, w, new_idx, arrived).
        """
        if not waypoints or idx >= len(waypoints):
            return 0.0, 0.0, idx, True

        # On to the next waypoint near this one, but cut the corner only when the straight line there is clear: in a
        # furnished room the cut line can run into the corner of a cabinet the path went round.
        while idx < len(waypoints) - 1 and (
                math.dist(pose[:2], waypoints[idx]) <= 0.1
                or math.dist(pose[:2], waypoints[idx]) <= blend_r and _clear(pose[:2], waypoints[idx + 1], occupied_fn)):
            idx += 1

        is_last = idx == len(waypoints) - 1
        dist_to_target = math.dist(pose[:2], waypoints[idx])

        if is_last and dist_to_target <= 0.08:
            return 0.0, 0.0, idx, True

        if is_last:
            speed = v_cruise * min(1.0, max(0.2, dist_to_target / 0.35))
        else:
            speed = v_cruise

        err = wrap(math.atan2(waypoints[idx][1] - pose[1], waypoints[idx][0] - pose[0]) - pose[2])
        # A waypoint well off to the side or behind: turn on the spot to face it first, as a differential drive can
        # (arcs need room a corner may not have), and keep turning until nearly facing it, not just under the limit.
        if abs(err) > 0.8 or (self._turning_to == waypoints[idx] and abs(err) > 0.15):
            self._turning_to = waypoints[idx]
            return 0.0, max(-1.5, min(1.5, 2.5 * err)), idx, False
        self._turning_to = None

        candidates = self.rollout_trajectories(
            pose, waypoints[idx], occupied_fn, people, v_max=speed, horizon_s=1.2, dt=0.1
        )
        best = candidates[0]

        if best.clearance > 0.15 or not occupied_fn(best.points[0][0], best.points[0][1]):
            v = best.v
            w = best.w
        else:
            w = max(-1.5, min(1.5, 2.5 * err))
            v = 0.05 if abs(err) < 0.6 else 0.0

        if not is_last and v < 0.1 and abs(err) < 0.6:
            v = 0.15

        return v, w, idx, False

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
