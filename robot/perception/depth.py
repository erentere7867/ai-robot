"""Metric depth calibration, ToF step/cliff hazard classification, and 360-degree synthetic scan."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robot.perception.rig import RIG, Camera


@dataclass(frozen=True)
class StepHazard:
    kind: str  # "clear" | "traversable" | "cliff" | "obstacle"
    delta_z: float  # metres: positive = step up / obstacle above floor, negative = drop below floor
    distance: float  # metres forward along sensor
    torque_boost: bool = False


def classify_tof_grid(tof_grid: np.ndarray, expected_floor: np.ndarray) -> StepHazard:
    """Classify 8x8 ToF laser matrix into cliff, obstacle, traversable step (rug/threshold), or clear.

    Height delta delta_z = expected_floor - tof_grid.
    - delta_z < -0.04 m: cliff (floor dropped below expected)
    - delta_z > +0.04 m: obstacle (surface raised above floor)
    - 0.004 m <= abs(delta_z) <= 0.022 m: traversable step/rug; torque_boost True when delta_z >= 0.008 m
    - abs(delta_z) < 0.004 m: clear floor
    """
    grid = np.asarray(tof_grid, dtype=np.float32)
    expected = np.asarray(expected_floor, dtype=np.float32)
    valid = np.isfinite(grid) & np.isfinite(expected) & (grid > 0.02) & (grid < 6.0)

    if not valid.any():
        return StepHazard("clear", 0.0, 0.0, False)

    dz = expected - grid

    # Priority 1: Cliff drop (delta_z < -0.04)
    cliff_mask = valid & (dz < -0.04)
    if cliff_mask.any():
        cliff_dz = np.where(cliff_mask, dz, 0.0)
        idx = int(np.argmin(cliff_dz))
        return StepHazard("cliff", float(dz.flat[idx]), float(grid.flat[idx]), False)

    # Priority 2: Obstacle above floor (delta_z > 0.04)
    obs_mask = valid & (dz > 0.04)
    if obs_mask.any():
        obs_dz = np.where(obs_mask, dz, -100.0)
        idx = int(np.argmax(obs_dz))
        return StepHazard("obstacle", float(dz.flat[idx]), float(grid.flat[idx]), False)

    # Priority 3: Traversable step or rug (0.004 <= abs(delta_z) <= 0.022)
    trav_mask = valid & (np.abs(dz) >= 0.004) & (np.abs(dz) <= 0.022)
    if trav_mask.any():
        max_pos = float(np.max(dz[trav_mask]))
        torque_boost = max_pos >= 0.008
        trav_abs = np.where(trav_mask, np.abs(dz), -1.0)
        idx = int(np.argmax(trav_abs))
        return StepHazard("traversable", float(dz.flat[idx]), float(grid.flat[idx]), torque_boost)

    # Priority 4: Clear floor
    median_dist = float(np.median(grid[valid]))
    return StepHazard("clear", 0.0, median_dist, False)


def _tilt_matrix(pitch_rad: float, roll_rad: float) -> np.ndarray:
    """Rotation matrix R_tilt = R_y(pitch) @ R_x(roll) where positive pitch tilts nose down."""
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    # R_y(pitch) with positive pitch tilting +X toward -Z (nose down)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    # R_x(roll) with positive roll tilting +Y toward +Z
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    return ry @ rx


def calibrate_depth(raw_depth: np.ndarray, cam: Camera, pitch_rad: float = 0.0, roll_rad: float = 0.0,
                    tof_grid: np.ndarray | None = None) -> np.ndarray:
    """Calibrate raw depth array to metric metres [0.05, 12.0] using floor plane trigonometry and optional ToF anchors."""
    raw = np.asarray(raw_depth, dtype=np.float32)
    h, w = raw.shape
    r_tilt = _tilt_matrix(pitch_rad, roll_rad)

    # Camera center in world frame
    cam_pos = np.array([cam.x, cam.y, cam.z], dtype=np.float32)
    cam_world = r_tilt @ cam_pos
    cam_world_z = float(cam_world[2])

    # Sample floor region (lower image: v > 0.6 * height)
    v_start = int(0.6 * h)
    v_grid, u_grid = np.mgrid[v_start:h:8, 0:w:16]
    v_flat, u_flat = v_grid.ravel(), u_grid.ravel()

    x_opt = (u_flat.astype(np.float32) - cam.width / 2) / cam.focal
    y_opt = (v_flat.astype(np.float32) - cam.height / 2) / cam.focal

    cos_yaw, sin_yaw = math.cos(cam.yaw), math.sin(cam.yaw)
    dx = cos_yaw + x_opt * sin_yaw
    dy = sin_yaw - x_opt * cos_yaw
    dz = -y_opt

    # Ray z in world frame
    v_world_z = r_tilt[2, 0] * dx + r_tilt[2, 1] * dy + r_tilt[2, 2] * dz

    # Rays pointing toward the floor (v_world_z < -1e-4)
    downward = v_world_z < -1e-4
    t_floor = np.full_like(v_world_z, np.nan)
    t_floor[downward] = -cam_world_z / v_world_z[downward]

    raw_sampled = raw[v_flat, u_flat]
    valid_floor = np.isfinite(raw_sampled) & np.isfinite(t_floor) & (raw_sampled > 0.05) & (t_floor >= 0.1) & (t_floor <= 8.0)

    anchor_x = list(raw_sampled[valid_floor])
    anchor_y = list(t_floor[valid_floor])

    # Extract ToF anchors if provided
    tof_x: list[float] = []
    tof_y: list[float] = []
    if tof_grid is not None:
        tof = np.asarray(tof_grid, dtype=np.float32)
        if tof.shape == (8, 8):
            tof_fov = 0.785  # ~45 deg FOV
            for r in range(8):
                for c in range(8):
                    dist_tof = float(tof[r, c])
                    if not (0.05 <= dist_tof <= 4.0 and np.isfinite(dist_tof)):
                        continue
                    theta_h = (c - 3.5) / 4.0 * (tof_fov / 2)
                    theta_v = (r - 3.5) / 4.0 * (tof_fov / 2)
                    u_t = int(cam.width / 2 + math.tan(theta_h) * cam.focal)
                    v_t = int(cam.height / 2 + math.tan(theta_v) * cam.focal)
                    if 0 <= u_t < w and 0 <= v_t < h:
                        d_raw = float(raw[v_t, u_t])
                        if np.isfinite(d_raw) and d_raw > 0.05:
                            tof_x.append(d_raw)
                            tof_y.append(dist_tof)

    a, b = 1.0, 0.0
    if len(tof_x) >= 4:
        tx = np.array(tof_x, dtype=np.float32)
        ty = np.array(tof_y, dtype=np.float32)
        span_x = float(np.max(tx) - np.min(tx))
        if span_x > 0.05:
            rng = np.random.default_rng(42)
            best_inliers: np.ndarray | None = None
            best_count = 0
            for _ in range(50):
                idx = rng.choice(len(tx), size=2, replace=False)
                x1, x2 = tx[idx[0]], tx[idx[1]]
                y1, y2 = ty[idx[0]], ty[idx[1]]
                if abs(x2 - x1) < 0.03:
                    continue
                cand_a = (y2 - y1) / (x2 - x1)
                cand_b = y1 - cand_a * x1
                if not (0.05 <= cand_a <= 15.0 and abs(cand_b) <= 8.0):
                    continue
                residuals = np.abs(cand_a * tx + cand_b - ty)
                inliers = residuals < 0.08
                count = int(np.count_nonzero(inliers))
                if count > best_count:
                    best_count = count
                    best_inliers = inliers
            if best_inliers is not None and best_count >= 3:
                poly = np.polyfit(tx[best_inliers], ty[best_inliers], deg=1)
                a, b = float(poly[0]), float(poly[1])
            else:
                poly = np.polyfit(tx, ty, deg=1)
                if 0.1 <= poly[0] <= 10.0:
                    a, b = float(poly[0]), float(poly[1])
        else:
            mean_x = float(np.mean(tx))
            mean_y = float(np.mean(ty))
            if mean_x > 0.01:
                a = max(0.1, min(10.0, mean_y / mean_x))
                b = 0.0

        # Refine with consistent floor inliers if present
        if len(anchor_x) > 0:
            fx = np.array(anchor_x, dtype=np.float32)
            fy = np.array(anchor_y, dtype=np.float32)
            floor_diff = np.abs(a * fx + b - fy)
            floor_ok = floor_diff < 0.10
            if np.count_nonzero(floor_ok) >= 10:
                all_x = np.concatenate([tx, fx[floor_ok]])
                all_y = np.concatenate([ty, fy[floor_ok]])
                if np.max(all_x) - np.min(all_x) > 0.05:
                    poly = np.polyfit(all_x, all_y, deg=1)
                    if 0.1 <= poly[0] <= 10.0:
                        a, b = float(poly[0]), float(poly[1])

    elif len(anchor_x) >= 6:
        ax = np.array(anchor_x, dtype=np.float32)
        ay = np.array(anchor_y, dtype=np.float32)

        rng = np.random.default_rng(42)
        n_samples = len(ax)
        best_inliers = None
        best_count = 0

        for _ in range(50):
            idx = rng.choice(n_samples, size=2, replace=False)
            x1, x2 = ax[idx[0]], ax[idx[1]]
            y1, y2 = ay[idx[0]], ay[idx[1]]
            if abs(x2 - x1) < 0.05:
                continue
            cand_a = (y2 - y1) / (x2 - x1)
            cand_b = y1 - cand_a * x1
            if not (0.05 <= cand_a <= 15.0 and abs(cand_b) <= 8.0):
                continue
            residuals = np.abs(cand_a * ax + cand_b - ay)
            inliers = residuals < 0.08
            count = int(np.count_nonzero(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is not None and best_count >= 4:
            poly = np.polyfit(ax[best_inliers], ay[best_inliers], deg=1)
            a, b = float(poly[0]), float(poly[1])
        else:
            poly = np.polyfit(ax, ay, deg=1)
            if 0.1 <= poly[0] <= 10.0:
                a, b = float(poly[0]), float(poly[1])

    calibrated = np.clip(a * raw + b, 0.05, 12.0).astype(np.float32)
    return calibrated


def synthetic_scan(depths: dict[str, np.ndarray], rig: dict[str, Camera] = RIG, num_beams: int = 360,
                   z_min: float = 0.03, z_max: float = 0.38, pitch_rad: float = 0.0,
                   roll_rad: float = 0.0) -> tuple[np.ndarray, float, float]:
    """Generate synthetic 360-degree LaserScan ranges from 3-CSI depth maps."""
    step = 2 * math.pi / num_beams
    angle_min = -math.pi
    ranges = np.full(num_beams, np.inf, dtype=np.float32)
    r_tilt = _tilt_matrix(pitch_rad, roll_rad)

    stride = 5  # strided sampling for <5ms execution

    for name, depth_img in depths.items():
        if name not in rig or depth_img is None:
            continue
        cam = rig[name]
        d = depth_img[::stride, ::stride]
        valid = np.isfinite(d) & (d >= 0.1) & (d <= 8.0)
        if not valid.any():
            continue

        h_sub, w_sub = d.shape
        v_idx, u_idx = np.mgrid[0:h_sub, 0:w_sub]
        u_orig = u_idx[valid].astype(np.float32) * stride
        v_orig = v_idx[valid].astype(np.float32) * stride
        d_val = d[valid].astype(np.float32)

        x_opt = (u_orig - cam.width / 2) / cam.focal
        y_opt = (v_orig - cam.height / 2) / cam.focal

        # Points in camera optical frame
        xc = d_val
        yc = -x_opt * d_val
        zc = -y_opt * d_val

        # Transform to base_link
        cos_yaw, sin_yaw = math.cos(cam.yaw), math.sin(cam.yaw)
        xb = cam.x + xc * cos_yaw - yc * sin_yaw
        yb = cam.y + xc * sin_yaw + yc * cos_yaw
        zb = cam.z + zc

        # Account for body pitch/roll relative to ground
        zw = r_tilt[2, 0] * xb + r_tilt[2, 1] * yb + r_tilt[2, 2] * zb

        # Filter obstacle height
        in_height = (zw >= z_min) & (zw <= z_max)
        if not in_height.any():
            continue

        xb_filt, yb_filt = xb[in_height], yb[in_height]
        r = np.hypot(xb_filt, yb_filt)

        # Filter radial distance
        in_range = (r >= 0.12) & (r <= 8.0)
        if not in_range.any():
            continue

        xb_ok, yb_ok, r_ok = xb_filt[in_range], yb_filt[in_range], r[in_range]
        theta = np.arctan2(yb_ok, xb_ok)

        # Map angles [-pi, pi) to beam indices
        k = np.floor((theta - angle_min) / step).astype(np.int32)
        k = np.clip(k, 0, num_beams - 1)

        np.minimum.at(ranges, k, r_ok)

    return ranges, angle_min, step
