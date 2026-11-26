"""Metric depth calibration, ToF step/cliff hazard classification, and 360-degree synthetic scan."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from robot.perception.rig import RIG, Camera
from robot.ros.model import CLEARANCE


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


@lru_cache(maxsize=16)
def _base_rays(cam: Camera, h: int, w: int, stride: int = 1):  # noqa: ANN202
    """Tilt-independent 1D ray components for `(cam, h, w, stride)` (~4 KB per entry)."""
    cam = cam.at(w, h)
    u = np.arange(0, w, stride, dtype=np.float32)[None, :] + 0.5
    v = np.arange(h, dtype=np.float32)[:, None] + 0.5
    x_opt = (u - cam.width / 2) / cam.focal
    y_opt = (v - cam.height / 2) / cam.focal
    c, s_ = math.cos(cam.yaw), math.sin(cam.yaw)
    fx, fy = c + x_opt * s_, s_ - x_opt * c  # the ray (forward 1, right x_opt) on the floor plane: 1 x columns
    fx.flags.writeable = fy.flags.writeable = y_opt.flags.writeable = False
    return cam, fx, fy, y_opt


@lru_cache(maxsize=8)
def _rays(cam: Camera, h: int, w: int, pitch_rad: float, roll_rad: float, stride: int = 1):  # noqa: ANN202
    """For a picture h x w from `cam` (columns every `stride`): each column's direction on the floor plane per metre
    of depth along the lens axis (fx, fy: 1 x columns), each pixel's rise per metre of depth (vz: rows x columns),
    and the lens's height above the floor, all with the body tipped by pitch and roll."""
    cam, fx, fy, y_opt = _base_rays(cam, h, w, stride)
    r_tilt = _tilt_matrix(pitch_rad, roll_rad)
    vz = (r_tilt[2, 0] * fx + r_tilt[2, 1] * fy) - r_tilt[2, 2] * y_opt
    cam_z = float((r_tilt @ np.array([cam.x, cam.y, cam.z], dtype=np.float32))[2])
    vz.flags.writeable = False
    return cam, fx, fy, vz, cam_z


def _nanmedian_axis1(win: np.ndarray) -> np.ndarray:
    """Vectorized row-wise nanmedian for small windows without per-row Python dispatch."""
    s = np.sort(win, axis=1)
    cnt = np.sum(~np.isnan(s), axis=1)
    idx = np.arange(len(cnt))
    lo = np.maximum(0, (cnt - 1) // 2)
    hi = np.maximum(0, cnt // 2)
    return np.where(cnt > 0, 0.5 * (s[idx, lo] + s[idx, hi]), np.nan)


def fit_to_floor(inverse: np.ndarray, cam: Camera, pitch_rad: float = 0.0, roll_rad: float = 0.0,
                 reach: float = 2.0) -> np.ndarray | None:
    """Metres from a relative inverse-depth picture (a monocular network's output: right up to a scale and a shift).

    The floor's inverse depth at each pixel is known from the camera's height and the body's tilt; where the lower
    picture shows floor (out to `reach` metres), the network's values must map onto it. A robust line fit (RANSAC:
    most of the lower picture is floor, and furniture there cannot agree with the floor's pattern) gives the scale
    and the shift. None when too little floor is in view to fit."""
    h, w = inverse.shape
    _, _, _, vz, cam_z = _rays(cam, h, w, pitch_rad, roll_rad)
    floor = -vz / cam_z  # 1 / depth of the floor at each pixel
    region = floor > 1.0 / reach
    x = np.asarray(inverse[region][::5], dtype=np.float32)
    y = np.asarray(floor[region][::5], dtype=np.float32)
    n = len(x)
    if n < 50:
        return None
    rng = np.random.default_rng(0)
    i = rng.integers(0, n, size=200)
    j = rng.integers(0, n - 1, size=200)
    j = np.where(j >= i, j + 1, j)
    dx = x[i] - x[j]
    valid = np.abs(dx) >= 1e-9
    a_cand = np.where(valid, (y[i] - y[j]) / np.where(valid, dx, 1.0), -1.0)
    keep = valid & (a_cand > 0)
    if not keep.any():
        return None
    a_c = a_cand[keep].astype(np.float32)
    b_c = (y[i[keep]] - a_c * x[i[keep]]).astype(np.float32)
    # Score hypotheses on ~1,500 strided samples (fits in L2 cache); if the winner fails the full inlier
    # threshold, fall back to full-set scoring so a borderline valid frame is never dropped.
    s_step = max(1, n // 1500)
    xs, ys = x[::s_step], y[::s_step]
    tol_s = np.float32(0.08) * ys
    best_k, best_s = -1, 0
    for s in range(0, len(a_c), 32):
        counts = np.count_nonzero(np.abs(a_c[s:s + 32, None] * xs + b_c[s:s + 32, None] - ys) < tol_s, axis=1)
        k = int(np.argmax(counts))
        if int(counts[k]) > best_s:
            best_k, best_s = s + k, int(counts[k])
    if best_k < 0:
        return None
    tol = np.float32(0.08) * y
    best = np.abs(a_c[best_k] * x + b_c[best_k] - y) < tol
    if int(np.count_nonzero(best)) < 0.3 * n:
        best_n = int(np.count_nonzero(best))
        for s in range(0, len(a_c), 16):
            counts = np.count_nonzero(np.abs(a_c[s:s + 16, None] * x + b_c[s:s + 16, None] - y) < tol, axis=1)
            k = int(np.argmax(counts))
            if int(counts[k]) > best_n:
                best_k, best_n = s + k, int(counts[k])
        if best_n < 0.3 * n:
            return None
        best = np.abs(a_c[best_k] * x + b_c[best_k] - y) < tol
    xb, yb = x[best].astype(np.float64), y[best].astype(np.float64)
    xm, ym = float(xb.mean()), float(yb.mean())
    dx_b = xb - xm
    denom = float(dx_b @ dx_b)
    if denom < 1e-12:
        return None
    a = np.float32(float(dx_b @ (yb - ym)) / denom)
    b = np.float32(ym - float(a) * xm)
    inv = a * np.asarray(inverse, dtype=np.float32) + b
    out = np.full(inv.shape, np.inf, dtype=np.float32)
    np.divide(np.float32(1.0), inv, out=out, where=inv > 1e-3)
    return out


FLOOR_TOL = 0.06  # metres: what rises less than this off the floor is floor to the cameras (lower things: the ToF)
FACE = 0.5  # a column whose inverse depth grows at under this share of the floor's rate is going up a face
SLOPE_ROWS = 2  # rows that rate is measured over: each row against the ones just above it
RUN = 3  # rows: a face at least this tall (and FLOOR_TOL high) is something standing there
OVERHANG = 0.10  # metres: a foot whose depth puts it higher than this is an overhang's edge, ranged by its depth
NEAR = 2.0  # metres: within this, depth is trusted to find an overhang in front of what stands behind it
AHEAD = 0.95  # an overhang counts when its depth puts it this much nearer than the foot behind (depth noise aside)


def synthetic_scan(depths: dict[str, np.ndarray], rig: dict[str, Camera] = RIG, num_beams: int = 360,
                   pitch_rad: float = 0.0, roll_rad: float = 0.0, clearance: float = CLEARANCE,
                   max_range: float = 8.0) -> tuple[np.ndarray, float, float]:
    """A 360-degree laser scan from the cameras' metric depth, found where the floor ends.

    Down a column of a picture, the floor's inverse depth grows row by row at a rate the camera's height and tilt
    fix; up the face of anything standing on the floor it stays nearly constant. A monocular network's depth is
    often wrong by a scale over a whole surface (a wall put twice as far as it is), but that surface still keeps a
    constant inverse depth up the column. So the foot of something is found from that rate, not from the depth's
    value: from the bottom of each column up, the lowest row whose inverse depth, against the SLOPE_ROWS rows above
    it, grows at under FACE of the floor's rate there, for RUN rows and FLOOR_TOL of height (lower things, a rug's
    edge, are floor). It is ranged where the ray through that row meets the floor (exact if the row is right), or,
    when depth puts its lowest edge clearly above the floor (a sofa's front over a gap, a shelf), by depth. Things
    whose foot is above `clearance` are passed under and hide what lies beyond them. Beams: the nearest thing; inf
    where the floor was seen clear as far as `max_range`; nan where nothing was seen either way.
    """
    step = 2 * math.pi / num_beams
    angle_min = -math.pi
    ranges = np.full(num_beams, np.inf, dtype=np.float32)
    hit = np.zeros(num_beams, dtype=bool)
    clear = np.zeros(num_beams, dtype=bool)
    stride = 2  # every other column of a 640-wide estimate
    offsets = np.arange(RUN, dtype=np.int32)[None, :]
    for name, depth_img in depths.items():
        if name not in rig or depth_img is None:
            continue
        h, w = depth_img.shape
        cam, fx, fy, vz, cam_z = _rays(rig[name], h, w, pitch_rad, roll_rad, stride)
        d = np.asarray(depth_img[:, ::stride], dtype=np.float32)
        ok = np.isfinite(d) & (d > 0.05)
        z = np.where(ok, cam_z + d * vz, np.nan)  # each point's height above the floor, by depth
        inv = np.where(ok, 1.0 / np.where(ok, d, 1.0), np.nan)
        floor_inv = -vz / cam_z  # the floor's inverse depth in each pixel (not positive above the horizon)
        n = SLOPE_ROWS
        rate = np.full_like(inv, np.nan)  # 1 on floor, about 0 up a face
        with np.errstate(invalid="ignore", divide="ignore"):
            rate[n:] = (inv[n:] - inv[:-n]) / (floor_inv[n:] - floor_inv[:-n])
        in_reach = floor_inv > 1.0 / max_range  # rows that would see the floor within the scan's range
        face = (rate < FACE) & in_reach
        # the foot: the lowest row with a face going up from it, RUN rows and FLOOR_TOL high at the floor's range there
        cs = np.cumsum(face, axis=0, dtype=np.int32)
        face_up = cs - np.maximum.accumulate(np.where(face, 0, cs), axis=0)
        per_row = np.full_like(vz, np.inf)  # metres a face at the floor's range in that row rises per row
        per_row[1:] = (cam_z / -np.minimum(vz[1:], -1e-6)) * np.maximum(vz[:-1] - vz[1:], 1e-9)
        foot_at = (face_up >= np.maximum(RUN, np.ceil(FLOOR_TOL / per_row))) & in_reach
        has = foot_at.any(axis=0)
        foot = h - 1 - np.argmax(foot_at[::-1], axis=0)  # row of the foot's lowest pixel in each column
        cols = np.arange(d.shape[1])
        seen_floor = ((rate >= FACE) & in_reach).any(axis=0)
        clear_cols = cols[~has & seen_floor]
        if clear_cols.size:  # floor all the way up: clear as far as the camera sees
            b = np.arctan2(cam.y + fy[0, clear_cols], cam.x + fx[0, clear_cols])
            clear[((b - angle_min) / step).astype(np.int32) % num_beams] = True
        has_cols = cols[has]
        if not has_cols.size:
            continue
        r0 = foot[has_cols]
        top = r0 - face_up[r0, has_cols] + 1
        # the face's own lowest rows: the rate in the SLOPE_ROWS rows under a face already reaches into it
        r_start = np.maximum(top, r0 - SLOPE_ROWS - RUN + 1)
        r_end = np.maximum(top, r0 - SLOPE_ROWS)
        row_idx = np.minimum(r_start[:, None] + offsets, r_end[:, None])
        valid_slot = (r_start[:, None] + offsets) <= r_end[:, None]
        zf = _nanmedian_axis1(np.where(valid_slot, z[row_idx, has_cols[:, None]], np.nan))
        keep_h = ~(zf > clearance)  # an overhang the robot passes under: nothing known beyond it
        if not keep_h.any():
            continue
        kc = has_cols[keep_h]
        r0_k = r0[keep_h]
        rs_k = r_start[keep_h]
        zf_k = zf[keep_h]
        seen = _nanmedian_axis1(np.where(valid_slot[keep_h], d[row_idx[keep_h], kc[:, None]], np.nan))
        vz_r0 = vz[r0_k, kc]
        on_floor = (vz_r0 < -1e-3) & (zf_k < OVERHANG)
        along = np.where(on_floor, cam_z / np.where(on_floor, -vz_r0, 1.0), seen)
        band = ok & (z > FLOOR_TOL) & (z <= clearance)
        band_run = band.copy()
        for k in range(1, RUN):
            band_run[:-k] &= band[k:]
        # something in the robot's height nearer than the foot, higher up (table edge, staircase underside)
        d_sub = d[:, kc]
        ray_xy = np.hypot(fx[0, kc], fy[0, kc])
        row_grid = np.arange(h, dtype=np.int32)[:, None]
        cand_over = (
            band_run[:, kc]
            & (row_grid < rs_k[None, :])
            & (d_sub * ray_xy[None, :] < NEAR)
            & (d_sub < AHEAD * seen[None, :])
        )
        if cand_over.any():
            along = np.minimum(along, np.where(cand_over, d_sub, np.inf).min(axis=0))
        px = cam.x + along * fx[0, kc]
        py = cam.y + along * fy[0, kc]
        r = np.hypot(px, py)
        in_r = r <= max_range
        if in_r.any():
            ks = ((np.arctan2(py[in_r], px[in_r]) - angle_min) / step).astype(np.int32) % num_beams
            np.minimum.at(ranges, ks, r[in_r])
            hit[ks] = True
    ranges[~hit & ~clear] = np.nan
    return ranges, angle_min, step


STEP_UP = 0.035  # metres: what rises more than this off the floor stops the wheels (they climb 3 cm)


_TOF_OFFSETS = ((np.arange(8, dtype=np.float32) + 0.5) - 4.0) / 4.0


def tof_scan(grid: np.ndarray, side: str, num_beams: int = 360, pitch_rad: float = 0.0, roll_rad: float = 0.0,
             clearance: float = CLEARANCE) -> np.ndarray:
    """Returns for a 360-degree scan from one 8x8 ToF grid: each zone whose point rises more than STEP_UP off the
    floor (and below the robot's top) is an obstacle at its horizontal range; nan elsewhere. The cameras count the
    lowest 6 cm as floor (their depth is not finer than that); the ToF sees what lies there: a shoe, a cable, a step.
    Zone distances are along the sensor's axis, as the sensor reports them."""
    from robot.ros.model import TOF

    x0, yaw, z0, pitch, hfov, max_range = TOF[side]
    out = np.full(num_beams, np.inf, dtype=np.float32)
    g = np.asarray(grid, dtype=np.float32).reshape(8, 8)
    valid = np.isfinite(g) & (g > 0.02) & (g < max_range)
    if not valid.any():
        return np.full(num_beams, np.nan, dtype=np.float32)
    r_tilt = _tilt_matrix(pitch_rad, roll_rad)
    t = math.tan(hfov / 2)
    cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    right = _TOF_OFFSETS[None, :] * t
    down = _TOF_OFFSETS[:, None] * t
    fwd, up = cp - down * sp, -sp - down * cp  # the zone's ray on the sensor tipped down by its pitch
    d = np.where(valid, g, np.float32(0.0))
    bx = x0 + d * (fwd * cy + right * sy)
    by = d * (fwd * sy - right * cy)
    bz = z0 + d * up
    z = r_tilt[2, 0] * bx + r_tilt[2, 1] * by + r_tilt[2, 2] * bz
    obs = valid & (z > STEP_UP) & (z <= clearance)
    if obs.any():
        step = 2 * math.pi / num_beams
        bx_o, by_o = bx[obs], by[obs]
        ks = ((np.arctan2(by_o, bx_o) + math.pi) / step).astype(np.int32) % num_beams
        np.minimum.at(out, ks, np.hypot(bx_o, by_o))
    out[np.isinf(out)] = np.nan
    return out


class MonoDepth:
    """Metric depth from one colour picture. Depth Anything V2 Base on the vision server (tools/vision_server.py, on
    the CPU too with tools/linux/vision.sh) gives relative inverse depth, and `fit_to_floor` makes it metres. Its input
    is `height` pixels tall; the result is at the size of a `width`-wide copy of the frame."""

    def __init__(self, server: Any, height: int = 364, width: int = 640) -> None:
        self.server, self.height, self.width = server, height, width

    def __call__(self, rgb: np.ndarray, cam: Camera, pitch_rad: float = 0.0,
                 roll_rad: float = 0.0) -> np.ndarray | None:
        import cv2

        h, w = rgb.shape[:2]
        out_w, out_h = self.width, round(self.width * h / w)
        w_in = int(round(w * self.height / h / 14)) * 14  # the network works in 14-pixel patches
        small = cv2.resize(rgb, (w_in, self.height), interpolation=cv2.INTER_AREA)
        raw = self.server.depth(small)
        inverse = cv2.resize(raw, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return fit_to_floor(inverse, cam, pitch_rad, roll_rad)
