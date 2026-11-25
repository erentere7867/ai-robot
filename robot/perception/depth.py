"""Metric depth calibration, ToF step/cliff hazard classification, and 360-degree synthetic scan."""

from __future__ import annotations

import math
from dataclasses import dataclass
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


def _rays(cam: Camera, h: int, w: int, pitch_rad: float, roll_rad: float, stride: int = 1):  # noqa: ANN202
    """For a picture h x w from `cam` (columns every `stride`): each column's direction on the floor plane per metre
    of depth along the lens axis (fx, fy: 1 x columns), each pixel's rise per metre of depth (vz: rows x columns),
    and the lens's height above the floor, all with the body tipped by pitch and roll."""
    cam = cam.at(w, h)
    r_tilt = _tilt_matrix(pitch_rad, roll_rad)
    u = np.arange(0, w, stride, dtype=np.float32)[None, :] + 0.5
    v = np.arange(h, dtype=np.float32)[:, None] + 0.5
    x_opt, y_opt = (u - cam.width / 2) / cam.focal, (v - cam.height / 2) / cam.focal
    c, s_ = math.cos(cam.yaw), math.sin(cam.yaw)
    fx, fy = c + x_opt * s_, s_ - x_opt * c  # the ray (forward 1, right x_opt) on the floor plane: 1 x columns
    fz = -y_opt * np.ones_like(fx)
    vz = r_tilt[2, 0] * fx + r_tilt[2, 1] * fy + r_tilt[2, 2] * fz
    cam_z = float((r_tilt @ np.array([cam.x, cam.y, cam.z], dtype=np.float32))[2])
    return cam, fx, fy, vz, cam_z


def fit_to_floor(inverse: np.ndarray, cam: Camera, pitch_rad: float = 0.0, roll_rad: float = 0.0,
                 reach: float = 2.0) -> np.ndarray | None:
    """Metres from a relative inverse-depth picture (a monocular network's output: right up to a scale and a shift).

    The floor's inverse depth at each pixel is known from the camera's height and the body's tilt; where the lower
    picture shows floor (out to `reach` metres), the network's values must map onto it. A robust line fit (RANSAC:
    most of the lower picture is floor, and furniture there cannot agree with the floor's pattern) gives the scale
    and the shift. None when too little floor is in view to fit."""
    h, w = inverse.shape
    _, _, _, vz, cam_z = _rays(cam, h, w, pitch_rad, roll_rad)
    floor = np.clip(-vz / cam_z, 0.0, None)  # 1 / depth of the floor at each pixel
    region = floor > 1.0 / reach
    x, y = inverse[region][::5].astype(np.float64), floor[region][::5].astype(np.float64)
    if len(x) < 50:
        return None
    rng = np.random.default_rng(0)
    best, best_n = None, 0
    for _ in range(200):
        i, j = rng.choice(len(x), 2, replace=False)
        if abs(x[i] - x[j]) < 1e-9:
            continue
        a = (y[i] - y[j]) / (x[i] - x[j])
        if a <= 0:  # nearer must stay nearer
            continue
        inliers = np.abs(a * x + (y[i] - a * x[i]) - y) < 0.08 * y
        if inliers.sum() > best_n:
            best, best_n = inliers, int(inliers.sum())
    if best is None or best_n < 0.3 * len(x):
        return None
    a, b = np.polyfit(x[best], y[best], 1)
    inv = a * inverse + b
    return np.where(inv > 1e-3, 1.0 / np.maximum(inv, 1e-3), np.inf).astype(np.float32)


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
    ranges = np.full(num_beams, np.nan, dtype=np.float32)
    clear = np.zeros(num_beams, dtype=bool)
    stride = 2  # every other column of a 640-wide estimate
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
        face_up = np.zeros(face.shape, np.int32)  # face rows from each row upward, unbroken
        face_up[0] = face[0]
        for r in range(1, h):
            face_up[r] = (face_up[r - 1] + 1) * face[r]
        per_row = np.full_like(vz, np.inf)  # metres a face at the floor's range in that row rises per row
        per_row[1:] = (cam_z / -np.minimum(vz[1:], -1e-6)) * np.maximum(vz[:-1] - vz[1:], 1e-9)
        foot_at = (face_up >= np.maximum(RUN, np.ceil(FLOOR_TOL / per_row))) & in_reach
        has = foot_at.any(axis=0)
        foot = h - 1 - np.argmax(foot_at[::-1], axis=0)  # row of the foot's lowest pixel in each column
        cols = np.arange(d.shape[1])
        seen_floor = ((rate >= FACE) & in_reach).any(axis=0)
        for j in cols[~has & seen_floor]:  # floor all the way up: clear as far as the camera sees
            b = math.atan2(cam.y + fy[0, j], cam.x + fx[0, j])
            clear[int((b - angle_min) / step) % num_beams] = True
        band = ok & (z > FLOOR_TOL) & (z <= clearance)
        band_run = band.copy()
        for k in range(1, RUN):
            band_run[:-k] &= band[k:]
        for j in cols[has]:
            r0 = foot[j]
            # the face's own lowest rows: the rate in the SLOPE_ROWS rows under a face already reaches into it
            top = r0 - face_up[r0, j] + 1
            rows = slice(max(top, r0 - SLOPE_ROWS - RUN + 1), max(top, r0 - SLOPE_ROWS) + 1)
            zf = float(np.nanmedian(z[rows, j]))
            if zf > clearance:
                continue  # an overhang the robot passes under: nothing known beyond it
            seen = float(np.nanmedian(d[rows, j]))  # the face's depth, as the network judges it
            if vz[r0, j] < -1e-3 and zf < OVERHANG:  # standing on the floor: its lowest row meets it there
                along = cam_z / -float(vz[r0, j])
            else:  # its lowest edge is clearly off the floor (or above the horizon): only depth can range it
                along = seen
            above = band_run[: rows.start, j]  # something in the robot's height nearer than the foot, higher up
            if above.any():  # (a table's edge, a staircase's underside, in front of the wall seen under it)
                nearer = d[: rows.start, j][above]  # depth against depth: a scale error the network makes on
                nearer = nearer[(nearer * math.hypot(fx[0, j], fy[0, j]) < NEAR) & (nearer < AHEAD * seen)]
                if nearer.size:  # the whole face is no overhang in front of itself
                    along = min(along, float(nearer.min()))
            px, py = cam.x + along * float(fx[0, j]), cam.y + along * float(fy[0, j])  # same in every row
            r = math.hypot(px, py)
            if r > max_range:
                continue
            k = int((math.atan2(py, px) - angle_min) / step) % num_beams
            if not r >= ranges[k]:  # nan or farther
                ranges[k] = r
    ranges[np.isnan(ranges) & clear] = np.inf
    return ranges, angle_min, step


STEP_UP = 0.035  # metres: what rises more than this off the floor stops the wheels (they climb 3 cm)


def tof_scan(grid: np.ndarray, side: str, num_beams: int = 360, pitch_rad: float = 0.0, roll_rad: float = 0.0,
             clearance: float = CLEARANCE) -> np.ndarray:
    """Returns for a 360-degree scan from one 8x8 ToF grid: each zone whose point rises more than STEP_UP off the
    floor (and below the robot's top) is an obstacle at its horizontal range; nan elsewhere. The cameras count the
    lowest 6 cm as floor (their depth is not finer than that); the ToF sees what lies there: a shoe, a cable, a step.
    Zone distances are along the sensor's axis, as the sensor reports them."""
    from robot.ros.model import TOF

    x0, yaw, z0, pitch, hfov, max_range = TOF[side]
    out = np.full(num_beams, np.nan, dtype=np.float32)
    g = np.asarray(grid, dtype=np.float32).reshape(8, 8)
    r_tilt = _tilt_matrix(pitch_rad, roll_rad)
    t = math.tan(hfov / 2)
    cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    step = 2 * math.pi / num_beams
    for row in range(8):
        for col in range(8):
            d = float(g[row, col])
            if not (math.isfinite(d) and 0.02 < d < max_range):
                continue
            right, down = ((col + 0.5) - 4) / 4 * t, ((row + 0.5) - 4) / 4 * t
            fwd, up = cp - down * sp, -sp - down * cp  # the zone's ray on the sensor tipped down by its pitch
            bx = x0 + d * (fwd * cy + right * sy)
            by = d * (fwd * sy - right * cy)
            bz = z0 + d * up
            z = float(r_tilt[2, 0] * bx + r_tilt[2, 1] * by + r_tilt[2, 2] * bz)
            if STEP_UP < z <= clearance:
                k = int((math.atan2(by, bx) + math.pi) / step) % num_beams
                r = math.hypot(bx, by)
                if not r >= out[k]:
                    out[k] = r
    return out


class MonoDepth:
    """Metric depth from one colour picture. Depth Anything V2 Base (ONNX) gives
    relative inverse depth, and `fit_to_floor` makes it metres. Its input is `height` pixels tall; the result is at
    the size of a `width`-wide copy of the frame."""

    def __init__(self, weights: str = "depth-anything-v2-base-onnx/model.onnx", height: int = 364, width: int = 640,
                 threads: int = 0, server: Any = None) -> None:
        """Use the vision server when supplied, otherwise run the network on the CPU with `threads`."""
        self.server, self.height, self.width = server, height, width
        if server is not None:
            return
        import onnxruntime as ort

        from robot.perception.detector import _resolve_weights

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.net = ort.InferenceSession(_resolve_weights(weights), options, providers=["CPUExecutionProvider"])
        self.input = self.net.get_inputs()[0].name

    def __call__(self, rgb: np.ndarray, cam: Camera, pitch_rad: float = 0.0,
                 roll_rad: float = 0.0) -> np.ndarray | None:
        import cv2

        h, w = rgb.shape[:2]
        out_w, out_h = self.width, round(self.width * h / w)
        w_in = int(round(w * self.height / h / 14)) * 14  # the network works in 14-pixel patches
        small = cv2.resize(rgb, (w_in, self.height), interpolation=cv2.INTER_AREA)
        if self.server is not None:
            raw = self.server.depth(small)
        else:
            x = ((small.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
            raw = self.net.run(None, {self.input: x})[0][0]
        inverse = cv2.resize(raw, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return fit_to_floor(inverse, cam, pitch_rad, roll_rad)


_MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # the network's input normalisation (ImageNet)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
