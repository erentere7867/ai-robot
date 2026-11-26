"""Perception on ROS: Sprocket's three colour cameras + ToF, depth from a monocular network, synthetic /scan, and the
YOLOE-26 tracker.

The cameras give colour only. The depth node (`python -m robot.ros.perceive --depth`, on the board's GPU) estimates a
depth picture for each camera's newest frame, publishes it, and turns the three, metric-calibrated against the floor
and the ToF grids with the IMU's tilt, into a synthetic 360-degree LaserScan on /scan for Nav2 and SLAM: there is no
lidar. In the robot's own process a worker thread takes the newest colour frame from each camera in turn, runs the
detector, places each detection on the map (depth + floor + /scan; robot pose from TF at the frame's time), and folds
them into the tracker.
"""

from __future__ import annotations

import argparse
import array
import math
import threading
import time
from collections import Counter, deque
from dataclasses import replace
from typing import Any

import numpy as np

from robot.mind.brain import _with_image
from robot.perception import colour
from robot.perception.camera import data_url, resize_rgb
from robot.perception.depth import StepHazard, classify_tof_grid, synthetic_scan, tof_scan
from robot.perception.locate import lidar_range, locate
from robot.perception.overlay import draw_set_of_mark
from robot.perception.pose import estimate_pose_from_keypoints
from robot.perception.rig import RIG, to_map, to_robot
from robot.perception.tracker import GONE_AFTER, Detection, Track, Tracker, canonical, look, pixels
from robot.ros.model import BODY_R, TOF
from robot.ros.motion import WHEELS

TOPICS = {name: f"/{name}/image" for name in RIG}  # the camera drivers' colour pictures
DEPTH_TOPICS = {name: f"/{name}/depth" for name in RIG}  # the depth node's estimates, at a smaller size
TOF_TOPICS = {"front": "/tof/front", "rear": "/tof/rear"}  # 8x8 grids from the MCU, metres
SEE_FAR = 3.5  # beyond this a miss is not evidence: the detector may simply not resolve it
PAST_EVERY, PAST_KEPT = 0.5, 60  # seconds between kept front pictures, and how many: the last 30 s, to look back on
PLACE_SLACK = 0.3  # metres round a mapped place the vision model is shown when the detector stops finding a thing
_, _, _TOF_HEIGHT, _TOF_PITCH, _TOF_HFOV, _ = TOF["front"]


def _image(msg: Any) -> np.ndarray:
    channels = 3 if msg.encoding in ("rgb8", "bgr8") else 1
    dtype = np.float32 if msg.encoding == "32FC1" else np.uint8
    array = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels) if channels == 3 else \
        np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
    return array[:, :, ::-1] if msg.encoding == "bgr8" else array


def _nanoseconds(stamp: Any) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _seconds(stamp: Any) -> float:
    return _nanoseconds(stamp) * 1e-9


def _only(rgb: np.ndarray, box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> np.ndarray:
    """The picture of one thing for the vision model to name: its box with a little margin, with any smaller thing
    detected in that part of the picture painted grey. A chair's crop showed the red ball in front of it, and the
    model named the chair a ball. A bigger box around this one (the table under a cup) is left alone."""
    x1, y1, x2, y2 = box
    pad = max(8, (x2 - x1) // 6)
    left, top = max(0, x1 - pad), max(0, y1 - pad)
    crop = rgb[top:y2 + pad, left:x2 + pad].copy()
    area = (x2 - x1) * (y2 - y1)
    for ox1, oy1, ox2, oy2 in others:
        if (ox2 - ox1) * (oy2 - oy1) >= area:
            continue
        a, b = max(ox1 - left, 0), max(oy1 - top, 0)
        c, d = min(ox2 - left, crop.shape[1]), min(oy2 - top, crop.shape[0])
        if a < c and b < d:
            crop[b:d, a:c] = 128
    return crop


def is_there(brain: Any, picture: np.ndarray, label: str) -> bool:
    """The vision model's yes to a thing in the middle of a picture."""
    question = f"Is there a {label} in the middle of this picture? Answer yes or no."
    return brain.yes_probability(_with_image(data_url(picture), question)) >= 0.5


def see_probability(brain: Any, picture: np.ndarray, label: str, max_dim: int = 512) -> float:
    """The vision model's yes probability to seeing `label` in `picture`, capped so tall strips stay small."""
    h, w = picture.shape[:2]
    scale = min(1.0, max_dim / max(1, h, w))
    img = resize_rgb(picture, max(1, round(w * scale)), max(1, round(h * scale))) if scale < 1.0 else picture
    question = f"Do you see {label} in this picture? Answer yes or no."
    return brain.yes_probability(_with_image(data_url(img, max_width=max_dim), question))


def locate_in_picture(brain: Any, rgb: np.ndarray, label: str) -> tuple[tuple[int, int, int, int], bool] | None:
    """Find where `label` appears in a full camera frame via yes/no crops (no numbered dots: the 2B model picks the
    last number). Overlapping thirds -> probability-weighted centre of two adjacent positives (cut close-up rug
    centre error from 0.78 m to 0.36 m), or refine the strongest third via overlapping halves
    (0.65 span: non-overlapping halves sliced a centred umbrella at u=0.50 from 96% to 4%/4%, while 0.65 overlap
    gave 100%/99%) -> upper/lower split on a 0.30-span context strip. Measured on 30 random apt_0 views (210 pairs)
    behind a 768px whole-image gate: 0/145 (0.0%) FP; blank-wall hallucinations (71% whole) dropped to 1%/0%."""
    h, w = rgb.shape[:2]
    thirds = ((0.0, 0.40), (0.30, 0.70), (0.60, 1.0))
    t_scores = [see_probability(brain, rgb[:, int(a * w):max(int(a * w) + 1, int(b * w))], label, max_dim=384)
                for a, b in thirds]
    pos = [i for i, s in enumerate(t_scores) if s >= 0.5]
    if not pos:
        return None
    best_i = max(pos, key=t_scores.__getitem__)
    # Merging three positives put no target pixels in the replay crop, 47° from the refined crop.
    # Keep the merged crop as a fallback when only the wider context is recognized.
    adjacent = len(pos) == 2 and pos[1] == pos[0] + 1
    for merge in (False, True) if len(pos) == 3 else (adjacent,):
        if merge:
            span = 0.40
            u_center = sum(t_scores[i] * 0.5 * (thirds[i][0] + thirds[i][1]) for i in pos) / sum(
                t_scores[i] for i in pos
            )
        else:
            a, b = thirds[best_i]
            span = b - a
            halves = ((a, a + 0.65 * span), (b - 0.65 * span, b))
            h_scores = [see_probability(brain, rgb[:, int(x0 * w):max(int(x0 * w) + 1, int(x1 * w))],
                                        label, max_dim=384) for x0, x1 in halves]
            if max(h_scores) < 0.5:
                continue
            total = h_scores[0] + h_scores[1]
            u_center = (h_scores[0] * (a + 0.25 * span) + h_scores[1] * (b - 0.25 * span)) / total
        sx0 = max(0, int(round((u_center - 0.30 * span) * w)))
        sx1 = min(w, max(sx0 + 1, int(round((u_center + 0.30 * span) * w))))
        strip = rgb[:, sx0:sx1]
        top_end, bot_start = max(1, int(0.55 * h)), min(h - 1, int(0.45 * h))
        v_top = see_probability(brain, strip[:top_end, :], label, max_dim=384)
        v_bot = see_probability(brain, strip[bot_start:, :], label, max_dim=384)
        if max(v_top, v_bot) < 0.5:
            continue
        # An umbrella can show in both halves (89%/100%) and must stay 3D.
        on_floor = v_bot >= 0.5 and v_top < 0.35
        y0, y1 = (bot_start, h) if on_floor else (0, max(1, int(0.60 * h))) if v_top > v_bot + 0.10 else (0, h)
        x0 = max(0, int(round((u_center - 0.25 * span) * w)))
        x1 = min(w, max(x0 + 1, int(round((u_center + 0.25 * span) * w))))
        return (x0, y0, x1, y1), on_floor
    return None


def place_picture(rgb: np.ndarray, cam: Any, bx: float, by: float, radius: float) -> np.ndarray | None:
    """The part of a picture where a thing mapped at robot-frame (bx, by) should be, PLACE_SLACK metres wider each
    way for its mapped place being a little off; None where it falls outside the picture."""
    cam = cam.at(rgb.shape[1], rgb.shape[0])
    middle = cam.project(bx, by, radius)  # half way up a thing about as tall as it is wide
    if middle is None:
        return None
    half = cam.focal * (radius + PLACE_SLACK) / max(0.1, math.hypot(bx - cam.x, by - cam.y))
    (u, v), (h, w) = middle, rgb.shape[:2]
    x1, x2, y1, y2 = int(max(0, u - half)), int(min(w, u + half)), int(max(0, v - half)), int(min(h, v + half))
    return rgb[y1:y2, x1:x2].copy() if x2 - x1 >= 16 and y2 - y1 >= 16 else None


def taken_with(buffer: deque, t: float, within: float) -> tuple[Any, float]:
    """Nearest item and signed stamp difference; no item outside the tolerance, (None, 0) for an empty buffer."""
    if not buffer:
        return None, 0.0
    stamp, item = min(list(buffer), key=lambda s: abs(s[0] - t))
    return (item, stamp - t) if abs(stamp - t) <= within else (None, stamp - t)


def _tilt_at(tilts: deque[tuple[float, float, float]], stamp: Any) -> tuple[float, float]:
    """Body roll and pitch (radians) nearest to a ROS time; level when IMU has not spoken."""
    if not tilts:
        return 0.0, 0.0
    t = _seconds(stamp)
    _, roll, pitch = min(list(tilts), key=lambda s: abs(s[0] - t))
    return roll, pitch


def _parse_tof(msg: Any) -> np.ndarray | None:
    """Convert an 8x8 ToF Float32MultiArray or depth message into a float32 (8, 8) matrix."""
    data = getattr(msg, "data", None)
    if data is None:
        return None
    if hasattr(msg, "height") and hasattr(msg, "width") and msg.height == 8 and msg.width == 8:
        return _image(msg).astype(np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 64:
        return arr.reshape(8, 8)
    return None


def _normalize_tof_grid(raw_grid: Any) -> np.ndarray | None:
    """Extract, resize to (8, 8) float32, and clamp finite values to [0.02, 4.0] metres."""
    arr = np.asarray(raw_grid, dtype=np.float32)
    if arr.ndim == 1 and arr.size == 64:
        arr = arr.reshape(8, 8)
    elif arr.ndim == 3 and arr.shape[2] >= 1:
        arr = arr[:, :, 0]
    elif arr.ndim != 2:
        return None
    if arr.shape != (8, 8):
        h, w = arr.shape
        if h <= 0 or w <= 0:
            return None
        ys = (np.arange(8) * h) // 8
        xs = (np.arange(8) * w) // 8
        arr = arr[np.ix_(ys, xs)]
    finite = np.isfinite(arr)
    return np.where(finite, np.clip(arr, 0.02, 4.0), arr).astype(np.float32)


def _expected_tof_floor(grid: np.ndarray, pitch_rad: float = 0.0, roll_rad: float = 0.0,
                        side: str = "front") -> np.ndarray:
    """Expected floor range matrix for an (8, 8) ToF grid, matching either angled Gazebo floor or flat tests."""
    valid = np.isfinite(grid) & (grid > 0.02) & (grid < 6.0)
    if not valid.any():
        return np.full((8, 8), np.nan, dtype=np.float32)

    expected_flat = np.full((8, 8), float(np.median(grid[valid])), dtype=np.float32)

    focal = 4.0 / math.tan(_TOF_HFOV / 2.0)
    pitch_eff = _TOF_PITCH + (pitch_rad if side == "front" else -pitch_rad)
    roll_eff = roll_rad if side == "front" else -roll_rad
    sp, cp = math.sin(pitch_eff), math.cos(pitch_eff)
    sr, cr = math.sin(roll_eff), math.cos(roll_eff)
    rows, cols = np.mgrid[0:8, 0:8]
    x_opt = (cols.astype(np.float32) + 0.5 - 4.0) / focal
    y_opt = (rows.astype(np.float32) + 0.5 - 4.0) / focal
    v_world_z = -sp + cp * (-x_opt * sr - y_opt * cr)
    expected_proj = np.full((8, 8), np.nan, dtype=np.float32)
    downward = v_world_z < -0.05
    d_proj = np.where(downward, _TOF_HEIGHT / np.maximum(-v_world_z, 1e-4), np.nan)
    keep = downward & (d_proj >= 0.05) & (d_proj <= 1.5)
    expected_proj[keep] = d_proj[keep]

    both = valid & np.isfinite(expected_proj)
    if np.count_nonzero(both) >= 8:
        err_proj = float(np.median(np.abs(grid[both] - expected_proj[both])))
        err_flat = float(np.median(np.abs(grid[both] - expected_flat[both])))
        if err_proj < err_flat:
            return expected_proj
    return expected_flat


SECTOR_AGE = 1.0  # seconds: a camera's sector older than this is left out of /scan
MEASURED_SCAN_TOPIC = "/scan_measured"
# Touching what no sensor saw (a table's tube frame, a chair leg): the motors are driven and the wheels do not turn.
STALL_V = 0.05  # m/s: a drive command at least this fast...
STALL_S = 0.3  # ...while the wheels turn at under a quarter of it for this long: something holds the body
CONTACT_KEEP = 120.0  # seconds a touched spot stays in /scan: the cameras will go on seeing clear floor there


def combine(parts: list[tuple[float, tuple[float, float, float], np.ndarray]], now: tuple[float, float, float],
            num_beams: int = 360) -> tuple[np.ndarray, float, float]:
    """One 360-degree scan from the cameras' sectors, each made from where the robot was then (its odometry pose),
    seen from where it is `now`: the nearest return per beam; inf where some sector saw clear floor; nan where none
    saw anything."""
    step = 2 * math.pi / num_beams
    out = np.full(num_beams, np.nan, np.float32)
    clear = np.zeros(num_beams, bool)
    angles = -math.pi + (np.arange(num_beams) + 0.5) * step
    for _, (px, py, pyaw), ranges in parts:
        dx, dy = px - now[0], py - now[1]  # then -> now, in the odometry frame, then into the robot's frame now
        c0, s0 = math.cos(-now[2]), math.sin(-now[2])
        ox, oy = dx * c0 - dy * s0, dx * s0 + dy * c0
        turn = pyaw - now[2]
        hit = np.isfinite(ranges)
        a = angles[hit] + turn
        x = ox + ranges[hit] * np.cos(a)
        y = oy + ranges[hit] * np.sin(a)
        k = (np.floor((np.arctan2(y, x) + math.pi) / step).astype(int)) % num_beams
        r = np.hypot(x, y).astype(np.float32)
        for kk, rr in zip(k, r, strict=True):
            if not rr >= out[kk]:
                out[kk] = rr
        free = np.isinf(ranges)
        k = (np.floor((angles[free] + turn + math.pi) / step).astype(int)) % num_beams
        clear[k] = True
    out[np.isnan(out) & clear] = np.inf
    return out, -math.pi, step


def _odometry_lookup(node: Any) -> Any:
    """stamp -> the robot's (x, y, yaw) in the odometry frame at that moment (the latest when not known yet)."""
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    buffer = Buffer()
    node._sprocket_tf = TransformListener(buffer, node)  # kept alive with the node

    def at(stamp: Any) -> tuple[float, float, float]:
        for when in (Time.from_msg(stamp), Time()):
            try:
                tr = buffer.lookup_transform("odom", "base_link", when).transform
            except Exception:  # noqa: BLE001 - not at that moment (yet)
                continue
            q = tr.rotation
            return tr.translation.x, tr.translation.y, math.atan2(2 * (q.w * q.z + q.x * q.y),
                                                                   1 - 2 * (q.y * q.y + q.z * q.z))
        return 0.0, 0.0, 0.0

    return at


class SyntheticScanPublisher:
    """Calibrates the cameras' depth estimates with IMU + ToF and publishes a 360-degree LaserScan on /scan."""

    def __init__(self, node: Any = None, publish_topic: str = "/scan", publish: bool = True) -> None:
        self.publish = publish
        self.cam_depths: dict[str, deque[tuple[int, np.ndarray]]] = {n: deque(maxlen=12) for n in RIG}
        self._depth_ready = threading.Condition()
        self.tof_grids: dict[str, np.ndarray] = {}
        self.hazards: dict[str, StepHazard] = {}
        self.tilts: deque[tuple[float, float, float]] = deque(maxlen=400)  # stamp s, roll, pitch (rad)
        self.pitch_range = [0.0, 0.0]
        self.turn_rate = 0.0
        self.scans: deque[tuple[float, tuple[np.ndarray, float, float]]] = deque(maxlen=12)
        self.scan: tuple[np.ndarray, float, float] | None = None
        self.parts: dict[str, tuple[float, tuple[float, float, float], np.ndarray]] = {}  # camera -> t, pose, sector
        self.odom_at: Any = None  # stamp -> the robot's (x, y, yaw) in the odometry frame then
        self._external_scan: bool = False
        self._lock = threading.Lock()
        self._pub: Any = None
        self._measured_pub: Any = None
        self._laser_scan_cls: Any = None
        self._last_pub_s: float = -1.0
        self._last_depth_s: float | None = None
        self._depth_dts: deque[float] = deque(maxlen=6)
        self._drive = (-1e9, 0.0)  # when, and the forward speed the motors were last told
        self._wheels_t: float | None = None
        self._stalled = 0.0  # seconds the wheels have not followed the drive
        self.contacts: deque[tuple[float, float, float]] = deque(maxlen=200)  # when, x, y in the odometry frame
        if node is not None:
            from geometry_msgs.msg import TwistStamped
            from nav_msgs.msg import Odometry
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image, Imu, LaserScan
            from std_msgs.msg import Float32MultiArray

            if self.publish and publish_topic:
                self._laser_scan_cls = LaserScan
                self._pub = node.create_publisher(LaserScan, publish_topic, 10)
                self._measured_pub = node.create_publisher(LaserScan, MEASURED_SCAN_TOPIC, 10)
                self.odom_at = _odometry_lookup(node)
                self._clock = node.get_clock
                now = lambda: _seconds(node.get_clock().now().to_msg())  # noqa: E731
                node.create_subscription(TwistStamped, WHEELS,
                                         lambda m: self.drive(now(), m.twist.linear.x), 10)
                node.create_subscription(Odometry, "/odom", lambda m: self.wheels(now(), m.twist.twist.linear.x),
                                         qos_profile_sensor_data)
            else:  # the depth node feeds its publisher directly; everyone else reads its depth pictures
                for name, topic in DEPTH_TOPICS.items():
                    node.create_subscription(Image, topic, lambda m, n=name: self._depth(n, m),
                                             qos_profile_sensor_data)
            for side, topic in TOF_TOPICS.items():
                node.create_subscription(Float32MultiArray, topic, lambda m, s=side: self._tof(s, m),
                                         qos_profile_sensor_data)
            node.create_subscription(Imu, "/imu", self._imu, qos_profile_sensor_data)

    def _sector_age(self) -> float:
        """Keep sectors for at least SECTOR_AGE, or a full slow-CPU camera cycle plus half a frame."""
        if len(self._depth_dts) < 2:
            return SECTOR_AGE
        return max(SECTOR_AGE, (len(self.cam_depths) + 0.5) * float(np.median(self._depth_dts)))

    def _fresh_parts(self, t: float) -> list[tuple[float, tuple[float, float, float], np.ndarray]]:
        age = self._sector_age()
        return [p for p in self.parts.values() if t - p[0] <= age]

    def tilt_at(self, stamp: Any) -> tuple[float, float]:
        return _tilt_at(self.tilts, stamp)

    # ---- touching what nothing saw ------------------------------------------------------------------------------
    def drive(self, t: float, v: float) -> None:
        self._drive = (t, v)

    def wheels(self, t: float, v: float, pose: tuple[float, float, float] | None = None) -> bool:
        """The wheels' speed at time t. When they have not followed a drive command for STALL_S, the body is held by
        something at its edge the way it was driving: that spot joins /scan for CONTACT_KEEP seconds, and a scan
        goes out at once. Returns True on a new contact."""
        dt = 0.0 if self._wheels_t is None else max(0.0, t - self._wheels_t)
        self._wheels_t = t
        when, want = self._drive
        if t - when < 0.5 and abs(want) >= STALL_V and abs(v) < 0.25 * abs(want):
            self._stalled += dt
        else:
            self._stalled = 0.0
            return False
        if self._stalled < STALL_S or self._stalled - dt >= STALL_S:
            return False
        if pose is None:
            pose = self.odom_at(self._clock().now().to_msg()) if self.odom_at is not None else (0.0, 0.0, 0.0)
        x, y, yaw = pose
        ahead = math.copysign(BODY_R + 0.03, want)
        c, s = math.cos(yaw), math.sin(yaw)
        with self._lock:
            for side in np.linspace(-0.8 * BODY_R, 0.8 * BODY_R, 5):  # across the body's front (or back) edge
                self.contacts.append((t, x + ahead * c - side * s, y + ahead * s + side * c))
        if self._pub is not None:
            stamp = self._clock().now().to_msg()
            with self._lock:
                parts = self._fresh_parts(t)
                scan = combine([*parts, self._contact_part(t, pose)], pose)
                self._last_pub_s = max(self._last_pub_s, t)
            self._publish_scan(self._pub, stamp, scan)
        return True

    def _contact_part(self, t: float, pose: tuple[float, float, float],
                      num_beams: int = 360) -> tuple[float, tuple[float, float, float], np.ndarray]:
        """The spots touched lately, as a scan sector from `pose` (odometry frame)."""
        ranges = np.full(num_beams, np.nan, np.float32)
        step = 2 * math.pi / num_beams
        c, s = math.cos(-pose[2]), math.sin(-pose[2])
        for when, px, py in self.contacts:
            if t - when > CONTACT_KEEP:
                continue
            dx, dy = px - pose[0], py - pose[1]
            bx, by = dx * c - dy * s, dx * s + dy * c
            r = math.hypot(bx, by)
            k = int((math.atan2(by, bx) + math.pi) / step) % num_beams
            if not r >= ranges[k]:
                ranges[k] = r
        return t, pose, ranges

    def _update_tof(self, side: str, grid: np.ndarray) -> StepHazard:
        _, roll, pitch = self.tilts[-1] if self.tilts else (0.0, 0.0, 0.0)
        expected = _expected_tof_floor(grid, pitch_rad=pitch, roll_rad=roll, side=side)
        hazard = classify_tof_grid(grid, expected)
        self.tof_grids[side] = grid
        self.hazards[side] = hazard
        if self.publish and self.odom_at is not None:  # low things the cameras take for floor join the scan
            stamp = self._clock().now().to_msg()
            with self._lock:
                self.parts[f"tof_{side}"] = (_seconds(stamp), self.odom_at(stamp),
                                              tof_scan(grid, side, pitch_rad=pitch, roll_rad=roll))
        return hazard

    def _tof(self, side: str, msg: Any) -> None:
        raw = _parse_tof(msg)
        if raw is None:
            return
        grid = _normalize_tof_grid(raw)
        if grid is not None:
            self._update_tof(side, grid)

    def _imu(self, msg: Any) -> None:
        self.turn_rate = msg.angular_velocity.z
        q = msg.orientation
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        self.tilts.append((_seconds(msg.header.stamp), roll, pitch))
        self.pitch_range[0] = min(self.pitch_range[0], pitch)
        self.pitch_range[1] = max(self.pitch_range[1], pitch)

    def _depth(self, name: str, msg: Any) -> None:
        self.ingest_depth(name, msg.header.stamp, _image(msg))

    def depth_for(self, name: str, stamp: Any) -> tuple[np.ndarray | None, float]:
        """Depth for this exposure only: a 0.2 s old image was 9 degrees off during a turn.
        The depth node copies the RGB stamp. Once it publishes a later frame, this one was skipped."""
        buf, t = self.cam_depths[name], _nanoseconds(stamp)
        with self._depth_ready:
            self._depth_ready.wait_for(lambda: buf and buf[-1][0] >= t, timeout=0.15)
            depth, gap = taken_with(buf, t, 0)
        return depth, gap * 1e-9

    def ingest_depth(self, name: str, stamp: Any, depth: np.ndarray) -> tuple[np.ndarray, float, float] | None:
        """Keep a camera's metric depth (for ranging what the detector finds); when publishing, add its sector to
        /scan. Each sector is made at its own picture's moment and carried to the newest one by the odometry: the
        cameras' pictures are taken at different moments, and while turning an old sector lies elsewhere now."""
        if name not in self.cam_depths:
            return None
        t = _seconds(stamp)
        with self._depth_ready:
            self.cam_depths[name].append((_nanoseconds(stamp), depth))
            self._depth_ready.notify_all()
        if not self.publish:
            return None
        roll, pitch = self.tilt_at(stamp)
        part = synthetic_scan({name: depth}, pitch_rad=pitch, roll_rad=roll)[0]  # this camera's sector
        pose = self.odom_at(stamp) if self.odom_at is not None else (0.0, 0.0, 0.0)
        with self._lock:
            if self._last_depth_s is not None and t > self._last_depth_s:
                self._depth_dts.append(t - self._last_depth_s)
            self._last_depth_s = max(self._last_depth_s, t) if self._last_depth_s is not None else t
            self.parts[name] = (t, pose, part)
            parts = self._fresh_parts(t)
            measured = combine(parts, pose)
            scan = measured
            if self.contacts:
                contacts, _, _ = combine([self._contact_part(t, pose)], pose)
                scan = np.fmin(measured[0], contacts), measured[1], measured[2]
            self.scan = scan
            self.scans.append((t, scan))
            should_pub = self._pub is not None and t > self._last_pub_s
            if should_pub:
                self._last_pub_s = t
        if should_pub:
            self._publish_scan(self._pub, stamp, scan)
            self._publish_scan(self._measured_pub, stamp, measured)
        return scan

    def _publish_scan(self, publisher: Any, stamp: Any, scan: tuple[np.ndarray, float, float]) -> None:
        ranges, angle_min, step = scan
        msg = self._laser_scan_cls()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.angle_min = float(angle_min)
        msg.angle_max = float(angle_min + (len(ranges) - 1) * step)
        msg.angle_increment = float(step)
        msg.range_min = 0.12
        msg.range_max = 8.0
        msg.ranges = array.array("f", np.asarray(ranges, np.float32).tobytes())  # inf: clear floor; nan: unseen
        publisher.publish(msg)


class Perception:
    def __init__(self, node: Any, detector: Any, pose_at: Any, brain: Any = None,
                 scan_bridge: SyntheticScanPublisher | None = None) -> None:
        """`pose_at(stamp)` gives the robot's (x, y, yaw) in the map at a ROS time, or None; `brain` names new
        objects when given."""
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, LaserScan

        self.detector, self.pose_at, self.brain = detector, pose_at, brain
        self.scan_bridge = scan_bridge if scan_bridge is not None else SyntheticScanPublisher(node, publish=False)
        self.tracker = Tracker()
        self.lock = threading.Lock()
        self.frames: dict[str, tuple[Any, np.ndarray]] = {}
        self.fresh: set[str] = set()
        # Share metric-calibrated 3-CSI depth, ToF, IMU tilt and synthetic /scan buffers with SyntheticScanPublisher.
        self.tof_grids = self.scan_bridge.tof_grids
        self.scans = self.scan_bridge.scans
        self.depth_pairs: deque[float] = deque(maxlen=500)  # 0 when a frame found its depth image, else how far off
        self.crops: dict[int, np.ndarray] = {}
        self.doubts: dict[int, tuple[str, np.ndarray]] = {}  # track id -> its label and the picture of its place
        self.past: deque[tuple[float, np.ndarray]] = deque(maxlen=PAST_KEPT)  # front pictures, 2 a second
        self.boxes: dict[str, list[tuple[tuple[int, int, int, int], int]]] = {}  # camera -> latest (box, track id)
        self.timings: list[float] = []
        self.looks: dict[str, Counter] = {}  # frames processed per camera, by each word the detector looked for
        self.served: dict[str, float] = {}  # when each camera's frame was last taken for detection
        self.detected: dict[str, tuple[float, np.ndarray]] = {}  # camera -> capture time, RGB paired with its boxes
        # every sighting: monotonic time, camera, track, map x, y, how its range was found, robot x, y, depth lag (s)
        self.sightings: deque[tuple] = deque(maxlen=2000)
        self.tilts = self.scan_bridge.tilts
        self.pitch_range = self.scan_bridge.pitch_range
        for name, topic in TOPICS.items():
            node.create_subscription(Image, topic, lambda m, n=name: self._frame(n, m), qos_profile_sensor_data)
        node.create_subscription(LaserScan, "/scan", self._scan, qos_profile_sensor_data)
        self.measured_scan: Any = None
        node.create_subscription(LaserScan, MEASURED_SCAN_TOPIC, lambda m: setattr(self, "measured_scan", m),
                                 qos_profile_sensor_data)

        self.pose_sigma = (0.05, math.radians(2.0))
        self.amcl_sigma: tuple[float, float] | None = None
        node.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self._amcl, 10)
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()
        if brain is not None:
            threading.Thread(target=self._verify_loop, daemon=True).start()

    @property
    def scan(self) -> tuple[np.ndarray, float, float] | None:
        return self.scan_bridge.scan

    @scan.setter
    def scan(self, value: tuple[np.ndarray, float, float] | None) -> None:
        self.scan_bridge.scan = value

    @property
    def turn_rate(self) -> float:
        return self.scan_bridge.turn_rate

    @turn_rate.setter
    def turn_rate(self, value: float) -> None:
        self.scan_bridge.turn_rate = value

    def _frame(self, name: str, msg: Any) -> None:
        rgb = _image(msg)
        self.frames[name] = (msg.header.stamp, rgb)
        self.fresh.add(name)
        t = _seconds(msg.header.stamp)
        if name == "front_camera" and (not self.past or t - self.past[-1][0] >= PAST_EVERY):  # for looking back
            import cv2

            self.past.append((t, cv2.resize(rgb, (768, round(rgb.shape[0] * 768 / rgb.shape[1])),
                                            interpolation=cv2.INTER_AREA)))

    def _depth(self, name_or_msg: Any, msg: Any = None) -> None:
        if msg is None:
            name, m = "front_camera", name_or_msg
        else:
            name, m = name_or_msg, msg
        self.scan_bridge.ingest_depth(name, m.header.stamp, _image(m))

    def _tof(self, side: str, msg: Any) -> None:
        self.scan_bridge._tof(side, msg)

    def _scan(self, msg: Any) -> None:
        scan = (np.asarray(msg.ranges, dtype=np.float32), msg.angle_min, msg.angle_increment)
        self.scan_bridge._external_scan = True
        self.scan_bridge.scan = scan
        self.scan_bridge.scans.append((_seconds(msg.header.stamp), scan))

    def _amcl(self, msg: Any) -> None:
        c = msg.pose.covariance
        self.pose_sigma = (math.sqrt(max(0.0, (c[0] + c[7]) / 2)), math.sqrt(max(0.0, c[35])))
        self.amcl_sigma = self.pose_sigma if all(math.isfinite(v) and v >= 0 for v in (c[0], c[7], c[35])) else None

    def _imu(self, msg: Any) -> None:
        self.scan_bridge._imu(msg)

    def tilt_at(self, stamp: Any) -> tuple[float, float]:
        """The body's roll and pitch (radians) nearest to a ROS time; level when the IMU has not spoken."""
        return _tilt_at(self.tilts, stamp)

    def stop(self) -> None:
        self._stop = True

    def _loop(self) -> None:
        while not self._stop:
            waiting = [n for n in RIG if n in self.fresh]  # fair turns: the camera served longest ago goes next
            name = min(waiting, key=lambda n: self.served.get(n, 0.0)) if waiting else None
            if name is None:
                time.sleep(0.02)
                continue
            self.fresh.discard(name)
            self.served[name] = time.monotonic()
            try:
                self.process(name)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not stop perception
                print(f"perception: {name}: {exc}", flush=True)

    def _verify_loop(self) -> None:
        """The vision model's yes or no on a picture, for a new thing the detector named (yes keeps it, no drops it:
        YOLOE fires on an asked word too easily, black panels as "tv", 12% of ball-free pictures as "ball"; renaming
        a no to another of the few asked words instead turned a false "tv" into a "sofa"), and for a thing the
        detector stopped finding where it is mapped, shown that place (no: gone; yes: still there). On the detector's
        misses alone the robot gave up real things mid-approach, seen up close or from aside: a bike seen 110 times,
        a plant 53 times, beanbags; false ones it named from afar (a table as "tv") go as before, on the model's no."""
        while not self._stop:
            if getattr(self.brain, "busy", False) is True:  # yield the GPU while an agent LLM turn is running
                time.sleep(0.15)
                continue
            # New things first: tools drive to them, and a doubted thing stays until it is answered. Asked after the
            # doubts, the bike YOLOE called an umbrella (the model's yes to that crop: 0%) was driven to unchecked.
            # Only background-verify labels in the active vocabulary (old-task tracks are vetted synchronously on demand).
            vocab = getattr(self.detector, "vocabulary", ())
            watched = {canonical(w) for w in vocab} if isinstance(vocab, (list, tuple, set, frozenset)) else set()
            with self.lock:
                active_crops = (
                    {tid: img for tid, img in self.crops.items()
                     if (tr := self.tracker.tracks.get(tid)) is not None and canonical(tr.label) in watched}
                    if watched else self.crops
                )
                track = self.tracker.to_name(active_crops)
                doubt = next(iter(self.doubts), None) if track is None else None
                if track is not None:
                    label = track.votes.most_common(1)[0][0] if track.votes else track.label
                    picture = self.crops[track.id]
                elif doubt is not None:
                    label, picture = self.doubts[doubt]
            if doubt is None and track is None:
                time.sleep(0.2)
                continue
            try:
                yes = is_there(self.brain, picture, label)
            except Exception as exc:  # noqa: BLE001 - unchecked when the model is unreachable: asked again later
                print(f"perception: checking {label}: {exc}", flush=True)
                time.sleep(1.0)
                continue
            with self.lock:
                if track is not None:
                    self.tracker.verify(track, label if yes else None, pixels(picture), sure=yes)
                else:
                    del self.doubts[doubt]
                    doubted = self.tracker.tracks.get(doubt)
                    if doubted is not None and doubted.misses >= GONE_AFTER:  # not seen again while the model looked
                        if yes:
                            doubted.misses = 0
                        else:
                            doubted.present = False
            time.sleep(0.15)

    def process(self, name: str) -> None:
        stamp, rgb = self.frames[name]
        pose = self.pose_at(stamp)
        if pose is None:
            return
        cam = RIG[name]
        started = time.perf_counter()
        words = list(getattr(self.detector, "vocabulary", ()))  # this frame's words: one asked for from now on is not
        seen = self.detector(rgb)
        self.timings = (self.timings + [time.perf_counter() - started])[-20:]
        self.looks.setdefault(name, Counter()).update(words)
        t = _seconds(stamp)
        roll, pitch = self.tilt_at(stamp)
        depth, gap = self.scan_bridge.depth_for(name, stamp)
        self.depth_pairs.append(gap)
        if depth is not None and depth.shape != rgb.shape[:2]:  # estimated smaller: one value per block of pixels
            ky, kx = rgb.shape[0] // depth.shape[0], rgb.shape[1] // depth.shape[1]
            depth = np.repeat(np.repeat(depth, ky, axis=0), kx, axis=1)
        scan, _ = taken_with(self.scans, t, 0.3)  # /scan from the depth node, near this frame's moment
        detections, hows = [], {}
        for s in seen:
            x1, y1, x2, y2 = (int(v) for v in s.box)
            cut = x1 <= 4 or x2 >= rgb.shape[1] - 4
            if cut and (x2 - x1) < 0.06 * rgb.shape[1]:
                continue
            place = locate(cam, s.box, depth, scan, s.foot, canonical(s.label) == "person",
                           mask=getattr(s, "mask", None))
            if place is None:
                continue
            reach = math.hypot(place.x, place.y)
            ray_bearing = math.atan2(place.y, place.x)
            if scan is not None:
                scan_r = lidar_range(scan, ray_bearing, spread=0.06)
                if scan_r is not None and reach > scan_r + max(0.40, place.radius + 0.25):
                    continue
            mx, my = to_map(pose, place.x, place.y)
            core = rgb[y1 + (y2 - y1) // 4:y2 - (y2 - y1) // 4, x1 + (x2 - x1) // 4:x2 - (x2 - x1) // 4]
            # how unsure this place is: the range finding's own error, plus the robot's pose error carried out to
            # the thing's distance (localisation's, and 50 ms of timing at the current turn rate)
            heading = self.pose_sigma[1] + abs(self.turn_rate) * 0.05
            center_x = (x1 + x2) / 2
            off_axis = abs(cam.bearing(center_x) - cam.yaw)
            if off_axis > math.radians(48.0):
                sigma_scale = 1.0 + 2.5 * ((off_axis - math.radians(48.0)) / math.radians(22.0)) ** 2
                if cut:
                    sigma_scale *= 1.8
            else:
                sigma_scale = 2.0 if cut else 1.0
            sigma = math.sqrt((place.sigma * sigma_scale) ** 2 + self.pose_sigma[0] ** 2 + (reach * heading) ** 2)
            is_person = canonical(s.label) == "person"
            human_pose = (estimate_pose_from_keypoints(s.keypoints, cam, reach, pose)
                          if is_person and getattr(s, "keypoints", None) is not None else None)
            ray = math.atan2(place.y - cam.y, place.x - cam.x) + pose[2] if cut else 0.0
            detection = Detection(s.label, mx, my, place.radius, colour.name(core), s.score, name, cut, sigma,
                                  look(core) if is_person else None,
                                  width=place.width, depth_extent=place.depth_extent,
                                  orientation_yaw=place.orientation_yaw, pose=human_pose, box=(x1, y1, x2, y2),
                                  normal=(-math.sin(ray), math.cos(ray)) if cut else None)
            detections.append(detection)
            hows[id(detection)] = place.how
        now = time.monotonic()
        with self.lock:
            pairs = self.tracker.update(detections, now)
            matched = {id(t) for _, t in pairs}
            # A split outline is one object. Keep all its pixels but only one badge and verification crop.
            boxes = {}
            for det, track in pairs:
                if det.pose is not None:
                    track.pose = det.pose
                x, y, xx, yy = det.box
                left, top, right, bottom = boxes.get(track.id, det.box)
                boxes[track.id] = (min(left, x), min(top, y), max(right, xx), max(bottom, yy))
            self.boxes[name] = [(box, tid) for tid, box in boxes.items()]
            self.detected[name] = (t, rgb)
            self.sightings.extend((now, name, track.id, det.x, det.y, hows[id(det)], pose[0], pose[1], gap)
                                  for det, track in pairs)
            for tid, box in boxes.items():
                if (box[0] <= 4 or box[2] >= cam.width - 4) and tid in self.crops:
                    continue  # a part of the thing makes a poor picture of it; keep the whole one
                self.crops[tid] = _only(rgb, box, [b for i, b in boxes.items() if i != tid])
            watched = {canonical(w) for w in words}
            for track in self.tracker.objects():  # negative evidence: viewed clearly, not seen, while looked for
                if (id(track) in matched or not track.present or track.person or track.visual
                        or canonical(track.label) not in watched):
                    continue  # (a ball seen while asked for is not gone because the detector stopped looking)
                bx, by = to_robot(pose, track.x, track.y)
                dist = math.hypot(bx, by)
                # a clear view: near enough to be detected, and far enough to fit in the picture (beside a chair the
                # front camera sees only legs, and three such misses took the chair off the map)
                fits = dist - track.radius > 0.6 and 2 * math.atan2(track.radius, dist) < 0.6 * cam.hfov
                if not (fits and dist < SEE_FAR and cam.sees(bx, by, margin=0.28)):
                    continue
                scan_r = lidar_range(self.scan, math.atan2(by, bx))
                if scan_r is not None and scan_r < dist - track.radius - 0.3:  # something nearer hides it
                    continue
                if self.tracker.missed(track, now):
                    where = place_picture(rgb, cam, bx, by, track.radius) if self.brain is not None else None
                    if where is None:
                        track.present = False
                    else:
                        self.doubts.setdefault(track.id, (track.label, where))

    def render_annotated_view(self, camera: str, width: int, height: int, min_stamp: float) -> np.ndarray | None:
        view = self.annotated_observation(camera, width, height, min_stamp)
        return view[0] if view is not None else None

    def annotated_observation(self, camera: str, width: int, height: int,
                              min_stamp: float) -> tuple[np.ndarray, frozenset[int], float] | None:
        """Show badges only on their own recent exposure. If detection stalls, show current unmarked RGB instead."""
        with self.lock:
            frame = self.detected.get(camera)
            if frame is not None and frame[0] >= min_stamp:
                stamp, rgb = frame
                raw_boxes = list(self.boxes.get(camera, ()))
            else:
                frame = self.frames.get(camera)
                if frame is None or _seconds(frame[0]) < min_stamp:
                    return None
                rgb, raw_boxes = frame[1], []
                stamp = _seconds(frame[0])
            labels_by_id = {t.id: t.label for t in self.tracker.tracks.values()}
            hidden = {t.id for t in self.tracker.tracks.values() if t.rejected or not t.present}
            raw_boxes = [(b, self.tracker.current(tid)) for b, tid in raw_boxes]
        orig_h, orig_w = rgb.shape[:2]
        target_w = width if width > 0 else orig_w
        target_h = height if height > 0 else (int(round(target_w * 9 / 16)) if width > 0 else orig_h)
        from robot.perception.camera import resize_rgb as _resize_rgb

        if target_w != orig_w or target_h != orig_h:
            img = _resize_rgb(rgb, target_w, target_h)
            sx = target_w / max(1, orig_w)
            sy = target_h / max(1, orig_h)
        else:
            img = rgb
            sx, sy = 1.0, 1.0

        boxes_with_labels: list[tuple[tuple[int, int, int, int], int, str]] = []
        for b, tid in raw_boxes:
            if tid in hidden or tid not in labels_by_id:  # rejected or marked absent: never lure go_to with dead IDs
                continue
            scaled_box = (
                int(round(b[0] * sx)),
                int(round(b[1] * sy)),
                int(round(b[2] * sx)),
                int(round(b[3] * sy)),
            )
            lbl = labels_by_id.get(tid, "")
            boxes_with_labels.append((scaled_box, tid, lbl))

        marks = frozenset(tid for _, tid, _ in boxes_with_labels)
        return draw_set_of_mark(img, boxes_with_labels), marks, stamp

    def objects(self) -> list[Track]:
        with self.lock:
            return [replace(t, colors=t.colors.copy(), votes=t.votes.copy()) for t in self.tracker.objects()]


class DepthNode:
    """Each camera's newest frame in turn becomes a depth estimate, published on DEPTH_TOPICS with that frame's stamp
    (for the detector to range what it finds) and fed straight to the /scan publisher."""

    def __init__(self, node: Any, model: Any) -> None:
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        self.model, self.image_cls = model, Image
        self.scan = SyntheticScanPublisher(node, publish=True)
        self.frames: dict[str, Any] = {}
        self.fresh: set[str] = set()
        self.served: dict[str, float] = {}
        self.timings: deque[float] = deque(maxlen=50)
        self.failures = 0  # frames the depth network failed on
        self.pubs = {name: node.create_publisher(Image, topic, 10) for name, topic in DEPTH_TOPICS.items()}
        for name, topic in TOPICS.items():
            node.create_subscription(Image, topic, lambda m, n=name: self._frame(n, m), qos_profile_sensor_data)

    def _frame(self, name: str, msg: Any) -> None:
        self.frames[name] = msg
        self.fresh.add(name)

    def once(self) -> bool:
        """Estimate one camera's newest frame (the one served longest ago goes first); False when none is new."""
        waiting = [n for n in RIG if n in self.fresh]
        if not waiting:
            return False
        name = min(waiting, key=lambda n: self.served.get(n, 0.0))
        self.fresh.discard(name)
        self.served[name] = time.monotonic()
        msg = self.frames[name]
        started = time.perf_counter()
        roll, pitch = self.scan.tilt_at(msg.header.stamp)
        try:
            depth = self.model(_image(msg), RIG[name], pitch, roll)
        except Exception as exc:  # noqa: BLE001 - one failed frame (a GPU hiccup) must not end depth for good
            self.failures += 1
            print(f"depth: {name}: {exc}", flush=True)
            return True
        self.timings.append(time.perf_counter() - started)
        if depth is None:  # too little floor in sight to scale it
            return True
        out = self.image_cls()
        out.header = msg.header
        out.height, out.width = depth.shape
        out.encoding, out.step = "32FC1", depth.shape[1] * 4
        out.data = array.array("B", depth.tobytes())
        self.pubs[name].publish(out)
        self.scan.ingest_depth(name, msg.header.stamp, depth)
        return True


def main() -> None:
    """The depth node: monocular depth for the three cameras, and the synthetic /scan made from it."""
    parser = argparse.ArgumentParser(description="Sprocket's depth node: depth estimates and synthetic /scan")
    parser.add_argument("--depth", action="store_true", help="run the depth node")
    args = parser.parse_args()
    if not args.depth:
        parser.error("pass --depth to run the depth node")

    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from rclpy.parameter import Parameter

    from robot.perception.depth import MonoDepth
    from robot.perception.gpu import VisionServer

    rclpy.init()
    node = rclpy.create_node("sprocket_depth", parameter_overrides=[Parameter("use_sim_time", value=True)])
    depth = DepthNode(node, MonoDepth(VisionServer()))
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    reported = time.monotonic()
    try:
        while rclpy.ok():
            if not depth.once():
                time.sleep(0.005)
            if time.monotonic() - reported > 30 and depth.timings:
                reported = time.monotonic()
                print(f"depth: median {1000 * sorted(depth.timings)[len(depth.timings) // 2]:.0f} ms per frame",
                      flush=True)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
