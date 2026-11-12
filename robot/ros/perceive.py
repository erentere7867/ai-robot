"""Perception on ROS: Sprocket's three cameras and lidar, through YOLOE-26, into one map-frame tracker.

A worker thread takes the newest frame from each camera in turn, runs the detector, places each detection on the map
(depth on the front camera; lidar or the floor on the rear ones; robot pose from TF at the frame's time), and folds
them into the tracker. A camera that clearly views where a known object should be, without seeing it, counts a miss
against it (negative evidence). Each object keeps its latest crop, so the vision model can answer about it; with a
brain, a second thread has the vision model confirm each new object's name from its crop ("is there a chair?"),
and name it from the robot's words only when it says no (tracker.verify).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import replace
from typing import Any

import numpy as np

from robot.perception import colour
from robot.perception.camera import data_url
from robot.perception.locate import lidar_range, locate
from robot.perception.rig import RIG, to_map, to_robot
from robot.perception.tracker import Detection, Track, Tracker, canonical, look, pixels

TOPICS = {"front_camera": "/rgbd_camera/image", "rear_left_camera": "/rear_left_camera/image",
          "rear_right_camera": "/rear_right_camera/image"}
SEE_FAR = 3.5  # beyond this a miss is not evidence: the detector may simply not resolve it


def _image(msg: Any) -> np.ndarray:
    channels = 3 if msg.encoding in ("rgb8", "bgr8") else 1
    dtype = np.float32 if msg.encoding == "32FC1" else np.uint8
    array = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels) if channels == 3 else \
        np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
    return array[:, :, ::-1] if msg.encoding == "bgr8" else array


def _seconds(stamp: Any) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


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


def taken_with(buffer: deque, t: float, within: float) -> tuple[Any, float]:
    """The item in a (stamp, item) buffer taken nearest to time t, and how much later it was taken; (None, 0) when
    nothing is within `within` seconds."""
    if not buffer:
        return None, 0.0
    stamp, item = min(list(buffer), key=lambda s: abs(s[0] - t))
    return (item, stamp - t) if abs(stamp - t) <= within else (None, stamp - t)


class Perception:
    def __init__(self, node: Any, detector: Any, pose_at: Any, brain: Any = None) -> None:
        """`pose_at(stamp)` gives the robot's (x, y, yaw) in the map at a ROS time, or None; `brain` names new
        objects when given."""
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, Imu, LaserScan

        self.detector, self.pose_at, self.brain = detector, pose_at, brain
        self.tracker = Tracker()
        self.lock = threading.Lock()
        self.frames: dict[str, tuple[Any, np.ndarray]] = {}
        self.fresh: set[str] = set()
        # The last second of depth images and lidar scans, by stamp: each colour frame is paired with the depth and the
        # scan taken with it. Paired with the newest instead, a depth image 200 ms older than its colour frame put
        # one front sighting in five over 30% too far while the robot turned.
        self.depths: deque[tuple[float, tuple[Any, np.ndarray]]] = deque(maxlen=12)  # stamp s, (stamp, image)
        self.scans: deque[tuple[float, tuple[np.ndarray, float, float]]] = deque(maxlen=12)
        self.scan: tuple[np.ndarray, float, float] | None = None  # the newest, for the live page
        self.depth_pairs: deque[float] = deque(maxlen=500)  # 0 when a frame found its depth image, else how far off
        self.crops: dict[int, np.ndarray] = {}
        self.boxes: dict[str, list[tuple[tuple[int, int, int, int], int]]] = {}  # camera -> latest (box, track id)
        self.timings: list[float] = []
        self.counts: dict[str, int] = {}  # frames processed per camera
        self.served: dict[str, float] = {}  # when each camera's frame was last taken for detection
        self.detected: dict[str, np.ndarray] = {}  # camera -> the frame the detector last ran on
        # every sighting: monotonic time, camera, track, map x, y, how its range was found, robot x, y, depth lag (s)
        self.sightings: deque[tuple] = deque(maxlen=20000)
        self.tilts: deque[tuple[float, float, float]] = deque(maxlen=400)  # IMU: stamp s, roll, pitch (radians)
        self.pitch_range = [0.0, 0.0]  # the most nose-down and nose-up pitch seen, radians
        for name, topic in TOPICS.items():
            node.create_subscription(Image, topic, lambda m, n=name: self._frame(n, m), qos_profile_sensor_data)
        node.create_subscription(Image, "/rgbd_camera/depth_image", self._depth, qos_profile_sensor_data)
        node.create_subscription(LaserScan, "/scan", self._scan, qos_profile_sensor_data)
        node.create_subscription(Imu, "/imu", self._imu, qos_profile_sensor_data)
        from geometry_msgs.msg import PoseWithCovarianceStamped

        self.pose_sigma = (0.05, math.radians(2.0))  # how unsure localisation is: metres, radians (from AMCL)
        self.turn_rate = 0.0  # radians a second, from the gyro
        node.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self._amcl, 10)
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()
        if brain is not None:
            threading.Thread(target=self._verify_loop, daemon=True).start()

    def _frame(self, name: str, msg: Any) -> None:
        self.frames[name] = (msg.header.stamp, _image(msg))
        self.fresh.add(name)

    def _depth(self, msg: Any) -> None:
        self.depths.append((_seconds(msg.header.stamp), (msg.header.stamp, _image(msg))))

    def _scan(self, msg: Any) -> None:
        self.scan = (np.asarray(msg.ranges, dtype=np.float32), msg.angle_min, msg.angle_increment)
        self.scans.append((_seconds(msg.header.stamp), self.scan))

    def _amcl(self, msg: Any) -> None:
        c = msg.pose.covariance
        self.pose_sigma = (math.sqrt(max(0.0, (c[0] + c[7]) / 2)), math.sqrt(max(0.0, c[35])))

    def _imu(self, msg: Any) -> None:
        self.turn_rate = msg.angular_velocity.z
        q = msg.orientation
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        self.tilts.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, roll, pitch))
        self.pitch_range = [min(self.pitch_range[0], pitch), max(self.pitch_range[1], pitch)]

    def tilt_at(self, stamp: Any) -> tuple[float, float]:
        """The body's roll and pitch (radians) nearest to a ROS time; level when the IMU has not spoken."""
        if not self.tilts:
            return 0.0, 0.0
        t = stamp.sec + stamp.nanosec * 1e-9
        _, roll, pitch = min(list(self.tilts), key=lambda s: abs(s[0] - t))
        return roll, pitch

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
        while not self._stop:
            with self.lock:
                track = self.tracker.to_name(self.crops)
                crop = self.crops.get(track.id) if track is not None else None
            if track is None:
                time.sleep(0.2)
                continue
            words = sorted({canonical(w) for w in self.detector.vocabulary})
            guess = track.votes.most_common(1)[0][0] if track.votes else track.label
            try:  # confirm the detector's name first: a yes or no is far surer than a pick from forty words
                url = data_url(crop)
                yes, _ = self.brain.check(url, f"Is there a {guess} in the middle of this picture?")
                label = guess if yes else self.brain.name(url, [w for w in words if w != guess])
                sure = bool(yes)
            except Exception as exc:  # noqa: BLE001 - the detector's label stands when the model is unreachable
                print(f"perception: naming {track.label} {track.id}: {exc}", flush=True)
                label, sure = track.label, False
            with self.lock:
                self.tracker.verify(track, label, pixels(crop), sure)

    def _depth_for(self, stamp: Any, pose: tuple[float, float, float]) -> tuple[np.ndarray | None, float]:
        """The depth image seen from where the colour frame was: taken within 30 ms of it, or taken while the robot
        stood (moved under 1 cm and half a degree in between). The depth stream runs slower than the colour one,
        so only one colour frame in four had a depth image of its own moment; standing still, any recent one serves,
        while turning a mismatched one would read the wrong pixels. Waits briefly for a depth image still on its way."""
        t, waited = _seconds(stamp), time.monotonic()
        while True:
            depth, gap = taken_with(self.depths, t, 0.03)
            if depth is not None:
                return depth[1], gap
            near, gap = taken_with(self.depths, t, 0.3)
            if near is not None:
                then = self.pose_at(near[0])
                if then is not None and math.dist(then[:2], pose[:2]) < 0.01 and \
                        abs(math.atan2(math.sin(then[2] - pose[2]), math.cos(then[2] - pose[2]))) < math.radians(0.5):
                    return near[1], gap
            if time.monotonic() - waited > 0.15:
                return None, gap
            time.sleep(0.01)

    def process(self, name: str) -> None:
        stamp, rgb = self.frames[name]
        pose = self.pose_at(stamp)
        if pose is None:
            return
        cam = RIG[name]
        started = time.perf_counter()
        seen = self.detector(rgb)
        self.timings = (self.timings + [time.perf_counter() - started])[-20:]
        self.counts[name] = self.counts.get(name, 0) + 1
        t = _seconds(stamp)
        depth, gap = None, 0.0  # gap: how much later than the colour frame the depth image was taken
        if cam.depth:
            depth, gap = self._depth_for(stamp, pose)
            self.depth_pairs.append(gap if depth is None else 0.0)  # for the report: misses and how far off
        scan, _ = taken_with(self.scans, t, 0.06)
        detections, boxes, hows = [], {}, {}
        for s in seen:
            place = locate(cam, s.box, depth, scan, s.foot, canonical(s.label) == "person")
            if place is None:
                continue
            mx, my = to_map(pose, place.x, place.y)
            x1, y1, x2, y2 = (int(v) for v in s.box)
            core = rgb[y1 + (y2 - y1) // 4:y2 - (y2 - y1) // 4, x1 + (x2 - x1) // 4:x2 - (x2 - x1) // 4]
            cut = x1 <= 2 or x2 >= rgb.shape[1] - 2
            # how unsure this place is: the range finding's own error, plus the robot's pose error carried out to
            # the thing's distance (localisation's, and 50 ms of timing at the current turn rate)
            reach = math.hypot(place.x, place.y)
            heading = self.pose_sigma[1] + abs(self.turn_rate) * 0.05
            sigma = math.sqrt((place.sigma * (2.0 if cut else 1.0)) ** 2 + self.pose_sigma[0] ** 2 + (reach * heading) ** 2)
            detection = Detection(s.label, mx, my, place.radius, colour.name(core), s.score, name, cut, sigma,
                                  look(core) if canonical(s.label) == "person" else None)
            detections.append(detection)
            boxes[id(detection)] = (x1, y1, x2, y2)
            hows[id(detection)] = place.how
        now = time.monotonic()
        with self.lock:
            pairs = self.tracker.update(detections, now)
            matched = {id(t) for _, t in pairs}
            self.boxes[name] = [(boxes[id(det)], track.id) for det, track in pairs]
            self.detected[name] = rgb  # the picture those boxes belong to (the newest may be a later one)
            self.sightings.extend((now, name, track.id, det.x, det.y, hows[id(det)], pose[0], pose[1], gap)
                                  for det, track in pairs)
            for det, track in pairs:  # pairs come sorted by score: find each detection's own box
                if det.cut and track.id in self.crops:
                    continue  # a part of the thing makes a poor picture of it; keep the whole one
                x1, y1, x2, y2 = boxes[id(det)]
                self.crops[track.id] = _only(rgb, (x1, y1, x2, y2), [boxes[id(d)] for d, _ in pairs if d is not det])
            for track in self.tracker.objects():  # negative evidence: viewed clearly, not seen
                if id(track) in matched or not track.present or track.person:
                    continue
                bx, by = to_robot(pose, track.x, track.y)
                dist = math.hypot(bx, by)
                # a clear view: near enough to be detected, and far enough to fit in the picture (beside a chair the
                # front camera sees only legs, and three such misses took the chair off the map)
                fits = dist - track.radius > 0.6 and 2 * math.atan2(track.radius, dist) < 0.6 * cam.hfov
                if not (fits and dist < SEE_FAR and cam.sees(bx, by, margin=0.15)):
                    continue
                lidar = lidar_range(self.scan, math.atan2(by, bx))
                if lidar is not None and lidar < dist - track.radius - 0.3:  # something nearer hides it
                    continue
                self.tracker.missed(track)

    def objects(self) -> list[Track]:
        with self.lock:
            return [replace(t, colors=t.colors.copy(), votes=t.votes.copy()) for t in self.tracker.objects()]
