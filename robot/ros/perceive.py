"""Perception on ROS: Sprocket's three cameras and lidar, through YOLOE-26, into one map-frame tracker.

A worker thread takes the newest frame from each camera in turn, runs the detector, places each detection on the map
(depth on the front camera; lidar or the floor on the rear ones; robot pose from TF at the frame's time), and folds
them into the tracker. A camera that clearly views where a known object should be, without seeing it, counts a miss
against it (negative evidence). Each object keeps its latest crop, so the vision model can answer about it; with a
brain, a second thread has the vision model name each new object from its crop (tracker.verify).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import replace
from typing import Any

import numpy as np

from robot.perception import colour
from robot.perception.camera import data_url
from robot.perception.locate import lidar_range, locate
from robot.perception.rig import RIG, to_map, to_robot
from robot.perception.tracker import Detection, Track, Tracker, canonical

TOPICS = {"front_camera": "/rgbd_camera/image", "rear_left_camera": "/rear_left_camera/image",
          "rear_right_camera": "/rear_right_camera/image"}
SEE_FAR = 3.5  # beyond this a miss is not evidence: the detector may simply not resolve it


def _image(msg: Any) -> np.ndarray:
    channels = 3 if msg.encoding in ("rgb8", "bgr8") else 1
    dtype = np.float32 if msg.encoding == "32FC1" else np.uint8
    array = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels) if channels == 3 else \
        np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
    return array[:, :, ::-1] if msg.encoding == "bgr8" else array


class Perception:
    def __init__(self, node: Any, detector: Any, pose_at: Any, brain: Any = None) -> None:
        """`pose_at(stamp)` gives the robot's (x, y, yaw) in the map at a ROS time, or None; `brain` names new
        objects when given."""
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, LaserScan

        self.detector, self.pose_at, self.brain = detector, pose_at, brain
        self.tracker = Tracker()
        self.lock = threading.Lock()
        self.frames: dict[str, tuple[Any, np.ndarray]] = {}
        self.fresh: set[str] = set()
        self.depth: tuple[Any, np.ndarray] | None = None
        self.scan: tuple[np.ndarray, float, float] | None = None
        self.crops: dict[int, np.ndarray] = {}
        self.boxes: dict[str, list[tuple[tuple[int, int, int, int], int]]] = {}  # camera -> latest (box, track id)
        self.timings: list[float] = []
        self.counts: dict[str, int] = {}  # frames processed per camera
        self.served: dict[str, float] = {}  # when each camera's frame was last taken for detection
        for name, topic in TOPICS.items():
            node.create_subscription(Image, topic, lambda m, n=name: self._frame(n, m), qos_profile_sensor_data)
        node.create_subscription(Image, "/rgbd_camera/depth_image", self._depth, qos_profile_sensor_data)
        node.create_subscription(LaserScan, "/scan", self._scan, qos_profile_sensor_data)
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()
        if brain is not None:
            threading.Thread(target=self._verify_loop, daemon=True).start()

    def _frame(self, name: str, msg: Any) -> None:
        self.frames[name] = (msg.header.stamp, _image(msg))
        self.fresh.add(name)

    def _depth(self, msg: Any) -> None:
        self.depth = (msg.header.stamp, _image(msg))

    def _scan(self, msg: Any) -> None:
        self.scan = (np.asarray(msg.ranges, dtype=np.float32), msg.angle_min, msg.angle_increment)

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
                track = next((t for t in sorted(self.tracker.objects(), key=lambda t: t.first_seen)
                              if not t.verified and t.id in self.crops), None)
                crop = self.crops.get(track.id) if track is not None else None
            if track is None:
                time.sleep(0.2)
                continue
            words = sorted({canonical(w) for w in self.detector.vocabulary})
            try:
                label = self.brain.name(data_url(crop), words)
            except Exception as exc:  # noqa: BLE001 - the detector's label stands when the model is unreachable
                print(f"perception: naming {track.label} {track.id}: {exc}", flush=True)
                label = track.label
            with self.lock:
                self.tracker.verify(track, label)

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
        depth = self.depth[1] if cam.depth and self.depth is not None else None
        detections, boxes = [], {}
        for s in seen:
            place = locate(cam, s.box, depth, self.scan)
            if place is None:
                continue
            bx, by, radius = place
            mx, my = to_map(pose, bx, by)
            x1, y1, x2, y2 = (int(v) for v in s.box)
            core = rgb[y1 + (y2 - y1) // 4:y2 - (y2 - y1) // 4, x1 + (x2 - x1) // 4:x2 - (x2 - x1) // 4]
            detection = Detection(s.label, mx, my, radius, colour.name(core), s.score, name)
            detections.append(detection)
            boxes[id(detection)] = (x1, y1, x2, y2)
        now = time.monotonic()
        with self.lock:
            pairs = self.tracker.update(detections, now)
            matched = {id(t) for _, t in pairs}
            self.boxes[name] = [(boxes[id(det)], track.id) for det, track in pairs]
            for det, track in pairs:  # pairs come sorted by score: find each detection's own box
                x1, y1, x2, y2 = boxes[id(det)]
                pad = max(8, (x2 - x1) // 6)
                self.crops[track.id] = rgb[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad].copy()
            for track in self.tracker.objects():  # negative evidence: viewed clearly, not seen
                if id(track) in matched or not track.present or track.person:
                    continue
                bx, by = to_robot(pose, track.x, track.y)
                dist = math.hypot(bx, by)
                if not (0.5 < dist < SEE_FAR and cam.sees(bx, by, margin=0.15)):
                    continue
                lidar = lidar_range(self.scan, math.atan2(by, bx))
                if lidar is not None and lidar < dist - track.radius - 0.3:  # something nearer hides it
                    continue
                self.tracker.missed(track)

    def objects(self) -> list[Track]:
        with self.lock:
            return [replace(t, colors=t.colors.copy(), votes=t.votes.copy()) for t in self.tracker.objects()]
