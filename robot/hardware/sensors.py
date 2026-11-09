"""The physical senses: cameras through the VENTUNO Q's ISP (or USB) and an LDROBOT LD19 / LD06 lidar.

The lidar is mounted on the mast above the head, so it sees walls, furniture and people's legs but not toys on
the floor (the cameras and the front range sensor cover those).
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from robot.core.angles import wrap_deg
from robot.perception.camera import Frame
from robot.perception.geometry import CameraModel
from robot.perception.lidar import Scan


class RealCamera:
    """A physical camera: a MIPI camera on the VENTUNO Q (GStreamer through OpenCV, `source="csi"`) or a USB
    camera (`source="usb"`), rectified to `model` when a calibration file exists.

    Capture runs in a background thread that keeps only the newest image, so a read never waits for the
    sensor and never returns a stale frame from a buffer queue.
    """

    def __init__(self, body: Any, clock: Any, model: CameraModel, *, source: str = "csi",
                 calibration_dir: Path | None = None) -> None:
        import cv2

        self.body, self.clock, self.model = body, clock, model
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        w, h = model.width, model.height
        if source == "csi":
            # MIPI camera through the SoC's ISP: Qualcomm's camera source first, libcamera as the fallback.
            # appsink keeps only the newest frame.
            tail = "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
            for src in (f"qtiqmmfsrc camera={model.device} ! video/x-raw,format=NV12,width=1280,height=720,"
                        "framerate=30/1 ! ", f"libcamerasrc camera-name={model.device} ! "):
                self._cam = cv2.VideoCapture(src + tail, cv2.CAP_GSTREAMER)
                if self._cam.isOpened():
                    break
        else:
            self._cam = cv2.VideoCapture(model.device)
            self._cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._maps = None
        calib = (calibration_dir or Path("~/robot-runtime/calibration").expanduser()) / f"{model.name}.npz"
        if calib.exists():
            data = np.load(calib)
            f = model.focal_px
            target = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
            size = tuple(int(v) for v in data["size"])  # the resolution it was calibrated at
            self._maps = cv2.fisheye.initUndistortRectifyMap(data["K"], data["D"], np.eye(3), target, (w, h),
                                                             cv2.CV_16SC2)
            self._calib_size = size

        def read() -> np.ndarray | None:
            ok, bgr = self._cam.read()
            if not ok:
                return None
            if self._maps is not None:
                if (bgr.shape[1], bgr.shape[0]) != self._calib_size:
                    bgr = cv2.resize(bgr, self._calib_size, interpolation=cv2.INTER_AREA)
                bgr = cv2.remap(bgr, *self._maps, interpolation=cv2.INTER_LINEAR)
            elif bgr.shape[1] != w or bgr.shape[0] != h:
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            return bgr[:, :, ::-1]

        self._read = read
        self._thread = threading.Thread(target=self._pump, name=f"camera-{model.name}", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        while not self._stop.is_set():
            try:
                rgb = self._read()
            except Exception:
                rgb = None
            if rgb is not None:
                with self._lock:
                    self._latest = np.ascontiguousarray(rgb)

    def _newest(self) -> np.ndarray:
        for _ in range(200):
            with self._lock:
                if self._latest is not None:
                    return self._latest
            self._stop.wait(0.01)
        raise RuntimeError(f"Camera {self.model.name} is not delivering frames")

    def capture(self) -> Frame:
        pose = self.body.pose()
        rgb = self._newest()
        return Frame(None, pose, self.model, self.clock.monotonic(), rgb)

    def close(self) -> None:
        self._stop.set()
        try:
            self._cam.release()
        except Exception:
            pass


def _crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x4D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


class Ld19Lidar:
    """LDROBOT LD19 / LD06 over a USB-UART. One reader thread assembles packets into full turns."""

    PACKET = 47

    def __init__(self, body: Any, clock: Any, *, port: str, mount_x_mm: float, baud: int = 230400,
                 mount_yaw_deg: float = 0.0, max_range_mm: float = 8000.0) -> None:
        import serial

        self.body, self.clock = body, clock
        self.mount_x, self.mount_yaw = mount_x_mm, mount_yaw_deg
        self.max_range = max_range_mm
        self._serial = serial.Serial(port, baud, timeout=0.1)
        self._lock = threading.Lock()
        self._turn: list[tuple[float, float]] = []
        self._last_angle = -1.0
        self._latest: Scan | None = None
        self._running = True
        self._thread = threading.Thread(target=self._read, name="lidar", daemon=True)
        self._thread.start()

    def latest(self) -> Scan | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._serial.close()

    def _read(self) -> None:
        buf = bytearray()
        while self._running:
            buf += self._serial.read(512)
            while len(buf) >= self.PACKET:
                start = buf.find(b"\x54\x2c")
                if start < 0:
                    buf.clear()
                    break
                if len(buf) - start < self.PACKET:
                    del buf[:start]
                    break
                packet = bytes(buf[start:start + self.PACKET])
                del buf[:start + self.PACKET]
                if _crc8(packet[:-1]) == packet[-1]:
                    self._on_packet(packet)

    def _on_packet(self, p: bytes) -> None:
        start = int.from_bytes(p[4:6], "little") / 100.0
        end = int.from_bytes(p[42:44], "little") / 100.0
        span = (end - start) % 360.0
        for k in range(12):
            off = 6 + 3 * k
            dist, conf = int.from_bytes(p[off:off + 2], "little"), p[off + 2]
            angle = (start + span * k / 11.0) % 360.0
            if angle < self._last_angle:  # wrapped: one full turn is done
                self._publish()
            self._last_angle = angle
            # The LD19 turns clockwise; robot angles are anticlockwise (+ = left).
            rel = wrap_deg(self.mount_yaw - angle)
            self._turn.append((rel, float(dist) if dist > 0 and conf > 20 else math.inf))

    def _publish(self) -> None:
        if len(self._turn) > 100:
            angles = np.array([a for a, _ in self._turn])
            ranges = np.array([r for _, r in self._turn])
            ranges = np.where(ranges < self.max_range, ranges, np.inf)
            with self._lock:
                self._latest = Scan(self.body.pose(), angles, ranges, time.monotonic(), self.mount_x)
        self._turn = []
