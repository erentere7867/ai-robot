"""Camera sources. Frames carry the body pose at capture so detections can be placed in the map.

`CameraRig` holds every camera in config/body.yaml: in the simulator each is a MuJoCo render, on the robot a
MIPI camera through the VENTUNO Q's ISP (or a USB camera). Wide lenses are rectified to the pinhole model the
rest of the code assumes, from a calibration file (scripts/ventuno/calibrate_camera.py).
"""

from __future__ import annotations

import base64
import struct
import threading
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot.body.hal import BodyPose
from robot.perception.geometry import CameraModel

VLM_MAX_WIDTH = 512  # the VLM reads 512-pixel tiles: a wider image only costs tokens


def encode_png(rgb: np.ndarray) -> bytes:
    """Fast numpy/zlib PNG encoder for an (H, W, 3) uint8 RGB image (~2 ms)."""
    h, w, _ = rgb.shape
    filter_col = np.zeros((h, 1), dtype=np.uint8)
    scanlines = np.hstack([filter_col, np.ascontiguousarray(rgb).reshape(h, w * 3)]).tobytes()

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(scanlines, level=1)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def fit_width(rgb: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale (block average) so the image is at most `max_width` wide."""
    h, w, _ = rgb.shape
    if w <= max_width:
        return rgb
    nh = max(1, round(h * max_width / w))
    ys = (np.arange(nh) * h / nh).astype(int)
    xs = (np.arange(max_width) * w / max_width).astype(int)
    return rgb[ys][:, xs]


def crop_png(rgb: np.ndarray, bbox: tuple[float, float, float, float], margin: float = 0.25) -> bytes:
    """PNG of a normalized box with some context around it (for verifying a detection)."""
    h, w, _ = rgb.shape
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    left, top = int(max(0.0, x1 - mx) * w), int(max(0.0, y1 - my) * h)
    right, bottom = int(min(1.0, x2 + mx) * w), int(min(1.0, y2 + my) * h)
    return encode_png(rgb[top:max(bottom, top + 1), left:max(right, left + 1)].copy())


@dataclass
class Frame:
    # PNG is encoded lazily: the fast detector reads raw pixels every frame, and only the occasional VLM
    # request needs a PNG (encoding costs a few milliseconds of a core per frame).
    _png: bytes | None
    pose: BodyPose
    model: CameraModel
    captured_at: float
    thumbnail: np.ndarray | None = None  # small grayscale image for cheap change detection
    rgb: np.ndarray | None = field(default=None, repr=False)  # raw pixels, for cropping detections
    _data_url: str | None = field(default=None, init=False, repr=False)

    @property
    def camera(self) -> str:
        return self.model.name

    @property
    def png_bytes(self) -> bytes:
        if self._png is None:
            self._png = encode_png(fit_width(self.rgb, VLM_MAX_WIDTH)) if self.rgb is not None else b""
        return self._png

    @property
    def data_url(self) -> str:
        if self._data_url is None:
            self._data_url = "data:image/png;base64," + base64.b64encode(self.png_bytes).decode("ascii")
        return self._data_url


def side_by_side(frames: list[Frame]) -> Frame:
    """Two or more views as one image (for the VLM), left to right; boxes and geometry stay per camera."""
    height = min(f.rgb.shape[0] for f in frames if f.rgb is not None)
    rgb = np.hstack([f.rgb[:height] for f in frames if f.rgb is not None])
    frame = Frame(None, frames[0].pose, frames[0].model, frames[0].captured_at, None, rgb)
    frame._png = encode_png(fit_width(rgb, 2 * VLM_MAX_WIDTH))
    return frame


def _thumbnail(rgb: np.ndarray, size: tuple[int, int] = (32, 24)) -> np.ndarray:
    """Block means (not point samples), so tiny render differences don't register as change."""
    h, w, _ = rgb.shape
    gray = rgb.astype(np.float32).mean(axis=2)
    bh, bw = max(1, h // size[1]), max(1, w // size[0])
    ys = (np.arange(size[1]) * h / size[1]).astype(int)
    xs = (np.arange(size[0]) * w / size[0]).astype(int)
    return np.array([[gray[y:y + bh, x:x + bw].mean() for x in xs] for y in ys], dtype=np.float32)


class SimCamera:
    """One of the robot's cameras rendered in MuJoCo from the body's *true* pose (what a real lens would see).

    Frames are tagged with the *estimated* pose, as on hardware where only odometry is known.
    """

    def __init__(self, sandbox: Any, body: Any, clock: Any, model: CameraModel) -> None:
        self.sandbox, self.body, self.clock, self.model = sandbox, body, clock, model

    def _true_pose(self) -> BodyPose:
        getter = getattr(self.body, "true_pose", None)
        return getter() if callable(getter) else self.body.pose()

    def capture(self) -> Frame:
        pose = self.body.pose()
        rgb = self.sandbox.render_rgb(self.model.name, self._true_pose())
        return Frame(None, pose, self.model, self.clock.monotonic(), _thumbnail(rgb), rgb)


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
        return Frame(None, pose, self.model, self.clock.monotonic(), _thumbnail(rgb), rgb)

    def close(self) -> None:
        self._stop.set()
        try:
            self._cam.release()
        except Exception:
            pass


class CameraRig:
    """All of the robot's cameras. The first is the front (head) camera: the default view."""

    def __init__(self, cameras: list[Any]) -> None:
        if not cameras:
            raise ValueError("a camera rig needs at least one camera")
        self.cameras = list(cameras)
        self._by_name = {c.model.name: c for c in self.cameras}

    @property
    def model(self) -> CameraModel:
        return self.cameras[0].model

    @property
    def models(self) -> list[CameraModel]:
        return [c.model for c in self.cameras]

    def capture(self, name: str | None = None) -> Frame:
        return (self._by_name[name] if name else self.cameras[0]).capture()

    def capture_all(self) -> list[Frame]:
        return [c.capture() for c in self.cameras]

    def thumbnail(self) -> np.ndarray:
        return self.capture().thumbnail

    def close(self) -> None:
        for cam in self.cameras:
            close = getattr(cam, "close", None)
            if callable(close):
                close()
