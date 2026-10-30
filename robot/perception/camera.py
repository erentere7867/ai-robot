"""Camera sources. Frames carry the body pose at capture so detections can be placed in the map."""

from __future__ import annotations

import base64
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from robot.body.hal import BodyPose
from robot.perception.geometry import CameraModel


def encode_png(rgb: np.ndarray) -> bytes:
    """Fast pure-Python/zlib PNG encoder for an (H, W, 3) uint8 RGB image (~2 ms)."""
    h, w, _ = rgb.shape
    filter_col = np.zeros((h, 1), dtype=np.uint8)
    scanlines = np.hstack([filter_col, np.ascontiguousarray(rgb).reshape(h, w * 3)]).tobytes()

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(scanlines, level=1)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


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
    png_bytes: bytes
    pose: BodyPose
    model: CameraModel
    captured_at: float
    thumbnail: np.ndarray | None = None  # small grayscale image for cheap change detection
    rgb: np.ndarray | None = field(default=None, repr=False)  # raw pixels, for cropping detections
    _data_url: str | None = field(default=None, init=False, repr=False)

    @property
    def data_url(self) -> str:
        if self._data_url is None:
            self._data_url = "data:image/png;base64," + base64.b64encode(self.png_bytes).decode("ascii")
        return self._data_url


class CameraSource(Protocol):
    model: CameraModel

    def capture(self) -> Frame: ...

    def thumbnail(self) -> np.ndarray: ...


def _thumbnail(rgb: np.ndarray, size: tuple[int, int] = (32, 24)) -> np.ndarray:
    h, w, _ = rgb.shape
    gray = rgb.astype(np.float32).mean(axis=2)
    ys = (np.arange(size[1]) * h / size[1]).astype(int)
    xs = (np.arange(size[0]) * w / size[0]).astype(int)
    # Block means rather than point samples so tiny renders differences don't register.
    bh, bw = max(1, h // size[1]), max(1, w // size[0])
    return np.array([[gray[y:y + bh, x:x + bw].mean() for x in xs] for y in ys], dtype=np.float32)


class SimCamera:
    """Renders the MuJoCo head camera from the body's *true* pose (what a real lens would see).

    Detections are attached to the *estimated* pose, as on hardware where only odometry is known.
    """

    def __init__(self, sandbox: Any, body: Any, clock: Any, model: CameraModel | None = None) -> None:
        self.sandbox = sandbox
        self.body = body
        self.clock = clock
        self.model = model or CameraModel(sandbox.cam_width, sandbox.cam_height)

    def _true_pose(self) -> BodyPose:
        getter = getattr(self.body, "true_pose", None)
        return getter() if callable(getter) else self.body.pose()

    def capture(self) -> Frame:
        rgb = self.sandbox.render_head_rgb(self._true_pose())
        return Frame(encode_png(rgb), self.body.pose(), self.model, self.clock.monotonic(), _thumbnail(rgb), rgb)

    def thumbnail(self) -> np.ndarray:
        return _thumbnail(self.sandbox.render_head_rgb(self._true_pose()))
