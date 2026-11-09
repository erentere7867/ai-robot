"""Camera sources. Frames carry the body pose at capture so detections can be placed in the map.

`CameraRig` holds every camera in config/body.yaml; each is a robot/sim/sensors.py render or a
robot/hardware/sensors.py device. Both deliver images in the pinhole model the rest of the code assumes.
"""

from __future__ import annotations

import base64
import struct
import zlib
from dataclasses import dataclass, field
from typing import Protocol

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
    rgb: np.ndarray | None = field(default=None, repr=False)  # raw pixels, for detection and cropping
    _data_url: str | None = field(default=None, init=False, repr=False)
    _thumbnail: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def camera(self) -> str:
        return self.model.name

    @property
    def png_bytes(self) -> bytes:
        if self._png is None:
            self._png = encode_png(fit_width(self.rgb, VLM_MAX_WIDTH)) if self.rgb is not None else b""
        return self._png

    @property
    def thumbnail(self) -> np.ndarray:
        """Small grayscale image for cheap change detection, computed when asked for."""
        if self._thumbnail is None:
            self._thumbnail = make_thumbnail(self.rgb)
        return self._thumbnail

    @property
    def data_url(self) -> str:
        if self._data_url is None:
            self._data_url = "data:image/png;base64," + base64.b64encode(self.png_bytes).decode("ascii")
        return self._data_url


def side_by_side(frames: list[Frame]) -> Frame:
    """Two or more views as one image (for the VLM), left to right; boxes and geometry stay per camera."""
    height = min(f.rgb.shape[0] for f in frames if f.rgb is not None)
    rgb = np.hstack([f.rgb[:height] for f in frames if f.rgb is not None])
    frame = Frame(None, frames[0].pose, frames[0].model, frames[0].captured_at, rgb)
    frame._png = encode_png(fit_width(rgb, 2 * VLM_MAX_WIDTH))
    return frame


def make_thumbnail(rgb: np.ndarray, size: tuple[int, int] = (32, 24)) -> np.ndarray:
    """Block means (not point samples), so tiny render differences don't register as change."""
    h, w, _ = rgb.shape
    gray = rgb.astype(np.float32).mean(axis=2)
    bh, bw = max(1, h // size[1]), max(1, w // size[0])
    ys = (np.arange(size[1]) * h / size[1]).astype(int)
    xs = (np.arange(size[0]) * w / size[0]).astype(int)
    return np.array([[gray[y:y + bh, x:x + bw].mean() for x in xs] for y in ys], dtype=np.float32)


class Camera(Protocol):
    model: CameraModel

    def capture(self) -> Frame: ...

    def close(self) -> None: ...


class CameraRig:
    """All of the robot's cameras. The first is the front (head) camera: the default view."""

    def __init__(self, cameras: list[Camera]) -> None:
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

    def thumbnail(self) -> np.ndarray:
        return self.capture().thumbnail

    def close(self) -> None:
        for cam in self.cameras:
            cam.close()
