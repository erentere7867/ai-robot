"""Camera sources. Frames carry the body pose at capture so detections can be placed in the map."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from robot.body.hal import BodyPose
from robot.perception.geometry import CameraModel


@dataclass
class Frame:
    png_bytes: bytes
    pose: BodyPose
    model: CameraModel
    captured_at: float
    thumbnail: np.ndarray | None = None  # small grayscale image for cheap change detection
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
        from robot.world.sandbox import _encode_rgb_to_png_bytes

        rgb = self.sandbox.render_head_rgb(self._true_pose())
        return Frame(_encode_rgb_to_png_bytes(rgb), self.body.pose(), self.model, self.clock.monotonic(),
                     _thumbnail(rgb))

    def thumbnail(self) -> np.ndarray:
        return _thumbnail(self.sandbox.render_head_rgb(self._true_pose()))
