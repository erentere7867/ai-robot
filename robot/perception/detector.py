"""The open-vocabulary detector: YOLOE-26 looking for the words a task asks for, and for people always.

No fixed household list: in the apartment a 53-word list added about 1.4 wrong names a picture to the model's
object list and to memory (five "balls" remembered in a flat with none), and big furniture, most of what is in view,
was found 0-8% of the time from the robot's 22 cm height anyway. The vision model sees the pictures itself; YOLOE
finds and tracks what a tool asks for ("ball", "keys", "the cat"). Each word is scored on its own, so asking for few
words loses nothing. The words' text features are made here (MobileCLIP, through ultralytics); the network runs on the
GPU vision server (robot/perception/gpu.py), on the board on its NPU.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot import paths
from robot.perception.tracker import canonical

ALWAYS = ("person",)  # looked for all the time: the user, and anyone the robot must not run into


@dataclass
class Seen:
    label: str
    score: float
    box: tuple[float, float, float, float]  # pixels: x1, y1, x2, y2
    foot: tuple[float, float] | None = None  # pixels: where its outline meets the floor (the middle of its lowest rows)
    keypoints: np.ndarray | None = None  # (17, 3) float32 COCO keypoints [u, v, conf] when person pose is available
    mask: np.ndarray | None = None  # bool, the box's size: which of its pixels are the thing


class Detector:
    def __init__(self, server: Any = None, text_weights: str = "yoloe-26l-seg.pt", always: tuple[str, ...] = ALWAYS,
                 confidence: float = 0.22, pose_weights: str = "yolo26n-pose.pt") -> None:
        """`server`: the GPU vision server (a VisionServer; one is made when None). It has a fixed number of word
        slots; the words asked for most recently fill the ones `always` leaves."""
        from robot.perception.gpu import VisionServer

        self.server = server if server is not None else VisionServer()
        self.always = [w.lower() for w in always]
        self.asked: list[str] = []  # newest last
        self.confidence = confidence
        self.text_weights, self.pose_weights = text_weights, pose_weights
        self._features: dict[str, np.ndarray] = {}
        self._text: Any = None
        self._pose_model: Any = None

    @property
    def vocabulary(self) -> list[str]:
        return [*self.always, *self.asked]

    def watch(self, *words: str) -> None:
        """Look for these words from now on, beside the other recent ones (the oldest give way)."""
        for w in (w.lower().strip() for w in words):
            if not w or w in self.always:
                continue
            if w in self.asked:
                self.asked.remove(w)
            self.asked.append(w)
            self._feature(w)  # made now, not on the next frame
        del self.asked[:max(0, len(self.asked) - (self.server.slots - len(self.always)))]

    def _feature(self, word: str) -> np.ndarray:
        if word not in self._features:
            if self._text is None:
                from ultralytics import YOLOE

                with _in_models_dir():  # elsewhere ultralytics downloads its 242 MB text encoder again
                    self._text = YOLOE(_resolve_weights(self.text_weights) or self.text_weights).model
                    self._text.get_text_pe(["person"], cache_clip_model=True, without_reprta=True)
            pe = self._text.get_text_pe([word], cache_clip_model=True, without_reprta=True)
            self._features[word] = pe.float().cpu().numpy().reshape(-1)
        return self._features[word]

    def __call__(self, rgb: np.ndarray) -> list[Seen]:
        """What the words find in one frame, in pixel boxes at the camera's own resolution, one box per thing."""
        words = self.vocabulary
        feats = [self._feature(w) for w in words]
        feats += [feats[0]] * (self.server.slots - len(feats))  # unused slots: filled, and ignored by the server
        out = self.server.detect(rgb, np.stack(feats)[None].astype(np.float32), len(words), self.confidence)
        seen, start = [], 0
        for k in range(len(out["scores"])):
            h, w = (int(v) for v in out["mask_shape"][k])
            patch = out["mask"][start:start + h * w].reshape(h, w)
            start += h * w
            box = tuple(float(v) for v in out["boxes"][k])
            mask = _box_mask(patch.astype(np.float32), out["mask_box"][k], box)
            seen.append(Seen(words[int(out["word"][k])], float(out["scores"][k]), box, _foot(mask, box), mask=mask))
        kept: list[Seen] = []  # one box per thing: a thing can score above the bar for two words ("cup" and "mug")
        for s in sorted(seen, key=lambda s: -s.score):
            if not any(_same_thing(k, s) for k in kept):
                kept.append(s)
        self._attach_poses(rgb, kept)
        return kept

    def _attach_poses(self, rgb: np.ndarray, kept: list[Seen]) -> None:
        people = [s for s in kept if canonical(s.label) == "person"]
        if not people:
            return
        if self._pose_model is None:
            self._pose_model = _load_yolo(self.pose_weights)
        if self._pose_model is None:
            return
        size = -(-max(rgb.shape[:2]) // 32) * 32
        pose_res = self._pose_model.predict(rgb[:, :, ::-1], conf=0.25, imgsz=size, verbose=False)[0]
        if pose_res.boxes is None or pose_res.keypoints is None:
            return
        pose_boxes = _to_numpy(pose_res.boxes.xyxy)
        pose_kpts = _extract_keypoints(pose_res.keypoints)
        if pose_kpts is None or pose_boxes.ndim != 2 or len(pose_boxes) == 0 or len(pose_kpts) == 0:
            return
        count = min(len(pose_boxes), len(pose_kpts))
        for s in people:
            best_idx = max(range(count), key=lambda i: _box_iou(s.box, pose_boxes[i]))
            if _box_iou(s.box, pose_boxes[best_idx]) > 0.2:
                s.keypoints = np.asarray(pose_kpts[best_idx, :, :3], dtype=np.float32)


def _box_mask(patch: np.ndarray, cover: Any, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """The mask over the box's own pixels, from probabilities at quarter resolution covering `cover` (frame pixels):
    scaled up smoothly, then cut to the box and thresholded, as ultralytics does with full-size masks."""
    import cv2

    cx0, cy0, cx1, cy1 = (int(round(float(v))) for v in cover)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    full = cv2.resize(patch, (cx1 - cx0, cy1 - cy0), interpolation=cv2.INTER_LINEAR)
    c0, r0 = int(box[0]), int(box[1])
    c1, r1 = max(int(box[2]), c0 + 1), max(int(box[3]), r0 + 1)
    part = full[max(0, r0 - cy0):max(0, r1 - cy0), max(0, c0 - cx0):max(0, c1 - cx0)]
    return part > 0.5 if part.size else None


def _foot(mask: np.ndarray | None, box: tuple[float, float, float, float], rows: int = 3) -> tuple[float, float] | None:
    """Where a thing meets the floor: the middle of its mask's lowest few rows (a person's feet, a chair's front legs),
    as a frame pixel."""
    if mask is None or not mask.any():
        return None
    filled = np.flatnonzero(mask.any(axis=1))
    lowest = int(filled[-1])
    band = mask[max(0, lowest - rows + 1):lowest + 1]
    per_column = band.sum(axis=0)
    middle = float((per_column * np.arange(mask.shape[1])).sum() / per_column.sum())
    return int(box[0]) + middle + 0.5, int(box[1]) + lowest + 1.0


@contextmanager
def _in_models_dir() -> Iterator[None]:
    here = os.getcwd()
    if paths.MODELS.is_dir():
        os.chdir(paths.MODELS)
    try:
        yield
    finally:
        os.chdir(here)


def _resolve_weights(weights: str | None) -> str | None:
    if not weights:
        return None
    p = Path(weights)
    if p.exists():
        return str(p)
    if not p.is_absolute():
        candidate = paths.MODELS / p
        if candidate.exists():
            return str(candidate)
    return weights


def _load_yolo(weights: str | None) -> Any:
    resolved = _resolve_weights(weights)
    if not resolved:
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    if not Path(resolved).is_file() and getattr(YOLO, "__module__", "").startswith("ultralytics"):
        return None
    return YOLO(resolved)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _extract_keypoints(keypoints_obj: Any) -> np.ndarray | None:
    data = getattr(keypoints_obj, "data", None)
    if data is not None:
        arr = _to_numpy(data)
        return arr if arr.ndim == 3 and arr.shape[-1] >= 3 else None
    xy = getattr(keypoints_obj, "xy", None)
    if xy is not None:
        xy_arr = _to_numpy(xy)
        if xy_arr.ndim != 3:
            return None
        conf = getattr(keypoints_obj, "conf", None)
        if conf is not None:
            conf_arr = _to_numpy(conf)
            if conf_arr.ndim == 2:
                conf_arr = conf_arr[..., None]
        else:
            conf_arr = np.ones((*xy_arr.shape[:2], 1), dtype=np.float32)
        return np.concatenate([xy_arr, conf_arr], axis=-1).astype(np.float32)
    return None


def _box_iou(a: tuple[float, float, float, float], b: Any) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _same_thing(a: Seen, b: Seen) -> bool:
    """One box mostly inside the other: the same thing when they share a name (a person's whole body and upper body,
    which otherwise became two people), or when they are about the same size (a cup's box inside a table's is
    another thing)."""
    (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = a.box, b.box
    area_a, area_b = (ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)
    if min(area_a, area_b) <= 0 or (a.label != b.label and max(area_a, area_b) > 2.5 * min(area_a, area_b)):
        return False
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    return ix * iy > 0.7 * min(area_a, area_b)
