"""The open-vocabulary detector: YOLOE-26 with a household vocabulary that grows when a task names something new.

A base vocabulary is always active; `watch()` adds task words ("keys", "elephant") at run time. Labels come back in
the planner's words (tracker.canonical). Runs on the CPU in the simulator; on the robot the same model runs on the
NPU after export.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot.perception.tracker import canonical

VOCABULARY = ("person", "chair", "couch", "sofa", "table", "coffee table", "desk", "bed", "tv", "laptop", "cup",
              "mug", "bottle", "backpack", "ball", "box", "cardboard box", "potted plant", "plant", "lamp",
              "trash can", "bin", "cabinet", "refrigerator", "shelf", "book", "shoe", "pillow", "basket", "door",
              "window", "dog", "cat", "toy", "keys", "phone", "remote", "bag", "umbrella", "clock", "vase")


@dataclass
class Seen:
    label: str
    score: float
    box: tuple[float, float, float, float]  # pixels: x1, y1, x2, y2
    foot: tuple[float, float] | None = None  # pixels: where its outline meets the floor (the middle of its lowest rows)
    keypoints: np.ndarray | None = None  # (17, 3) float32 COCO keypoints [u, v, conf] when person pose is available
    mask: np.ndarray | None = None  # bool 2D patch matching box (or full frame) for pixel-accurate 3D localization


class Detector:
    def __init__(self, weights: str = "yoloe-26s-seg.pt", vocabulary: tuple[str, ...] = VOCABULARY,
                 confidence: float = 0.22, threads: int = 0, pose_weights: str = "yolo26n-pose.pt",
                 depth_weights: str = "yolo26n-depth.pt", mono_depth: bool = False) -> None:
        """`threads` caps the CPU threads (0: all), so on a CPU the detector leaves room for the model and the
        simulator. With CUDA it runs on the GPU, in full precision: on the laptop's GTX 1650 (no tensor cores) half
        precision was three times slower (213 against 64 ms for a 1280-pixel frame). On the robot it runs on the NPU."""
        import torch
        from ultralytics import YOLOE

        if threads:
            torch.set_num_threads(threads)
        self.model = YOLOE(_resolve_weights(weights) or weights)
        self.vocabulary = list(dict.fromkeys(vocabulary))
        self.confidence = confidence
        self.pose_weights = pose_weights
        self.depth_weights = depth_weights
        self.mono_depth = mono_depth
        self._pose_model: Any = None
        self._depth_model: Any = None
        self.model.set_classes(self.vocabulary)

    def watch(self, *words: str) -> None:
        """Make new words detectable (a few hundred ms for the text encoder, then free per frame)."""
        new = [w.lower() for w in words if w and w.lower() not in self.vocabulary]
        if new:
            self.vocabulary += new
            self.model.set_classes(self.vocabulary)

    def __call__(self, rgb: np.ndarray) -> list[Seen]:
        """Detections in pixel boxes, at the camera's own resolution: shrinking a wide 1280-pixel frame to the usual
        640 loses far things (a person 4 m away scored 0.13 at 640 and 0.84 at 1280 in the simulator)."""
        size = -(-max(rgb.shape[:2]) // 32) * 32
        result = self.model.predict(rgb[:, :, ::-1], conf=self.confidence, imgsz=size, retina_masks=True,
                                    verbose=False)[0]  # BGR in; masks at the camera's own pixels
        if result.boxes is None:
            return []
        feet = _feet(result.masks.data) if result.masks is not None else [None] * len(result.boxes)
        masks_list: list[np.ndarray | None] = [None] * len(result.boxes)
        if result.masks is not None and len(result.masks.data) == len(result.boxes):
            mdata = result.masks.data
            if hasattr(mdata, "cpu"):
                mdata = mdata.cpu().numpy()
            for idx, b in enumerate(result.boxes.xyxy.tolist()):
                c0, r0 = int(b[0]), int(b[1])
                c1, r1 = max(int(b[2]), c0 + 1), max(int(b[3]), r0 + 1)
                h, w = mdata[idx].shape
                y1_c, y2_c = max(0, min(h, r0)), max(0, min(h, r1))
                x1_c, x2_c = max(0, min(w, c0)), max(0, min(w, c1))
                if y2_c > y1_c and x2_c > x1_c:
                    masks_list[idx] = mdata[idx, y1_c:y2_c, x1_c:x2_c] > 0.5
        seen = [Seen(result.names[int(c)], float(s), tuple(b), f, mask=m)
                for b, c, s, f, m in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                      result.boxes.conf.tolist(), feet, masks_list, strict=True)]
        kept: list[Seen] = []  # one box per thing: the model sometimes boxes a thing twice, or under two names
        for s in sorted(seen, key=lambda s: -s.score):
            if not any(_same_thing(k, s) for k in kept):
                kept.append(s)
        self._attach_poses(rgb, size, kept)
        return kept

    def _attach_poses(self, rgb: np.ndarray, size: int, kept: list[Seen]) -> None:
        people = [s for s in kept if canonical(s.label) == "person" or s.label == "person"]
        if not people:
            return
        if self._pose_model is None:
            self._pose_model = _load_yolo(self.pose_weights)
        if self._pose_model is None:
            return
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

    def estimate_depth(self, rgb: np.ndarray) -> np.ndarray | None:
        """Estimate a float32 monocular depth map sized to `rgb.shape[:2]` with `yolo26n-depth.pt`."""
        if self._depth_model is None:
            self._depth_model = _load_yolo(self.depth_weights)
        if self._depth_model is None:
            return None
        size = -(-max(rgb.shape[:2]) // 32) * 32
        res = self._depth_model.predict(rgb[:, :, ::-1], imgsz=size, verbose=False)[0]
        depth_obj = getattr(res, "depth", None)
        if depth_obj is None:
            return None
        depth = _to_numpy(getattr(depth_obj, "data", depth_obj))
        if depth.ndim == 3:
            depth = depth[0]
        if depth.ndim != 2:
            return None
        h, w = rgb.shape[:2]
        if depth.shape != (h, w):
            dh, dw = depth.shape
            ys = (np.arange(h) * dh) // h
            xs = (np.arange(w) * dw) // w
            depth = depth[np.ix_(ys, xs)]
        return np.asarray(depth, dtype=np.float32)


def _resolve_weights(weights: str | None) -> str | None:
    if not weights:
        return None
    p = Path(weights)
    if p.exists():
        return str(p)
    if not p.is_absolute():
        models_dir = Path(os.environ.get("SPROCKET_MODELS_DIR", "~/sprocket-models")).expanduser()
        candidate = models_dir / p
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


def _feet(masks: Any, rows: int = 3) -> list[tuple[float, float] | None]:
    """Where each thing meets the floor: the middle of its mask's lowest few rows (a person's feet, a chair's front
    legs), as a pixel. Worked out on the device the masks are on; only two numbers a thing come back."""
    import torch

    inside = masks > 0.5  # (things, height, width)
    if inside.shape[0] == 0:
        return []
    filled = inside.any(dim=2)  # which rows hold some of the thing
    height = inside.shape[1]
    lowest = height - 1 - filled.flip(1).int().argmax(dim=1)
    band = filled & (torch.arange(height, device=inside.device)[None, :] >= (lowest - rows + 1)[:, None])
    per_column = (inside & band[:, :, None]).sum(dim=1)  # (things, width)
    count = per_column.sum(dim=1)
    columns = torch.arange(inside.shape[2], device=inside.device, dtype=torch.float32)
    middle = (per_column.float() * columns).sum(dim=1) / count.clamp(min=1)
    return [(u + 0.5, v + 1.0) if n > 0 else None  # pixel centres; the bottom edge of the lowest row
            for u, v, n in zip(middle.tolist(), lowest.tolist(), count.tolist(), strict=True)]


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
