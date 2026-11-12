"""The open-vocabulary detector: YOLOE-26 with a household vocabulary that grows when a task names something new.

A base vocabulary is always active; `watch()` adds task words ("keys", "elephant") at run time. Labels come back in
the planner's words (tracker.canonical). Runs on the CPU in the simulator; on the robot the same model runs on the
NPU after export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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


class Detector:
    def __init__(self, weights: str = "yoloe-26s-seg.pt", vocabulary: tuple[str, ...] = VOCABULARY,
                 confidence: float = 0.15, threads: int = 0) -> None:  # low: the tracker's two sightings filter noise
        """`threads` caps the CPU threads (0: all), so on a CPU the detector leaves room for the model and the
        simulator. With CUDA it runs on the GPU, in full precision: on the laptop's GTX 1650 (no tensor cores) half
        precision was three times slower (213 against 64 ms for a 1280-pixel frame). On the robot it runs on the NPU."""
        import torch
        from ultralytics import YOLOE

        if threads:
            torch.set_num_threads(threads)
        self.model = YOLOE(weights)
        self.vocabulary = list(dict.fromkeys(vocabulary))
        self.confidence = confidence
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
        seen = [Seen(result.names[int(c)], float(s), tuple(b), f)
                for b, c, s, f in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                      result.boxes.conf.tolist(), feet, strict=True)]
        kept: list[Seen] = []  # one box per thing: the model sometimes boxes a thing twice, or under two names
        for s in sorted(seen, key=lambda s: -s.score):
            if not any(_same_thing(k, s) for k in kept):
                kept.append(s)
        return kept


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
