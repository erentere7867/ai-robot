"""The open-vocabulary detector: YOLOE-26 with a household vocabulary that grows when a task names something new.

A base vocabulary is always active; `watch()` adds task words ("keys", "elephant") at run time. Labels come back in
the planner's words (tracker.canonical). Runs on the CPU in the simulator; on the robot the same model runs on the
NPU after export.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        result = self.model.predict(rgb[:, :, ::-1], conf=self.confidence, imgsz=size, verbose=False)[0]  # BGR in
        if result.boxes is None:
            return []
        seen = [Seen(result.names[int(c)], float(s), tuple(b))
                for b, c, s in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                   strict=True)]
        kept: list[Seen] = []  # one box per thing: the model sometimes boxes a thing twice, or under two names
        for s in sorted(seen, key=lambda s: -s.score):
            if not any(_same_thing(k.box, s.box) for k in kept):
                kept.append(s)
        return kept


def _same_thing(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Boxes of about the same size, one mostly inside the other (a cup's box inside a table's is another thing)."""
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    if min(area_a, area_b) <= 0 or max(area_a, area_b) > 2.5 * min(area_a, area_b):
        return False
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy > 0.7 * min(area_a, area_b)
