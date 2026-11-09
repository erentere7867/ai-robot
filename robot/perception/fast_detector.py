"""The fast layer's eyes: a small detector run on every camera frame.

YOLO11n trained on this robot's objects and people (scripts/train_detector.py), run as its NCNN export: no
PyTorch on the robot, the fastest CPU runtime for YOLO on ARM, and with `gpu=True` the GPU through Vulkan
(the VENTUNO Q's Adreno; any Vulkan GPU on a laptop). Anything outside its classes comes out as "object", and
the slow layer (the VLM) can name it when that matters.
"""

from __future__ import annotations

import atexit
import threading
import time
import weakref
from pathlib import Path

import numpy as np

from robot.perception.camera import Frame
from robot.perception.detection import Detection

CATCH_ALL = "object"


def _overlap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection over the smaller box's area."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / smaller if smaller > 0 else 0.0


def suppress_catch_all(dets: list[Detection], threshold: float = 0.6) -> list[Detection]:
    """Drop an "object" box that mostly covers a named one: the same thing seen twice, not a second thing.

    Class-wise NMS keeps both when the detector is unsure whether a duck is a "yellow duck" or just an
    "object", and the map would then hold a phantom next to the real duck.
    """
    named = [d for d in dets if d.label != CATCH_ALL]
    return [d for d in dets if d.label != CATCH_ALL or all(_overlap(d.bbox, n.bbox) < threshold for n in named)]


_GPU_NETS: weakref.WeakSet = weakref.WeakSet()


def _release_gpu() -> None:
    """Vulkan must be torn down after every net that used it, or the interpreter crashes on exit."""
    if not _GPU_NETS:
        return
    import ncnn

    for det in list(_GPU_NETS):
        det._net = None
    ncnn.destroy_gpu_instance()


atexit.register(_release_gpu)


class NcnnDetector:
    """The detector without PyTorch: the NCNN export run directly (what the robot runs).

    Importing ultralytics pulls in PyTorch (~400 MB of RAM and seconds of start-up next to the language model).
    This runs the exported graph with ncnn, letterboxes exactly as ultralytics does, and decodes the boxes and
    class scores itself. `gpu` runs it through Vulkan when a GPU is there (falls back to the CPU otherwise).
    """

    def __init__(self, model_dir: Path | str, *, conf: float = 0.4, iou: float = 0.5, threads: int = 4,
                 gpu: bool = False) -> None:
        import ncnn
        import yaml

        self.model_dir = Path(model_dir)
        meta = yaml.safe_load((self.model_dir / "metadata.yaml").read_text(encoding="utf-8"))
        names = meta["names"]
        self.labels = [str(names[i]) for i in sorted(names)]
        self.vocabulary = [label for label in self.labels if label != CATCH_ALL]
        imgsz = meta.get("imgsz", [320, 320])
        self.size = int(imgsz[0] if isinstance(imgsz, list) else imgsz)
        self.conf, self.iou = conf, iou
        self._net = ncnn.Net()
        self.gpu = bool(gpu) and ncnn.get_gpu_count() > 0
        self._net.opt.use_vulkan_compute = self.gpu
        self._net.opt.num_threads = threads
        if self.gpu:
            _GPU_NETS.add(self)
        self._net.load_param(str(self.model_dir / "model.ncnn.param"))
        self._net.load_model(str(self.model_dir / "model.ncnn.bin"))
        self._ncnn = ncnn
        self._lock = threading.Lock()
        self.last_ms = 0.0

    def detect_rgb(self, rgb: np.ndarray) -> list[Detection]:
        import cv2

        h, w = rgb.shape[:2]
        scale = min(self.size / w, self.size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        pad_x, pad_y = (self.size - nw) / 2.0, (self.size - nh) / 2.0
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
        canvas[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        with self._lock:
            started = time.perf_counter()
            mat = self._ncnn.Mat.from_pixels(np.ascontiguousarray(canvas), self._ncnn.Mat.PixelType.PIXEL_RGB,
                                             self.size, self.size)
            mat.substract_mean_normalize([], [1 / 255.0] * 3)
            ex = self._net.create_extractor()
            ex.input("in0", mat)
            _, out = ex.extract("out0")
            pred = np.array(out)  # (4 + classes, anchors): cx, cy, w, h in input pixels, then class scores
            self.last_ms = 1000.0 * (time.perf_counter() - started)
        scores = pred[4:]
        cls = scores.argmax(axis=0)
        best = scores[cls, np.arange(scores.shape[1])]
        keep = best >= self.conf
        if not keep.any():
            return []
        cx, cy, bw, bh = pred[0, keep], pred[1, keep], pred[2, keep], pred[3, keep]
        boxes = np.stack([(cx - bw / 2 - left) / scale, (cy - bh / 2 - top) / scale, bw / scale, bh / scale], axis=1)
        conf, cls = best[keep], cls[keep]
        out_dets = []
        for c in np.unique(cls):  # class-wise NMS, as ultralytics does
            idx = np.nonzero(cls == c)[0]
            chosen = cv2.dnn.NMSBoxes(boxes[idx].tolist(), conf[idx].tolist(), self.conf, self.iou)
            for k in np.array(chosen).reshape(-1):
                x, y, bw_, bh_ = (float(v) for v in boxes[idx[k]])
                out_dets.append(Detection(self.labels[int(c)], (max(0.0, x / w), max(0.0, y / h),
                                                                 min(1.0, (x + bw_) / w), min(1.0, (y + bh_) / h)),
                                          float(conf[idx[k]])))
        return suppress_catch_all(sorted(out_dets, key=lambda d: -d.score))

    def detect_frame(self, frame: Frame) -> list[Detection]:
        if frame.rgb is None:
            return []
        return self.detect_rgb(frame.rgb)
