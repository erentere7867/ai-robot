"""The fast layer's eyes: a small detector run on every camera frame.

YOLO11n trained on this robot's objects (scripts/train_detector.py). On the laptop it runs the PyTorch
weights (CUDA if available); on a Raspberry Pi 5 it runs the NCNN export, the fastest CPU runtime for
YOLO on ARM. Anything outside its classes comes out as "object", and the slow layer (the VLM) can name
it when that matters.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from robot.perception.camera import Frame
from robot.perception.detector import Detection

logger = logging.getLogger(__name__)


class FastDetector:
    def __init__(self, weights: Path | str, *, imgsz: int = 320, conf: float = 0.4, device: str | None = None
                 ) -> None:
        from ultralytics import YOLO

        self.weights = Path(weights)
        self.imgsz = imgsz
        self.conf = conf
        self._model = YOLO(str(self.weights), task="detect")
        self.device = device
        if device is None and self.weights.suffix == ".pt":
            try:
                import torch

                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        self.labels = [str(self._model.names[i]) for i in sorted(self._model.names)]
        self.vocabulary = [label for label in self.labels if label != "object"]  # what it can search for
        self._lock = threading.Lock()  # one inference at a time; the model is not re-entrant
        self.last_ms = 0.0
        self.warm()

    @classmethod
    def load_default(cls, models_dir: Path, *, prefer_ncnn: bool = False, **kwargs) -> FastDetector | None:
        """The trained detector from ~/robot-runtime/models, or None if it has not been trained yet."""
        candidates = [models_dir / "fast_detector_ncnn_model", models_dir / "fast_detector.pt"]
        if not prefer_ncnn:
            candidates.reverse()
        for path in candidates:
            if path.exists():
                try:
                    if path.is_dir():  # NCNN export: run it without PyTorch
                        return NcnnDetector(path, **{k: v for k, v in kwargs.items() if k != "imgsz"})
                    return cls(path, **kwargs)
                except Exception:
                    logger.warning("Could not load the fast detector from %s", path, exc_info=True)
        return None

    def warm(self) -> None:
        self.detect_rgb(np.zeros((240, 320, 3), dtype=np.uint8))

    def detect_rgb(self, rgb: np.ndarray) -> list[Detection]:
        """Detections with normalized boxes for an (H, W, 3) RGB image. Thread-safe; blocking."""
        h, w = rgb.shape[:2]
        with self._lock:
            started = time.perf_counter()
            result = self._model.predict(np.ascontiguousarray(rgb[:, :, ::-1]), imgsz=self.imgsz, conf=self.conf,
                                         device=self.device, verbose=False)[0]
            self.last_ms = 1000.0 * (time.perf_counter() - started)
        boxes = result.boxes
        out = []
        for (x1, y1, x2, y2), cls, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(),
                                                 strict=True):
            out.append(Detection(self.labels[int(cls)], (max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w),
                                                          min(1.0, y2 / h)), float(score)))
        return out

    def detect_frame(self, frame: Frame) -> list[Detection]:
        if frame.rgb is None:
            return []
        return self.detect_rgb(frame.rgb)


class NcnnDetector:
    """The same detector without PyTorch: the NCNN export run directly (for the Raspberry Pi).

    Importing ultralytics pulls in PyTorch (~400 MB of RAM and seconds of start-up on a Pi that also holds
    the language model). This runs the exported graph with ncnn, letterboxes exactly as ultralytics does,
    and decodes the boxes and class scores itself.
    """

    def __init__(self, model_dir: Path | str, *, conf: float = 0.4, iou: float = 0.5, threads: int = 4,
                 **_: object) -> None:
        import ncnn
        import yaml

        self.model_dir = Path(model_dir)
        meta = yaml.safe_load((self.model_dir / "metadata.yaml").read_text(encoding="utf-8"))
        names = meta["names"]
        self.labels = [str(names[i]) for i in sorted(names)]
        self.vocabulary = [label for label in self.labels if label != "object"]
        imgsz = meta.get("imgsz", [320, 320])
        self.size = int(imgsz[0] if isinstance(imgsz, list) else imgsz)
        self.conf, self.iou = conf, iou
        self._net = ncnn.Net()
        self._net.opt.use_vulkan_compute = False
        self._net.opt.num_threads = threads
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
        return sorted(out_dets, key=lambda d: -d.score)

    def detect_frame(self, frame: Frame) -> list[Detection]:
        if frame.rgb is None:
            return []
        return self.detect_rgb(frame.rgb)
