"""The vision server: YOLOE (open-vocabulary detection) and Depth Anything (depth) through ONNX Runtime, for the
robot's software. On the development PC it runs on the Windows GPU with DirectML (tools/win/vision.ps1), as WSL has no
GPU compute for its RX 6600; there YOLOE takes 75 ms a frame against about 400 ms on the CPU. On a Linux machine it
runs on the CPU or an NVIDIA GPU (tools/linux/vision.sh, `--provider`). On the board the same ONNX files are meant
for its NPU.

Requests and answers are numpy archives (np.savez) over HTTP on 127.0.0.1:
  POST /detect?conf=0.22&n=3   picture uint8 720x1280x3 RGB, words float32 1xSLOTSx512 (the first n are real)
       -> boxes Mx4 (frame pixels x1 y1 x2 y2), scores M, word M (index into the words), and each box's mask as
          probabilities at the prototypes' quarter resolution: mask (flat float16), mask_shape Mx2, mask_box Mx4
          (the frame pixels the patch covers)
  POST /depth                   picture uint8 364x644x3 RGB -> inverse float16 364x644 (relative inverse depth)
  GET  /health

Each word is scored on its own (YOLOE gives every box an independent score per word), so a box is kept under every
word that scores it above `conf`, with non-max suppression within each word: asking for "plant" this way found 71%
of plants against 76% for "potted plant" in a 53-word list where each box keeps only its best word. Needs only numpy
and onnxruntime (onnxruntime-directml on Windows):  python vision_server.py --models <repo>/models/vision
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import onnxruntime as ort

ROWS, COLS, PAD = 736, 1280, 8  # the model's input; a 720-row frame sits on rows 8..727 of grey 114, as ultralytics
IOU, MAX_DET = 0.7, 100         # pads it
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # Depth Anything's input normalisation (ImageNet)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_DEPTH_SCALE = (1.0 / (255.0 * STD)).astype(np.float32)
_DEPTH_BIAS = (-MEAN / STD).astype(np.float32)
_BOX_MAX = np.array([COLS, ROWS - 2 * PAD, COLS, ROWS - 2 * PAD], np.float32)


def session(path: Path, provider: str, shape: tuple[int, ...] | None = None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.enable_mem_pattern = False  # DirectML wants these two
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if shape is not None:  # open input sizes fixed: DirectML runs about 3x faster with them known
        probe = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for name, size in zip(probe.get_inputs()[0].shape, shape, strict=True):
            if isinstance(name, str):
                options.add_free_dimension_override_by_name(name, size)
    return ort.InferenceSession(str(path), options, providers=[provider])


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = IOU, limit: int = MAX_DET) -> np.ndarray:
    order, keep = np.argsort(-scores), []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size and len(keep) < limit:
        i, order = order[0], order[1:]
        keep.append(i)
        xx1, yy1 = np.maximum(boxes[i, 0], boxes[order, 0]), np.maximum(boxes[i, 1], boxes[order, 1])
        xx2, yy2 = np.minimum(boxes[i, 2], boxes[order, 2]), np.minimum(boxes[i, 3], boxes[order, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = order[inter / (area[i] + area[order] - inter + 1e-9) <= iou]
    return np.array(keep, int)


def decode(pred: np.ndarray, protos: np.ndarray, n: int, conf: float) -> dict[str, np.ndarray]:
    """Raw predictions (4 + slots + 32, boxes) and prototypes (32, h, w) -> each real word's boxes, as `detect`
    answers them."""
    slots = pred.shape[0] - 4 - protos.shape[0]
    scores, coefs = pred[4:4 + slots], pred[4 + slots:]
    kept_boxes, kept_scores, kept_words, idx = [], [], [], []
    for j in range(min(n, slots)):
        ok = np.flatnonzero(scores[j] >= conf)
        if not ok.size:
            continue
        cx, cy, w, h = pred[0, ok], pred[1, ok], pred[2, ok], pred[3, ok]
        cand = np.clip(np.stack([cx - w / 2, cy - h / 2 - PAD, cx + w / 2, cy + h / 2 - PAD], 1), 0, _BOX_MAX)
        sel = nms(cand, scores[j, ok])
        keep = ok[sel]
        kept_boxes.append(cand[sel])
        kept_scores.append(scores[j, keep])
        kept_words.append(np.full(len(keep), j, np.int32))
        idx.append(keep)
    keep = np.concatenate(idx) if idx else np.zeros(0, int)
    out = {"boxes": np.concatenate(kept_boxes).astype(np.float32) if idx else np.zeros((0, 4), np.float32),
           "scores": np.concatenate(kept_scores).astype(np.float32) if idx else np.zeros(0, np.float32),
           "word": np.concatenate(kept_words).astype(np.int32) if idx else np.zeros(0, np.int32)}
    scale = COLS / protos.shape[2]  # frame pixels per prototype cell (4)
    patches, shapes, cover = [], [], []
    for b, i in zip(out["boxes"], keep, strict=True):
        c0, c1 = int(b[0] // scale), int(-(-b[2] // scale))
        r0, r1 = int((b[1] + PAD) // scale), int(-(-(b[3] + PAD) // scale))
        c1, r1 = max(c1, c0 + 1), max(r1, r0 + 1)
        region = protos[:, r0:r1, c0:c1]
        logit = np.tensordot(coefs[:, i], region, axes=1)
        patches.append((1 / (1 + np.exp(-logit))).astype(np.float16).ravel())
        shapes.append(logit.shape)
        cover.append((c0 * scale, r0 * scale - PAD, c1 * scale, r1 * scale - PAD))
    out["mask"] = np.concatenate(patches) if patches else np.zeros(0, np.float16)
    out["mask_shape"] = np.array(shapes, np.int32).reshape(-1, 2)
    out["mask_box"] = np.array(cover, np.float32).reshape(-1, 4)
    return out


class Models:
    def __init__(self, folder: Path, provider: str, yoloe: str, depth: str) -> None:
        self.yoloe = session(folder / yoloe, provider)
        self.depth = session(folder / depth, provider, (1, 3, 364, 644))
        self.depth_input = self.depth.get_inputs()[0].name
        self.slots = self.yoloe.get_inputs()[1].shape[1]
        self.gpu = threading.Lock()  # one run on the GPU at a time: in the 3D run, YOLOE and depth at once beside the
        # simulator's rendering and the language model failed once in DirectML ("Unspecified error") and stalled 20 s
        self._detect_buf = np.full((1, 3, ROWS, COLS), 114.0 / 255.0, np.float32)
        self.times: dict[str, list[float]] = {"detect": [], "depth": []}

    def detect(self, picture: np.ndarray, words: np.ndarray, n: int, conf: float) -> dict[str, np.ndarray]:
        if picture.shape != (ROWS - 2 * PAD, COLS, 3) or words.shape != (1, self.slots, 512):
            raise ValueError(f"want picture {(ROWS - 2 * PAD, COLS, 3)} and words (1, {self.slots}, 512), got "
                             f"{picture.shape} and {words.shape}")
        words_f = np.ascontiguousarray(words, np.float32)
        with self.gpu:
            np.multiply(picture.transpose(2, 0, 1), np.float32(1.0 / 255.0), out=self._detect_buf[0, :, PAD:ROWS - PAD])
            t = time.perf_counter()
            pred, protos = self.yoloe.run(None, {"picture": self._detect_buf, "words": words_f})
            self._took("detect", t)
        return decode(pred[0], protos[0], n, conf)

    def run_depth(self, picture: np.ndarray) -> dict[str, np.ndarray]:
        """A picture already at the network's size (uint8, 364x644x3 RGB): normalised here, so a quarter of the
        bytes cross over, and the answer goes back in half precision."""
        x = (picture * _DEPTH_SCALE + _DEPTH_BIAS).transpose(2, 0, 1)[None]
        with self.gpu:
            t = time.perf_counter()
            inverse = self.depth.run(None, {self.depth_input: np.ascontiguousarray(x, np.float32)})[0][0]
            self._took("depth", t)
        return {"inverse": inverse.astype(np.float16)}

    def _took(self, name: str, t: float) -> None:
        self.times[name] = (self.times[name] + [time.perf_counter() - t])[-200:]


def handler(models: Models) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        # connections stay open (every answer has its length): a new connection per request, 16 a second from WSL,
        # now and then took 20-35 s to answer even /health, which never waits for the GPU
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:  # one line per request is noise at 20 a second
            pass

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - the http.server name
            if urlparse(self.path).path != "/health":
                return self._send(404, b"{}", "application/json")
            median = {k: round(1000 * float(np.median(v)), 1) if v else None for k, v in models.times.items()}
            self._send(200, json.dumps({"status": "ok", "slots": models.slots, "median_ms": median,
                                        "providers": models.yoloe.get_providers()}).encode(), "application/json")

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                body = np.load(io.BytesIO(self.rfile.read(int(self.headers["Content-Length"]))))
                if url.path == "/detect":
                    out = models.detect(body["picture"], body["words"], int(query.get("n", models.slots)),
                                        float(query.get("conf", 0.22)))
                elif url.path == "/depth":
                    out = models.run_depth(body["picture"])
                else:
                    return self._send(404, b"{}", "application/json")
            except Exception as exc:  # noqa: BLE001 - the caller gets the reason, the server keeps serving
                return self._send(400, json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode(),
                                  "application/json")
            buf = io.BytesIO()
            np.savez(buf, **out)
            self._send(200, buf.getvalue(), "application/octet-stream")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=str(Path(__file__).resolve().parents[1] / "models" / "vision"))
    parser.add_argument("--yoloe", default="yoloe26l_8.onnx")
    parser.add_argument("--depth", default="depth_small.onnx")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--provider", default="DmlExecutionProvider")
    args = parser.parse_args()
    models = Models(Path(args.models), args.provider, args.yoloe, args.depth)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(models))
    print(f"vision server on 127.0.0.1:{args.port}: {args.yoloe} ({models.slots} word slots) and {args.depth} on "
          f"{models.yoloe.get_providers()[0]}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
