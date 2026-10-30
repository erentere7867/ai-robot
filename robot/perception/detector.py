"""Open-vocabulary object detection with the vision-language model's grounding ability.

LFM-VL answers a detection request with a JSON list of labels and boxes on a 0..1000 grid.
The detector is model-agnostic: it accepts 0..1 fractions too and validates every box.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass

from robot.cognition.backend import InferenceBackend
from robot.perception.camera import Frame, crop_png

DETECT_ALL_PROMPT = (
    "Detect every distinct physical object in this image (ignore the floor, walls and sky). "
    "Use short common names with a colour, for example \"red ball\" or \"brown cardboard box\". "
    'Reply with only a JSON list of {"label": str, "bbox": [x1, y1, x2, y2]}.'
)
DETECT_QUERY_PROMPT = (
    "Detect every {query} in this image. If there is none, reply []. "
    'Reply with only a JSON list of {{"label": str, "bbox": [x1, y1, x2, y2]}}.'
)

_LABEL_CLEAN = re.compile(r"[^a-z0-9 \-']+")
_OBJECT = re.compile(r"\{[^{}]*\}")
_PAIR = re.compile(r'"(?:label|name)"\s*:\s*"([^"]*)"[^\[{}]*?"(?:bbox|box|bbox_2d)"\s*:\s*\[([^\]\[]*)\]')


@dataclass(frozen=True)
class Detection:
    label: str
    bbox: tuple[float, float, float, float]  # normalized x1, y1, x2, y2

    @property
    def center_u(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def clean_label(label: str) -> str:
    text = _LABEL_CLEAN.sub(" ", str(label).lower()).strip()
    return re.sub(r"\s+", " ", text)[:48]


def parse_detections(text: str, *, max_items: int = 16) -> list[Detection]:
    """Parse the model's JSON detection list defensively."""
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except ValueError:
        # Small models often slip on the syntax ("}}]", "767]]}"); read each object on its own, and as a
        # last resort pull out label/box pairs directly.
        items = []
        for chunk in _OBJECT.findall(match.group(0)):
            try:
                items.append(json.loads(chunk))
            except ValueError:
                continue
        if not items:
            for label, numbers in _PAIR.findall(match.group(0)):
                try:
                    items.append({"label": label, "bbox": [float(v) for v in numbers.split(",")]})
                except ValueError:
                    continue
    if not isinstance(items, list):
        return []
    if items and all(isinstance(v, (int, float)) for v in items) and len(items) == 4:
        items = [{"label": "object", "bbox": items}]
    boxes: list[Detection] = []
    for item in items[: max_items * 2]:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox") or item.get("box") or item.get("bbox_2d")
        label = clean_label(item.get("label") or item.get("name") or "")
        if not label or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            coords = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        scale = 1000.0 if max(coords) > 1.5 else 1.0
        x1, y1, x2, y2 = (max(0.0, min(1.0, v / scale)) for v in coords)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 - x1 < 0.008 or y2 - y1 < 0.008 or (x2 - x1) * (y2 - y1) > 0.9:
            continue
        det = Detection(label, (x1, y1, x2, y2))
        if any(_iou(det.bbox, other.bbox) > 0.7 for other in boxes):
            continue
        boxes.append(det)
        if len(boxes) >= max_items:
            break
    return boxes


class VLMDetector:
    def __init__(self, backend: InferenceBackend, *, max_tokens: int = 320) -> None:
        self.backend = backend
        self.max_tokens = max_tokens
        self.calls = 0

    async def detect(self, frame: Frame, query: str | None = None) -> list[Detection]:
        query = clean_label(query or "")
        prompt = DETECT_QUERY_PROMPT.format(query=query) if query else DETECT_ALL_PROMPT
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": frame.data_url}},
            {"type": "text", "text": prompt},
        ]}]
        self.calls += 1
        result = await self.backend.chat(messages, max_tokens=self.max_tokens)
        detections = parse_detections(result.content)
        if query:
            # "Detect every duck" is a leading question: a small VLM often boxes *something*. Each hit is
            # confirmed on a crop with a neutral yes/no question before the robot believes it.
            detections = [Detection(d.label if d.label != "object" else query, d.bbox) for d in detections[:3]]
            detections = [d for d in detections if await self.verify(frame, d.bbox, query)]
        return detections

    async def verify(self, frame: Frame, bbox: tuple[float, float, float, float], query: str) -> bool:
        if frame.rgb is None:
            return True  # nothing to crop (e.g. a test frame); keep the detection
        url = "data:image/png;base64," + base64.b64encode(crop_png(frame.rgb, bbox)).decode("ascii")
        self.calls += 1
        result = await self.backend.chat([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": f"Is this a {query}? Answer only yes or no."},
        ]}], max_tokens=4)
        return result.content.strip().lower().startswith("yes")
