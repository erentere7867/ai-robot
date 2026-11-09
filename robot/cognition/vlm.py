"""Open-vocabulary detection with the vision-language model's grounding ability.

LFM-VL answers a detection request with a JSON list of labels and boxes on a 0..1000 grid; anything the fast
detector has no class for ("my keys") is found here, more slowly.
"""

from __future__ import annotations

import base64

from robot.cognition.backend import BackendUnavailable, InferenceBackend
from robot.perception.camera import Frame, crop_png
from robot.perception.detection import Detection, clean_label, parse_detections

NAME_PROMPT = "What is this object? Answer with two or three words, such as \"orange traffic cone\"."
DETECT_ALL_PROMPT = (
    "Detect every distinct physical object in this image (ignore the floor, walls and sky). "
    "Use short common names with a colour, for example \"red ball\" or \"brown cardboard box\". "
    'Reply with only a JSON list of {"label": str, "bbox": [x1, y1, x2, y2]}.'
)
DETECT_QUERY_PROMPT = (
    "Detect every {query} in this image. If there is none, reply []. "
    'Reply with only a JSON list of {{"label": str, "bbox": [x1, y1, x2, y2]}}.'
)


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

    async def name_object(self, frame: Frame, bbox: tuple[float, float, float, float]) -> str | None:
        """What is this thing the fast detector only knows as "object"? None when the model cannot say."""
        if frame.rgb is None:
            return None
        url = "data:image/png;base64," + base64.b64encode(crop_png(frame.rgb, bbox)).decode("ascii")
        try:
            reply = await self.backend.chat([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": NAME_PROMPT},
            ]}], max_tokens=12)
        except BackendUnavailable:
            return None
        name = " ".join(reply.content.strip().strip(".\"'").lower().split()[:4])
        return name if name and len(name) < 40 else None

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
