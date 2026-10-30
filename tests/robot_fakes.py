"""Shared fakes and builders for the test suite (no model server, no real camera, no audio devices)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np

from robot.cognition.backend import ChatResult, NullBackend, ScriptedBackend
from robot.core.clock import RealClock, ReplayClock
from robot.core.config import RobotConfig
from robot.main import RobotHarness
from robot.perception.camera import Frame
from robot.perception.detector import Detection
from robot.perception.geometry import CameraModel
from robot.world.sandbox import _encode_rgb_to_png_bytes
from robot.world.spatial_map import label_score

TINY_PNG = _encode_rgb_to_png_bytes(np.zeros((4, 4, 3), dtype=np.uint8))


def make_config(tmp_path: Path, **overrides: Any) -> RobotConfig:
    """Default config with every persistent path inside `tmp_path` and no model server."""
    cfg = RobotConfig.from_yaml()
    cfg.database_path = tmp_path / "memory.db"
    cfg.recordings_dir = tmp_path / "recordings"
    cfg.spatial_map_path = tmp_path / "spatial_map.json"
    cfg.record_events = False
    cfg.llm.backend = "null"
    cfg.llm.auto_start_server = False
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_harness(tmp_path: Path, backend: Any = None, *, real: bool = False, dashboard: bool = False,
                 clock: Any = None, fake_perception: bool = True, config: RobotConfig | None = None,
                 **kwargs: Any) -> RobotHarness:
    harness = RobotHarness(
        config or make_config(tmp_path),
        clock=clock or (RealClock() if real else ReplayClock()),
        backend=backend if backend is not None else NullBackend(),
        enable_dashboard=dashboard,
        dashboard_port=0,
        tts_mode="warble",
        realtime_audio_sync=False,
        **kwargs,
    )
    harness.face_bridge._send_udp = lambda payload: None
    if fake_perception:
        use_ground_truth_perception(harness)
    return harness


def use_ground_truth_perception(harness: RobotHarness) -> GroundTruthDetector:
    """Swap the MuJoCo render + VLM detector for a cheap camera and a ground-truth detector."""
    detector = GroundTruthDetector(harness.sandbox)
    harness.camera = FakeCamera(harness.body, harness.clock, CameraModel(harness.sandbox.cam_width,
                                                                         harness.sandbox.cam_height))
    harness.skill_context.camera = harness.camera
    harness.skill_context.detector = detector
    harness.detector = detector
    harness.autonomy.camera = harness.camera
    return detector


class FakeCamera:
    """Returns tiny frames carrying the body's pose; no rendering."""

    def __init__(self, body: Any, clock: Any, model: CameraModel | None = None) -> None:
        self.body = body
        self.clock = clock
        self.model = model or CameraModel()
        self.captures = 0
        self.thumb = np.zeros((24, 32), dtype=np.float32)

    def capture(self) -> Frame:
        self.captures += 1
        return Frame(TINY_PNG + bytes([self.captures % 256]), self.body.pose(), self.model,
                     self.clock.monotonic(), self.thumb)

    def thumbnail(self) -> np.ndarray:
        return self.thumb


class GroundTruthDetector:
    """Detections built from the simulator's projected ground-truth boxes (what an ideal VLM would say)."""

    def __init__(self, sandbox: Any, *, labels: dict[str, str] | None = None) -> None:
        self.sandbox = sandbox
        self.labels = labels or {}
        self.calls: list[str | None] = []

    async def detect(self, frame: Frame, query: str | None = None) -> list[Detection]:
        self.calls.append(query)
        w, h = self.sandbox.cam_width, self.sandbox.cam_height
        out = []
        for d in self.sandbox.detect_objects(frame.pose):
            label = self.labels.get(d.id, d.label.lower())
            if query and label_score(label, query) == 0:
                continue
            x1, y1, x2, y2 = d.bbox_px
            out.append(Detection(label, (x1 / w, y1 / h, x2 / w, y2 / h)))
        return out


class FixedDetector:
    """Always returns the same detections (or a queue of lists, one per call)."""

    def __init__(self, *views: list[Detection]) -> None:
        self.views = list(views)
        self.calls: list[str | None] = []

    async def detect(self, frame: Frame, query: str | None = None) -> list[Detection]:
        self.calls.append(query)
        if len(self.views) > 1:
            return list(self.views.pop(0))
        return list(self.views[0]) if self.views else []


class GatedBackend(ScriptedBackend):
    """ScriptedBackend whose first `gated_calls` requests wait for `release` (optionally ignoring cancel)."""

    name = "gated"

    def __init__(self, script: list[Any] | None = None, *, default: Any = "", gated_calls: int = 1,
                 ignore_cancel: bool = False) -> None:
        super().__init__(script, default=default)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.gated_calls = gated_calls
        self.ignore_cancel = ignore_cancel

    async def chat(self, messages, *, tools=None, max_tokens=None) -> ChatResult:
        if tools is not None and self.gated_calls > 0:
            self.gated_calls -= 1
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                if not self.ignore_cancel:
                    raise
                await self.release.wait()
        return await super().chat(messages, tools=tools, max_tokens=max_tokens)


def clear_scene(sandbox: Any, keep: tuple[str, ...] = ()) -> None:
    for obj in list(sandbox.list_objects()):
        if obj["id"] not in keep:
            sandbox.remove_object(obj["id"])


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return str(content)
    return ""


def tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in messages if m.get("role") == "tool"]
