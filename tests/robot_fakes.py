"""Shared fakes and builders for the test suite (no model server, no real camera, no audio devices)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

from robot.body.hal import BodyCapabilities
from robot.cognition.backend import ChatResult, NullBackend, ToolCall
from robot.cognition.repair import repair_call
from robot.core.clock import RealClock, ReplayClock
from robot.core.config import RobotConfig
from robot.main import RobotHarness
from robot.perception.camera import CameraRig, Frame, encode_png
from robot.perception.detection import Detection
from robot.perception.geometry import CameraModel
from robot.platform import Platform
from robot.sim.platform import build_sim
from robot.skills.base import SkillResult
from robot.world.labels import label_score

TINY_PNG = encode_png(np.zeros((4, 4, 3), dtype=np.uint8))


def make_config(tmp_path: Path, **overrides: Any) -> RobotConfig:
    """Default config with every persistent path inside `tmp_path` and no model server."""
    cfg = RobotConfig.from_yaml()
    cfg.database_path = tmp_path / "memory.db"
    cfg.recordings_dir = tmp_path / "recordings"
    cfg.spatial_map_path = tmp_path / "spatial_map.json"
    cfg.record_events = False
    cfg.llm.backend = "null"
    cfg.llm.auto_start_server = False
    cfg.perception.fast_detector = "off"  # hermetic: tests inject the ground-truth detector when they need one
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_harness(tmp_path: Path, backend: Any = None, *, real: bool = False, dashboard: bool = False,
                 clock: Any = None, fake_perception: bool = True, config: RobotConfig | None = None,
                 person: bool = False, **kwargs: Any) -> RobotHarness:
    """A harness on the simulated robot. The room's person (and their voice) only with `person=True`."""
    eye = GroundTruthDetector()

    def platform(config: RobotConfig, bus: Any, clock: Any, capabilities: BodyCapabilities,
                 rig: tuple[CameraModel, ...], localizer: Any) -> Platform:
        built = build_sim(config, bus, clock, capabilities, rig, localizer)
        if fake_perception:
            eye.sandbox = built.sandbox
            built.camera = CameraRig([FakeCamera(built.body, clock, model) for model in rig])
            built.detector = eye
        return built

    harness = RobotHarness(
        config or make_config(tmp_path),
        clock=clock or (RealClock() if real else ReplayClock()),
        backend=backend if backend is not None else NullBackend(),
        platform_factory=platform,
        vlm_detector=eye if fake_perception else None,
        enable_dashboard=dashboard,
        dashboard_port=0,
        tts_mode="warble",
        realtime_audio_sync=False,
        **kwargs,
    )
    harness.face_bridge._send_udp = lambda payload: None
    if not person:
        harness.sandbox.remove_object("person")
    return harness


class FakeCamera:
    """Returns tiny frames carrying the body's pose; no rendering."""

    def __init__(self, body: Any, clock: Any, model: CameraModel | None = None) -> None:
        self.body = body
        self.clock = clock
        self.model = model or CameraModel()
        self.captures = 0

    def capture(self) -> Frame:
        self.captures += 1
        frame = Frame(TINY_PNG + bytes([self.captures % 256]), self.body.pose(), self.model, self.clock.monotonic())
        frame._thumbnail = np.zeros((24, 32), dtype=np.float32)
        return frame

    def close(self) -> None:
        pass


class GroundTruthDetector:
    """Detections built from the simulator's projected ground-truth boxes (what an ideal detector would say)."""

    def __init__(self, sandbox: Any = None, *, labels: dict[str, str] | None = None) -> None:
        self.sandbox = sandbox
        self.labels = labels or {}
        self.calls: list[str | None] = []

    @property
    def vocabulary(self) -> list[str]:
        return [self.labels.get(o["id"], o["label"].lower()) for o in self.sandbox.list_objects()]

    def _boxes(self, frame: Frame, query: str | None) -> list[Detection]:
        out = []
        for d in self.sandbox.detect_objects(frame.pose, frame.model):
            label = self.labels.get(d.id, d.label.lower())
            if query and label_score(label, query) == 0:
                continue
            out.append(Detection(label, d.bbox))
        return out

    def detect_frame(self, frame: Frame) -> list[Detection]:
        self.calls.append(None)
        return self._boxes(frame, None)

    async def detect(self, frame: Frame, query: str | None = None) -> list[Detection]:
        self.calls.append(query)
        return self._boxes(frame, query)

    async def name_object(self, frame: Frame, bbox: tuple[float, float, float, float]) -> str | None:
        return None


class FixedDetector:
    """Always returns the same detections (or a queue of lists, one per call)."""

    vocabulary: list[str] = []

    def __init__(self, *views: list[Detection]) -> None:
        self.views = list(views)
        self.calls: list[str | None] = []

    def detect_frame(self, frame: Frame) -> list[Detection]:
        self.calls.append(None)
        if len(self.views) > 1:
            return list(self.views.pop(0))
        return list(self.views[0]) if self.views else []

    async def detect(self, frame: Frame, query: str | None = None) -> list[Detection]:
        self.calls.append(query)
        if len(self.views) > 1:
            return list(self.views.pop(0))
        return list(self.views[0]) if self.views else []

    async def name_object(self, frame: Frame, bbox: tuple[float, float, float, float]) -> str | None:
        return None


class ScriptedBackend:
    """Deterministic backend for tests: each entry is a reply, or a callable receiving the messages.

    Reply shorthand: a string is plain text; a dict {"call": name, "args": {...}} is one tool call;
    {"calls": [[name, args], ...]} is several; {"say": text} is plain text.
    """

    name = "scripted"

    def __init__(self, script: list[Any] | None = None, *, default: Any = "") -> None:
        self.script = list(script or [])
        self.default = default
        self.requests: list[dict[str, Any]] = []
        self._counter = 0

    async def is_available(self) -> bool:
        return True

    async def chat(self, messages, *, tools=None, max_tokens=None, tool_choice="auto") -> ChatResult:
        self.requests.append({"messages": json.loads(json.dumps(messages, default=str)), "tools": tools,
                              "tool_choice": tool_choice})
        entry = self.script.pop(0) if self.script else self.default
        if callable(entry):
            entry = entry(messages, tools)
        return self._to_result(entry)

    def _to_result(self, entry: Any) -> ChatResult:
        if isinstance(entry, ChatResult):
            return entry
        if isinstance(entry, str):
            return ChatResult(entry, backend_name=self.name)
        calls = []
        pairs = entry.get("calls") or ([[entry["call"], entry.get("args", {})]] if "call" in entry else [])
        for name, args in pairs:
            self._counter += 1
            calls.append(ToolCall(f"call_{self._counter}", name, dict(args)))
        return ChatResult(str(entry.get("say", "")), calls, backend_name=self.name)


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

    async def chat(self, messages, *, tools=None, max_tokens=None, tool_choice="auto") -> ChatResult:
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


async def run_skill(harness: RobotHarness, name: str, **args: Any) -> SkillResult:
    """Call a skill as the agent does: repaired from what the human said (`harness.asked`), if anything."""
    call = repair_call(name, args, getattr(harness, "asked", ""), harness.skill_context)
    if call.refusal:
        return SkillResult(False, call.refusal, status="rejected")
    result = await harness.skills.run(call.tool, call.args, harness.skill_context)
    result.message += call.note
    return result


def clear_scene(sandbox: Any, keep: tuple[str, ...] = (), spatial_map: Any = None) -> None:
    for obj in list(sandbox.list_objects()):
        if obj["id"] not in keep:
            sandbox.remove_object(obj["id"])
    if spatial_map is not None:
        spatial_map.grid.clear_hits()  # range hits on the removed objects would be ghost obstacles


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
