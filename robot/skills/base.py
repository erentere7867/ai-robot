"""Skill framework: general capabilities the language model invokes as tools.

A skill says what to do (go somewhere, look, express) and starts closed-loop behaviours in the fast layer to do
it. It never sees the words of the request; those are for robot/cognition/repair.py.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from robot.body.hal import BodyCapabilities, BodyPose
from robot.control.behaviors import Done, Turn
from robot.core.angles import wrap_deg


class SkillError(Exception):
    """Invalid arguments or an unmet precondition; reported to the model, not raised to users."""


@dataclass
class SkillResult:
    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    status: str = ""

    def to_model(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": self.ok, "result": self.message}
        out.update({k: v for k, v in self.data.items() if not k.startswith("_")})
        return out


@dataclass
class SkillContext:
    """Everything a skill may touch. Built once by the harness; `begin_turn` resets the per-turn fields."""

    bus: Any
    body: Any
    camera: Any
    vlm: Any  # the open-vocabulary eye (robot.cognition.vlm.VLMDetector)
    spatial_map: Any
    planner: Any
    memory: Any
    emotion: Any
    clock: Any
    capabilities: BodyCapabilities
    fast: Any = None  # the FastLoop: continuous perception + behaviours
    correlation_id: str | None = None
    attached_image: Any = None  # latest Frame for the model to look at
    cancelled: Callable[[], bool] = lambda: False
    on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
    wake_request_s: float | None = None
    expression: dict[str, Any] = field(default_factory=dict)
    turn_cache: dict[str, Any] = field(default_factory=dict)
    last_look: Any = None  # pose key of the last observe, kept across turns

    def begin_turn(self, correlation_id: str, cancelled: Callable[[], bool]) -> None:
        self.correlation_id, self.cancelled = correlation_id, cancelled
        self.attached_image, self.wake_request_s = None, None
        self.expression, self.turn_cache = {}, {}

    def pose(self) -> BodyPose:
        return self.body.pose()

    def check(self) -> None:
        if self.cancelled():
            raise asyncio.CancelledError

    async def behave(self, behavior: Any) -> Done:
        """Run a fast-layer behaviour to completion (cancellable like any other step of the turn)."""
        self.check()
        done = await self.fast.run(behavior)
        self.check()
        return done

    async def turn_by(self, angle_deg: float) -> Done | None:
        angle = wrap_deg(angle_deg) if abs(angle_deg) < 359.0 else angle_deg
        if abs(angle) < 2.0:
            return None
        return await self.behave(Turn(round(angle, 1)))

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.on_event is not None:
            await self.on_event(event_type, payload)


SkillFn = Callable[..., Awaitable[SkillResult]]


@dataclass
class Skill:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: SkillFn
    required: list[str] = field(default_factory=list)
    acts: bool = False  # moves the body or looks: what a reply claiming an action needs behind it
    persistent: bool = False  # keeps running after the reply; anything after it in the same turn would end it
    ends_part: bool = False  # its success finishes one part of a several-part request
    report_is_answer: bool = False  # its own result is the reply: no closing sentence from the model
    idempotent: bool = False  # a second run from the same spot can only repeat the first

    def tool_schema(self) -> dict[str, Any]:
        # "aliases" is harness-side tolerance for near-miss words; the model only sees the clean schema.
        properties = {k: {kk: vv for kk, vv in v.items() if kk != "aliases"} for k, v in self.parameters.items()}
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": properties, "required": self.required,
                           "additionalProperties": False},
        }}


def _coerce(name: str, spec: dict[str, Any], value: Any) -> Any:
    kind = spec.get("type")
    if kind == "number":
        if isinstance(value, bool):
            raise SkillError(f"{name} must be a number")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise SkillError(f"{name} must be a number") from exc
        if not math.isfinite(number):
            raise SkillError(f"{name} must be finite")
        lo, hi = spec.get("minimum"), spec.get("maximum")
        if lo is not None:
            number = max(float(lo), number)
        if hi is not None:
            number = min(float(hi), number)
        return number
    if kind == "string":
        text = str(value).strip()[:120]
        options = spec.get("enum")
        if options:
            lowered = text.lower().replace(" ", "_").replace("-", "_")
            lowered = spec.get("aliases", {}).get(lowered, lowered)
            if lowered not in options:
                raise SkillError(f"{name} must be one of {options}")
            return lowered
        return text
    return value


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._skills)

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [s.tool_schema() for s in self._skills.values()]

    def names_where(self, flag: str) -> set[str]:
        return {s.name for s in self._skills.values() if getattr(s, flag)}

    async def run(self, name: str, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        skill = self._skills.get(name)
        if skill is None:
            return SkillResult(False, f"There is no tool named '{name}'. Available: {', '.join(self.names)}.",
                               status="rejected")
        clean: dict[str, Any] = {}
        try:
            for key, value in (args or {}).items():
                if key not in skill.parameters or value is None:
                    continue  # tolerate stray arguments from a small model
                clean[key] = _coerce(key, skill.parameters[key], value)
            missing = [k for k in skill.required if k not in clean]
            if missing:
                raise SkillError(f"Missing argument(s): {', '.join(missing)}")
            result = await skill.fn(ctx, **clean)
        except SkillError as exc:
            return SkillResult(False, str(exc), status="rejected")
        if not result.status:
            result.status = "completed" if result.ok else "failed"
        return result
