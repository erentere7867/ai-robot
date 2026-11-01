"""Skill framework: closed-loop behaviours the language model invokes as tools.

A skill is a generic capability (observe, go somewhere, turn, express), never a task script.
Skills run motor primitives through the ActionExecutor, so joint limits, reflexes and cancellation
apply exactly as for any other command.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from robot.actions.tools import ToolCallRequest
from robot.body.hal import BodyPose, CommandResult


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
    """Everything a skill may touch. Built once by the harness; per-turn fields are reset by the agent."""

    bus: Any
    executor: Any
    body: Any
    camera: Any
    detector: Any
    spatial_map: Any
    planner: Any
    memory: Any
    emotion: Any
    clock: Any
    registry: Any
    correlation_id: str | None = None
    attached_image: Any = None  # latest Frame for the model to look at
    cancelled: Callable[[], bool] = lambda: False
    on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
    wake_request_s: float | None = None
    expression: dict[str, Any] = field(default_factory=dict)
    turn_cache: dict[str, Any] = field(default_factory=dict)  # per-turn memo, cleared by the arbiter
    request: str = ""  # what the human asked this turn (empty on background turns), set by the agent

    def pose(self) -> BodyPose:
        return self.body.pose()

    def check(self) -> None:
        if self.cancelled():
            raise asyncio.CancelledError

    async def primitive(self, tool: str, **args: Any) -> CommandResult:
        self.check()
        result = await self.executor.execute_call(ToolCallRequest(tool, args), correlation_id=self.correlation_id)
        self.check()
        return result

    async def spin(self, angle_deg: float) -> CommandResult | None:
        angle = (angle_deg + 180.0) % 360.0 - 180.0 if abs(angle_deg) < 359.0 else angle_deg
        if abs(angle) < 2.0:
            return None
        return await self.primitive("spin", angle_deg=round(angle, 1))

    async def drive(self, distance_mm: float) -> CommandResult | None:
        if abs(distance_mm) < 10:
            return None
        return await self.primitive("drive", direction="forward" if distance_mm > 0 else "backward",
                                    distance_mm=round(abs(distance_mm), 1))

    async def face_point(self, x_mm: float, y_mm: float) -> CommandResult | None:
        pose = self.pose()
        bearing = math.degrees(math.atan2(y_mm - pose.y_mm, x_mm - pose.x_mm)) - pose.heading_deg
        return await self.spin(bearing)

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
    background_ok: bool = True

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

    async def run(self, name: str, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        skill = self._skills.get(name)
        if skill is None:
            return SkillResult(False, f"There is no tool named '{name}'. Available: {', '.join(self.names)}.",
                               status="rejected")
        clean: dict[str, Any] = {}
        try:
            for key, value in (args or {}).items():
                if key not in skill.parameters:
                    continue  # tolerate stray arguments from a small model
                if value is None:
                    continue
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
