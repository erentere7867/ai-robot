"""Action executor service: validates, queues, runs, and cancels body commands."""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from collections.abc import Callable
from typing import Any

from robot.actions.registry import CapabilityRegistry
from robot.actions.tools import ToolCallRequest
from robot.body.hal import BodyCommand, BodyHAL, CommandResult
from robot.core.bus import EventBus
from robot.core.events import Event, EventPriority
from robot.core.service import ServiceHealth

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Validates tool requests against CapabilityRegistry and dispatches them to BodyHAL."""

    name = "action_executor"

    def __init__(
        self,
        bus: EventBus,
        body: BodyHAL,
        registry: CapabilityRegistry,
        *,
        default_timeout_s: float = 5.0,
        camera_observer: Callable[[], list[str]] | None = None,
    ) -> None:
        self.bus = bus
        self.body = body
        self.registry = registry
        self.default_timeout_s = default_timeout_s
        self.camera_observer = camera_observer
        self.completed_count: int = 0
        self.rejected_count: int = 0
        self.failed_count: int = 0
        self._sub_ids: list[str] = []
        self._generation = 0
        self._active_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._sub_ids.append(
            self.bus.subscribe("action.request", self._on_action_request, name="executor:req")
        )
        self._sub_ids.append(
            self.bus.subscribe(
                "action.cancel", self._on_action_cancel, name="executor:cancel", immediate=True
            )
        )

    async def stop(self) -> None:
        for sid in self._sub_ids:
            self.bus.unsubscribe(sid)
        self._sub_ids.clear()
        await self.cancel_all()

    async def cancel_all(self, command_id: str | None = None) -> None:
        self._generation += 1
        self.bus.discard_pending("action.request")
        await self.body.cancel(command_id)
        if command_id is None:
            for task in tuple(self._active_tasks):
                task.cancel()

    async def health(self) -> ServiceHealth:
        return ServiceHealth(
            name=self.name,
            healthy=True,
            status="ok",
            details={
                "completed": self.completed_count,
                "rejected": self.rejected_count,
                "failed": self.failed_count,
            },
        )

    async def execute_call(
        self,
        call: ToolCallRequest,
        *,
        correlation_id: str | None = None,
    ) -> CommandResult:
        """Validate and execute a single ToolCallRequest, emitting lifecycle events."""
        cmd_id = uuid.uuid4().hex[:10]
        generation = self._generation
        if call.tool == "stop":
            await self.cancel_all()
            generation = self._generation
        outcome = self.registry.validate_tool_call(call)

        if not outcome.valid:
            self.rejected_count += 1
            msg = outcome.error or f"Invalid tool call '{call.tool}'"
            await self.bus.emit(
                "action.rejected",
                source=self.name,
                payload={
                    "command_id": cmd_id,
                    "tool": call.tool,
                    "args": call.args,
                    "error": msg,
                },
                priority=EventPriority.NORMAL,
                correlation_id=correlation_id,
            )
            return CommandResult(
                command_id=cmd_id,
                tool=call.tool,
                ok=False,
                status="rejected",
                message=msg,
                pose=self.body.pose().to_dict(),
            )

        exec_args = dict(outcome.normalized_args)
        # Preserve requested angle if tilt_head was clamped so SimBody or feedback can report both
        # Keep angle_deg bounded; requested metadata never controls a motor.

        cmd = BodyCommand(
            id=cmd_id,
            tool=outcome.tool,
            args=exec_args,
            timeout_s=self._timeout_for(outcome.tool, exec_args),
        )

        await self.bus.emit(
            "action.started",
            source=self.name,
            payload={
                "command_id": cmd_id,
                "tool": cmd.tool,
                "args": outcome.normalized_args,
                "clamped": outcome.clamped,
                "warnings": outcome.warnings,
            },
            priority=EventPriority.NORMAL,
            correlation_id=correlation_id,
        )

        task = None
        observed: set[str] = set()
        last_pose = self.body.pose().to_dict()
        sampled_pose = None
        rotation = 0.0

        def sample(event: Event) -> None:
            nonlocal last_pose, sampled_pose, rotation
            if event.payload.get("command_id") != cmd_id:
                return
            pose = event.payload["pose"]
            rotation += (pose["heading_deg"] - last_pose["heading_deg"] + 180) % 360 - 180
            last_pose = pose
            moved = (math.hypot(pose["x_mm"] - sampled_pose["x_mm"], pose["y_mm"] - sampled_pose["y_mm"])
                     if sampled_pose else math.inf)
            turned = (abs((pose["heading_deg"] - sampled_pose["heading_deg"] + 180) % 360 - 180)
                      if sampled_pose else math.inf)
            if moved >= 150 or turned >= 30:
                sampled_pose = pose
                observed.update(self.camera_observer())

        observer_id = (self.bus.subscribe("body.pose", sample, immediate=True)
                       if self.camera_observer is not None else None)
        try:
            if generation != self._generation:
                raise asyncio.CancelledError
            task = asyncio.create_task(self.body.execute(cmd), name=f"body:{cmd_id}")
            self._active_tasks.add(task)
            result = await asyncio.wait_for(task, timeout=cmd.timeout_s)
        except asyncio.CancelledError:
            await self.body.cancel(cmd_id)
            if asyncio.current_task().cancelling():
                raise
            result = CommandResult(
                cmd_id, cmd.tool, False, "cancelled", "Action cancelled.", self.body.pose().to_dict()
            )
        except TimeoutError:
            await self.body.cancel(cmd_id)
            result = CommandResult(
                command_id=cmd_id,
                tool=cmd.tool,
                ok=False,
                status="timeout",
                message=f"Action '{cmd.tool}' timed out after {cmd.timeout_s:.1f}s.",
                pose=self.body.pose().to_dict(),
            )
        except Exception as exc:
            logger.exception("Unexpected error executing %s", cmd.tool)
            await self.body.cancel(cmd_id)
            result = CommandResult(
                command_id=cmd_id,
                tool=cmd.tool,
                ok=False,
                status="failed",
                message=f"Action '{cmd.tool}' failed: {exc}",
                pose=self.body.pose().to_dict(),
            )
        finally:
            if observer_id:
                self.bus.unsubscribe(observer_id)
            if task is not None:
                self._active_tasks.discard(task)

        if self.camera_observer is not None:
            observed.update(self.camera_observer())
            result.details.update(observed_objects=sorted(observed)[:32], rotation_deg=round(rotation, 1))

        if result.ok:
            self.completed_count += 1
            await self.bus.emit(
                "action.completed",
                source=self.name,
                payload={
                    "command_id": result.command_id,
                    "tool": result.tool,
                    "status": result.status,
                    "message": result.message,
                    "pose": result.pose,
                    "warnings": outcome.warnings,
                    "details": result.details,
                },
                priority=EventPriority.NORMAL,
                correlation_id=correlation_id,
            )
        else:
            self.failed_count += 1
            await self.bus.emit(
                "action.failed",
                source=self.name,
                payload={
                    "command_id": result.command_id,
                    "tool": result.tool,
                    "status": result.status,
                    "message": result.message,
                    "pose": result.pose,
                },
                priority=EventPriority.HIGH,
                correlation_id=correlation_id,
            )
        return result

    def _timeout_for(self, tool: str, args: dict) -> float:
        caps = self.registry.caps
        if tool == "drive":
            return min(
                30.0,
                max(
                    self.default_timeout_s,
                    args["distance_mm"] / args.get("speed_mm_s", caps.default_linear_speed_mm_s) + 1.0,
                ),
            )
        if tool == "spin":
            return min(
                30.0,
                max(self.default_timeout_s, abs(args["angle_deg"]) / caps.default_angular_speed_deg_s + 1.0),
            )
        if tool == "perform_trick":
            return min(30.0, max(self.default_timeout_s, caps.tricks[args["trick"]].duration_s + 5.0))
        return min(30.0, self.default_timeout_s)

    async def _on_action_request(self, event: Event) -> None:
        tool = str(event.payload.get("tool", ""))
        args: dict[str, Any] = dict(event.payload.get("args") or {})
        await self.execute_call(
            ToolCallRequest(tool=tool, args=args),
            correlation_id=event.correlation_id or event.id,
        )

    async def _on_action_cancel(self, event: Event) -> None:
        cmd_id = event.payload.get("command_id")
        await self.cancel_all(str(cmd_id) if cmd_id else None)
        await self.bus.emit(
            "action.cancelled",
            source=self.name,
            payload={"command_id": cmd_id},
            priority=EventPriority.HIGH,
            correlation_id=event.correlation_id or event.id,
        )
