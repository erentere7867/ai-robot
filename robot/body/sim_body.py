"""Kinematic simulation of the companion_r1 body with safety reflexes."""

from __future__ import annotations

import asyncio
import math

from robot.body.hal import (
    BodyCapabilities,
    BodyCommand,
    BodyPose,
    CommandResult,
    SensorReadings,
)
from robot.core.bus import EventBus
from robot.core.events import EventPriority
from robot.core.service import ServiceHealth


def _normalize_angle_deg(angle: float) -> float:
    """Wrap angle into (-180, 180] degrees."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


class SimBody:
    """Kinematic simulation of 4-wheel skid-steer drive, head tilt, encoders, IMU, and safety reflexes."""

    name = "sim_body"

    def __init__(
        self,
        bus: EventBus,
        capabilities: BodyCapabilities | None = None,
        *,
        step_dt_s: float = 0.05,
    ) -> None:
        self.bus = bus
        self._caps = capabilities or BodyCapabilities()
        if not math.isfinite(step_dt_s) or not 0.001 <= step_dt_s <= 1.0:
            raise ValueError("step_dt_s must be between 0.001 and 1 second")
        self.step_dt_s = step_dt_s
        self._pose = BodyPose()
        self._sensors = SensorReadings(front_tof_mm=self._caps.default_front_tof_mm)
        self._active_cmd_id: str | None = None
        self._cancel_requested: bool = False
        self._cancel_generation = 0
        self._lock = asyncio.Lock()
        self._running = False
        from robot.actions.registry import CapabilityRegistry

        self._registry = CapabilityRegistry(self._caps)
        self._ticks_left = 0.0
        self._ticks_right = 0.0
        self._manual_tof_override = False
        self._manual_cliff_override = False
        self._sensor_model = None

    def attach_sensor_model(self, model) -> None:
        """Sample simulated sensors in the motor loop, independent of event delivery."""
        self._sensor_model = model

    def _refresh_sensors(self) -> None:
        if self._sensor_model is not None:
            self.set_sandbox_sensors(
                front_tof_mm=self._sensor_model.compute_front_tof_mm(self._pose),
                cliff=self._sensor_model.check_cliff_triggered(self._pose),
            )

    def capabilities(self) -> BodyCapabilities:
        return self._caps

    def pose(self) -> BodyPose:
        return BodyPose(
            x_mm=self._pose.x_mm,
            y_mm=self._pose.y_mm,
            heading_deg=self._pose.heading_deg,
            head_tilt_deg=self._pose.head_tilt_deg,
            linear_vel_mm_s=self._pose.linear_vel_mm_s,
            angular_vel_deg_s=self._pose.angular_vel_deg_s,
        )

    def sensors(self) -> SensorReadings:
        self._refresh_sensors()
        return SensorReadings(
            front_tof_mm=self._sensors.front_tof_mm,
            cliff_triggered=self._sensors.cliff_triggered,
            estop_active=self._sensors.estop_active,
            picked_up=self._sensors.picked_up,
            wheel_ticks_left=self._sensors.wheel_ticks_left,
            wheel_ticks_right=self._sensors.wheel_ticks_right,
            imu_yaw_deg=self._sensors.imu_yaw_deg,
            imu_pitch_deg=self._sensors.imu_pitch_deg,
        )

    async def start(self) -> None:
        self._running = True
        await self._emit_pose()

    async def stop(self) -> None:
        await self.cancel()
        self._running = False

    async def health(self) -> ServiceHealth:
        if self._sensors.estop_active:
            return ServiceHealth(
                name=self.name, healthy=True, status="estop_active", details=self._pose.to_dict()
            )
        return ServiceHealth(name=self.name, healthy=True, status="ok", details=self._pose.to_dict())

    def set_estop(self, active: bool) -> None:
        self._sensors.estop_active = active
        if active:
            self._cancel_generation += 1
            self._cancel_requested = True
            self._pose.linear_vel_mm_s = 0.0
            self._pose.angular_vel_deg_s = 0.0

    def set_simulated_obstacle(self, front_tof_mm: float | None = None, cliff: bool | None = None) -> None:
        """Fault-injection helper for testing obstacle and cliff reflexes."""
        if front_tof_mm is not None:
            value = float(front_tof_mm)
            if not math.isfinite(value) or value < 0:
                raise ValueError("Range must be a finite nonnegative number")
            self._sensors.front_tof_mm = value
            self._manual_tof_override = True
        if cliff is not None:
            self._sensors.cliff_triggered = bool(cliff)
            self._manual_cliff_override = True

    def set_sandbox_sensors(self, front_tof_mm: float | None = None, cliff: bool | None = None) -> None:
        """Update raycast sensors from the 3D sandbox unless manually overridden by fault injection."""
        if front_tof_mm is not None and not self._manual_tof_override:
            self._sensors.front_tof_mm = float(front_tof_mm)
        if cliff is not None and not self._manual_cliff_override:
            self._sensors.cliff_triggered = bool(cliff)

    async def cancel(self, command_id: str | None = None) -> None:
        if command_id is None or command_id == self._active_cmd_id:
            self._cancel_generation += 1
            self._cancel_requested = True
            self._pose.linear_vel_mm_s = 0.0
            self._pose.angular_vel_deg_s = 0.0

    async def execute(self, cmd: BodyCommand) -> CommandResult:
        """Execute a validated command on the kinematic model with safety reflex checks."""
        if cmd.tool == "stop":
            await self.cancel()
            await self._emit_pose()
            return CommandResult(
                cmd.id, "stop", True, "completed", "All motion stopped.", self._pose.to_dict()
            )
        from robot.actions.tools import ToolCallRequest

        outcome = self._registry.validate_tool_call(ToolCallRequest(cmd.tool, cmd.args))
        if not outcome.valid:
            return CommandResult(
                cmd.id, cmd.tool, False, "rejected", outcome.error or "Invalid command", self._pose.to_dict()
            )
        args = dict(outcome.normalized_args)
        if "_requested_angle_deg" in cmd.args:
            args["_requested_angle_deg"] = cmd.args["_requested_angle_deg"]
        cmd = BodyCommand(cmd.id, outcome.tool, args, cmd.timeout_s)
        generation = self._cancel_generation
        async with self._lock:
            if generation != self._cancel_generation:
                return self._cancelled_result(cmd)
            self._active_cmd_id = cmd.id
            self._cancel_requested = False
            try:
                await self._emit_pose()
                if self._sensors.estop_active and cmd.tool != "stop":
                    await self._emit_reflex("estop_active", "Emergency stop is active; motion blocked")
                    return CommandResult(
                        command_id=cmd.id,
                        tool=cmd.tool,
                        ok=False,
                        status="reflex_stopped",
                        message="Motion blocked because emergency stop is active.",
                        pose=self._pose.to_dict(),
                    )

                if cmd.tool == "stop":
                    self._pose.linear_vel_mm_s = 0.0
                    self._pose.angular_vel_deg_s = 0.0
                    await self._emit_pose()
                    return CommandResult(
                        command_id=cmd.id,
                        tool=cmd.tool,
                        ok=True,
                        status="completed",
                        message="All motion stopped.",
                        pose=self._pose.to_dict(),
                    )

                if cmd.tool == "tilt_head":
                    return await self._exec_tilt_head(cmd)
                if cmd.tool == "drive":
                    return await self._exec_drive(cmd)
                if cmd.tool == "spin":
                    return await self._exec_spin(cmd)
                if cmd.tool == "perform_trick":
                    return await self._exec_trick(cmd)
                if cmd.tool == "set_gaze":
                    target = str(cmd.args.get("target", "center")).lower()
                    await self.bus.emit(
                        "expression.updated",
                        source=self.name,
                        payload={"gaze_target": target, "reason": "tool:set_gaze"},
                        priority=EventPriority.NORMAL,
                    )
                    return CommandResult(
                        command_id=cmd.id,
                        tool=cmd.tool,
                        ok=True,
                        status="completed",
                        message=f"Set gaze to '{target}'.",
                        pose=self._pose.to_dict(),
                        details={"target": target},
                    )
                if cmd.tool == "set_eye_color":
                    color = str(cmd.args.get("color", "cyan")).lower()
                    await self.bus.emit(
                        "face.theme",
                        source=self.name,
                        payload={"theme": color},
                        priority=EventPriority.HIGH,
                    )
                    return CommandResult(
                        command_id=cmd.id,
                        tool=cmd.tool,
                        ok=True,
                        status="completed",
                        message=f"Set eye color theme to '{color}'.",
                        pose=self._pose.to_dict(),
                        details={"color": color},
                    )

                return CommandResult(
                    command_id=cmd.id,
                    tool=cmd.tool,
                    ok=False,
                    status="failed",
                    message=f"Unknown body tool '{cmd.tool}'.",
                    pose=self._pose.to_dict(),
                )
            finally:
                self._pose.linear_vel_mm_s = 0.0
                self._pose.angular_vel_deg_s = 0.0
                self._active_cmd_id = None
                await self._emit_pose()

    async def _safety_result(
        self, cmd: BodyCommand, *, base: bool = False, forward: bool = False
    ) -> CommandResult | None:
        self._refresh_sensors()
        reason = None
        if self._sensors.estop_active:
            reason = ("estop_active", "Emergency stop is active")
        elif self._cancel_requested:
            return self._cancelled_result(cmd)
        elif base and self._sensors.picked_up:
            reason = ("picked_up", "Robot was picked up")
        elif base and self._sensors.cliff_triggered:
            reason = ("cliff_detected", "Cliff sensor triggered")
        elif forward and (
            not math.isfinite(self._sensors.front_tof_mm)
            or self._sensors.front_tof_mm <= self._caps.min_front_tof_stop_mm
        ):
            reason = ("obstacle_stop", "Front obstacle or invalid range reading")
        if reason:
            await self._emit_reflex(*reason)
            await self._emit_pose()
            return CommandResult(cmd.id, cmd.tool, False, "reflex_stopped", reason[1], self._pose.to_dict())
        return None

    async def _move(self, cmd: BodyCommand, amount: float, speed: float, kind: str) -> CommandResult | None:
        """Integrate at the configured rate, with safety before and after each wait."""
        remaining = abs(amount)
        sign = -1.0 if amount < 0 else 1.0
        while remaining > 1e-8:
            result = await self._safety_result(cmd, base=kind != "head", forward=kind == "drive" and sign > 0)
            if result:
                return result
            step = min(remaining, speed * self.step_dt_s)
            if kind == "drive":
                self._pose.linear_vel_mm_s = sign * speed
            elif kind == "spin":
                self._pose.angular_vel_deg_s = sign * speed
            await self.bus.clock.sleep(step / speed)
            result = await self._safety_result(cmd, base=kind != "head", forward=kind == "drive" and sign > 0)
            if result:
                return result
            if kind == "drive":
                rad = math.radians(self._pose.heading_deg)
                self._pose.x_mm += sign * step * math.cos(rad)
                self._pose.y_mm += sign * step * math.sin(rad)
                self._add_wheel_ticks(sign * step, sign * step)
            elif kind == "spin":
                self._apply_rotation(sign * step)
            else:
                self._pose.head_tilt_deg += sign * step
                self._sensors.imu_pitch_deg = self._pose.head_tilt_deg
            remaining -= step
            await self._emit_pose()
        self._pose.linear_vel_mm_s = 0.0
        self._pose.angular_vel_deg_s = 0.0
        return None

    async def _exec_tilt_head(self, cmd: BodyCommand) -> CommandResult:
        spec = self._caps.joints["head_tilt"]
        requested = float(cmd.args.get("_requested_angle_deg", cmd.args["angle_deg"]))
        target = max(spec.min_deg, min(spec.max_deg, float(cmd.args["angle_deg"])))
        result = await self._move(cmd, target - self._pose.head_tilt_deg, spec.max_speed_deg_s, "head")
        if result:
            return result
        clamped = abs(target - requested) > 1e-3
        message = (
            f"Head tilt clamped from {requested:.1f} to physical limit {target:.1f} degrees."
            if clamped
            else f"Tilted head to {target:.1f} degrees."
        )
        return CommandResult(
            cmd.id,
            cmd.tool,
            True,
            "clamped" if clamped else "completed",
            message,
            self._pose.to_dict(),
            {"requested_deg": requested, "applied_deg": target, "clamped": clamped},
        )

    async def _exec_drive(self, cmd: BodyCommand) -> CommandResult:
        distance = float(cmd.args["distance_mm"])
        direction = str(cmd.args.get("direction", "forward"))
        sign = -1 if direction == "backward" else 1
        speed = float(cmd.args.get("speed_mm_s", self._caps.default_linear_speed_mm_s))
        result = await self._move(cmd, sign * distance, speed, "drive")
        if result:
            return result
        return CommandResult(
            cmd.id,
            cmd.tool,
            True,
            "completed",
            f"Drove {direction} {distance:.0f} mm.",
            self._pose.to_dict(),
            {"distance_mm": distance, "direction": direction},
        )

    async def _exec_spin(self, cmd: BodyCommand) -> CommandResult:
        angle = float(cmd.args["angle_deg"])
        result = await self._move(cmd, angle, self._caps.default_angular_speed_deg_s, "spin")
        if result:
            return result
        return CommandResult(
            cmd.id,
            cmd.tool,
            True,
            "completed",
            f"Spun {angle:.0f} degrees.",
            self._pose.to_dict(),
            {"applied_deg": angle},
        )

    async def _exec_trick(self, cmd: BodyCommand) -> CommandResult:
        trick = str(cmd.args["trick"])
        # Every segment shares the same safety and joint-limit path as explicit tools.
        segments = {
            "nod": (("head", 15), ("head", 0)),
            "shake": (("spin", 15), ("spin", -15)),
            "curious_tilt": (("head", 20),),
            "wiggle": (("spin", 20), ("spin", -20)),
            "dance": (("spin", 90), ("head", 18), ("spin", -90), ("head", 5)),
            "spin": (("spin", 360),),
            "sleepy_droop": (("head", -18),),
            "perk_up": (("head", 14),),
            "stomp": (("head", -8), ("spin", 12), ("spin", -12)),
        }
        if trick not in segments:
            return CommandResult(
                cmd.id, cmd.tool, False, "rejected", "Trick has no implementation.", self._pose.to_dict()
            )
        for kind, amount in segments[trick]:
            if kind == "head":
                spec = self._caps.joints["head_tilt"]
                target = max(spec.min_deg, min(spec.max_deg, amount))
                result = await self._move(cmd, target - self._pose.head_tilt_deg, spec.max_speed_deg_s, kind)
            else:
                result = await self._move(cmd, amount, self._caps.default_angular_speed_deg_s, kind)
            if result:
                return result
        return CommandResult(
            cmd.id,
            cmd.tool,
            True,
            "completed",
            f"Performed trick '{trick}'.",
            self._pose.to_dict(),
            {"trick": trick},
        )

    def _apply_rotation(self, delta_deg: float) -> None:
        self._pose.heading_deg = _normalize_angle_deg(self._pose.heading_deg + delta_deg)
        self._sensors.imu_yaw_deg = self._pose.heading_deg
        arc_mm = math.radians(delta_deg) * (self._caps.track_width_mm / 2.0)
        self._add_wheel_ticks(-arc_mm, arc_mm)

    def _add_wheel_ticks(self, left_mm: float, right_mm: float) -> None:
        circumference = math.pi * max(1.0, self._caps.wheel_diameter_mm)
        ticks_per_mm = self._caps.encoder_ticks_per_rev / circumference
        self._ticks_left += left_mm * ticks_per_mm
        self._ticks_right += right_mm * ticks_per_mm
        self._sensors.wheel_ticks_left = round(self._ticks_left)
        self._sensors.wheel_ticks_right = round(self._ticks_right)

    def _cancelled_result(self, cmd: BodyCommand) -> CommandResult:
        return CommandResult(
            command_id=cmd.id,
            tool=cmd.tool,
            ok=False,
            status="cancelled",
            message=f"Action '{cmd.tool}' was cancelled.",
            pose=self._pose.to_dict(),
        )

    async def _emit_pose(self) -> None:
        self._refresh_sensors()
        await self.bus.emit(
            "body.pose",
            source=self.name,
            payload={
                "command_id": self._active_cmd_id,
                "pose": self._pose.to_dict(),
                "sensors": self._sensors.to_dict(),
            },
            priority=EventPriority.TELEMETRY,
        )

    async def _emit_reflex(self, reflex_type: str, reason: str) -> None:
        self._pose.linear_vel_mm_s = 0.0
        self._pose.angular_vel_deg_s = 0.0
        await self.bus.emit(
            "reflex.triggered",
            source=self.name,
            payload={
                "reflex": reflex_type,
                "reason": reason,
                "pose": self._pose.to_dict(),
                "sensors": self._sensors.to_dict(),
            },
            priority=EventPriority.CRITICAL,
        )
