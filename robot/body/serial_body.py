"""The real robot's body: a motor microcontroller on USB serial, driven by speed set points.

The microcontroller (firmware/base_controller) runs the wheels' speed loops and the reflexes that must
never wait for Linux: forward stop at the range sensor, bumper, cliff, emergency stop, and a watchdog
that stops the motors 300 ms after the last command. This class keeps odometry, exposes the same API as
the simulated bodies (`set_velocity`/`update_motion` for the fast layer, `execute` for discrete actions)
and turns telemetry flags into reflex events.

Protocol (ASCII lines, 115200 baud):
  host -> MCU   V <mm/s> <mdeg/s>     speed set point (watchdog: stop 300 ms after the last one)
                H <deg*10>            head tilt
                S                     stop everything
                E <0|1>               emergency stop off/on
  MCU -> host   T <ms> <ticks_l> <ticks_r> <gyro_mdeg_s> <tof_mm> <flags> <battery_mv>   at 50 Hz
                flags: 1 bumper, 2 cliff, 4 estop, 8 picked up, 16 charging, 32 forward blocked
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from typing import Any

from robot.body.hal import BodyCapabilities, BodyCommand, BodyPose, CommandResult, SensorReadings
from robot.core.events import EventPriority

logger = logging.getLogger(__name__)

BUMPER, CLIFF, ESTOP, PICKED_UP, CHARGING, BLOCKED = 1, 2, 4, 8, 16, 32
REFLEX_FROM_FLAG = ((ESTOP, "estop_active", "Emergency stop is active"),
                    (CLIFF, "cliff_detected", "Cliff sensor triggered"),
                    (PICKED_UP, "picked_up", "Robot was picked up"),
                    (BUMPER, "bumper", "Bumped into something"),
                    (BLOCKED, "obstacle_stop", "Front obstacle or invalid range reading"))


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


class SerialBody:
    name = "serial_body"
    VELOCITY_TIMEOUT_S = 0.35

    def __init__(self, bus: Any, capabilities: BodyCapabilities | None = None, *, port: str = "/dev/ttyACM0",
                 baud: int = 115200, transport: Any = None, battery_full_mv: int = 8400,
                 battery_empty_mv: int = 6400) -> None:
        self.bus = bus
        self._caps = capabilities or BodyCapabilities()
        self.port, self.baud = port, baud
        self._transport = transport  # anything with write(bytes) and readline() -> bytes (tests: a fake MCU)
        self._lock = threading.Lock()
        self._pose = BodyPose()
        self._sensors = SensorReadings(front_tof_mm=self._caps.default_front_tof_mm)
        self._flags = 0
        self._last_ticks: tuple[int, int] | None = None
        self._last_ms: int | None = None
        self._telemetry_at = 0.0
        self._reflex: str | None = None
        self._active_cmd_id: str | None = None
        self._cancel_generation = 0
        self._reader: threading.Thread | None = None
        self._running = False
        self._battery = (battery_empty_mv, battery_full_mv)
        mm_per_tick = math.pi * self._caps.wheel_diameter_mm / self._caps.encoder_ticks_per_rev
        self._mm_per_tick = mm_per_tick
        from robot.actions.registry import CapabilityRegistry

        self._registry = CapabilityRegistry(self._caps)

    # -- link -------------------------------------------------------------------------------------
    async def start(self) -> None:
        if self._transport is None:
            import serial  # pip install pyserial

            self._transport = serial.Serial(self.port, self.baud, timeout=0.2)
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, name="serial-body", daemon=True)
        self._reader.start()
        self._send("S")
        await self._emit_pose()

    async def stop(self) -> None:
        self._send("S")
        self._running = False
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        close = getattr(self._transport, "close", None)
        if callable(close):
            close()

    async def health(self) -> Any:
        from robot.core.service import ServiceHealth

        fresh = time.monotonic() - self._telemetry_at < 0.5
        return ServiceHealth(name=self.name, healthy=fresh, status="ok" if fresh else "no telemetry",
                             details=self._pose.to_dict())

    def _send(self, line: str) -> None:
        try:
            self._transport.write((line + "\n").encode("ascii"))
        except Exception:
            logger.warning("Serial write failed", exc_info=True)

    def _read_loop(self) -> None:
        while self._running:
            try:
                raw = self._transport.readline()
            except Exception:
                logger.warning("Serial read failed", exc_info=True)
                time.sleep(0.1)
                continue
            if raw:
                self._on_line(raw.decode("ascii", errors="ignore").strip())

    def _on_line(self, line: str) -> None:
        parts = line.split()
        if len(parts) != 8 or parts[0] != "T":
            return
        try:
            ms, ticks_l, ticks_r, gyro, tof, flags, batt = (int(p) for p in parts[1:])
        except ValueError:
            return
        with self._lock:
            if self._last_ticks is not None and self._last_ms is not None:
                dt = max(1e-3, (ms - self._last_ms) / 1000.0)
                dl = (ticks_l - self._last_ticks[0]) * self._mm_per_tick
                dr = (ticks_r - self._last_ticks[1]) * self._mm_per_tick
                # Skid steer: wheels slip when turning, so heading comes from the gyro, distance from encoders.
                dyaw = gyro / 1000.0 * dt
                mid = math.radians(self._pose.heading_deg + 0.5 * dyaw)
                ds = 0.5 * (dl + dr)
                self._pose.x_mm += ds * math.cos(mid)
                self._pose.y_mm += ds * math.sin(mid)
                self._pose.heading_deg = _wrap(self._pose.heading_deg + dyaw)
                self._pose.linear_vel_mm_s = ds / dt
                self._pose.angular_vel_deg_s = gyro / 1000.0
            self._last_ticks, self._last_ms = (ticks_l, ticks_r), ms
            self._flags = flags
            s = self._sensors
            s.front_tof_mm, s.wheel_ticks_left, s.wheel_ticks_right = float(tof), ticks_l, ticks_r
            s.imu_yaw_deg = self._pose.heading_deg
            s.cliff_triggered, s.picked_up = bool(flags & CLIFF), bool(flags & PICKED_UP)
            s.estop_active, s.charging = bool(flags & ESTOP), bool(flags & CHARGING)
            empty, full = self._battery
            s.battery_pct = max(0.0, min(100.0, 100.0 * (batt - empty) / (full - empty)))
            self._telemetry_at = time.monotonic()

    # -- state ------------------------------------------------------------------------------------
    def capabilities(self) -> BodyCapabilities:
        return self._caps

    def pose(self) -> BodyPose:
        with self._lock:
            p = self._pose
            return BodyPose(p.x_mm, p.y_mm, p.heading_deg, p.head_tilt_deg, p.linear_vel_mm_s, p.angular_vel_deg_s)

    def sensors(self) -> SensorReadings:
        with self._lock:
            s = self._sensors
            return SensorReadings(**{k: getattr(s, k) for k in s.__dataclass_fields__})

    def reset_pose(self, pose: BodyPose | None = None) -> None:
        with self._lock:
            self._pose = pose or BodyPose()

    def set_estop(self, active: bool) -> None:
        self._send(f"E {1 if active else 0}")
        if active:
            self._cancel_generation += 1

    async def cancel(self, command_id: str | None = None) -> None:
        if command_id is None or command_id == self._active_cmd_id:
            self._cancel_generation += 1
            self._send("S")

    # -- speed control (fast layer) ----------------------------------------------------------------
    async def set_velocity(self, linear_mm_s: float, angular_deg_s: float, *, timeout_s: float | None = None
                           ) -> str | None:
        v_max, w_max = self._caps.max_linear_speed_mm_s, self._caps.max_angular_speed_deg_s
        v = max(-v_max, min(v_max, float(linear_mm_s))) if math.isfinite(linear_mm_s) else 0.0
        w = max(-w_max, min(w_max, float(angular_deg_s))) if math.isfinite(angular_deg_s) else 0.0
        self._send(f"V {round(v)} {round(w * 1000)}")
        return await self.update_motion()

    async def update_motion(self) -> str | None:
        flags = self._flags
        reason = next(((name, text) for bit, name, text in REFLEX_FROM_FLAG if flags & bit), None)
        if reason is not None and reason[0] != self._reflex:
            await self.bus.emit("reflex.triggered", self.name, {"reflex": reason[0], "reason": reason[1],
                                                                "pose": self.pose().to_dict(),
                                                                "sensors": self.sensors().to_dict()},
                                priority=EventPriority.CRITICAL)
        self._reflex = reason[0] if reason else None
        await self._emit_pose()
        return reason[0] if reason else None

    async def _emit_pose(self) -> None:
        await self.bus.emit("body.pose", self.name, {"command_id": self._active_cmd_id,
                                                     "pose": self.pose().to_dict(),
                                                     "sensors": self.sensors().to_dict()},
                            priority=EventPriority.TELEMETRY)

    # -- discrete actions ---------------------------------------------------------------------------
    async def _profile(self, cmd: BodyCommand, kind: str, amount: float, speed: float) -> CommandResult:
        """Drive or spin a set amount on odometry, ramping down near the end."""
        generation = self._cancel_generation
        start = self.pose()
        sign = 1.0 if amount >= 0 else -1.0
        turned, last_heading = 0.0, start.heading_deg  # accumulated, so turns past 180 degrees work
        deadline = time.monotonic() + 3.0 * abs(amount) / max(1.0, speed) + 3.0
        while True:
            if self._cancel_generation != generation:
                self._send("S")
                return CommandResult(cmd.id, cmd.tool, False, "cancelled", f"Action '{cmd.tool}' was cancelled.",
                                     self.pose().to_dict())
            p = self.pose()
            if kind == "drive":
                yaw = math.radians(start.heading_deg)
                done = (p.x_mm - start.x_mm) * math.cos(yaw) + (p.y_mm - start.y_mm) * math.sin(yaw)
                remaining = abs(amount) - sign * done
                rate = min(speed, max(40.0, 2.5 * remaining))
                reflex = await self.set_velocity(sign * rate, 0.0)
                tolerance = 8.0
            else:
                turned += _wrap(p.heading_deg - last_heading)
                last_heading = p.heading_deg
                remaining = abs(amount) - sign * turned
                rate = min(speed, max(15.0, 2.5 * remaining))
                reflex = await self.set_velocity(0.0, sign * rate)
                tolerance = 1.5
            if reflex is not None and (kind != "drive" or reflex != "obstacle_stop" or sign > 0):
                self._send("S")
                return CommandResult(cmd.id, cmd.tool, False, "reflex_stopped",
                                     next(t for _, n, t in REFLEX_FROM_FLAG if n == reflex), self.pose().to_dict())
            if remaining <= tolerance:
                self._send("S")
                verb = "Drove" if kind == "drive" else "Spun"
                unit = "mm" if kind == "drive" else "degrees"
                return CommandResult(cmd.id, cmd.tool, True, "completed", f"{verb} {abs(amount):.0f} {unit}.",
                                     self.pose().to_dict())
            if time.monotonic() > deadline:
                self._send("S")
                return CommandResult(cmd.id, cmd.tool, False, "failed",
                                     "Motion did not converge: wheels slipping or base obstructed.",
                                     self.pose().to_dict())
            await asyncio.sleep(0.05)

    async def execute(self, cmd: BodyCommand) -> CommandResult:
        from robot.actions.tools import ToolCallRequest

        if cmd.tool == "stop":
            await self.cancel()
            return CommandResult(cmd.id, "stop", True, "completed", "All motion stopped.", self.pose().to_dict())
        outcome = self._registry.validate_tool_call(ToolCallRequest(cmd.tool, cmd.args))
        if not outcome.valid:
            return CommandResult(cmd.id, cmd.tool, False, "rejected", outcome.error or "Invalid command",
                                 self.pose().to_dict())
        args = dict(outcome.normalized_args)
        self._active_cmd_id = cmd.id
        try:
            if cmd.tool == "drive":
                sign = -1.0 if args.get("direction") == "backward" else 1.0
                return await self._profile(cmd, "drive", sign * float(args.get("distance_mm", 0.0)),
                                           float(args.get("speed_mm_s", self._caps.default_linear_speed_mm_s)))
            if cmd.tool == "spin":
                return await self._profile(cmd, "spin", float(args.get("angle_deg", 0.0)),
                                           float(args.get("speed_deg_s", self._caps.default_angular_speed_deg_s)))
            if cmd.tool == "tilt_head":
                angle = float(args.get("angle_deg", 0.0))
                self._send(f"H {round(angle * 10)}")
                spec = self._caps.joints["head_tilt"]
                await asyncio.sleep(abs(angle - self._pose.head_tilt_deg) / spec.max_speed_deg_s + 0.1)
                with self._lock:
                    self._pose.head_tilt_deg = angle
                return CommandResult(cmd.id, cmd.tool, True, "completed", f"Tilted head to {angle:.1f} degrees.",
                                     self.pose().to_dict())
            if cmd.tool == "perform_trick":
                return await self._trick(cmd, str(args.get("trick", "")))
            if cmd.tool in ("set_eye_color", "set_gaze"):
                if cmd.tool == "set_eye_color":
                    await self.bus.emit("face.theme", self.name, {"theme": str(args.get("color", "cyan"))},
                                        priority=EventPriority.HIGH)
                else:
                    await self.bus.emit("expression.updated", self.name,
                                        {"gaze_target": str(args.get("target", "center")), "reason": "tool:set_gaze"})
                return CommandResult(cmd.id, cmd.tool, True, "completed", "Done.", self.pose().to_dict())
            return CommandResult(cmd.id, cmd.tool, False, "failed", f"Unknown body tool '{cmd.tool}'.",
                                 self.pose().to_dict())
        finally:
            self._active_cmd_id = None

    async def _trick(self, cmd: BodyCommand, trick: str) -> CommandResult:
        """Tricks as short speed patterns (the same moves the simulator plays)."""
        patterns = {
            "wiggle": [(0, 90, 0.25), (0, -90, 0.5), (0, 90, 0.25)] * 2,
            "dance": [(0, 120, 0.6), (0, -120, 0.6), (80, 0, 0.3), (-80, 0, 0.3), (0, 120, 0.6)],
            "spin": [(0, 120, 3.0)],
            "stomp": [(120, 0, 0.25), (-120, 0, 0.25)] * 2,
            "shake": [(0, 100, 0.15), (0, -100, 0.3), (0, 100, 0.15)] * 2,
        }
        for v, w, seconds in patterns.get(trick, []):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                reflex = await self.set_velocity(v, w)
                if reflex in ("bumper", "cliff_detected", "estop_active", "picked_up"):
                    self._send("S")
                    return CommandResult(cmd.id, cmd.tool, False, "reflex_stopped", "A reflex stopped the trick.",
                                         self.pose().to_dict())
                await asyncio.sleep(0.05)
        self._send("S")
        return CommandResult(cmd.id, cmd.tool, True, "completed", f"Performed {trick}.", self.pose().to_dict())
