"""The real robot on an Arduino VENTUNO Q: Linux talks to the on-board STM32H5 through the Arduino Router.

The MCU (firmware/ventuno_mcu) runs the 200 Hz wheel loops, odometry (encoders + gyro, bias re-learned at
rest), the braking-distance safety envelope, reflex latches and the command watchdog. This body sends it
speed set points, reads its telemetry, and offers the same API as the simulated bodies: `set_velocity` /
`update_motion` for the fast layer, `execute` for discrete actions (drive, spin, head, tricks).

Link: msgpack-RPC over the router's unix socket (/var/run/arduino-router.sock). Messages are
[0, id, method, params] requests, [1, id, error, result] responses and [2, method, params] notifications;
this process registers `telemetry` with the router (`$/register`) so the MCU's notifications reach it.

  host -> MCU   cmd_vel(mm/s, mdeg/s)   head(deg*10)   stop()   estop(bool)   reset_odom()
  MCU -> host   telemetry(ms, x_mm, y_mm, heading_mdeg, v_mm_s, w_mdeg_s, ticks_l, ticks_r, range_mm, flags,
                          battery_mv, head_ddeg)   at 50 Hz
                flags: 1 bumper, 2 cliff, 4 estop, 8 picked up, 16 charging, 32 forward blocked,
                       64 wheel stall, 128 IMU fault, 256 range sensor fault, 512 battery low
"""

from __future__ import annotations

import asyncio
import logging
import math
import socket
import threading
import time
from typing import Any

from robot.body.hal import BodyCapabilities, BodyCommand, BodyPose, CommandResult, SensorReadings
from robot.core.events import EventPriority

logger = logging.getLogger(__name__)

ROUTER_SOCKET = "/var/run/arduino-router.sock"
BUMPER, CLIFF, ESTOP, PICKED_UP, CHARGING, BLOCKED = 1, 2, 4, 8, 16, 32
STALL, IMU_FAULT, RANGE_FAULT, BATTERY_LOW = 64, 128, 256, 512
REFLEX_FROM_FLAG = ((ESTOP, "estop_active", "Emergency stop is active"),
                    (CLIFF, "cliff_detected", "Cliff sensor triggered"),
                    (PICKED_UP, "picked_up", "Robot was picked up"),
                    (BUMPER, "bumper", "Bumped into something"),
                    (BLOCKED, "obstacle_stop", "Front obstacle or invalid range reading"))
TRICKS = {  # short speed patterns (v mm/s, w deg/s, seconds): the same moves the simulator plays
    "wiggle": [(0, 90, 0.25), (0, -90, 0.5), (0, 90, 0.25)] * 2,
    "dance": [(0, 120, 0.6), (0, -120, 0.6), (80, 0, 0.3), (-80, 0, 0.3), (0, 120, 0.6)],
    "spin": [(0, 120, 3.0)],
    "stomp": [(120, 0, 0.25), (-120, 0, 0.25)] * 2,
    "shake": [(0, 100, 0.15), (0, -100, 0.3), (0, 100, 0.15)] * 2,
}


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


class RouterLink:
    """A minimal msgpack-RPC client for the Arduino Router (one socket, a reader thread)."""

    def __init__(self, path: str = ROUTER_SOCKET) -> None:
        import msgpack  # pip install msgpack

        self._msgpack = msgpack
        self.path = path
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._next_id = 1
        self._waiting: dict[int, tuple[threading.Event, list[Any]]] = {}
        self.handlers: dict[str, Any] = {}
        self._running = False
        self._thread: threading.Thread | None = None

    def open(self) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self.path)
        self._running = True
        self._thread = threading.Thread(target=self._read, name="router-link", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _write(self, message: list[Any]) -> None:
        data = self._msgpack.packb(message, use_bin_type=True)
        with self._lock:
            if self._sock is None:
                raise ConnectionError("router link is closed")
            self._sock.sendall(data)

    def notify(self, method: str, *params: Any) -> None:
        self._write([2, method, list(params)])

    def call(self, method: str, *params: Any, timeout_s: float = 2.0) -> Any:
        with self._lock:
            msg_id = self._next_id
            self._next_id += 1
        done, box = threading.Event(), []
        self._waiting[msg_id] = (done, box)
        self._write([0, msg_id, method, list(params)])
        if not done.wait(timeout_s):
            self._waiting.pop(msg_id, None)
            raise TimeoutError(f"router call {method} timed out")
        error, result = box
        if error:
            raise RuntimeError(f"router call {method} failed: {error}")
        return result

    def provide(self, method: str, handler: Any) -> None:
        """Receive `method` calls/notifications from the MCU."""
        self.handlers[method] = handler
        self.call("$/register", method)

    def _read(self) -> None:
        unpacker = self._msgpack.Unpacker(raw=False)
        while self._running and self._sock is not None:
            try:
                chunk = self._sock.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            unpacker.feed(chunk)
            for message in unpacker:
                self._dispatch(message)
        self._running = False

    def _dispatch(self, message: Any) -> None:
        if not isinstance(message, (list, tuple)) or not message:
            return
        kind = message[0]
        if kind == 1 and len(message) == 4:  # response
            waiting = self._waiting.pop(message[1], None)
            if waiting is not None:
                waiting[1].extend([message[2], message[3]])
                waiting[0].set()
        elif kind == 2 and len(message) == 3:  # notification
            handler = self.handlers.get(message[1])
            if handler is not None:
                handler(*message[2])
        elif kind == 0 and len(message) == 4:  # request: answer it
            handler = self.handlers.get(message[2])
            error, result = None, None
            try:
                result = handler(*message[3]) if handler else None
                if handler is None:
                    error = f"no method {message[2]}"
            except Exception as exc:  # report, never kill the reader
                error = str(exc)
            self._write([1, message[1], error, result])


class VentunoBody:
    name = "ventuno_body"

    def __init__(self, bus: Any, capabilities: BodyCapabilities | None = None, *, link: Any = None,
                 socket_path: str = ROUTER_SOCKET, battery_full_mv: int = 8400, battery_empty_mv: int = 6400
                 ) -> None:
        from robot.actions.registry import CapabilityRegistry

        self.bus = bus
        self._caps = capabilities or BodyCapabilities()
        self._registry = CapabilityRegistry(self._caps)
        self._link = link  # tests pass a fake MCU with notify/provide
        self._socket_path = socket_path
        self._lock = threading.Lock()
        self._pose = BodyPose()
        self._sensors = SensorReadings(front_tof_mm=self._caps.default_front_tof_mm)
        self._origin: tuple[float, float, float, float, float, float] | None = None
        self._flags = 0
        self.faults = 0
        self._telemetry_at = 0.0
        self._reflex: str | None = None
        self._active_cmd_id: str | None = None
        self._cancel_generation = 0
        self._battery = (battery_empty_mv, battery_full_mv)

    # -- link -------------------------------------------------------------------------------------
    async def start(self) -> None:
        if self._link is None:
            self._link = RouterLink(self._socket_path)
        if hasattr(self._link, "open"):
            self._link.open()
        self._link.provide("telemetry", self._on_telemetry)
        self._notify("stop")
        await self._emit_pose()

    async def stop(self) -> None:
        self._notify("stop")
        close = getattr(self._link, "close", None)
        if callable(close):
            close()

    def _notify(self, method: str, *params: Any) -> None:
        try:
            self._link.notify(method, *params)
        except Exception:
            logger.warning("MCU link write failed", exc_info=True)

    def _on_telemetry(self, ms: int, x_mm: int, y_mm: int, heading_mdeg: int, v_mm_s: int, w_mdeg_s: int,
                      ticks_l: int, ticks_r: int, range_mm: int, flags: int, battery_mv: int,
                      head_ddeg: int = 0) -> None:
        mx, my, mth = float(x_mm), float(y_mm), heading_mdeg / 1000.0
        with self._lock:
            if self._origin is None:
                p = self._pose  # MCU odometry frame -> the harness's frame, anchored where they first met
                self._origin = (mx, my, mth, p.x_mm, p.y_mm, p.heading_deg)
            ox, oy, oth, px, py, pth = self._origin
            rot = math.radians(pth - oth)
            dx, dy = mx - ox, my - oy
            self._pose.x_mm = px + dx * math.cos(rot) - dy * math.sin(rot)
            self._pose.y_mm = py + dx * math.sin(rot) + dy * math.cos(rot)
            self._pose.heading_deg = _wrap(pth + mth - oth)
            self._pose.linear_vel_mm_s = float(v_mm_s)
            self._pose.angular_vel_deg_s = w_mdeg_s / 1000.0
            self._pose.head_tilt_deg = head_ddeg / 10.0
            # A stalled wheel is a contact: stop like a bumper. Other faults go to the health check.
            self._flags = flags & (ESTOP | CLIFF | PICKED_UP | CHARGING | BLOCKED | BUMPER)
            self.faults = flags & (STALL | IMU_FAULT | RANGE_FAULT | BATTERY_LOW)
            if flags & STALL:
                self._flags |= BUMPER
            s = self._sensors
            s.front_tof_mm, s.wheel_ticks_left, s.wheel_ticks_right = float(range_mm), ticks_l, ticks_r
            s.imu_yaw_deg = self._pose.heading_deg
            s.cliff_triggered, s.picked_up = bool(flags & CLIFF), bool(flags & PICKED_UP)
            s.estop_active, s.charging = bool(flags & ESTOP), bool(flags & CHARGING)
            empty, full = self._battery
            s.battery_pct = max(0.0, min(100.0, 100.0 * (battery_mv - empty) / (full - empty)))
            self._telemetry_at = time.monotonic()

    async def health(self) -> Any:
        from robot.core.service import ServiceHealth

        fresh = time.monotonic() - self._telemetry_at < 0.5
        names = [n for bit, n in ((STALL, "wheel stall"), (IMU_FAULT, "IMU"), (RANGE_FAULT, "range sensor"),
                                  (BATTERY_LOW, "battery low")) if self.faults & bit]
        status = "no telemetry" if not fresh else ("faults: " + ", ".join(names) if names else "ok")
        return ServiceHealth(name=self.name, healthy=fresh and not (self.faults & (STALL | IMU_FAULT)),
                             status=status, details={**self._pose.to_dict(), "faults": names})

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
            self._origin = None  # re-anchor MCU odometry to the new pose at the next telemetry

    def set_estop(self, active: bool) -> None:
        self._notify("estop", bool(active))
        if active:
            self._cancel_generation += 1

    async def cancel(self, command_id: str | None = None) -> None:
        if command_id is None or command_id == self._active_cmd_id:
            self._cancel_generation += 1
            self._notify("stop")

    # -- speed control (fast layer) ----------------------------------------------------------------
    async def set_velocity(self, linear_mm_s: float, angular_deg_s: float, *, timeout_s: float | None = None
                           ) -> str | None:
        v_max, w_max = self._caps.max_linear_speed_mm_s, self._caps.max_angular_speed_deg_s
        v = max(-v_max, min(v_max, float(linear_mm_s))) if math.isfinite(linear_mm_s) else 0.0
        w = max(-w_max, min(w_max, float(angular_deg_s))) if math.isfinite(angular_deg_s) else 0.0
        self._notify("cmd_vel", round(v), round(w * 1000))
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
    def _result(self, cmd: BodyCommand, ok: bool, status: str, message: str) -> CommandResult:
        return CommandResult(cmd.id, cmd.tool, ok, status, message, self.pose().to_dict())

    async def _profile(self, cmd: BodyCommand, kind: str, amount: float, speed: float) -> CommandResult:
        """Drive or spin a set amount on odometry, ramping down near the end."""
        generation = self._cancel_generation
        start = self.pose()
        sign = 1.0 if amount >= 0 else -1.0
        turned, last_heading = 0.0, start.heading_deg  # accumulated, so turns past 180 degrees work
        deadline = time.monotonic() + 3.0 * abs(amount) / max(1.0, speed) + 3.0
        while True:
            if self._cancel_generation != generation:
                self._notify("stop")
                return self._result(cmd, False, "cancelled", f"Action '{cmd.tool}' was cancelled.")
            p = self.pose()
            if kind == "drive":
                yaw = math.radians(start.heading_deg)
                done = (p.x_mm - start.x_mm) * math.cos(yaw) + (p.y_mm - start.y_mm) * math.sin(yaw)
                remaining = abs(amount) - sign * done
                reflex = await self.set_velocity(sign * min(speed, max(40.0, 2.5 * remaining)), 0.0)
                tolerance = 8.0
            else:
                turned += _wrap(p.heading_deg - last_heading)
                last_heading = p.heading_deg
                remaining = abs(amount) - sign * turned
                reflex = await self.set_velocity(0.0, sign * min(speed, max(15.0, 2.5 * remaining)))
                tolerance = 1.5
            if reflex is not None and (kind != "drive" or reflex != "obstacle_stop" or sign > 0):
                self._notify("stop")
                return self._result(cmd, False, "reflex_stopped",
                                    next(t for _, n, t in REFLEX_FROM_FLAG if n == reflex))
            if remaining <= tolerance:
                self._notify("stop")
                verb, unit = ("Drove", "mm") if kind == "drive" else ("Spun", "degrees")
                return self._result(cmd, True, "completed", f"{verb} {abs(amount):.0f} {unit}.")
            if time.monotonic() > deadline:
                self._notify("stop")
                return self._result(cmd, False, "failed", "Motion did not converge: wheels slipping or base obstructed.")
            await asyncio.sleep(0.05)

    async def execute(self, cmd: BodyCommand) -> CommandResult:
        from robot.actions.tools import ToolCallRequest

        if cmd.tool == "stop":
            await self.cancel()
            return self._result(cmd, True, "completed", "All motion stopped.")
        outcome = self._registry.validate_tool_call(ToolCallRequest(cmd.tool, cmd.args))
        if not outcome.valid:
            return self._result(cmd, False, "rejected", outcome.error or "Invalid command")
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
                self._notify("head", round(angle * 10))
                spec = self._caps.joints["head_tilt"]
                await asyncio.sleep(abs(angle - self.pose().head_tilt_deg) / spec.max_speed_deg_s + 0.1)
                with self._lock:
                    self._pose.head_tilt_deg = angle
                return self._result(cmd, True, "completed", f"Tilted head to {angle:.1f} degrees.")
            if cmd.tool == "perform_trick":
                return await self._trick(cmd, str(args.get("trick", "")))
            if cmd.tool == "set_eye_color":
                await self.bus.emit("face.theme", self.name, {"theme": str(args.get("color", "cyan"))},
                                    priority=EventPriority.HIGH)
                return self._result(cmd, True, "completed", "Done.")
            if cmd.tool == "set_gaze":
                await self.bus.emit("expression.updated", self.name,
                                    {"gaze_target": str(args.get("target", "center")), "reason": "tool:set_gaze"})
                return self._result(cmd, True, "completed", "Done.")
            return self._result(cmd, False, "failed", f"Unknown body tool '{cmd.tool}'.")
        finally:
            self._active_cmd_id = None

    async def _trick(self, cmd: BodyCommand, trick: str) -> CommandResult:
        for v, w, seconds in TRICKS.get(trick, []):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                reflex = await self.set_velocity(v, w)
                if reflex in ("bumper", "cliff_detected", "estop_active", "picked_up"):
                    self._notify("stop")
                    return self._result(cmd, False, "reflex_stopped", "A reflex stopped the trick.")
                await asyncio.sleep(0.05)
        self._notify("stop")
        return self._result(cmd, True, "completed", f"Performed {trick}.")
