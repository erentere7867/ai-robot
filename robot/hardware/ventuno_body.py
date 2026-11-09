"""The real robot on an Arduino VENTUNO Q: Linux talks to the on-board STM32H5 through the Arduino Router.

The MCU (firmware/ventuno_mcu) runs the 200 Hz wheel loops, odometry (encoders + gyro, bias re-learned at
rest), the braking-distance safety envelope, reflex latches and the command watchdog. This body sends it
speed set points and head angles and reads its telemetry.

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

import logging
import math
import socket
import threading
import time
from typing import Any

from robot.body.hal import BodyCapabilities, BodyPose, Reflex, SensorReadings
from robot.core.angles import wrap_deg
from robot.core.events import EventPriority
from robot.core.service import ServiceHealth

logger = logging.getLogger(__name__)

ROUTER_SOCKET = "/var/run/arduino-router.sock"
BUMPER, CLIFF, ESTOP, PICKED_UP, CHARGING, BLOCKED = 1, 2, 4, 8, 16, 32
STALL, IMU_FAULT, RANGE_FAULT, BATTERY_LOW = 64, 128, 256, 512
REFLEX_FROM_FLAG = ((ESTOP, Reflex.ESTOP), (CLIFF, Reflex.CLIFF), (PICKED_UP, Reflex.PICKED_UP),
                    (BUMPER, Reflex.BUMPER), (BLOCKED, Reflex.OBSTACLE_AHEAD))


class RouterLink:
    """A minimal msgpack-RPC client for the Arduino Router (one socket, a reader thread)."""

    def __init__(self, path: str = ROUTER_SOCKET) -> None:
        import msgpack

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

    def __init__(self, bus: Any, capabilities: BodyCapabilities, *, link: Any = None,
                 socket_path: str = ROUTER_SOCKET, battery_full_mv: int = 8400, battery_empty_mv: int = 6400
                 ) -> None:
        self.bus = bus
        self._caps = capabilities
        self._link = link if link is not None else RouterLink(socket_path)
        self._lock = threading.Lock()
        self._pose = BodyPose()
        self._sensors = SensorReadings(front_tof_mm=capabilities.default_front_tof_mm)
        self._origin: tuple[float, float, float, float, float, float] | None = None
        self._flags = 0
        self.faults = 0
        self._telemetry_at = 0.0
        self._reflex: Reflex | None = None
        self._battery = (battery_empty_mv, battery_full_mv)

    async def start(self) -> None:
        self._link.open()
        self._link.provide("telemetry", self._on_telemetry)
        self._notify("stop")
        await self.publish_state()

    async def stop(self) -> None:
        self._notify("stop")
        self._link.close()

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
            self._pose.heading_deg = wrap_deg(pth + mth - oth)
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

    async def health(self) -> ServiceHealth:
        fresh = time.monotonic() - self._telemetry_at < 0.5
        names = [n for bit, n in ((STALL, "wheel stall"), (IMU_FAULT, "IMU"), (RANGE_FAULT, "range sensor"),
                                  (BATTERY_LOW, "battery low")) if self.faults & bit]
        status = "no telemetry" if not fresh else ("faults: " + ", ".join(names) if names else "ok")
        return ServiceHealth(name=self.name, healthy=fresh and not (self.faults & (STALL | IMU_FAULT)),
                             status=status, details={**self._pose.to_dict(), "faults": names})

    def capabilities(self) -> BodyCapabilities:
        return self._caps

    def pose(self) -> BodyPose:
        with self._lock:
            return self._pose.copy()

    def sensors(self) -> SensorReadings:
        with self._lock:
            return self._sensors.copy()

    def reset_pose(self, pose: BodyPose | None = None) -> None:
        with self._lock:
            self._pose = pose or BodyPose()
            self._origin = None  # re-anchor MCU odometry to the new pose at the next telemetry

    def set_estop(self, active: bool) -> None:
        self._notify("estop", bool(active))

    async def halt(self) -> None:
        self._notify("stop")

    async def set_velocity(self, linear_mm_s: float, angular_deg_s: float) -> Reflex | None:
        v, w = self._caps.clamp_velocity(linear_mm_s, angular_deg_s)
        self._notify("cmd_vel", round(v), round(w * 1000))
        return await self.update_motion()

    def set_head(self, tilt_deg: float) -> None:
        self._notify("head", round(self._caps.clamp_head(tilt_deg) * 10))

    async def update_motion(self) -> Reflex | None:
        flags = self._flags
        reflex = next((r for bit, r in REFLEX_FROM_FLAG if flags & bit), None)
        if reflex is not None and reflex != self._reflex:
            await self.bus.emit("reflex.triggered", self.name, {"reflex": str(reflex), "reason": reflex.reason,
                                                                "pose": self.pose().to_dict(),
                                                                "sensors": self.sensors().to_dict()},
                                priority=EventPriority.CRITICAL)
        self._reflex = reflex
        await self.publish_state()
        return reflex

    async def publish_state(self) -> None:
        await self.bus.emit("body.pose", self.name, {"pose": self.pose().to_dict(),
                                                     "sensors": self.sensors().to_dict()},
                            priority=EventPriority.TELEMETRY)
