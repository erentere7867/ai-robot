"""The real robot on an Arduino VENTUNO Q: Linux talks to the on-board STM32H5 through the Arduino Router.

The MCU (firmware/ventuno_mcu) runs the 200 Hz wheel loops, odometry (encoders + gyro, bias re-learned at
rest), the braking-distance safety envelope, reflex latches and the command watchdog. This body sends it
speed set points and reads its telemetry; everything else (discrete moves, tricks, reflex events, the HAL
API the fast layer uses) is SerialBody's, so the robot behaves the same on any controller.

Link: msgpack-RPC over the router's unix socket (/var/run/arduino-router.sock). Messages are
[0, id, method, params] requests, [1, id, error, result] responses and [2, method, params] notifications;
this process registers `telemetry` with the router (`$/register`) so the MCU's notifications reach it.
"""

from __future__ import annotations

import logging
import math
import socket
import threading
import time
from typing import Any

from robot.body.serial_body import BLOCKED, CHARGING, CLIFF, ESTOP, PICKED_UP, SerialBody

logger = logging.getLogger(__name__)

ROUTER_SOCKET = "/var/run/arduino-router.sock"
STALL, IMU_FAULT, RANGE_FAULT, BATTERY_LOW = 64, 128, 256, 512


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


class McuBody(SerialBody):
    """SerialBody's behaviour on the VENTUNO Q's real-time controller (odometry comes from the MCU)."""

    name = "mcu_body"

    def __init__(self, bus: Any, capabilities: Any = None, *, link: Any = None, socket_path: str = ROUTER_SOCKET,
                 **kwargs: Any) -> None:
        super().__init__(bus, capabilities, transport=object(), **kwargs)
        self._link = link
        self._socket_path = socket_path
        self.faults = 0

    async def start(self) -> None:
        if self._link is None:
            self._link = RouterLink(self._socket_path)
        if hasattr(self._link, "open"):
            self._link.open()
        self._running = True
        self._link.provide("telemetry", self._on_telemetry)
        self._send("S")
        await self._emit_pose()

    async def stop(self) -> None:
        self._send("S")
        self._running = False
        close = getattr(self._link, "close", None)
        if callable(close):
            close()

    def _send(self, line: str) -> None:
        """SerialBody speaks in protocol lines; here each becomes a Bridge notification."""
        parts = line.split()
        try:
            if parts[0] == "V":
                self._link.notify("cmd_vel", int(parts[1]), int(parts[2]))
            elif parts[0] == "H":
                self._link.notify("head", int(parts[1]))
            elif parts[0] == "S":
                self._link.notify("stop")
            elif parts[0] == "E":
                self._link.notify("estop", parts[1] == "1")
        except Exception:
            logger.warning("MCU link write failed", exc_info=True)

    def reset_pose(self, pose: Any = None) -> None:
        super().reset_pose(pose)
        self._origin = None  # re-anchor MCU odometry to the new pose at the next telemetry

    _origin: tuple[float, float, float, float, float, float] | None = None

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
            self._pose.heading_deg = (pth + mth - oth + 180.0) % 360.0 - 180.0
            self._pose.linear_vel_mm_s = float(v_mm_s)
            self._pose.angular_vel_deg_s = w_mdeg_s / 1000.0
            self._pose.head_tilt_deg = head_ddeg / 10.0
            # Faults the reflex table does not name are reported through the health check.
            self._flags = flags & (ESTOP | CLIFF | PICKED_UP | CHARGING | BLOCKED | 1)
            self.faults = flags & (STALL | IMU_FAULT | RANGE_FAULT | BATTERY_LOW)
            if flags & STALL:
                self._flags |= 1  # a stalled wheel is a contact: stop like a bumper
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
