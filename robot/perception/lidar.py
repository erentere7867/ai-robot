"""A 2D 360-degree lidar: walls, furniture and legs in every direction, ten times a second.

The camera finds and names things in front; the lidar knows where solid things are all around, which is
what navigation needs (a single forward range sensor discovers a wall only by driving at it). Scans are
fused into the occupancy grid by the fast loop (SpatialMap.update_scan), tagged with the pose at capture.

Sources: `SimLidar` (raycast in the sandbox) and `Ld19Lidar` (LDROBOT LD19 / LD06 on a USB-UART, the
common ~$70 units: 230400 baud, 47-byte packets of 12 points). Mounted on the mast above the head, so it
sees walls and furniture but not toys on the floor (the camera and the front range sensor cover those).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Scan:
    pose: Any  # BodyPose when the scan was taken
    angles_deg: np.ndarray  # relative to the robot's heading, + = left
    ranges_mm: np.ndarray  # inf = no return within range
    stamp: float


class SimLidar:
    """Raycasts the simulated room: walls plus every object taller than the lidar's mounting height."""

    def __init__(self, sandbox: Any, body: Any, clock: Any, *, beams: int = 360, max_range_mm: float = 8000.0,
                 height_mm: float = 330.0, rate_hz: float = 10.0, noise_mm: float = 8.0) -> None:
        self.sandbox, self.body, self.clock = sandbox, body, clock
        self.angles = np.linspace(-180.0, 180.0, beams, endpoint=False)
        self.max_range = max_range_mm
        self.height = height_mm
        self.period = 1.0 / rate_hz
        self.noise = noise_mm
        self._rng = np.random.default_rng(0)
        self._last: Scan | None = None

    def latest(self) -> Scan | None:
        now = self.clock.monotonic()
        if self._last is not None and now - self._last.stamp < self.period:
            return self._last
        truth = self.body.true_pose() if hasattr(self.body, "true_pose") else self.body.pose()
        ranges = self._cast(truth)
        self._last = Scan(self.body.pose(), self.angles, ranges, now)
        return self._last

    def _cast(self, pose: Any) -> np.ndarray:
        th = np.radians(pose.heading_deg + self.angles)
        dx, dy = np.cos(th), np.sin(th)
        ox, oy = pose.x_mm, pose.y_mm
        half = self.sandbox.room_half_extent_mm
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx > 0, (half - ox) / dx, np.where(dx < 0, (-half - ox) / dx, np.inf))
            ty = np.where(dy > 0, (half - oy) / dy, np.where(dy < 0, (-half - oy) / dy, np.inf))
        best = np.minimum(tx, ty)
        for obj in getattr(self.sandbox, "_objects", {}).values():
            if obj.height_mm + obj.z_mm < self.height:
                continue
            cx, cy = obj.x_mm - ox, obj.y_mm - oy
            r = obj.radius_mm
            along = cx * dx + cy * dy
            perp2 = (cx * cx + cy * cy) - along * along
            hit = (along > 0) & (perp2 <= r * r)
            t = along - np.sqrt(np.maximum(0.0, r * r - perp2))
            best = np.where(hit & (t > 0) & (t < best), t, best)
        best = best + self._rng.normal(0.0, self.noise, best.shape)
        return np.where(best < self.max_range, best, np.inf)


def _crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x4D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


class Ld19Lidar:
    """LDROBOT LD19 / LD06 over a USB-UART. One reader thread assembles packets into full turns."""

    PACKET = 47

    def __init__(self, body: Any, clock: Any, *, port: str = "/dev/ttyUSB0", baud: int = 230400,
                 mount_yaw_deg: float = 0.0, max_range_mm: float = 8000.0) -> None:
        import serial  # pip install pyserial

        self.body, self.clock = body, clock
        self.mount_yaw = mount_yaw_deg
        self.max_range = max_range_mm
        self._serial = serial.Serial(port, baud, timeout=0.1)
        self._lock = threading.Lock()
        self._turn: list[tuple[float, float]] = []
        self._last_angle = -1.0
        self._latest: Scan | None = None
        self._running = True
        self._thread = threading.Thread(target=self._read, name="lidar", daemon=True)
        self._thread.start()

    def latest(self) -> Scan | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._serial.close()

    def _read(self) -> None:
        buf = bytearray()
        while self._running:
            buf += self._serial.read(512)
            while len(buf) >= self.PACKET:
                start = buf.find(b"\x54\x2c")
                if start < 0:
                    buf.clear()
                    break
                if len(buf) - start < self.PACKET:
                    del buf[:start]
                    break
                packet = bytes(buf[start:start + self.PACKET])
                del buf[:start + self.PACKET]
                if _crc8(packet[:-1]) == packet[-1]:
                    self._on_packet(packet)

    def _on_packet(self, p: bytes) -> None:
        start = int.from_bytes(p[4:6], "little") / 100.0
        end = int.from_bytes(p[42:44], "little") / 100.0
        span = (end - start) % 360.0
        for k in range(12):
            off = 6 + 3 * k
            dist, conf = int.from_bytes(p[off:off + 2], "little"), p[off + 2]
            angle = (start + span * k / 11.0) % 360.0
            if angle < self._last_angle:  # wrapped: one full turn is done
                self._publish()
            self._last_angle = angle
            # The LD19 turns clockwise; robot angles are anticlockwise (+ = left).
            rel = (-(angle) + self.mount_yaw + 180.0) % 360.0 - 180.0
            self._turn.append((rel, float(dist) if dist > 0 and conf > 20 else math.inf))

    def _publish(self) -> None:
        if len(self._turn) > 100:
            angles = np.array([a for a, _ in self._turn])
            ranges = np.array([r for _, r in self._turn])
            ranges = np.where(ranges < self.max_range, ranges, np.inf)
            with self._lock:
                self._latest = Scan(self.body.pose(), angles, ranges, time.monotonic())
        self._turn = []
