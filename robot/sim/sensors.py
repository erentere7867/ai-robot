"""The simulated robot's senses: cameras rendered in MuJoCo, a raycast lidar, and the person's voice.

Each sees the world from the body's true pose and tags its data with the estimated one, as on hardware.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from robot.audio.doa import SPEED_OF_SOUND_MM_S, MicArray, srp_phat
from robot.perception.camera import Frame
from robot.perception.geometry import CameraModel
from robot.perception.lidar import Scan


class SimCamera:
    """One of the robot's cameras rendered in MuJoCo from the body's *true* pose (what a real lens would see).

    Frames are tagged with the *estimated* pose, as on hardware where only odometry is known.
    """

    def __init__(self, sandbox: Any, body: Any, clock: Any, model: CameraModel) -> None:
        self.sandbox, self.body, self.clock, self.model = sandbox, body, clock, model

    def capture(self) -> Frame:
        pose = self.body.pose()
        rgb = self.sandbox.render_rgb(self.model.name, self.body.true_pose())
        return Frame(None, pose, self.model, self.clock.monotonic(), rgb)

    def close(self) -> None:
        pass


class SimLidar:
    """Raycasts the simulated room: walls plus every object taller than the lidar's mounting height."""

    def __init__(self, sandbox: Any, body: Any, clock: Any, *, mount_x_mm: float, height_mm: float, beams: int = 360,
                 max_range_mm: float = 8000.0, rate_hz: float = 10.0, noise_mm: float = 8.0) -> None:
        self.sandbox, self.body, self.clock = sandbox, body, clock
        self.angles = np.linspace(-180.0, 180.0, beams, endpoint=False)
        self.max_range = max_range_mm
        self.height = height_mm
        self.period = 1.0 / rate_hz
        self.noise = noise_mm
        self.mount_x = mount_x_mm
        self._rng = np.random.default_rng(0)
        self._last: Scan | None = None

    def latest(self) -> Scan | None:
        now = self.clock.monotonic()
        if self._last is not None and now - self._last.stamp < self.period:
            return self._last
        ranges = self._cast(self.body.true_pose())
        self._last = Scan(self.body.pose(), self.angles, ranges, now, self.mount_x)
        return self._last

    def close(self) -> None:
        pass

    def _cast(self, pose: Any) -> np.ndarray:
        th = np.radians(pose.heading_deg + self.angles)
        dx, dy = np.cos(th), np.sin(th)
        yaw = math.radians(pose.heading_deg)
        ox, oy = pose.x_mm + self.mount_x * math.cos(yaw), pose.y_mm + self.mount_x * math.sin(yaw)
        half = self.sandbox.room_half_extent_mm
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx > 0, (half - ox) / dx, np.where(dx < 0, (-half - ox) / dx, np.inf))
            ty = np.where(dy > 0, (half - oy) / dy, np.where(dy < 0, (-half - oy) / dy, np.inf))
        best = np.minimum(tx, ty)
        for x, y, r in self.sandbox.lidar_circles(self.height):  # a person is two legs at this height
            cx, cy = x - ox, y - oy
            along = cx * dx + cy * dy
            perp2 = (cx * cx + cy * cy) - along * along
            hit = (along > 0) & (perp2 <= r * r)
            t = along - np.sqrt(np.maximum(0.0, r * r - perp2))
            best = np.where(hit & (t > 0) & (t < best), t, best)
        best = best + self._rng.normal(0.0, self.noise, best.shape)
        return np.where(best < self.max_range, best, np.inf)


def _speech_like(n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Band-limited noise with a syllable-rate envelope: enough like speech for direction finding."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sample_rate)
    spectrum *= ((f > 150.0) & (f < 4500.0)) / np.maximum(1.0, f / 600.0)  # voices fall off above ~600 Hz
    voice = np.fft.irfft(spectrum, n)
    t = np.arange(n) / sample_rate
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 2 * np.pi))  # ~4 syllables a second
    return voice * envelope / (np.std(voice) + 1e-9)


def simulate_voice(source_mm: tuple[float, float, float], pose: Any, array: MicArray, *, sample_rate: int = 16000,
                   seconds: float = 0.6, room_half_mm: float | None = 1500.0, reflection: float = 0.45,
                   snr_db: float = 20.0, head_z_mm: float = 300.0, rng: np.random.Generator | None = None
                   ) -> np.ndarray:
    """The array's signals (mics, samples) for a voice at `source_mm`, heard from robot `pose` (true pose).

    Direct path plus the four first wall reflections (image sources) and uncorrelated mic noise.
    """
    rng = rng or np.random.default_rng()
    n = int(seconds * sample_rate)
    pad = 2048
    voice = _speech_like(n + pad, sample_rate, rng)
    spectrum = np.fft.rfft(voice)
    freqs = np.fft.rfftfreq(n + pad, 1.0 / sample_rate)
    h = math.radians(pose.heading_deg)
    cx = pose.x_mm + array.center_x_mm * math.cos(h)
    cy = pose.y_mm + array.center_x_mm * math.sin(h)
    mics = [(cx + px * math.cos(h) - py * math.sin(h), cy + px * math.sin(h) + py * math.cos(h))
            for px, py in array.positions_mm]
    sx, sy, sz = source_mm
    sources = [(sx, sy, 1.0)]
    if room_half_mm is not None:
        sources += [(2 * room_half_mm - sx, sy, reflection), (-2 * room_half_mm - sx, sy, reflection),
                    (sx, 2 * room_half_mm - sy, reflection), (sx, -2 * room_half_mm - sy, reflection)]
    out = np.zeros((array.n, n))
    for m, (mx, my) in enumerate(mics):
        acc = np.zeros_like(spectrum)
        for qx, qy, gain in sources:
            dist = math.sqrt((qx - mx) ** 2 + (qy - my) ** 2 + (sz - head_z_mm) ** 2)
            acc += gain * (1000.0 / max(dist, 100.0)) * np.exp(-2j * np.pi * freqs * dist / SPEED_OF_SOUND_MM_S) \
                * spectrum
        out[m] = np.fft.irfft(acc, n + pad)[pad // 2:pad // 2 + n]
    level = float(np.std(out))
    out += rng.standard_normal(out.shape) * level * 10 ** (-snr_db / 20.0)
    return out


class SimVoice:
    """The simulated person's voice on the simulated array: what the microphones would hear."""

    def __init__(self, sandbox: Any, body: Any, array: MicArray, *, seed: int = 0) -> None:
        self.sandbox, self.body, self.array = sandbox, body, array
        self._rng = np.random.default_rng(seed)

    def locate(self) -> dict[str, float] | None:
        head = self.sandbox.person_head()
        if head is None:
            return None
        signals = simulate_voice(head, self.body.true_pose(), self.array, room_half_mm=self.sandbox.room_half_extent_mm,
                                 rng=self._rng)
        bearing, confidence = srp_phat(signals, 16000, self.array)
        return {"bearing_deg": round(bearing, 1), "confidence": round(confidence, 2)}
