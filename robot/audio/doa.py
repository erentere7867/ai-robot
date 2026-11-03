"""Where a voice came from: direction of arrival on the microphone array (SRP-PHAT).

The array on the head (config/body.yaml, `sensors.microphone_array`) hears a voice a fraction of a millisecond
earlier on the near side. For every candidate direction, SRP-PHAT adds up the phase-normalised cross-spectra of
all microphone pairs shifted by the delays that direction would cause; the true direction lines them all up.
Delays are applied in the frequency domain, so they need not be whole samples (a 64 mm array spans only three
samples at 16 kHz). Speech frequencies only: below 300 Hz the array is too small to tell directions apart.

The simulator renders the same array's signals from where the person stands (direct path, first wall
reflections, noise), so "come here" in the simulator goes through the same estimate as on the robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SPEED_OF_SOUND_MM_S = 343_000.0


@dataclass(frozen=True)
class MicArray:
    positions_mm: tuple[tuple[float, float], ...]  # x forward, y left, relative to the array centre
    center_x_mm: float = 25.0  # array centre from the robot's centre (above the head pivot)

    @classmethod
    def from_body_config(cls, path: Path | str) -> MicArray | None:
        from robot.core.config import load_yaml_file

        raw = ((load_yaml_file(path).get("sensors") or {}).get("microphone_array")) or {}
        positions = raw.get("positions_mm")
        if not raw.get("enabled", False) or not positions:
            return None
        return cls(tuple((float(p[0]), float(p[1])) for p in positions), float(raw.get("center_x_mm", 25.0)))

    @property
    def n(self) -> int:
        return len(self.positions_mm)


def _steering(array: MicArray, angles_deg: np.ndarray) -> np.ndarray:
    """Arrival time of a far-field plane wave at each mic, relative to the centre: (angles, mics) seconds."""
    th = np.radians(angles_deg)[:, None]
    p = np.asarray(array.positions_mm)
    return -(p[None, :, 0] * np.cos(th) + p[None, :, 1] * np.sin(th)) / SPEED_OF_SOUND_MM_S


def srp_phat(signals: np.ndarray, sample_rate: int, array: MicArray, *, frame: int = 512,
             band_hz: tuple[float, float] = (300.0, 5000.0), step_deg: float = 1.0) -> tuple[float, float]:
    """(bearing in degrees relative to the robot's heading, + = left; confidence 0..1) of the strongest source.

    `signals` is (mics, samples). Confidence is how much the best direction stands out from the average one.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != array.n or x.shape[1] < frame:
        raise ValueError("signals must be (mics, samples) with at least one frame")
    hop = frame // 2
    window = np.hanning(frame)
    starts = range(0, x.shape[1] - frame + 1, hop)
    spectra = np.stack([np.fft.rfft(x[:, s:s + frame] * window, axis=1) for s in starts])  # (frames, mics, bins)
    freqs = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    band = (freqs >= band_hz[0]) & (freqs <= min(band_hz[1], 0.5 * sample_rate))
    spectra, freqs = spectra[:, :, band], freqs[band]
    pairs = [(i, j) for i in range(array.n) for j in range(i + 1, array.n)]
    cross = np.stack([(spectra[:, i] * np.conj(spectra[:, j])).sum(axis=0) for i, j in pairs])  # (pairs, bins)
    cross /= np.maximum(np.abs(cross), 1e-12)  # PHAT: phase only, so loud low frequencies do not dominate
    angles = np.arange(-180.0, 180.0, step_deg)
    tau = _steering(array, angles)  # (angles, mics)
    dtau = np.stack([tau[:, i] - tau[:, j] for i, j in pairs], axis=1)  # (angles, pairs)
    # A source at angle a: X_i ~ e^{-j w tau_i}, so X_i X_j* ~ e^{-j w (tau_i - tau_j)}; undo that shift.
    power = np.real(np.einsum("pf,apf->a", cross, np.exp(2j * np.pi * freqs[None, None, :] * dtau[:, :, None])))
    best = int(np.argmax(power))
    spread = float(power.max() - power.mean())
    confidence = spread / (spread + float(power.std()) + 1e-9) if spread > 0 else 0.0
    return float(angles[best]), max(0.0, min(1.0, confidence))


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
        getter = getattr(self.body, "true_pose", None)
        pose = getter() if callable(getter) else self.body.pose()
        signals = simulate_voice(head, pose, self.array, room_half_mm=self.sandbox.room_half_extent_mm,
                                 rng=self._rng)
        bearing, confidence = srp_phat(signals, 16000, self.array)
        return {"bearing_deg": round(bearing, 1), "confidence": round(confidence, 2)}
