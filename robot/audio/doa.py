"""Where a voice came from: direction of arrival on the microphone array (SRP-PHAT).

The array on the head (config/body.yaml, `sensors.microphone_array`) hears a voice a fraction of a millisecond
earlier on the near side. For every candidate direction, SRP-PHAT adds up the phase-normalised cross-spectra of
all microphone pairs shifted by the delays that direction would cause; the true direction lines them all up.
Delays are applied in the frequency domain, so they need not be whole samples (a 64 mm array spans only three
samples at 16 kHz). Speech frequencies only: below 300 Hz the array is too small to tell directions apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot.core.config import load_yaml_file

SPEED_OF_SOUND_MM_S = 343_000.0


@dataclass(frozen=True)
class MicArray:
    positions_mm: tuple[tuple[float, float], ...]  # x forward, y left, relative to the array centre
    center_x_mm: float  # array centre from the robot's centre (above the head pivot)

    @classmethod
    def from_body_config(cls, path: Path | str) -> MicArray:
        raw = load_yaml_file(path)["sensors"]["microphone_array"]
        return cls(tuple((float(p[0]), float(p[1])) for p in raw["positions_mm"]), float(raw["center_x_mm"]))

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
