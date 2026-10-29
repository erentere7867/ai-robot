"""Procedural NumPy DSP synthesizer for expressive non-verbal companion robot sounds."""

from __future__ import annotations

import io
import math
import wave

import numpy as np

SAMPLE_RATE = 24_000


def _envelope(n: int, attack_ratio: float = 0.08, release_ratio: float = 0.22) -> np.ndarray:
    """Create a smooth attack-sustain-release amplitude envelope."""
    env = np.ones(n, dtype=np.float32)
    a = max(1, int(n * attack_ratio))
    r = max(1, int(n * release_ratio))
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    env[-r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def _chirp_segment(
    f_start: float,
    f_end: float,
    duration_s: float,
    *,
    vibrato_hz: float = 0.0,
    vibrato_depth_hz: float = 0.0,
    harmonics: tuple[tuple[int, float], ...] = ((1, 1.0), (2, 0.22), (3, 0.08)),
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Synthesize a frequency-swept musical chirp with optional vibrato and harmonics."""
    n = max(1, int(duration_s * sr))
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float32)
    # Exponential/smooth pitch curve feels more organic than linear
    inst_freq = f_start * ((f_end / max(1.0, f_start)) ** (t / max(1e-4, duration_s)))
    if vibrato_hz > 0.0 and vibrato_depth_hz > 0.0:
        inst_freq = inst_freq + vibrato_depth_hz * np.sin(2.0 * math.pi * vibrato_hz * t)

    phase = 2.0 * math.pi * np.cumsum(inst_freq) / sr
    sig = np.zeros(n, dtype=np.float32)
    for mult, amp in harmonics:
        sig += float(amp) * np.sin(mult * phase).astype(np.float32)

    peak = float(np.max(np.abs(sig))) if n > 0 else 1.0
    if peak > 1e-5:
        sig /= peak
    return sig * _envelope(n)


class PetSoundSynthesizer:
    """Synthesizes expressive low-latency waveforms for all allowed robot sounds."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, master_gain: float = 0.45) -> None:
        self.sample_rate = sample_rate
        self.master_gain = master_gain

    def synthesize(self, sound_name: str) -> np.ndarray:
        """Return a 1D float32 mono waveform in [-1.0, 1.0] for `sound_name`."""
        name = (sound_name or "none").strip().lower()
        sr = self.sample_rate

        if name in ("none", ""):
            return np.zeros(0, dtype=np.float32)

        if name == "ack_chirp":
            # Quick two-note attentive perk-up ("bip-boop!")
            s1 = _chirp_segment(540.0, 680.0, 0.055, sr=sr)
            gap = np.zeros(int(0.015 * sr), dtype=np.float32)
            s2 = _chirp_segment(740.0, 980.0, 0.070, sr=sr)
            wav = np.concatenate([s1, gap, s2])

        elif name == "happy_trill":
            # Rising playful 4-note arpeggio with warm shimmer
            notes = [
                _chirp_segment(523.25, 560.0, 0.065, sr=sr),
                _chirp_segment(659.25, 700.0, 0.065, sr=sr),
                _chirp_segment(783.99, 830.0, 0.070, sr=sr),
                _chirp_segment(1046.5, 1120.0, 0.120, vibrato_hz=28.0, vibrato_depth_hz=35.0, sr=sr),
            ]
            wav = np.concatenate(notes)

        elif name == "curious_chirp":
            # Questioning upward inflection ("hmm-wee?")
            s1 = _chirp_segment(420.0, 460.0, 0.075, sr=sr)
            s2 = _chirp_segment(460.0, 960.0, 0.165, vibrato_hz=18.0, vibrato_depth_hz=20.0, sr=sr)
            wav = np.concatenate([s1, s2])

        elif name == "sad_whimper":
            # Soft descending minor sigh with gentle tremolo
            s1 = _chirp_segment(640.0, 440.0, 0.18, vibrato_hz=10.0, vibrato_depth_hz=12.0, sr=sr)
            gap = np.zeros(int(0.03 * sr), dtype=np.float32)
            s2 = _chirp_segment(420.0, 270.0, 0.28, vibrato_hz=8.0, vibrato_depth_hz=10.0, sr=sr)
            wav = np.concatenate([s1, gap, s2]) * 0.85

        elif name == "angry_grumble":
            # Gritty low FM growl-stomp ("grrr-bzt!")
            rich = ((1, 1.0), (2, 0.65), (3, 0.45), (4, 0.30), (5, 0.20))
            s1 = _chirp_segment(
                175.0, 120.0, 0.16, vibrato_hz=42.0, vibrato_depth_hz=28.0, harmonics=rich, sr=sr
            )
            gap = np.zeros(int(0.025 * sr), dtype=np.float32)
            s2 = _chirp_segment(
                155.0, 98.0, 0.20, vibrato_hz=48.0, vibrato_depth_hz=32.0, harmonics=rich, sr=sr
            )
            wav = np.tanh(1.6 * np.concatenate([s1, gap, s2])).astype(np.float32)

        elif name == "sleepy_yawn":
            # Long gentle downward glide
            wav = (
                _chirp_segment(
                    410.0,
                    195.0,
                    0.52,
                    harmonics=((1, 1.0), (2, 0.12)),
                    sr=sr,
                )
                * 0.75
            )

        elif name == "startle_pip":
            # High surprised double-pip ("eep!")
            s1 = _chirp_segment(980.0, 1480.0, 0.065, sr=sr)
            gap = np.zeros(int(0.015 * sr), dtype=np.float32)
            s2 = _chirp_segment(1320.0, 1180.0, 0.060, sr=sr)
            wav = np.concatenate([s1, gap, s2])

        elif name == "giggle":
            # Bouncy 4-bubble staccato giggle
            parts = []
            for base_f in (680.0, 790.0, 740.0, 920.0):
                parts.append(_chirp_segment(base_f, base_f * 1.22, 0.065, sr=sr))
                parts.append(np.zeros(int(0.018 * sr), dtype=np.float32))
            wav = np.concatenate(parts)

        elif name == "purr":
            # Warm rhythmic amplitude-modulated purr
            dur = 0.48
            n = int(dur * sr)
            t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
            carrier = (
                np.sin(2.0 * math.pi * 105.0 * t)
                + 0.45 * np.sin(2.0 * math.pi * 210.0 * t)
                + 0.20 * np.sin(2.0 * math.pi * 315.0 * t)
            ).astype(np.float32)
            am = 0.55 + 0.45 * np.sin(2.0 * math.pi * 24.0 * t).astype(np.float32)
            wav = (carrier / 1.65) * am * _envelope(n, 0.12, 0.25)

        else:
            # Default: "soft_beep"
            wav = _chirp_segment(660.0, 660.0, 0.11, sr=sr)

        return (wav * float(self.master_gain)).astype(np.float32)

    def synthesize_wav_bytes(self, sound_name: str) -> bytes:
        """Synthesize `sound_name` and return a complete 16-bit PCM WAV file in memory."""
        samples = self.synthesize(sound_name)
        pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()
