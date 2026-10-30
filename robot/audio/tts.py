"""Local TTS engine with companion-robot DSP voice stylizer and 60 Hz RMS mouth envelope."""

from __future__ import annotations

import atexit
import io
import math
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot.audio.sfx import SAMPLE_RATE, _chirp_segment

try:
    import ptts as _ptts_mod
except ImportError:  # pragma: no cover
    _ptts_mod = None  # type: ignore[assignment]

_PHONON_LOCK = threading.Lock()
_PHONON_ENGINES: dict[str, Any] = {}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class SynthesizedSpeech:
    """Synthesized speech audio waveform paired with its 60 Hz mouth-opening envelope."""

    text: str
    samples: np.ndarray  # 1D float32 mono waveform in [-1.0, 1.0]
    sample_rate: int
    mouth_envelope: list[float]  # 0.05..0.95 values sampled at `envelope_fps`
    envelope_fps: int = 60
    backend_name: str = "unknown"

    @property
    def duration_s(self) -> float:
        return len(self.samples) / max(1, self.sample_rate)

    def to_wav_bytes(self) -> bytes:
        pcm16 = np.clip(self.samples * 32767.0, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()


def compute_mouth_envelope(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    fps: int = 60,
) -> list[float]:
    """Compute a smooth 60 Hz RMS amplitude envelope in [0.04, 0.95] to drive `FaceAnimator.mouth_open`."""
    if len(samples) == 0:
        return [0.04]

    hop = max(1, sample_rate // max(1, fps))
    n_frames = max(1, math.ceil(len(samples) / hop))
    raw_rms: list[float] = []
    for i in range(n_frames):
        chunk = samples[i * hop : (i + 1) * hop]
        if len(chunk) == 0:
            raw_rms.append(0.0)
        else:
            rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
            raw_rms.append(rms)

    peak = max(raw_rms) if raw_rms else 1.0
    if peak < 1e-4:
        return [0.04 for _ in raw_rms]

    # Normalize and apply slight attack/release smoothing so mouth movement looks organic
    smoothed: list[float] = []
    prev = 0.04
    for r in raw_rms:
        norm = min(1.0, (r / peak) ** 0.75)
        target = 0.04 + 0.88 * norm if norm > 0.08 else 0.04
        alpha = 0.55 if target > prev else 0.32
        cur = prev + alpha * (target - prev)
        smoothed.append(round(cur, 3))
        prev = cur
    return smoothed


class RobotVoiceStylizer:
    """Applies pitch-shifting and subtle harmonic ring modulation for a cute companion-robot voice."""

    def __init__(
        self,
        *,
        pitch_factor: float = 1.18,
        ring_mod_hz: float = 85.0,
        ring_mod_mix: float = 0.07,
        master_gain: float = 0.75,
    ) -> None:
        self.pitch_factor = pitch_factor
        self.ring_mod_hz = ring_mod_hz
        self.ring_mod_mix = ring_mod_mix
        self.master_gain = master_gain

    def process(self, samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        if len(samples) == 0:
            return samples, sample_rate

        # 1. Resample for pitch/formant shift upward (classic cute companion robot timbre)
        if abs(self.pitch_factor - 1.0) > 0.01:
            n_out = max(1, int(len(samples) / self.pitch_factor))
            x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
            x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
            pitched = np.interp(x_new, x_old, samples).astype(np.float32)
        else:
            pitched = samples.astype(np.float32)

        # 2. Subtle metallic/robotic harmonic warmth via gentle ring modulation
        if self.ring_mod_mix > 0.0:
            t = np.arange(len(pitched), dtype=np.float32) / float(sample_rate)
            mod = np.sin(2.0 * math.pi * self.ring_mod_hz * t).astype(np.float32)
            pitched = (1.0 - self.ring_mod_mix) * pitched + self.ring_mod_mix * (pitched * mod)

        # 3. Normalize & soft-clip
        peak = float(np.max(np.abs(pitched)))
        if peak > 1e-4:
            pitched = (pitched / peak) * float(self.master_gain)
        return pitched.astype(np.float32), sample_rate


def _clean_text_for_speech(text: str) -> str:
    """Strip stage directions like `*Droops head*` and limit notes before speaking."""
    cleaned = re.sub(r"\*[^*]+\*", " ", text)
    cleaned = re.sub(r"\(Note:.*?\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "Beep boop!"


def _synthesize_syllable_warble(text: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Fast pure-NumPy melodic syllable warble fallback (works anywhere with zero external binaries)."""
    words = re.findall(r"[a-zA-Z0-9']+", text)
    if not words:
        return _chirp_segment(520.0, 680.0, 0.14, sr=sample_rate) * 0.4

    parts: list[np.ndarray] = []
    scale = (440.0, 493.88, 523.25, 587.33, 659.25, 783.99, 880.0)
    for idx, word in enumerate(words[:18]):
        n_syllables = max(1, min(3, len(re.findall(r"[aeiouyAEIOUY]+", word)) or 1))
        for s in range(n_syllables):
            h = (sum(ord(c) for c in word) + idx * 7 + s * 3) % len(scale)
            f0 = scale[h]
            f1 = f0 * (1.14 if text.strip().endswith("?") else 1.04)
            seg = _chirp_segment(f0, f1, 0.068, vibrato_hz=18.0, vibrato_depth_hz=12.0, sr=sample_rate)
            parts.append(seg * 0.42)
            parts.append(np.zeros(int(0.012 * sample_rate), dtype=np.float32))
        parts.append(np.zeros(int(0.032 * sample_rate), dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _read_wav_mono_float32(wav_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1).astype(np.float32)
    return data, framerate


_SAPI_LOCK = threading.Lock()
_SAPI_PROC: subprocess.Popen[str] | None = None
_SAPI_TMPDIR: tempfile.TemporaryDirectory[str] | None = None
_SAPI_SEQ: int = 0


def _ensure_sapi_worker() -> tuple[subprocess.Popen[str], Path] | None:
    """Start or reuse a persistent Windows System.Speech worker process to avoid cold subprocess startup."""
    global _SAPI_PROC, _SAPI_TMPDIR
    if _SAPI_PROC is not None and _SAPI_PROC.poll() is None and _SAPI_TMPDIR is not None:
        return _SAPI_PROC, Path(_SAPI_TMPDIR.name)
    # Release a dead worker's pipes and WAV directory before starting its replacement.
    _cleanup_sapi_worker()

    _SAPI_TMPDIR = tempfile.TemporaryDirectory(prefix="sprocket_tts_")
    # Console.In decodes with the console's OEM code page, mangling the UTF-8 text and
    # paths we write (curly quotes, accented names); read stdin as UTF-8 explicitly.
    ps_loop = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = 2; "
        "$in = New-Object System.IO.StreamReader([Console]::OpenStandardInput(), "
        "(New-Object System.Text.UTF8Encoding $false)); "
        "[Console]::Out.WriteLine('READY'); [Console]::Out.Flush(); "
        "while (($line = $in.ReadLine()) -ne $null) { "
        "  if ($line -eq 'QUIT') { break } "
        "  $p = $line -split '\\|', 2; "
        "  if ($p.Length -eq 2) { "
        "    try { $s.SetOutputToWaveFile($p[0]); $s.Speak($p[1]); $s.SetOutputToNull(); [Console]::Out.WriteLine('OK') } "
        "    catch { [Console]::Out.WriteLine('ERR') } "
        "    [Console]::Out.Flush() "
        "  } "
        "} "
        "$s.Dispose();"
    )
    try:
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_loop],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=_NO_WINDOW,
        )
        watchdog = threading.Timer(4.0, proc.kill)
        watchdog.daemon = True
        watchdog.start()
        try:
            ready = proc.stdout.readline().strip() if proc.stdout else ""
        finally:
            watchdog.cancel()
        if ready == "READY":
            _SAPI_PROC = proc
            return proc, Path(_SAPI_TMPDIR.name)
        proc.kill()
    except Exception:
        pass
    return None


def _cleanup_sapi_worker() -> None:
    global _SAPI_PROC, _SAPI_TMPDIR
    proc, _SAPI_PROC = _SAPI_PROC, None
    if proc is not None:
        try:
            if proc.poll() is None:
                if proc.stdin:
                    proc.stdin.write("QUIT\n")
                    proc.stdin.flush()
                proc.terminate()
                proc.wait(timeout=2.0)
        except Exception:
            pass
        for pipe in (proc.stdin, proc.stdout):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:
                pass
    if _SAPI_TMPDIR is not None:
        try:
            _SAPI_TMPDIR.cleanup()
        except Exception:
            pass
        _SAPI_TMPDIR = None


atexit.register(_cleanup_sapi_worker)


class RobotTTSEngine:
    """Synthesizes companion-robot speech via model.q8.gguf (Phonon 40M), LFM3-Omni, SAPI, or warble fallback."""

    def __init__(
        self,
        *,
        mode: str = "auto",  # "auto" | "phonon_q8" | "lfm3_omni" | "sapi" | "warble"
        stylizer: RobotVoiceStylizer | None = None,
        q8_model_path: Path | None = None,
        phonon_dir: Path | None = None,
        model_path: Path | None = None,
        mmproj_path: Path | None = None,
        tts_bin: Path | None = None,
    ) -> None:
        self.mode = mode
        self.stylizer = stylizer or RobotVoiceStylizer(
            pitch_factor=1.10,
            ring_mod_hz=85.0,
            ring_mod_mix=0.03,
            master_gain=0.82,
        )
        self.q8_model_path = q8_model_path or Path("~/Downloads/model.q8.gguf").expanduser()
        self.phonon_dir = phonon_dir or Path("~/robot-runtime/phonon-40m-en").expanduser()
        self.model_path = model_path or Path("~/Downloads/LFM3-Omni-2B-Q4_K_M.gguf").expanduser()
        self.mmproj_path = mmproj_path or Path("~/Downloads/mmproj-LFM3-Omni-2B-Q8_0.gguf").expanduser()
        self.tts_bin = tts_bin or Path("~/robot-runtime/llama.cpp/llama-tts.exe").expanduser()

    def synthesize(self, text: str) -> SynthesizedSpeech:
        spoken_text = _clean_text_for_speech(text[:400])
        samples: np.ndarray | None = None
        sr = SAMPLE_RATE
        backend_used = "melodic_warble"

        if self.mode in ("auto", "phonon_q8"):
            q8_res = self._try_phonon_q8_tts(spoken_text)
            if q8_res is not None:
                samples, sr = self.stylizer.process(*q8_res)
                backend_used = "phonon_q8_gguf"

        if samples is None and self.mode in ("auto", "lfm3_omni"):
            omni_res = self._try_lfm3_omni_tts(spoken_text)
            if omni_res is not None:
                samples, sr = omni_res
                peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
                if peak > 1e-4:
                    samples = (samples / peak * 0.85).astype(np.float32)
                backend_used = "lfm3_omni_tts"

        if samples is None and self.mode in ("auto", "sapi") and sys.platform == "win32":
            sapi_res = self._try_windows_sapi(spoken_text)
            if sapi_res is not None:
                samples, sr = self.stylizer.process(*sapi_res)
                backend_used = "windows_sapi_stylized"

        if samples is None or len(samples) == 0:
            samples = _synthesize_syllable_warble(spoken_text, sample_rate=SAMPLE_RATE)
            sr = SAMPLE_RATE
            backend_used = "melodic_warble"

        envelope = compute_mouth_envelope(samples, sr, fps=60)
        return SynthesizedSpeech(
            text=spoken_text,
            samples=samples,
            sample_rate=sr,
            mouth_envelope=envelope,
            envelope_fps=60,
            backend_name=backend_used,
        )

    def _try_phonon_q8_tts(self, text: str) -> tuple[np.ndarray, int] | None:
        """Synthesize 24 kHz speech in-process using `model.q8.gguf` (Gradium Phonon 40M Q8_0)."""
        if _ptts_mod is None:
            return None
        gguf_target = self.phonon_dir / "model.q8.gguf"
        if not gguf_target.exists() and self.q8_model_path.exists():
            try:
                self.phonon_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.q8_model_path, gguf_target)
            except Exception:
                return None
        if not (gguf_target.exists() and (self.phonon_dir / "config.json").exists()):
            return None

        key = str(self.phonon_dir.resolve())
        with _PHONON_LOCK:
            engine = _PHONON_ENGINES.get(key)
            if engine is None:
                try:
                    engine = _ptts_mod.TTS(key, lang="en")
                    _PHONON_ENGINES[key] = engine
                except Exception:
                    return None
            try:
                pcm = np.asarray(engine.synth(text), dtype=np.float32)
                if len(pcm) > 0:
                    return pcm, 24000
            except Exception:
                return None
        return None

    def _try_lfm3_omni_tts(self, text: str) -> tuple[np.ndarray, int] | None:
        """Synthesize speech using LFM3-Omni-2B native audio decoder (`llama-tts.exe` with `-c 512`)."""
        if not (self.tts_bin.exists() and self.model_path.exists() and self.mmproj_path.exists()):
            return None
        with tempfile.TemporaryDirectory(prefix="sprocket_omni_tts_") as tmpdir:
            wav_path = Path(tmpdir) / "omni_out.wav"
            cmd = [
                str(self.tts_bin),
                "-m",
                str(self.model_path),
                "-mm",
                str(self.mmproj_path),
                "-c",
                "512",
                "-b",
                "256",
                "-ub",
                "256",
                "-ngl",
                "99",
                "-p",
                text,
                "-o",
                str(wav_path),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=15.0, creationflags=_NO_WINDOW)
                if wav_path.exists() and wav_path.stat().st_size > 200:
                    return _read_wav_mono_float32(wav_path)
            except Exception:
                return None
        return None

    @staticmethod
    def _try_windows_sapi(text: str) -> tuple[np.ndarray, int] | None:
        """Synthesize speech offline via persistent Windows System.Speech worker into a temporary WAV."""
        global _SAPI_SEQ
        safe_text = text.replace("|", " ").replace("\r", " ").replace("\n", " ").strip()
        if not safe_text:
            return None
        with _SAPI_LOCK:
            worker = _ensure_sapi_worker()
            if worker is None:
                return None
            proc, tmpdir = worker
            _SAPI_SEQ += 1
            wav_path = tmpdir / f"utt_{_SAPI_SEQ % 8}.wav"
            try:
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(f"{wav_path}|{safe_text}\n")
                proc.stdin.flush()
                watchdog = threading.Timer(6.0, proc.kill)
                watchdog.daemon = True
                watchdog.start()
                try:
                    status = proc.stdout.readline().strip()
                finally:
                    watchdog.cancel()
                if status == "OK" and wav_path.exists() and wav_path.stat().st_size > 100:
                    return _read_wav_mono_float32(wav_path)
            except Exception:
                _cleanup_sapi_worker()
                return None
        return None
