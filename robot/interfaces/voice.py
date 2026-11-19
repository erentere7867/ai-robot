"""Speech in and out, expressive non-verbal procedural chimes, and wake-word-locked DOA.

Components:
- WAV I/O and resampling: `read_wav`, `to_16k`, `write_wav`.
- Utterance segmentation and wake-word gating: `Listener`, `addressed`, `microphone`.
- Fast voiceprint extraction: `extract_voiceprint`.
- DOA ring buffer locking onto utterance onset: `DOARingBuffer`.
- Expressive procedural chimes: `chime`.
- TTS and STT interface integration: `attach`, `Ears`, `Mouth`.
"""

from __future__ import annotations

import io
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from robot import paths

VOICE = paths.MODELS / "voice" / "en_US-lessac-medium.onnx"


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Mono float32 samples in [-1, 1] and the sample rate of a 16-bit PCM WAV."""
    with wave.open(io.BytesIO(data)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit samples, got {8 * width}-bit")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return samples.reshape(-1, channels).mean(axis=1), rate


def to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    """Resample to the 16 kHz Whisper/voiceprint listens at (linear interpolation)."""
    if rate == 16000 or samples.size == 0:
        return samples
    n = int(samples.size * 16000 / rate)
    return np.interp(np.linspace(0, samples.size - 1, n), np.arange(samples.size), samples).astype(np.float32)


def write_wav(samples: np.ndarray, rate: int) -> bytes:
    """A 16-bit PCM mono WAV of float samples in [-1, 1]."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Mel Filterbank & Voiceprint Extraction
# ---------------------------------------------------------------------------

_MEL_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _mel_filterbank(n_mels: int = 44, n_fft: int = 512, rate: int = 16000) -> np.ndarray:
    """Construct triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
    key = (n_mels, n_fft, rate)
    if key in _MEL_CACHE:
        return _MEL_CACHE[key]

    def hz_to_mel(hz: float) -> float:
        return float(2595.0 * np.log10(1.0 + hz / 700.0))

    def mel_to_hz(mel: float) -> float:
        return float(700.0 * (10.0 ** (mel / 2595.0) - 1.0))

    mel_min = hz_to_mel(100.0)
    mel_max = hz_to_mel(7500.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = [mel_to_hz(m) for m in mel_points]
    bin_points = np.clip(np.floor((n_fft + 1) * np.array(hz_points) / rate).astype(int), 0, n_fft // 2)

    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_prev = bin_points[m - 1]
        f_curr = bin_points[m]
        f_next = bin_points[m + 1]
        if f_curr > f_prev:
            fbank[m - 1, f_prev:f_curr] = (np.arange(f_prev, f_curr) - f_prev) / (f_curr - f_prev)
        if f_next > f_curr:
            fbank[m - 1, f_curr:f_next] = (f_next - np.arange(f_curr, f_next)) / (f_next - f_curr)

    _MEL_CACHE[key] = fbank
    return fbank


def extract_voiceprint(samples: np.ndarray, rate: int = 16000, dim: int = 64) -> np.ndarray:
    """Compute a normalized 64-D spectral log-mel and pitch-autocorrelation voiceprint vector."""
    samples = np.asarray(samples, dtype=np.float32).ravel()
    if samples.size == 0:
        return np.zeros(dim, dtype=np.float32)
    if rate != 16000:
        samples = to_16k(samples, rate)
        rate = 16000

    peak = float(np.max(np.abs(samples))) if samples.size > 0 else 0.0
    if peak < 1e-4 or samples.size < 320:
        return np.zeros(dim, dtype=np.float32)

    frame_len = 400  # 25 ms at 16 kHz
    hop_len = 160    # 10 ms at 16 kHz
    n_frames = (samples.size - frame_len) // hop_len + 1
    if n_frames <= 0:
        return np.zeros(dim, dtype=np.float32)

    idx = np.arange(frame_len)[None, :] + np.arange(n_frames)[:, None] * hop_len
    frames = samples[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    thresh = max(0.005, 0.10 * float(np.max(rms)))
    voiced = frames[rms > thresh]
    if voiced.shape[0] == 0:
        return np.zeros(dim, dtype=np.float32)

    n_mels = int(dim * 0.6875)  # 44 for dim=64
    n_pitch = dim - n_mels      # 20 for dim=64
    fbank = _mel_filterbank(n_mels=n_mels, n_fft=512, rate=rate)

    win = np.hanning(frame_len).astype(np.float32)
    win_frames = (voiced - np.mean(voiced, axis=1, keepdims=True)) * win
    specs = np.abs(np.fft.rfft(win_frames, n=512)) ** 2
    mel_energies = np.dot(specs, fbank.T)

    max_e = np.max(mel_energies, axis=1, keepdims=True) + 1e-9
    rel_mel = np.log10(np.maximum(mel_energies / max_e, 1e-3))
    mean_mel = np.mean(rel_mel, axis=0)
    mean_mel = mean_mel - float(np.mean(mean_mel))
    norm_m = float(np.linalg.norm(mean_mel))
    if norm_m > 1e-9:
        mean_mel = mean_mel / norm_m

    # Pitch autocorrelation over voiced frames for lags 70 Hz to 400 Hz (lags 35 to 235)
    lag_min, lag_max = 35, 235
    lags = np.linspace(lag_min, lag_max, n_pitch, dtype=int)
    norm_denom = np.sum(win_frames ** 2, axis=1) + 1e-9
    pitch_vals = np.zeros(n_pitch, dtype=np.float32)
    for i, lag in enumerate(lags):
        corr = np.sum(win_frames[:, :-lag] * win_frames[:, lag:], axis=1) / norm_denom
        pitch_vals[i] = float(np.mean(corr))
    norm_p = float(np.linalg.norm(pitch_vals))
    if norm_p > 1e-9:
        pitch_vals = pitch_vals / norm_p

    feat = np.concatenate([mean_mel, pitch_vals]).astype(np.float32)
    norm = float(np.linalg.norm(feat))
    return (feat / norm).astype(np.float32) if norm > 1e-9 else np.zeros(dim, dtype=np.float32)


# ---------------------------------------------------------------------------
# DOA Ring Buffer
# ---------------------------------------------------------------------------

class DOARingBuffer:
    """Sliding ring buffer tracking audio chunks and Direction of Arrival (DOA) angles over window_s."""

    def __init__(self, window_s: float = 2.5) -> None:
        self.window_s = window_s
        self._buffer: list[tuple[float, np.ndarray, float]] = []

    def push(self, samples: np.ndarray, doa_rad: float, now: float | None = None) -> None:
        """Push an audio chunk with its instantaneous DOA angle into the sliding buffer."""
        if now is None:
            now = time.monotonic()
        self._buffer.append((now, np.asarray(samples, dtype=np.float32).copy(), float(doa_rad)))
        cutoff = now - self.window_s
        self._buffer = [e for e in self._buffer if e[0] >= cutoff]

    def doa_at(self, timestamp: float) -> float | None:
        """Resolve the DOA angle closest to the target timestamp within the sliding window."""
        if not self._buffer:
            return None
        best = min(self._buffer, key=lambda entry: abs(entry[0] - timestamp))
        if abs(best[0] - timestamp) > self.window_s:
            return None
        return best[2]

    def utterance_with_doa(
        self,
        listener: Listener,
        samples: np.ndarray,
        doa_rad: float,
        now: float | None = None,
    ) -> list[tuple[np.ndarray, float, np.ndarray]]:
        """Push samples with DOA, feed listener, resolve onset angle, and extract voiceprint."""
        if now is None:
            now = time.monotonic()
        self.push(samples, doa_rad, now=now)
        utts = listener.feed(samples)
        results: list[tuple[np.ndarray, float, np.ndarray]] = []
        for utt in utts:
            duration = len(utt) / listener.rate
            onset_time = now - duration + 0.15
            speaker_doa = self.doa_at(onset_time)
            if speaker_doa is None:
                speaker_doa = doa_rad
            vprint = extract_voiceprint(utt, rate=listener.rate)
            results.append((utt, float(speaker_doa), vprint))
        return results


# ---------------------------------------------------------------------------
# Procedural Audio Chimes
# ---------------------------------------------------------------------------

def _tone(freq: float, duration: float, rate: int, amp: float = 0.4, harmonic: float = 0.25) -> np.ndarray:
    n = int(duration * rate)
    t = np.arange(n) / rate
    fade = int(0.008 * rate)
    env = np.ones(n, dtype=np.float32)
    if 2 * fade < n:
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
    decay = np.exp(-2.8 * t / duration)
    sig = amp * (np.sin(2 * np.pi * freq * t) + harmonic * np.sin(4 * np.pi * freq * t)) * env * decay
    return sig.astype(np.float32)


def _glide(f0: float, f1: float, duration: float, rate: int, amp: float = 0.4) -> np.ndarray:
    n = int(duration * rate)
    t = np.arange(n) / rate
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * (t ** 2) / duration)
    fade = int(0.008 * rate)
    env = np.ones(n, dtype=np.float32)
    if 2 * fade < n:
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
    decay = np.exp(-1.4 * t / duration)
    sig = amp * (np.sin(phase) + 0.2 * np.sin(2 * phase)) * env * decay
    return sig.astype(np.float32)


def chime(kind: str, rate: int = 16000) -> bytes:
    """Generate expressive, clean non-verbal audio chimes as 16-bit PCM WAV bytes."""
    kind = kind.lower().strip()
    pause = np.zeros(int(0.018 * rate), dtype=np.float32)

    if kind == "wake":
        # bright rising two-note chirp (523 Hz -> 784 Hz)
        s = np.concatenate([_tone(523.25, 0.10, rate, amp=0.4), pause, _tone(783.99, 0.16, rate, amp=0.45)])
    elif kind == "listen":
        # soft attentive two-tone pulse (660 Hz -> 880 Hz)
        s = np.concatenate([_tone(660.0, 0.08, rate, amp=0.3), pause, _tone(880.0, 0.12, rate, amp=0.35)])
    elif kind == "curious":
        # ascending inflected glide (440 Hz -> 680 Hz)
        s = _glide(440.0, 680.0, 0.24, rate, amp=0.38)
    elif kind == "happy":
        # cheerful 3-note arpeggio trill (587 -> 784 -> 1046 Hz)
        p_short = np.zeros(int(0.012 * rate), dtype=np.float32)
        s = np.concatenate([
            _tone(587.33, 0.08, rate, amp=0.38),
            p_short,
            _tone(783.99, 0.08, rate, amp=0.42),
            p_short,
            _tone(1046.50, 0.16, rate, amp=0.45),
        ])
    elif kind == "uh_oh":
        # gentle descending two-tone (420 Hz -> 290 Hz)
        s = np.concatenate([_tone(420.0, 0.13, rate, amp=0.35), pause, _tone(290.0, 0.20, rate, amp=0.32)])
    elif kind == "yield":
        # soft descending courtesy chirp (520 Hz -> 390 Hz)
        s = _glide(520.0, 390.0, 0.18, rate, amp=0.28)
    else:
        raise ValueError(f"Unknown chime kind: '{kind}'. Expected wake/listen/curious/happy/uh_oh/yield.")

    return write_wav(s, rate)


# ---------------------------------------------------------------------------
# Voice Segmentation & Interface Helpers
# ---------------------------------------------------------------------------

class Listener:
    """Cuts a microphone stream into speech utterances at silence pauses."""

    def __init__(self, rate: int = 16000, pause: float = 0.6, shortest: float = 0.25, loudness: float = 3.0) -> None:
        self.rate = rate
        self.frame = int(0.03 * rate)
        self.pause_frames = int(pause / 0.03)
        self.shortest = shortest
        self.loudness = loudness
        self.floor = 1e-3
        self._buffer = np.zeros(0, dtype=np.float32)
        self._speech: list[np.ndarray] = []
        self._quiet = 0

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        """Add samples; returns utterances that finished in them."""
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32)])
        done: list[np.ndarray] = []
        while self._buffer.size >= self.frame:
            frame, self._buffer = self._buffer[: self.frame], self._buffer[self.frame :]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            loud = rms > self.loudness * self.floor
            if not loud and not self._speech:
                self.floor = 0.95 * self.floor + 0.05 * max(rms, 1e-5)
            if loud or self._speech:
                self._speech.append(frame)
                self._quiet = 0 if loud else self._quiet + 1
                if self._quiet >= self.pause_frames:
                    utterance = np.concatenate(self._speech[: len(self._speech) - self._quiet])
                    self._speech, self._quiet = [], 0
                    if utterance.size >= self.shortest * self.rate:
                        done.append(utterance)
        return done


def addressed(text: str, name: str = "sprocket", waiting: bool = False) -> str | None:
    """Check if utterance text was addressed to the robot; returns the instruction."""
    from robot.agent.instant import instant

    words = text.strip()
    if instant(words) is not None or waiting:
        return words
    low = words.lower()
    for call in (name, f"hey {name}", f"ok {name}", f"okay {name}"):
        if low.startswith(call):
            rest = words[len(call) :].lstrip(" ,.!:")
            return rest or None
    return None


def microphone(inbox: Any, ears: Ears, waiting: Any = lambda: False, device: Any = None) -> None:
    """Continuous microphone listener stream sending recognized instructions to inbox."""
    import queue

    import sounddevice

    chunks: queue.Queue[np.ndarray] = queue.Queue()
    listener = Listener()
    with sounddevice.InputStream(
        samplerate=16000,
        channels=1,
        dtype="float32",
        device=device,
        callback=lambda data, *_: chunks.put(data[:, 0].copy()),
    ):
        while True:
            for utterance in listener.feed(chunks.get()):
                meant = addressed(ears.hear(write_wav(utterance, 16000)), waiting=bool(waiting()))
                if meant:
                    inbox.put(("instruct", meant[:300]))


def attach(live: Any) -> bool:
    """Attach Ears and Mouth to the live interface when Whisper/Piper are available."""
    try:
        import faster_whisper  # noqa: F401
        import piper  # noqa: F401
    except ImportError:
        return False
    if not VOICE.exists():
        return False
    live.ears, live.mouth = Ears(), Mouth()
    live.update(voice=True)
    return True


class Ears:
    """Whisper STT transcription."""

    def __init__(self, model: str = "base.en", device: str = "cpu") -> None:
        self.model_name = model
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def hear(self, wav: bytes) -> str:
        """Transcribe speech in WAV bytes."""
        from faster_whisper import WhisperModel

        samples, rate = read_wav(wav)
        with self._lock:
            if self._model is None:
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type="int8" if self.device == "cpu" else "float16",
                )
            segments, _ = self._model.transcribe(to_16k(samples, rate), language="en", beam_size=1, vad_filter=True)
            return " ".join(s.text.strip() for s in segments).strip()


class Mouth:
    """Piper TTS synthesis."""

    def __init__(self, voice: Path = VOICE) -> None:
        self.voice_path = voice
        self._voice = None
        self._lock = threading.Lock()
        self._cache: dict[str, bytes] = {}

    def say(self, text: str) -> bytes:
        """Synthesize spoken text as WAV bytes."""
        from piper import PiperVoice

        with self._lock:
            if text not in self._cache:
                if self._voice is None:
                    self._voice = PiperVoice.load(str(self.voice_path))
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    speak = getattr(self._voice, "synthesize_wav", None) or self._voice.synthesize
                    speak(text, w)
                self._cache[text] = buf.getvalue()
                if len(self._cache) > 200:
                    self._cache.pop(next(iter(self._cache)))
            return self._cache[text]
