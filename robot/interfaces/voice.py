"""Speech in and out: Whisper hears, Piper speaks (docs/architecture.md, speech; LFM3-Omni-2B is the challenger).

    ears, mouth = Ears(), Mouth()
    text = ears.hear(wav_bytes)          # 16-bit PCM WAV, any rate, mono or stereo
    wav = mouth.say("I'm on my way.")    # 16-bit PCM WAV

Whisper because its small models have NPU builds for the board's Qualcomm chip; Piper because it runs fast on an ARM
CPU. Each loads its model on first use. On the robot, audio comes from the microphone array after voice activity
detection and goes to the speaker; on the laptop it comes from the live page's talk button and goes back to the page
(robot/interfaces/live.py). What is heard goes the same way as a typed instruction: "stop", "wait" and "come back" act
at once, an answer goes to the program that asked, and anything else goes to the planner.

Always listening (`microphone`), the robot cuts the stream into utterances at pauses (`Listener`) and acts only on
what is meant for it (`addressed`): what follows its name ("Sprocket, go to the chair"), "stop" from anyone, and
anything while it waits for an answer. Talk in the room is ignored.
"""

from __future__ import annotations

import io
import threading
import wave
from pathlib import Path

import numpy as np

VOICE = Path("~/sprocket-models/voice/en_US-lessac-medium.onnx").expanduser()


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
    """Resample to the 16 kHz Whisper listens at (linear interpolation: enough for speech)."""
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


class Listener:
    """Cuts a microphone's stream into utterances: speech starts when a 30 ms frame is clearly louder than the room's
    noise floor (which it keeps learning from quiet frames), and ends after `pause` seconds of quiet."""

    def __init__(self, rate: int = 16000, pause: float = 0.6, shortest: float = 0.25, loudness: float = 3.0) -> None:
        self.rate, self.frame = rate, int(0.03 * rate)
        self.pause_frames, self.shortest = int(pause / 0.03), shortest
        self.loudness = loudness
        self.floor = 1e-3  # the room's noise level (RMS), learned
        self._buffer = np.zeros(0, dtype=np.float32)
        self._speech: list[np.ndarray] = []
        self._quiet = 0

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        """Add samples; returns the utterances that ended in them."""
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32)])
        done = []
        while self._buffer.size >= self.frame:
            frame, self._buffer = self._buffer[:self.frame], self._buffer[self.frame:]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            loud = rms > self.loudness * self.floor
            if not loud and not self._speech:
                self.floor = 0.95 * self.floor + 0.05 * max(rms, 1e-5)
            if loud or self._speech:
                self._speech.append(frame)
                self._quiet = 0 if loud else self._quiet + 1
                if self._quiet >= self.pause_frames:
                    utterance = np.concatenate(self._speech[:len(self._speech) - self._quiet])
                    self._speech, self._quiet = [], 0
                    if utterance.size >= self.shortest * self.rate:
                        done.append(utterance)
        return done


def addressed(text: str, name: str = "sprocket", waiting: bool = False) -> str | None:
    """What of an utterance is meant for the robot: the words after its name ("Sprocket, go to the chair"), an instant
    command from anyone ("stop"), or anything at all while a program waits for an answer; None for talk in the room."""
    from robot.agent.instant import instant

    words = text.strip()
    if instant(words) is not None or waiting:
        return words
    low = words.lower()
    for call in (name, f"hey {name}", f"ok {name}", f"okay {name}"):
        if low.startswith(call):
            rest = words[len(call):].lstrip(" ,.!:")
            return rest or None
    return None


def microphone(inbox: object, ears: Ears, waiting: object = lambda: False, device: object = None) -> None:
    """The robot's ear, for ever: utterances from the microphone meant for the robot go to `inbox` as instructions
    (`waiting()`: a program is waiting for an answer, so anything said is the answer). Needs `sounddevice`."""
    import queue

    import sounddevice

    chunks: queue.Queue[np.ndarray] = queue.Queue()
    listener = Listener()
    with sounddevice.InputStream(samplerate=16000, channels=1, dtype="float32", device=device,
                                 callback=lambda data, *_: chunks.put(data[:, 0].copy())):
        while True:
            for utterance in listener.feed(chunks.get()):
                meant = addressed(ears.hear(write_wav(utterance, 16000)), waiting=bool(waiting()))
                if meant:
                    inbox.put(("instruct", meant[:300]))


def attach(live: object) -> bool:
    """Give the live page ears and a mouth when Whisper, Piper and the voice are installed; False when they are not."""
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
    def __init__(self, model: str = "base.en", device: str = "cpu") -> None:
        self.model_name, self.device = model, device
        self._model = None
        self._lock = threading.Lock()

    def hear(self, wav: bytes) -> str:
        """What was said in a WAV recording; empty when it held no speech."""
        from faster_whisper import WhisperModel

        samples, rate = read_wav(wav)
        with self._lock:
            if self._model is None:
                self._model = WhisperModel(self.model_name, device=self.device,
                                           compute_type="int8" if self.device == "cpu" else "float16")
            segments, _ = self._model.transcribe(to_16k(samples, rate), language="en", beam_size=1, vad_filter=True)
            return " ".join(s.text.strip() for s in segments).strip()


class Mouth:
    def __init__(self, voice: Path = VOICE) -> None:
        self.voice_path = voice
        self._voice = None
        self._lock = threading.Lock()
        self._cache: dict[str, bytes] = {}

    def say(self, text: str) -> bytes:
        """The sentence spoken, as a WAV."""
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
