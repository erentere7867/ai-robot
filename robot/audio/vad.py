"""Voice Activity Detector (VAD) with pre-roll ring buffer and barge-in detection."""

from __future__ import annotations

import io
import wave
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VADEvent:
    """Event emitted by VoiceActivityDetector when speech starts, ends, or barges in."""

    kind: str  # "speech_start" | "speech_end" | "barge_in"
    rms: float
    audio_samples: np.ndarray | None = None
    sample_rate: int = 16_000

    def to_wav_bytes(self) -> bytes:
        if self.audio_samples is None or len(self.audio_samples) == 0:
            return b""
        pcm16 = np.clip(self.audio_samples * 32767.0, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()


class VoiceActivityDetector:
    """Frame-based RMS energy VAD with pre-roll buffering and echo-aware barge-in threshold."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_ms: int = 30,
        start_rms_threshold: float = 0.025,
        barge_in_rms_threshold: float = 0.065,
        min_speech_frames: int = 3,
        silence_hangover_frames: int = 18,
        pre_roll_frames: int = 8,
        max_utterance_frames: int = 350,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.start_rms_threshold = start_rms_threshold
        self.barge_in_rms_threshold = barge_in_rms_threshold
        self.min_speech_frames = min_speech_frames
        self.silence_hangover_frames = silence_hangover_frames
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_frames)
        self.max_utterance_frames = max_utterance_frames

        self.in_speech: bool = False
        self._consecutive_voiced: int = 0
        self._consecutive_silence: int = 0
        self._utterance_frames: list[np.ndarray] = []

    def reset(self) -> None:
        self.in_speech = False
        self._consecutive_voiced = 0
        self._consecutive_silence = 0
        self._utterance_frames.clear()
        self._pre_roll.clear()

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms // 1000)

    @staticmethod
    def compute_rms(frame: np.ndarray) -> float:
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        robot_is_speaking: bool = False,
    ) -> list[VADEvent]:
        """Process a single mono float32 audio frame and return any triggered VAD events."""
        mono = np.asarray(frame, dtype=np.float32).reshape(-1)
        rms = self.compute_rms(mono)
        events: list[VADEvent] = []

        # Higher threshold while robot speaker is active to prevent self-triggering,
        # while still allowing deliberate user barge-in!
        threshold = self.barge_in_rms_threshold if robot_is_speaking else self.start_rms_threshold
        voiced = rms >= threshold

        if not self.in_speech:
            self._pre_roll.append(mono)
            if voiced:
                self._consecutive_voiced += 1
                if self._consecutive_voiced >= self.min_speech_frames:
                    self.in_speech = True
                    self._consecutive_silence = 0
                    self._utterance_frames = list(self._pre_roll)
                    if robot_is_speaking:
                        events.append(VADEvent(kind="barge_in", rms=rms, sample_rate=self.sample_rate))
                    events.append(VADEvent(kind="speech_start", rms=rms, sample_rate=self.sample_rate))
            else:
                self._consecutive_voiced = 0
        else:
            self._utterance_frames.append(mono)
            if voiced:
                self._consecutive_silence = 0
            else:
                self._consecutive_silence += 1

            if (
                self._consecutive_silence >= self.silence_hangover_frames
                or len(self._utterance_frames) >= self.max_utterance_frames
            ):
                full_audio = np.concatenate(self._utterance_frames).astype(np.float32)
                self.reset()
                events.append(
                    VADEvent(
                        kind="speech_end",
                        rms=rms,
                        audio_samples=full_audio,
                        sample_rate=self.sample_rate,
                    )
                )

        return events
