"""Composable expression and emotion state with smooth decay toward neutral."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from robot.core.clock import Clock, RealClock


@dataclass
class ExpressionSnapshot:
    """Composable facial/body expression parameters."""

    activity: str = "idle"  # idle | listening | thinking | speaking | acting
    emotion: str = (
        "neutral"  # neutral | happy | excited | curious | confused | surprised | sleepy | concerned
    )
    intensity: float = 0.2  # 0.0 .. 1.0
    gaze_target: str = "center"  # center | speaker | left | right | up | down
    blink_rate_hz: float = 0.25
    manual_override: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "activity": self.activity,
            "emotion": self.emotion,
            "intensity": round(self.intensity, 2),
            "gaze_target": self.gaze_target,
            "blink_rate_hz": round(self.blink_rate_hz, 2),
            "manual_override": self.manual_override,
        }


class EmotionEngine:
    """Maintains composable activity/emotion/gaze state with time-based decay to neutral."""

    def __init__(
        self,
        clock: Clock | None = None,
        *,
        allowed_emotions: list[str] | None = None,
        decay_rate_per_s: float = 0.08,
    ) -> None:
        self.clock: Clock = clock or RealClock()
        self.allowed_emotions = set(
            allowed_emotions
            or ["neutral", "happy", "excited", "curious", "confused", "surprised", "sleepy", "concerned"]
        )
        self.decay_rate_per_s = decay_rate_per_s
        self._state = ExpressionSnapshot()
        self._last_update_ts = self.clock.monotonic()

    def set_activity(self, activity: str, *, gaze_target: str | None = None) -> ExpressionSnapshot:
        self.step_decay()
        self._state.activity = activity
        if gaze_target is not None:
            self._state.gaze_target = gaze_target
        return self.snapshot()

    def set_emotion(
        self,
        emotion: str,
        intensity: float = 0.7,
        *,
        gaze_target: str | None = None,
        manual_override: bool = False,
    ) -> ExpressionSnapshot:
        self.step_decay()
        norm = emotion.strip().lower()
        if not math.isfinite(float(intensity)):
            raise ValueError("Emotion intensity must be finite")
        if norm not in self.allowed_emotions:
            norm = "neutral"
        self._state.emotion = norm
        self._state.intensity = max(0.0, min(1.0, float(intensity)))
        self._state.manual_override = manual_override
        if gaze_target is not None:
            self._state.gaze_target = gaze_target
        self._last_update_ts = self.clock.monotonic()
        return self.snapshot()

    def clear_manual_override(self) -> None:
        self._state.manual_override = False

    def step_decay(self) -> ExpressionSnapshot:
        """Decay reactive emotion intensity back toward neutral over elapsed time."""
        now = self.clock.monotonic()
        dt = max(0.0, now - self._last_update_ts)
        self._last_update_ts = now
        if self._state.manual_override or self._state.emotion in ("neutral", "sleepy"):
            return self.snapshot()

        if dt > 0 and self.decay_rate_per_s > 0:
            new_intensity = self._state.intensity - (dt * self.decay_rate_per_s)
            if new_intensity <= 0.22:
                self._state.emotion = "neutral"
                self._state.intensity = 0.2
            else:
                self._state.intensity = new_intensity
        return self.snapshot()

    def snapshot(self) -> ExpressionSnapshot:
        return ExpressionSnapshot(
            activity=self._state.activity,
            emotion=self._state.emotion,
            intensity=self._state.intensity,
            gaze_target=self._state.gaze_target,
            blink_rate_hz=self._state.blink_rate_hz,
            manual_override=self._state.manual_override,
        )
