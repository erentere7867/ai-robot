"""Composable 60 Hz procedural face animator with smooth interpolation and blinking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

EYE_COLOR_THEMES: dict[str, tuple[int, int, int]] = {
    "cyan": (0, 228, 255),
    "green": (46, 235, 138),
    "emerald": (46, 235, 138),
    "amber": (255, 184, 46),
    "gold": (255, 184, 46),
    "blue": (68, 150, 255),
    "violet": (182, 114, 255),
    "purple": (182, 114, 255),
    "coral": (255, 108, 118),
    "red": (255, 92, 92),
}

GAZE_PRESETS: dict[str, tuple[float, float]] = {
    "center": (0.0, 0.0),
    "speaker": (0.0, -0.08),
    "left": (-0.65, 0.0),
    "right": (0.65, 0.0),
    "up": (0.0, -0.55),
    "down": (0.0, 0.55),
}


def resolve_theme_color(name: str) -> tuple[str, tuple[int, int, int]]:
    """Return canonical theme name and RGB tuple."""
    key = (name or "cyan").strip().lower()
    for token in key.split():
        if token in EYE_COLOR_THEMES:
            canonical = (
                "green"
                if token == "emerald"
                else ("amber" if token == "gold" else ("violet" if token == "purple" else token))
            )
            return canonical, EYE_COLOR_THEMES[token]
    return "cyan", EYE_COLOR_THEMES["cyan"]


@dataclass
class FaceFrameParams:
    """Interpolated per-frame geometric parameters for the procedural vector renderer."""

    activity: str = "idle"
    emotion: str = "neutral"
    intensity: float = 0.2
    gaze_target: str = "center"
    theme: str = "cyan"
    color_rgb: tuple[int, int, int] = (0, 228, 255)
    gaze_x: float = 0.0  # -1.0 (left) .. +1.0 (right)
    gaze_y: float = 0.0  # -1.0 (up)   .. +1.0 (down)
    left_openness: float = 1.0  # 0.0 (closed) .. 1.25 (wide)
    right_openness: float = 1.0
    left_brow_deg: float = 0.0  # +angled inward/down, -raised outward
    right_brow_deg: float = 0.0
    lower_lid_squint: float = 0.0  # 0.0 .. 0.65 (happy cheek raise)
    pupil_scale: float = 1.0  # 0.7 .. 1.35
    mouth_open: float = 0.05  # 0.0 .. 1.0
    mouth_curve: float = 0.2  # -1.0 (frown) .. +1.0 (smile)
    blink_amount: float = 0.0  # 0.0 (open) .. 1.0 (fully blinked)

    def to_dict(self) -> dict[str, Any]:
        return {
            "activity": self.activity,
            "emotion": self.emotion,
            "intensity": round(self.intensity, 2),
            "gaze_target": self.gaze_target,
            "theme": self.theme,
            "color_rgb": list(self.color_rgb),
            "gaze_x": round(self.gaze_x, 3),
            "gaze_y": round(self.gaze_y, 3),
            "left_openness": round(self.left_openness, 3),
            "right_openness": round(self.right_openness, 3),
            "left_brow_deg": round(self.left_brow_deg, 2),
            "right_brow_deg": round(self.right_brow_deg, 2),
            "lower_lid_squint": round(self.lower_lid_squint, 3),
            "pupil_scale": round(self.pupil_scale, 3),
            "mouth_open": round(self.mouth_open, 3),
            "mouth_curve": round(self.mouth_curve, 3),
            "blink_amount": round(self.blink_amount, 3),
        }


def _lerp(current: float, target: float, alpha: float) -> float:
    return current + (target - current) * max(0.0, min(1.0, alpha))


class FaceAnimator:
    """Computes smooth 60 Hz procedural eye and mouth geometry from high-level states."""

    def __init__(self, *, blink_period_s: float = 3.6) -> None:
        self.blink_period_s = blink_period_s
        self.blink_duration_s = 0.16
        self._elapsed_s: float = 0.0
        self._since_blink_s: float = 0.0
        self.external_mouth_open: float | None = None
        self._current = FaceFrameParams()

    @property
    def current(self) -> FaceFrameParams:
        return self._current

    def set_mouth_open(self, value: float | None) -> None:
        """Set or clear an external real-time audio RMS mouth openness override (0.0..1.0)."""
        if value is None:
            self.external_mouth_open = None
        else:
            clamped = max(0.0, min(1.0, float(value)))
            self.external_mouth_open = clamped
            self._current.mouth_open = clamped

    def set_state(
        self,
        *,
        activity: str | None = None,
        emotion: str | None = None,
        intensity: float | None = None,
        gaze_target: str | None = None,
        theme: str | None = None,
    ) -> None:
        if activity is not None:
            self._current.activity = str(activity).strip().lower()
        if emotion is not None:
            self._current.emotion = str(emotion).strip().lower()
        if intensity is not None:
            self._current.intensity = max(0.0, min(1.0, float(intensity)))
        if gaze_target is not None:
            self._current.gaze_target = str(gaze_target).strip().lower()
        if theme is not None:
            tname, rgb = resolve_theme_color(theme)
            self._current.theme = tname
            self._current.color_rgb = rgb

    def trigger_blink(self) -> None:
        """Immediately start a natural blink cycle."""
        self._since_blink_s = self.blink_period_s

    def step(self, dt_s: float = 1.0 / 60.0) -> FaceFrameParams:
        """Advance procedural animation by `dt_s` seconds and return updated frame parameters."""
        dt = max(0.001, min(0.25, float(dt_s)))
        self._elapsed_s += dt
        self._since_blink_s += dt

        emo = self._current.emotion
        act = self._current.activity
        inten = self._current.intensity

        # 1. Target emotion geometry
        t_left_open = 1.0
        t_right_open = 1.0
        t_left_brow = 0.0
        t_right_brow = 0.0
        t_squint = 0.05
        t_pupil = 1.0
        t_mouth_curve = 0.18

        if emo == "happy":
            t_squint = 0.25 + 0.30 * inten
            t_mouth_curve = 0.45 + 0.45 * inten
            t_pupil = 1.05 + 0.10 * inten
        elif emo == "excited":
            t_left_open = 1.08 + 0.12 * inten
            t_right_open = 1.08 + 0.12 * inten
            t_squint = 0.30 + 0.25 * inten
            t_pupil = 1.15 + 0.15 * inten
            t_mouth_curve = 0.65 + 0.35 * inten
        elif emo == "curious":
            # Asymmetric curious eye & brow tilt!
            t_left_open = 1.08 + 0.10 * inten
            t_right_open = 0.85 - 0.10 * inten
            t_left_brow = -10.0 * inten
            t_right_brow = 8.0 * inten
            t_pupil = 1.10
            t_mouth_curve = 0.15
        elif emo == "confused":
            t_left_open = 0.78
            t_right_open = 1.06
            t_left_brow = 14.0 * inten
            t_right_brow = -12.0 * inten
            t_mouth_curve = -0.25 * inten
        elif emo == "surprised":
            t_left_open = 1.18 + 0.10 * inten
            t_right_open = 1.18 + 0.10 * inten
            t_left_brow = -14.0 * inten
            t_right_brow = -14.0 * inten
            t_pupil = 0.82 - 0.10 * inten
            t_mouth_curve = 0.0
        elif emo == "sleepy":
            t_left_open = 0.35 - 0.15 * inten
            t_right_open = 0.35 - 0.15 * inten
            t_left_brow = -6.0
            t_right_brow = -6.0
            t_squint = 0.10
            t_mouth_curve = 0.05
        elif emo == "sad":
            t_left_open = 0.62 - 0.12 * inten
            t_right_open = 0.62 - 0.12 * inten
            t_left_brow = -18.0 * inten
            t_right_brow = -18.0 * inten
            t_pupil = 0.92
            t_mouth_curve = -0.65 * inten
        elif emo in ("angry", "grumpy"):
            t_left_open = 0.76 - 0.10 * inten
            t_right_open = 0.76 - 0.10 * inten
            t_left_brow = 22.0 * inten
            t_right_brow = 22.0 * inten
            t_squint = 0.18 * inten
            t_pupil = 0.86
            t_mouth_curve = -0.55 * inten
        elif emo == "concerned":
            t_left_open = 0.88
            t_right_open = 0.88
            t_left_brow = -15.0 * inten
            t_right_brow = -15.0 * inten
            t_mouth_curve = -0.35 * inten
        elif emo == "proud":
            t_left_open = 0.95
            t_right_open = 0.95
            t_squint = 0.32 * inten
            t_left_brow = -6.0 * inten
            t_right_brow = -6.0 * inten
            t_mouth_curve = 0.60 * inten
        elif emo == "shy":
            t_left_open = 0.80
            t_right_open = 0.80
            t_squint = 0.30 * inten
            t_left_brow = -10.0 * inten
            t_right_brow = -10.0 * inten
            t_mouth_curve = 0.25 * inten
        elif emo == "scared":
            t_left_open = 1.20
            t_right_open = 1.20
            t_left_brow = -16.0 * inten
            t_right_brow = -16.0 * inten
            t_pupil = 0.72
            t_mouth_curve = -0.30 * inten
        elif emo == "love":
            t_left_open = 1.02
            t_right_open = 1.02
            t_squint = 0.42 * inten
            t_pupil = 1.22
            t_mouth_curve = 0.85 * inten

        # 2. Layer composable activity state
        base_gx, base_gy = GAZE_PRESETS.get(self._current.gaze_target, (0.0, 0.0))
        t_mouth_open = 0.04

        if act == "listening":
            t_left_open = min(1.25, t_left_open + 0.08)
            t_right_open = min(1.25, t_right_open + 0.08)
            t_pupil = min(1.30, t_pupil + 0.08)
        elif act == "thinking":
            # Gaze drifts upward and scans gently side-to-side while thinking
            base_gx += 0.25 * math.sin(self._elapsed_s * 2.4)
            base_gy = -0.38 + 0.08 * math.cos(self._elapsed_s * 1.8)
            t_left_open *= 0.88
            t_right_open *= 0.94
        elif act == "speaking":
            if self.external_mouth_open is not None:
                t_mouth_open = max(0.04, min(1.0, float(self.external_mouth_open)))
                t_squint = min(0.65, t_squint + 0.06 * t_mouth_open)
            else:
                # Rhythmic mouth oscillation synchronized with speaking state
                osc = 0.5 * (1.0 + math.sin(self._elapsed_s * 18.0))
                t_mouth_open = 0.18 + 0.55 * osc
                t_squint = min(0.65, t_squint + 0.06 * osc)
        elif act == "idle":
            # Subtle idle breathing micro-motion
            base_gx += 0.06 * math.sin(self._elapsed_s * 0.9)
            base_gy += 0.04 * math.cos(self._elapsed_s * 0.7)

        if self.external_mouth_open is not None:
            t_mouth_open = max(0.04, min(1.0, float(self.external_mouth_open)))
        elif emo == "surprised" and act != "speaking":
            t_mouth_open = 0.35 * inten

        # 3. Blinking envelope
        blink_amt = 0.0
        if self._since_blink_s >= self.blink_period_s:
            b_progress = (self._since_blink_s - self.blink_period_s) / self.blink_duration_s
            if b_progress >= 1.0:
                self._since_blink_s = 0.0
                blink_amt = 0.0
            else:
                # Smooth triangle/sine blink curve
                blink_amt = math.sin(b_progress * math.pi)

        # 4. Smooth exponential interpolation
        rate = min(1.0, dt * 12.0)
        gaze_rate = min(1.0, dt * 10.0)
        self._current.gaze_x = _lerp(self._current.gaze_x, base_gx, gaze_rate)
        self._current.gaze_y = _lerp(self._current.gaze_y, base_gy, gaze_rate)
        self._current.left_openness = _lerp(self._current.left_openness, t_left_open, rate) * (
            1.0 - 0.92 * blink_amt
        )
        self._current.right_openness = _lerp(self._current.right_openness, t_right_open, rate) * (
            1.0 - 0.92 * blink_amt
        )
        self._current.left_brow_deg = _lerp(self._current.left_brow_deg, t_left_brow, rate)
        self._current.right_brow_deg = _lerp(self._current.right_brow_deg, t_right_brow, rate)
        self._current.lower_lid_squint = _lerp(self._current.lower_lid_squint, t_squint, rate)
        self._current.pupil_scale = _lerp(self._current.pupil_scale, t_pupil, rate)
        self._current.mouth_open = _lerp(self._current.mouth_open, t_mouth_open, min(1.0, dt * 20.0))
        self._current.mouth_curve = _lerp(self._current.mouth_curve, t_mouth_curve, rate)
        self._current.blink_amount = blink_amt

        return self._current
