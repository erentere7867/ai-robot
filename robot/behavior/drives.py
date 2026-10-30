"""Homeostatic drives: the pet's own reasons to act when nobody is talking to it.

Drives are cheap numbers that rise over time and fall when satisfied. When one crosses its
threshold the autonomy controller wakes the model with a feeling ("you feel curious...") and the
model decides what, if anything, to do about it. Nothing here prescribes an action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SATISFIERS = {
    "curiosity": {"observe": 0.25, "look_around": 0.6, "explore": 0.8, "go_to": 0.2},
    "play": {"express": 0.5, "turn": 0.1, "move": 0.1},
    "social": {},
    "energy": {},
}


@dataclass
class Drive:
    name: str
    level: float
    rate_per_min: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {"level": round(self.level, 2), "threshold": self.threshold, "rate_per_min": self.rate_per_min}


class DriveSystem:
    def __init__(self, *, rate_scale: float = 1.0, low_battery_pct: float = 30.0) -> None:
        self.rate_scale = rate_scale
        self.low_battery_pct = low_battery_pct
        self.drives = {
            "curiosity": Drive("curiosity", 0.55, 0.06, 0.7),
            "social": Drive("social", 0.0, 0.04, 0.8),
            "play": Drive("play", 0.2, 0.03, 0.8),
            "energy": Drive("energy", 0.0, 0.0, 0.6),  # driven by the battery, not by time
        }
        self.minutes_since_human = 0.0
        self.unexplored_fraction = 1.0
        self.battery_pct: float | None = None
        self.charging = False

    def step(self, dt_s: float, *, battery_pct: float | None = None, charging: bool = False,
             unexplored_fraction: float | None = None) -> None:
        minutes = max(0.0, dt_s) / 60.0
        self.minutes_since_human += minutes
        for drive in self.drives.values():
            drive.level = min(1.0, drive.level + drive.rate_per_min * self.rate_scale * minutes)
        if unexplored_fraction is not None:
            self.unexplored_fraction = unexplored_fraction
            # A fully explored room is less interesting.
            cap = 0.5 + 0.5 * min(1.0, unexplored_fraction * 2.5)
            self.drives["curiosity"].level = min(self.drives["curiosity"].level, cap)
        self.battery_pct, self.charging = battery_pct, charging
        if battery_pct is not None:
            energy = 0.0 if charging else max(0.0, (self.low_battery_pct + 20.0 - battery_pct) / 40.0)
            self.drives["energy"].level = min(1.0, energy)

    def on_turn(self, result: Any, *, autonomous: bool) -> None:
        if not autonomous:
            self.minutes_since_human = 0.0
            self.drives["social"].level = 0.0
        for entry in getattr(result, "tool_log", []):
            for name, amounts in SATISFIERS.items():
                if entry.get("ok") and entry.get("tool") in amounts:
                    self.drives[name].level = max(0.0, self.drives[name].level - amounts[entry["tool"]])
        if autonomous and not getattr(result, "tool_log", None):
            # Choosing to do nothing still settles the urge for a while.
            for drive in self.drives.values():
                if drive.level >= drive.threshold:
                    drive.level = max(0.0, drive.threshold - 0.25)

    def urgent(self) -> tuple[str, str] | None:
        """Most pressing drive above threshold and a feeling the model can act on."""
        over = [d for d in self.drives.values() if d.level >= d.threshold]
        if not over:
            return None
        drive = max(over, key=lambda d: d.level - d.threshold + (0.5 if d.name == "energy" else 0.0))
        return drive.name, self._feeling(drive)

    def _feeling(self, drive: Drive) -> str:
        if drive.name == "curiosity":
            return (f"You feel curious (curiosity {drive.level:.1f}). About {self.unexplored_fraction:.0%} of "
                    "the area around you is still unexplored.")
        if drive.name == "social":
            return (f"You feel a bit lonely: the human has not talked to you for {self.minutes_since_human:.0f} "
                    "minutes. You could say one short, friendly thing to them, or simply wait.")
        if drive.name == "play":
            return "You feel playful and full of energy."
        return (f"Your battery is low ({self.battery_pct:.0f}%). Driving onto your charging dock recharges "
                "you; you may need to find it first.")

    def snapshot(self) -> dict[str, Any]:
        return {"drives": {k: d.to_dict() for k, d in self.drives.items()},
                "minutes_since_human": round(self.minutes_since_human, 1),
                "unexplored_fraction": round(self.unexplored_fraction, 2),
                "battery_pct": self.battery_pct, "charging": self.charging}
