"""Direct commands from the dashboard's buttons, as the same behaviours everything else runs."""

from __future__ import annotations

from typing import Any

from robot.body.hal import BodyCapabilities
from robot.control.behaviors import Behavior, Gesture, Move, TiltHead, Turn


def behavior_for(tool: str, args: dict[str, Any], capabilities: BodyCapabilities) -> Behavior:
    if tool == "drive":
        sign = -1.0 if args.get("direction") == "backward" else 1.0
        return Move(sign * float(args["distance_mm"]))
    if tool == "spin":
        return Turn(float(args["angle_deg"]))
    if tool == "tilt_head":
        return TiltHead(capabilities.clamp_head(float(args["angle_deg"])))
    if tool == "perform_trick":
        return Gesture(capabilities.gestures[str(args["trick"])])
    raise ValueError(f"Unknown command '{tool}'")
