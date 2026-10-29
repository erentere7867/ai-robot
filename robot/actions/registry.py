"""Capability registry generated dynamically from body.yaml."""

from __future__ import annotations

import math
from typing import Any

from robot.actions.tools import ToolCallRequest, ToolValidationOutcome
from robot.body.hal import BodyCapabilities
from robot.face.animator import EYE_COLOR_THEMES, GAZE_PRESETS


class CapabilityRegistry:
    """Derives available tools, bounds, and prompt summaries from BodyCapabilities."""

    def __init__(self, capabilities: BodyCapabilities) -> None:
        self.caps = capabilities

    @property
    def tool_names(self) -> list[str]:
        return [
            "perform_trick",
            "tilt_head",
            "spin",
            "drive",
            "set_gaze",
            "set_eye_color",
            "stop",
        ]

    @property
    def trick_names(self) -> list[str]:
        return list(self.caps.tricks.keys())

    @property
    def allowed_emotions(self) -> list[str]:
        return list(self.caps.allowed_emotions)

    @property
    def allowed_sounds(self) -> list[str]:
        return list(self.caps.allowed_sounds)

    def to_dict(self) -> dict[str, Any]:
        """Return structured capability registry metadata for dashboard observability."""
        tilt = self.caps.joints["head_tilt"]
        return {
            "robot_id": self.caps.robot_id,
            "description": self.caps.description,
            "tools": [
                {
                    "name": "perform_trick",
                    "args": {"trick": self.trick_names},
                    "description": "Execute a coordinated multi-step body/head trick",
                },
                {
                    "name": "tilt_head",
                    "args": {"angle_deg": f"{tilt.min_deg:.0f}..{tilt.max_deg:.0f}"},
                    "description": "Tilt head pitch up (+) or down (-) within joint limits",
                },
                {
                    "name": "spin",
                    "args": {
                        "angle_deg": f"-{self.caps.max_single_spin_deg:.0f}..+{self.caps.max_single_spin_deg:.0f}"
                    },
                    "description": "Rotate 4WD skid-steer base in place (+CCW/left, -CW/right)",
                },
                {
                    "name": "drive",
                    "args": {
                        "direction": ["forward", "backward"],
                        "distance_mm": f"10..{self.caps.max_single_move_mm:.0f}",
                    },
                    "description": "Drive forward or backward with obstacle & cliff reflexes",
                },
                {
                    "name": "set_gaze",
                    "args": {"target": list(GAZE_PRESETS.keys())},
                    "description": "Direct eye gaze toward a target direction",
                },
                {
                    "name": "set_eye_color",
                    "args": {"color": sorted(set(EYE_COLOR_THEMES.keys()))},
                    "description": "Change procedural eye display color theme",
                },
                {
                    "name": "stop",
                    "args": {},
                    "description": "Immediately halt all motors and active actions",
                },
            ],
            "tricks": {
                k: {"description": v.description, "duration_s": v.duration_s}
                for k, v in self.caps.tricks.items()
            },
            "joints": {
                k: {"min_deg": v.min_deg, "max_deg": v.max_deg, "max_speed_deg_s": v.max_speed_deg_s}
                for k, v in self.caps.joints.items()
            },
            "drive_limits": {
                "max_linear_speed_mm_s": self.caps.max_linear_speed_mm_s,
                "max_angular_speed_deg_s": self.caps.max_angular_speed_deg_s,
                "max_single_move_mm": self.caps.max_single_move_mm,
                "max_single_spin_deg": self.caps.max_single_spin_deg,
                "min_front_tof_stop_mm": self.caps.min_front_tof_stop_mm,
            },
            "allowed_emotions": self.allowed_emotions,
            "allowed_sounds": self.allowed_sounds,
        }

    def format_for_prompt(self) -> str:
        """Render a compact, token-budget-friendly summary of tools and limits for the LLM."""
        tilt = self.caps.joints["head_tilt"]
        return (
            f"Tools: perform_trick(trick in [{', '.join(self.caps.tricks)}]); "
            f"tilt_head(angle_deg {tilt.min_deg:g}..{tilt.max_deg:g}); "
            f"spin(angle_deg {-self.caps.max_single_spin_deg:g}..{self.caps.max_single_spin_deg:g}, +left); "
            f"drive(direction forward/backward, distance_mm >0..{self.caps.max_single_move_mm:g}); "
            "set_gaze(target center/speaker/left/right/up/down); "
            "set_eye_color(color cyan/green/amber/blue/violet/coral/red); stop().\n"
            f"Emotions: {', '.join(self.caps.allowed_emotions)}.\n"
            f"Sounds: {', '.join(self.caps.allowed_sounds)}."
        )

    def action_schemas(self) -> list[dict[str, Any]]:
        """Grammar and execution derive tool arguments from the same capabilities."""
        caps = self.caps
        tilt = caps.joints["head_tilt"]
        arguments = {
            "perform_trick": {"trick": {"type": "string", "enum": self.trick_names}},
            "tilt_head": {"angle_deg": {"type": "number", "minimum": tilt.min_deg, "maximum": tilt.max_deg}},
            "spin": {
                "angle_deg": {
                    "type": "number",
                    "minimum": -caps.max_single_spin_deg,
                    "maximum": caps.max_single_spin_deg,
                }
            },
            "drive": {
                "direction": {"type": "string", "enum": ["forward", "backward"]},
                "distance_mm": {"type": "number", "exclusiveMinimum": 0, "maximum": caps.max_single_move_mm},
            },
            "set_gaze": {"target": {"type": "string", "enum": list(GAZE_PRESETS)}},
            "set_eye_color": {"color": {"type": "string", "enum": list(EYE_COLOR_THEMES)}},
            "stop": {},
        }
        return [
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["tool", "args"],
                "properties": {
                    "tool": {"type": "string", "enum": [tool]},
                    "args": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(args),
                        "properties": args,
                    },
                },
            }
            for tool, args in arguments.items()
        ]

    def validate_tool_call(self, call: ToolCallRequest) -> ToolValidationOutcome:
        """Validate and normalize a proposed tool call against hardware limits."""
        tool = str(call.tool or "").strip().lower()
        if not isinstance(call.args, dict):
            return ToolValidationOutcome(valid=False, tool=tool, error="Tool args must be an object")
        args = dict(call.args)
        warnings: list[str] = []
        clamped = False

        if tool not in self.tool_names:
            return ToolValidationOutcome(
                valid=False,
                tool=tool,
                error=f"Unknown tool '{tool}'. Available tools: {', '.join(self.tool_names)}.",
            )

        required = {
            "tilt_head": ("angle_deg", "angle"),
            "spin": ("angle_deg", "angle"),
            "drive": ("distance_mm", "distance"),
            "perform_trick": ("trick",),
        }
        if tool in required and not any(key in args for key in required[tool]):
            return ToolValidationOutcome(
                valid=False, tool=tool, error=f"Missing required argument for {tool}"
            )
        for key in ("angle_deg", "angle", "distance_mm", "distance", "speed_mm_s", "_requested_angle_deg"):
            if key in args:
                try:
                    if isinstance(args[key], bool) or not math.isfinite(float(args[key])):
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    return ToolValidationOutcome(
                        valid=False, tool=tool, error=f"{key} must be a finite number"
                    )

        if tool == "stop":
            return ToolValidationOutcome(valid=True, tool="stop", normalized_args={})

        if tool == "set_gaze":
            target = str(args.get("target", args.get("gaze", "center"))).strip().lower()
            if target not in GAZE_PRESETS:
                return ToolValidationOutcome(
                    valid=False,
                    tool=tool,
                    error=f"Invalid gaze target '{target}'. Allowed: {list(GAZE_PRESETS.keys())}.",
                )
            return ToolValidationOutcome(valid=True, tool=tool, normalized_args={"target": target})

        if tool == "set_eye_color":
            color = str(args.get("color", args.get("theme", "cyan"))).strip().lower()
            if color not in EYE_COLOR_THEMES:
                return ToolValidationOutcome(
                    valid=False,
                    tool=tool,
                    error=f"Unknown eye color '{color}'. Allowed: {sorted(EYE_COLOR_THEMES.keys())}.",
                )
            return ToolValidationOutcome(valid=True, tool=tool, normalized_args={"color": color})

        if tool == "perform_trick":
            trick = str(args.get("trick", "")).strip().lower()
            if trick not in self.caps.tricks:
                return ToolValidationOutcome(
                    valid=False,
                    tool=tool,
                    error=f"Unknown trick '{trick}'. Allowed tricks: {', '.join(self.trick_names)}.",
                )
            return ToolValidationOutcome(
                valid=True,
                tool=tool,
                normalized_args={"trick": trick},
            )

        if tool == "tilt_head":
            spec = self.caps.joints["head_tilt"]
            raw_angle = args.get("angle_deg", args.get("angle", 0.0))
            try:
                angle = float(raw_angle)
            except (TypeError, ValueError):
                return ToolValidationOutcome(
                    valid=False,
                    tool=tool,
                    error=f"Invalid angle_deg '{raw_angle}' for tilt_head.",
                )
            bounded = max(spec.min_deg, min(spec.max_deg, angle))
            if abs(bounded - angle) > 1e-3:
                clamped = True
                warnings.append(
                    f"head_tilt requested {angle:.1f}° exceeds physical range "
                    f"[{spec.min_deg:.0f}°, {spec.max_deg:.0f}°]; clamped to {bounded:.1f}°."
                )
            return ToolValidationOutcome(
                valid=True,
                tool=tool,
                normalized_args={"angle_deg": bounded, "_requested_angle_deg": angle},
                clamped=clamped,
                warnings=warnings,
            )

        if tool == "spin":
            raw_angle = args.get("angle_deg", args.get("angle", 90.0))
            try:
                angle = float(raw_angle)
            except (TypeError, ValueError):
                return ToolValidationOutcome(
                    valid=False,
                    tool=tool,
                    error=f"Invalid angle_deg '{raw_angle}' for spin.",
                )
            max_spin = self.caps.max_single_spin_deg
            bounded = max(-max_spin, min(max_spin, angle))
            if abs(bounded - angle) > 1e-3:
                clamped = True
                warnings.append(
                    f"spin requested {angle:.1f}° exceeds single-move limit ±{max_spin:.0f}°; clamped to {bounded:.1f}°."
                )
            return ToolValidationOutcome(
                valid=True,
                tool=tool,
                normalized_args={"angle_deg": bounded},
                clamped=clamped,
                warnings=warnings,
            )

        if tool == "drive":
            direction = str(args.get("direction", "forward")).strip().lower()
            if direction in ("back", "reverse"):
                direction = "backward"
            if direction not in ("forward", "backward"):
                return ToolValidationOutcome(
                    valid=False,
                    tool=tool,
                    error=f"Invalid drive direction '{direction}'. Must be 'forward' or 'backward'.",
                )
            raw_dist = args.get("distance_mm", args.get("distance", 200.0))
            try:
                dist = float(raw_dist)
            except (TypeError, ValueError):
                return ToolValidationOutcome(
                    valid=False,
                    tool=tool,
                    error=f"Invalid distance_mm '{raw_dist}' for drive.",
                )
            if dist <= 0:
                return ToolValidationOutcome(
                    valid=False,
                    tool=tool,
                    error=f"drive distance_mm must be positive (got {dist}).",
                )
            bounded = min(dist, self.caps.max_single_move_mm)
            if abs(bounded - dist) > 1e-3:
                clamped = True
                warnings.append(
                    f"drive distance {dist:.0f} mm exceeds max single move "
                    f"{self.caps.max_single_move_mm:.0f} mm; clamped to {bounded:.0f} mm."
                )
            normalized = {"direction": direction, "distance_mm": bounded}
            if "speed_mm_s" in args:
                speed = float(args["speed_mm_s"])
                if speed <= 0:
                    return ToolValidationOutcome(valid=False, tool=tool, error="speed_mm_s must be positive")
                normalized["speed_mm_s"] = min(speed, self.caps.max_linear_speed_mm_s)
            return ToolValidationOutcome(
                valid=True,
                tool=tool,
                normalized_args=normalized,
                clamped=clamped,
                warnings=warnings,
            )

        return ToolValidationOutcome(valid=False, tool=tool, error=f"Unsupported tool '{tool}'.")
