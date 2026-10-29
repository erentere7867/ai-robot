"""JSON schema and typed structures for constrained model output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robot.actions.tools import ToolCallRequest
from robot.memory.models import MemoryProposal


@dataclass(frozen=True)
class EmotionSpec:
    name: str = "neutral"
    intensity: float = 0.5


@dataclass
class ValidatedResponse:
    """Validated, safe cognitive response ready for execution by the BehaviorArbiter."""

    say: str
    sound: str = "none"
    emotion: EmotionSpec = field(default_factory=EmotionSpec)
    actions: list[ToolCallRequest] = field(default_factory=list)
    memory: list[MemoryProposal] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rejected_actions: list[dict[str, Any]] = field(default_factory=list)
    clamped_notes: list[str] = field(default_factory=list)
    raw_text: str = ""
    backend_name: str = "unknown"
    latency_ms: float = 0.0
    fallback_used: bool = False
    status: str = "completed"
    next_wake_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "say": self.say,
            "sound": self.sound,
            "emotion": {"name": self.emotion.name, "intensity": round(self.emotion.intensity, 2)},
            "actions": [{"tool": a.tool, "args": a.args} for a in self.actions],
            "memory": [{"op": m.op, "text": m.text, "subject": m.subject} for m in self.memory],
            "warnings": list(self.warnings),
            "rejected_actions": list(self.rejected_actions),
            "clamped_notes": list(self.clamped_notes),
            "backend": self.backend_name,
            "latency_ms": round(self.latency_ms, 1),
            "fallback_used": self.fallback_used,
            "status": self.status,
            "next_wake_s": self.next_wake_s,
        }


def build_response_json_schema(
    allowed_emotions: list[str] | None = None,
    allowed_sounds: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    action_schemas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the OpenAI-compatible JSON schema for llama.cpp grammar-constrained output."""
    emotions = allowed_emotions or [
        "neutral",
        "happy",
        "excited",
        "curious",
        "confused",
        "surprised",
        "sleepy",
        "concerned",
    ]
    sounds = allowed_sounds or [
        "none",
        "happy_trill",
        "curious_chirp",
        "soft_beep",
        "sleepy_yawn",
        "startle_pip",
        "ack_chirp",
    ]
    tools = allowed_tools or ["perform_trick", "tilt_head", "spin", "drive", "stop"]

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["say", "actions", "status"],
        "properties": {
            "say": {
                "type": "string",
                "maxLength": 400,
                "description": "Short spoken reply (1-2 concise sentences).",
            },
            "sound": {
                "type": "string",
                "enum": sounds,
            },
            "emotion": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "intensity"],
                "properties": {
                    "name": {"type": "string", "enum": emotions},
                    "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
            "actions": {
                "type": "array",
                "maxItems": 3,
                "items": {"oneOf": action_schemas}
                if action_schemas
                else {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["tool", "args"],
                    "properties": {
                        "tool": {"type": "string", "enum": tools},
                        "args": {"type": "object"},
                    },
                },
            },
            "memory": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["op", "text"],
                    "properties": {
                        "op": {"type": "string", "enum": ["remember", "forget"]},
                        "text": {"type": "string"},
                    },
                },
            },
            "status": {
                "type": "string",
                "enum": ["completed", "continue", "blocked"],
                "description": "completed ends this request. continue requests another observation and decision.",
            },
            "next_wake_s": {
                "type": "number", "minimum": 1.0, "maximum": 300.0,
                "description": "Seconds until another idle decision after completion; no movement is implied.",
            },
        },
    }
