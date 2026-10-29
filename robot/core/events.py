"""Typed event schema, priorities, and drop policies."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any


class EventPriority(IntEnum):
    """Priority levels for internal events (lower numeric value = higher urgency)."""

    CRITICAL = 0  # E-stop, cliff/obstacle reflex, hardware fault
    HIGH = 10  # Explicit user commands, cancellation, wake word
    NORMAL = 20  # Cognitive responses, state transitions, memory writes
    LOW = 30  # Idle autonomous behaviors, background consolidation
    TELEMETRY = 40  # High-frequency sensor readings, periodic health pings


class DropPolicy(StrEnum):
    """Backpressure behavior when a subscriber queue is full."""

    NEVER_DROP = "never_drop"
    DROP_OLDEST = "drop_oldest"


def default_drop_policy(event_type: str, priority: EventPriority) -> DropPolicy:
    """Determine default backpressure policy for an event."""
    if priority <= EventPriority.NORMAL:
        return DropPolicy.NEVER_DROP
    if event_type.startswith(("sensor.", "telemetry.", "body.pose")):
        return DropPolicy.DROP_OLDEST
    return DropPolicy.NEVER_DROP


@dataclass(slots=True, frozen=True)
class Event:
    """Immutable, serializable event message routed on the EventBus."""

    type: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "ts": self.ts,
            "source": self.source,
            "priority": int(self.priority),
            "correlation_id": self.correlation_id,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        """Serialize event to a single-line JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Deserialize an Event from a dictionary."""
        raw_priority = data.get("priority", int(EventPriority.NORMAL))
        try:
            priority = EventPriority(int(raw_priority))
        except ValueError:
            priority = EventPriority.NORMAL
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            type=str(data["type"]),
            ts=float(data.get("ts", 0.0)),
            source=str(data.get("source", "unknown")),
            priority=priority,
            correlation_id=data.get("correlation_id"),
            payload=dict(data.get("payload") or {}),
        )

    @classmethod
    def from_json(cls, line: str) -> Event:
        """Deserialize an Event from a JSON string."""
        return cls.from_dict(json.loads(line))
