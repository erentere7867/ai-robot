"""Data models for semantic/episodic memories, tombstones, and write proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MemoryKind(StrEnum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class MemoryProvenance(StrEnum):
    USER_EXPLICIT = "user_explicit"
    MODEL_PROPOSED = "model_proposed"
    SYSTEM = "system"


@dataclass(frozen=True)
class MemoryRecord:
    """Durable memory item with temporal validity and provenance."""

    id: str
    subject: str
    content: str
    kind: MemoryKind = MemoryKind.SEMANTIC
    provenance: MemoryProvenance = MemoryProvenance.USER_EXPLICIT
    confidence: float = 1.0
    valid_from: float = 0.0
    valid_to: float | None = None
    supersedes: str | None = None
    sensitivity: str = "shared"
    updated_at: float = 0.0
    importance: float = 0.5

    @property
    def is_active(self) -> bool:
        return self.valid_to is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "content": self.content,
            "kind": str(self.kind),
            "provenance": str(self.provenance),
            "confidence": self.confidence,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "supersedes": self.supersedes,
            "sensitivity": self.sensitivity,
            "updated_at": self.updated_at,
            "importance": self.importance,
        }


@dataclass(frozen=True)
class TombstoneRecord:
    """Tombstone preventing forgotten topics from being silently re-learned."""

    id: str
    topic: str
    created_at: float
    reason: str = "user_forget_request"


@dataclass(frozen=True)
class EpisodeRecord:
    """Compact episodic log entry."""

    id: str
    ts: float
    kind: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryProposal:
    """Proposed memory operation (`remember` or `forget`)."""

    op: str  # "remember" | "forget"
    text: str
    subject: str | None = None
    provenance: MemoryProvenance = MemoryProvenance.USER_EXPLICIT
    confidence: float = 1.0
    importance: float | None = None
