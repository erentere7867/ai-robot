"""Idle memory consolidation pipeline (ADD / UPDATE / DELETE / NOOP with tombstone protection)."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from robot.core.bus import EventBus
from robot.core.events import EventPriority
from robot.memory.models import MemoryProvenance
from robot.memory.policy import (
    extract_explicit_memory_proposals,
    infer_subject,
    normalize_fact_text,
)
from robot.memory.store import MemoryStore

# Implicit durable fact patterns extracted during idle review
_IMPLICIT_FACT_PATTERNS = [
    # "My cat's name is Mochi", "Our wifi guest room is upstairs", "My favorite drink is coffee"
    re.compile(
        r"\b((?:my|our)\s+[a-z0-9 '_-]{2,35}?\s+(?:is|are)\s+[^.!?,;]{2,50})",
        re.IGNORECASE,
    ),
    # "I prefer green eye color", "I really love synthwave music", "I like warm tea"
    re.compile(
        r"\b(i\s+(?:really\s+)?(?:prefer|love|like)\s+[^.!?,;]{2,50})",
        re.IGNORECASE,
    ),
    # "The charging dock is in the living room"
    re.compile(
        r"\b(the\s+(?:charging\s+dock|home\s+base|kitchen|living\s+room|office)\s+is\s+[^.!?,;]{2,50})",
        re.IGNORECASE,
    ),
]

_TRANSIENT_SUBJECTS = {
    "general",
    "doing",
    "going",
    "here",
    "ready",
    "tired",
    "bored",
    "back",
}


@dataclass(frozen=True)
class ConsolidationDecision:
    """Single candidate decision produced by idle memory consolidation."""

    op: str  # "ADD" | "UPDATE" | "DELETE" | "NOOP"
    subject: str
    text: str
    reason: str
    record_id: str | None = None
    supersedes_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "subject": self.subject,
            "text": self.text,
            "reason": self.reason,
            "record_id": self.record_id,
            "supersedes_id": self.supersedes_id,
        }


@dataclass
class ConsolidationReport:
    """Summary of an idle memory consolidation pass."""

    turns_reviewed: int = 0
    added: int = 0
    updated: int = 0
    deleted: int = 0
    noop: int = 0
    blocked_by_tombstone: int = 0
    decisions: list[ConsolidationDecision] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns_reviewed": self.turns_reviewed,
            "added": self.added,
            "updated": self.updated,
            "deleted": self.deleted,
            "noop": self.noop,
            "blocked_by_tombstone": self.blocked_by_tombstone,
            "decisions": [d.to_dict() for d in self.decisions],
        }


class MemoryConsolidator:
    """Queues conversational turns and extracts/updates durable memories during idle periods."""

    def __init__(self, bus: EventBus, store: MemoryStore, *, max_pending: int = 32) -> None:
        self.bus = bus
        self.store = store
        self._pending_turns: deque[tuple[str, str]] = deque(maxlen=max_pending)
        self.total_passes: int = 0
        self.last_report: ConsolidationReport = ConsolidationReport()

    @property
    def pending_count(self) -> int:
        return len(self._pending_turns)

    def enqueue_turn(self, user_text: str, assistant_say: str) -> None:
        """Queue a completed conversation turn for background consolidation."""
        cleaned = user_text.strip()
        if not cleaned or "?" in cleaned:
            return
        # Skip turns that were already explicit "remember ..." / "forget ..." commands
        if extract_explicit_memory_proposals(cleaned):
            return
        self._pending_turns.append((cleaned[:4096], ""))

    def clear_pending(self) -> None:
        self._pending_turns.clear()

    def extract_candidates(self, user_text: str) -> list[tuple[str, str]]:
        """Extract `(subject, normalized_fact)` pairs from a user statement."""
        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()
        # Split compound clauses on ' and ' / ' but ' / semicolons first
        clauses = re.split(r"\s+(?:and|but)\s+|[.;!?]+", user_text, flags=re.IGNORECASE)
        for clause in clauses:
            clause_str = clause.strip()
            if not clause_str:
                continue
            for pat in _IMPLICIT_FACT_PATTERNS:
                for match in pat.finditer(clause_str):
                    snippet = match.group(1).strip().rstrip(".!?,;")
                    subj = infer_subject(snippet).strip().lower()
                    if subj in _TRANSIENT_SUBJECTS or len(subj) < 2:
                        continue
                    norm_fact = normalize_fact_text(snippet)
                    key = f"{subj}::{norm_fact.lower()}"
                    if key not in seen:
                        seen.add(key)
                        candidates.append((subj, norm_fact))
        return candidates

    async def consolidate(self, *, emit_when_empty: bool = False) -> ConsolidationReport:
        """Process queued turns into ADD / UPDATE / NOOP memory operations."""
        report = ConsolidationReport(turns_reviewed=len(self._pending_turns))
        if not self._pending_turns:
            self.last_report = report
            return report

        while self._pending_turns:
            user_text, _assistant_say = self._pending_turns.popleft()
            for subj, norm_fact in self.extract_candidates(user_text):
                if self.store.is_tombstoned(subj):
                    report.noop += 1
                    report.blocked_by_tombstone += 1
                    report.decisions.append(
                        ConsolidationDecision(
                            op="NOOP",
                            subject=subj,
                            text=norm_fact,
                            reason="blocked_by_tombstone",
                        )
                    )
                    continue

                active = self.store.get_active(subj)
                active_for_subj = [active] if active else []
                if (
                    active_for_subj
                    and active_for_subj[0].content.strip().lower() == norm_fact.strip().lower()
                ):
                    report.noop += 1
                    report.decisions.append(
                        ConsolidationDecision(
                            op="NOOP",
                            subject=subj,
                            text=norm_fact,
                            reason="duplicate_active_fact",
                            record_id=active_for_subj[0].id,
                        )
                    )
                    continue

                rec = self.store.remember(
                    norm_fact,
                    subject=subj,
                    provenance=MemoryProvenance.MODEL_PROPOSED,
                    confidence=0.82,
                )
                if rec is None:
                    report.noop += 1
                    continue

                if rec.supersedes:
                    report.updated += 1
                    decision = ConsolidationDecision(
                        op="UPDATE",
                        subject=subj,
                        text=rec.content,
                        reason="superseded_previous_fact",
                        record_id=rec.id,
                        supersedes_id=rec.supersedes,
                    )
                else:
                    report.added += 1
                    decision = ConsolidationDecision(
                        op="ADD",
                        subject=subj,
                        text=rec.content,
                        reason="extracted_implicit_fact",
                        record_id=rec.id,
                    )
                report.decisions.append(decision)
                await self.bus.emit(
                    "memory.updated",
                    source="consolidator",
                    payload={"op": "remember", "status": "consolidated", "record": rec.to_dict()},
                    priority=EventPriority.LOW,
                )

        self.total_passes += 1
        self.last_report = report
        if report.decisions or emit_when_empty:
            await self.bus.emit(
                "memory.consolidated",
                source="consolidator",
                payload=report.to_dict(),
                priority=EventPriority.LOW,
            )
        return report
