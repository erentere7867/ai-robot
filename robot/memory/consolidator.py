"""Idle memory consolidation pipeline (ADD / UPDATE / DELETE / NOOP with tombstone protection)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from robot.core.bus import EventBus
from robot.core.events import EventPriority
from robot.memory.models import MemoryProvenance
from robot.memory.policy import (
    FactCandidate,
    canonical_subject,
    extract_explicit_memory_proposals,
    extract_fact_candidates,
)
from robot.memory.store import MemoryStore

# Below this share of max_episodes the store only compresses days older than the normal window.
_EPISODE_PRESSURE = 0.8
_PRESSURE_COMPRESS_AFTER_S = 3600.0


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
    episodes_compressed: int = 0
    episode_summaries: int = 0
    decisions: list[ConsolidationDecision] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns_reviewed": self.turns_reviewed,
            "added": self.added,
            "updated": self.updated,
            "deleted": self.deleted,
            "noop": self.noop,
            "blocked_by_tombstone": self.blocked_by_tombstone,
            "episodes_compressed": self.episodes_compressed,
            "episode_summaries": self.episode_summaries,
            "decisions": [d.to_dict() for d in self.decisions],
        }


class MemoryConsolidator:
    """Queues conversational turns, extracts durable facts, and compresses old episodes during idle periods."""

    def __init__(
        self,
        bus: EventBus,
        store: MemoryStore,
        *,
        max_pending: int = 32,
        episode_compress_after_s: float = 86400.0,
    ) -> None:
        self.bus = bus
        self.store = store
        self._pending_turns: deque[tuple[str, str]] = deque(maxlen=max_pending)
        self.episode_compress_after_s = episode_compress_after_s
        self.total_passes: int = 0
        self.last_report: ConsolidationReport = ConsolidationReport()

    @property
    def pending_count(self) -> int:
        return len(self._pending_turns)

    def enqueue_turn(self, user_text: str, assistant_say: str) -> None:
        """Queue a completed conversation turn for background consolidation."""
        cleaned = user_text.strip()
        if not cleaned:
            return
        explicit = extract_explicit_memory_proposals(cleaned)
        if any(p.op == "forget" for p in explicit):
            return
        candidates = extract_fact_candidates(cleaned)
        if explicit:
            # The arbiter already applied the explicit fact; queue only if the turn carries others.
            covered = {canonical_subject(p.subject or "") for p in explicit}
            if all(c.subject in covered for c in candidates):
                return
        elif "?" in cleaned and not candidates:
            return
        self._pending_turns.append((cleaned[:4096], ""))

    def clear_pending(self) -> None:
        self._pending_turns.clear()

    def extract_candidate_facts(self, user_text: str) -> list[FactCandidate]:
        """Extract durable facts (with provenance and importance) from a user statement."""
        return extract_fact_candidates(user_text)

    def extract_candidates(self, user_text: str) -> list[tuple[str, str]]:
        """Extract `(subject, normalized_fact)` pairs from a user statement."""
        return [(c.subject, c.text) for c in self.extract_candidate_facts(user_text)]

    def compress_episodes(self) -> dict[str, int]:
        """Fold old episodes into per-day summaries; compress sooner when near the episode cap."""
        older_than = self.episode_compress_after_s
        if self.store.episode_count >= _EPISODE_PRESSURE * self.store.max_episodes:
            older_than = min(older_than, _PRESSURE_COMPRESS_AFTER_S)
        return self.store.compress_episodes(older_than_s=older_than)

    async def consolidate(self, *, emit_when_empty: bool = False) -> ConsolidationReport:
        """Process queued turns into ADD / UPDATE / NOOP memory operations, then compress old episodes."""
        report = ConsolidationReport(turns_reviewed=len(self._pending_turns))
        reviewed = bool(self._pending_turns)

        while self._pending_turns:
            user_text, _assistant_say = self._pending_turns.popleft()
            for cand in self.extract_candidate_facts(user_text):
                await self._consolidate_candidate(cand, report)

        compression = self.compress_episodes()
        report.episodes_compressed = compression["compressed"]
        report.episode_summaries = compression["summaries"]
        if reviewed:
            self.total_passes += 1
        self.last_report = report
        if report.decisions or report.episodes_compressed or emit_when_empty:
            await self.bus.emit(
                "memory.consolidated",
                source="consolidator",
                payload=report.to_dict(),
                priority=EventPriority.LOW,
            )
        return report

    async def _consolidate_candidate(self, cand: FactCandidate, report: ConsolidationReport) -> None:
        subj, norm_fact = cand.subject, cand.text
        if self.store.is_tombstoned(subj):
            report.noop += 1
            report.blocked_by_tombstone += 1
            report.decisions.append(
                ConsolidationDecision(op="NOOP", subject=subj, text=norm_fact, reason="blocked_by_tombstone")
            )
            return

        active = self.store.get_active(subj)
        if active and active.content.strip().lower() == norm_fact.strip().lower():
            report.noop += 1
            report.decisions.append(
                ConsolidationDecision(
                    op="NOOP",
                    subject=subj,
                    text=norm_fact,
                    reason="duplicate_active_fact",
                    record_id=active.id,
                )
            )
            return

        user_stated = cand.provenance == MemoryProvenance.USER_EXPLICIT
        rec = self.store.remember(
            norm_fact,
            subject=subj,
            provenance=cand.provenance,
            confidence=0.9 if user_stated else 0.82,
            importance=cand.importance,
        )
        if rec is None:
            report.noop += 1
            return

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
                reason="extracted_user_fact" if user_stated else "extracted_implicit_fact",
                record_id=rec.id,
            )
        report.decisions.append(decision)
        await self.bus.emit(
            "memory.updated",
            source="consolidator",
            payload={"op": "remember", "status": "consolidated", "record": rec.to_dict()},
            priority=EventPriority.LOW,
        )
