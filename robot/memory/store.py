"""SQLite + FTS5 persistent memory store with temporal validity and tombstones."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from robot.core.clock import Clock, RealClock
from robot.memory.models import (
    EpisodeRecord,
    MemoryKind,
    MemoryProposal,
    MemoryProvenance,
    MemoryRecord,
    TombstoneRecord,
)
from robot.memory.policy import infer_subject, normalize_fact_text

_QUERY_STOPWORDS = {
    "what",
    "whats",
    "who",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "you",
    "your",
    "my",
    "our",
    "the",
    "a",
    "an",
    "remember",
    "know",
    "about",
    "tell",
    "me",
}


class MemoryStore:
    """Durable SQLite + FTS5 store for semantic facts, tombstones, and episodic logs."""

    def __init__(self, db_path: Path | str, clock: Clock | None = None, *, max_episodes: int = 1000) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.clock: Clock = clock or RealClock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self.max_episodes = max_episodes
        self.revision = 0
        self._conn.execute("PRAGMA cache_size=-2048")
        self._conn.execute("PRAGMA mmap_size=0")
        self._episode_count = self._conn.execute("SELECT count(*) FROM episodes").fetchone()[0]

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    valid_from REAL NOT NULL,
                    valid_to REAL,
                    supersedes TEXT,
                    sensitivity TEXT NOT NULL DEFAULT 'shared',
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_subject_active
                    ON memories(subject, valid_to);

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    memory_id UNINDEXED,
                    subject,
                    content
                );

                CREATE TABLE IF NOT EXISTS tombstones (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    reason TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tombstones_topic
                    ON tombstones(topic);

                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts DESC);
                CREATE INDEX IF NOT EXISTS idx_memories_active_updated ON memories(updated_at DESC) WHERE valid_to IS NULL;
                """
            )

    def close(self) -> None:
        self._conn.close()

    def apply(self, proposal: MemoryProposal) -> dict[str, Any]:
        """Apply a `remember` or `forget` proposal and return a structured result."""
        op = proposal.op.lower().strip()
        if op == "forget":
            count, tomb = self.forget(proposal.text, subject=proposal.subject)
            return {
                "op": "forget",
                "forgotten_count": count,
                "topic": tomb.topic,
                "tombstone_id": tomb.id,
            }
        if op == "remember":
            rec = self.remember(
                proposal.text,
                subject=proposal.subject,
                provenance=proposal.provenance,
                confidence=proposal.confidence,
            )
            if rec is None:
                return {"op": "remember", "status": "blocked_by_tombstone", "text": proposal.text}
            return {
                "op": "remember",
                "status": "stored",
                "record": rec.to_dict(),
            }
        raise ValueError(f"Unsupported memory operation: {proposal.op}")

    def remember(
        self,
        text: str,
        *,
        subject: str | None = None,
        kind: MemoryKind = MemoryKind.SEMANTIC,
        provenance: MemoryProvenance = MemoryProvenance.USER_EXPLICIT,
        confidence: float = 1.0,
    ) -> MemoryRecord | None:
        """Store or update a semantic fact, superseding prior active facts on the same subject."""
        now = self.clock.now()
        norm_text = normalize_fact_text(text)[:2048]
        subj = (subject or infer_subject(text)).strip().lower()[:160]
        if not norm_text or not subj:
            raise ValueError("Memory content and subject must not be empty")
        topics = self._matching_tombstones(subj)
        if topics and provenance != MemoryProvenance.USER_EXPLICIT:
            return None
        prior = self.get_active(subj)
        if prior and prior.content.casefold() == norm_text.casefold():
            with self._conn:
                self._conn.execute(
                    "UPDATE memories SET updated_at=?, confidence=MAX(confidence,?) WHERE id=?",
                    (now, confidence, prior.id),
                )
            self.revision += 1
            return self.get_active(subj)
        rec = MemoryRecord(
            id=uuid.uuid4().hex[:12],
            subject=subj,
            content=norm_text,
            kind=kind,
            provenance=provenance,
            confidence=float(confidence),
            valid_from=now,
            valid_to=None,
            supersedes=prior.id if prior else None,
            sensitivity="shared",
            updated_at=now,
        )
        # Superseding and insertion are one transaction: failure cannot lose the active fact.
        with self._conn:
            for topic in topics:
                self._conn.execute("DELETE FROM tombstones WHERE topic=?", (topic,))
            if prior:
                self._conn.execute(
                    "UPDATE memories SET valid_to=?, updated_at=? WHERE subject=? AND valid_to IS NULL",
                    (now, now, subj),
                )
                self._conn.execute("DELETE FROM memories_fts WHERE memory_id=?", (prior.id,))
            self._conn.execute(
                "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rec.id,
                    subj,
                    norm_text,
                    str(kind),
                    str(provenance),
                    rec.confidence,
                    now,
                    None,
                    rec.supersedes,
                    "shared",
                    now,
                ),
            )
            self._conn.execute(
                "INSERT INTO memories_fts (memory_id,subject,content) VALUES (?,?,?)",
                (rec.id, subj, norm_text),
            )
        self.revision += 1
        return rec

    def get_active(self, subject: str) -> MemoryRecord | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE subject=? AND valid_to IS NULL ORDER BY valid_from DESC LIMIT 1",
            (subject.lower(),),
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def _matching_tombstones(self, subject: str) -> list[str]:
        # Boundary matching covers broad forget requests without matching 'cat' in 'education'.
        prefixes = {subject} | {subject[:i] for i, char in enumerate(subject) if char in " '" and i}
        placeholders = ",".join("?" for _ in prefixes)
        return [
            str(row[0])
            for row in self._conn.execute(
                f"SELECT DISTINCT topic FROM tombstones WHERE topic IN ({placeholders})", tuple(prefixes)
            )
        ]

    def forget(self, query_or_topic: str, *, subject: str | None = None) -> tuple[int, TombstoneRecord]:
        """Invalidate matching active memories and record a tombstone."""
        if not query_or_topic.strip():
            raise ValueError("Forget topic must not be empty")
        now = self.clock.now()
        subj = (subject or infer_subject(query_or_topic)).strip().lower()
        tokens = [
            t
            for t in re.findall(r"[a-z0-9]+", query_or_topic.lower())
            if t not in _QUERY_STOPWORDS and len(t) > 1
        ]

        active_rows = self._conn.execute("SELECT id, subject, content FROM memories WHERE valid_to IS NULL")

        matched_ids: list[str] = []
        matched_subjects: set[str] = set()
        for row in active_rows:
            row_subj = str(row["subject"]).lower()
            row_content = str(row["content"]).lower()
            if row_subj == subj or subj in row_subj or subj in row_content:
                matched_ids.append(str(row["id"]))
                matched_subjects.add(row_subj)
            elif tokens and all(tok in f"{row_subj} {row_content}" for tok in tokens):
                matched_ids.append(str(row["id"]))
                matched_subjects.add(row_subj)

        with self._conn:
            for mid in matched_ids:
                self._conn.execute(
                    "UPDATE memories SET valid_to = ?, updated_at = ? WHERE id = ?",
                    (now, now, mid),
                )
                self._conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (mid,))

            tomb = TombstoneRecord(
                id=uuid.uuid4().hex[:12],
                topic=subj,
                created_at=now,
                reason="user_forget_request",
            )
            for topic in matched_subjects | {subj}:
                self._conn.execute(
                    "INSERT INTO tombstones (id,topic,created_at,reason) SELECT ?,?,?,? WHERE NOT EXISTS (SELECT 1 FROM tombstones WHERE topic=?)",
                    (tomb.id if topic == subj else uuid.uuid4().hex[:12], topic, now, tomb.reason, topic),
                )

        self.revision += 1
        return len(matched_ids), tomb

    def is_tombstoned(self, subject_or_topic: str) -> bool:
        return bool(self._matching_tombstones(subject_or_topic.strip().lower()))

    def search(self, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        """Search active memories using FTS5 keyword matching with fallback to active facts."""
        tokens = [
            t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in _QUERY_STOPWORDS and len(t) > 1
        ]
        results: list[MemoryRecord] = []
        seen_ids: set[str] = set()

        if tokens:
            fts_query = " OR ".join(f'"{t}"*' for t in tokens[:6])
            rows = self._conn.execute(
                """
                SELECT m.*, bm25(memories_fts) AS rank
                FROM memories_fts f
                JOIN memories m ON m.id = f.memory_id
                WHERE memories_fts MATCH ? AND m.valid_to IS NULL
                ORDER BY rank ASC, m.updated_at DESC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
            for r in rows:
                rec = self._row_to_memory(r)
                results.append(rec)
                seen_ids.add(rec.id)

        if not tokens and len(results) < limit:
            fallback_rows = self._conn.execute(
                """
                SELECT * FROM memories
                WHERE valid_to IS NULL
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            for r in fallback_rows:
                rec = self._row_to_memory(r)
                if rec.id not in seen_ids:
                    results.append(rec)
                    seen_ids.add(rec.id)
                if len(results) >= limit:
                    break

        return results

    def list_active(self, limit: int | None = None) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories WHERE valid_to IS NULL ORDER BY updated_at DESC"
        rows = self._conn.execute(
            sql + (" LIMIT ?" if limit is not None else ""), (limit,) if limit is not None else ()
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def list_tombstones(self, limit: int | None = None) -> list[TombstoneRecord]:
        sql = "SELECT * FROM tombstones ORDER BY created_at DESC"
        rows = self._conn.execute(
            sql + (" LIMIT ?" if limit is not None else ""), (limit,) if limit is not None else ()
        ).fetchall()
        return [
            TombstoneRecord(
                id=str(r["id"]),
                topic=str(r["topic"]),
                created_at=float(r["created_at"]),
                reason=str(r["reason"]),
            )
            for r in rows
        ]

    def log_episode(
        self,
        summary: str,
        *,
        kind: str = "interaction",
        details: dict[str, Any] | None = None,
    ) -> EpisodeRecord:
        ep = EpisodeRecord(
            id=uuid.uuid4().hex[:12],
            ts=self.clock.now(),
            kind=kind,
            summary=summary,
            details=dict(details or {}),
        )
        with self._conn:
            self._conn.execute(
                "INSERT INTO episodes (id, ts, kind, summary, details_json) VALUES (?, ?, ?, ?, ?)",
                (ep.id, ep.ts, ep.kind, ep.summary, json.dumps(ep.details)),
            )
            if self._episode_count >= self.max_episodes:
                self._conn.execute(
                    "DELETE FROM episodes WHERE id IN (SELECT id FROM episodes ORDER BY ts DESC, rowid DESC LIMIT -1 OFFSET ?)",
                    (self.max_episodes,),
                )
                self._episode_count = self.max_episodes
            else:
                self._episode_count += 1
        self.revision += 1
        return ep

    def recent_episodes(self, limit: int = 5) -> list[EpisodeRecord]:
        rows = self._conn.execute(
            "SELECT * FROM episodes ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            EpisodeRecord(
                id=str(r["id"]),
                ts=float(r["ts"]),
                kind=str(r["kind"]),
                summary=str(r["summary"]),
                details=json.loads(str(r["details_json"] or "{}")),
            )
            for r in rows
        ]

    def dashboard_snapshot(self) -> dict[str, Any]:
        return {
            "active_memories": [m.to_dict() for m in self.list_active(limit=100)],
            "tombstones": [t.__dict__ for t in self.list_tombstones(limit=100)],
            "recent_episodes": [e.__dict__ for e in self.recent_episodes(limit=10)],
        }

    def export_all(self) -> dict[str, Any]:
        all_rows = self._conn.execute("SELECT * FROM memories ORDER BY valid_from ASC").fetchall()
        return {
            "active_memories": [m.to_dict() for m in self.list_active()],
            "all_memories": [self._row_to_memory(r).to_dict() for r in all_rows],
            "tombstones": [t.__dict__ for t in self.list_tombstones()],
            "recent_episodes": [e.__dict__ for e in self.recent_episodes(limit=10)],
        }

    @staticmethod
    def _row_to_memory(row: sqlite3.Row, *, updated_at: float | None = None) -> MemoryRecord:
        return MemoryRecord(
            id=str(row["id"]),
            subject=str(row["subject"]),
            content=str(row["content"]),
            kind=MemoryKind(str(row["kind"])),
            provenance=MemoryProvenance(str(row["provenance"])),
            confidence=float(row["confidence"]),
            valid_from=float(row["valid_from"]),
            valid_to=float(row["valid_to"]) if row["valid_to"] is not None else None,
            supersedes=str(row["supersedes"]) if row["supersedes"] is not None else None,
            sensitivity=str(row["sensitivity"]),
            updated_at=float(updated_at if updated_at is not None else row["updated_at"]),
        )
