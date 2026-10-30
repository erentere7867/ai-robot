"""SQLite + FTS5 persistent memory store with temporal validity, importance, and tombstones."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
import uuid
from collections import Counter
from datetime import datetime
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
from robot.memory.policy import (
    DEFAULT_RESERVED_TERMS,
    NAME_SUBJECT,
    canonical_fact,
    canonical_subject,
    default_importance,
    fact_rejection_reason,
    infer_subject,
    is_instruction_like,
    normalize_fact_text,
)

DAILY_SUMMARY_KIND = "daily_summary"
_FACT_HALF_LIFE_S = 30 * 86400.0
_EPISODE_HALF_LIFE_S = 3 * 86400.0
_MAX_QUERY_TERMS = 8
_MAX_HIGHLIGHTS = 6
_MAX_SUMMARY_CHARS = 600

_QUERY_STOPWORDS = {
    "what", "whats", "who", "where", "when", "why", "how", "which", "is", "are", "was", "were", "am", "be", "been",
    "do", "does", "did", "you", "your", "yours", "my", "mine", "our", "the", "a", "an", "remember", "know",
    "about", "tell", "me", "i", "im", "it", "its", "of", "to", "in", "on", "for", "and", "or", "that", "this",
    "these", "those", "there", "here", "can", "could", "would", "should", "will", "please", "robot", "sprocket",
    "hey", "hi", "hello", "any", "some", "have", "has", "had", "with", "at", "from", "by", "as", "if", "so", "not",
    "no", "yes", "ok", "okay", "user", "users", "just", "really", "very", "now", "then", "again", "go", "get",
    "got", "let", "lets", "us", "we", "they", "them", "he", "she", "him", "her", "said", "say", "thing", "things",
    "anything", "something", "everything", "all", "stuff", "ever", "still",
}  # fmt: skip
_SUMMARY_STOPWORDS = _QUERY_STOPWORDS | {
    "background", "event", "see", "look", "like", "think", "want", "okay", "yeah", "sure", "got", "right",
    "going", "done", "back", "one", "all", "out", "up", "down", "over", "more", "now", "let", "lets", "nothing",
    "don", "doesn", "didn", "isn", "aren", "won", "can't", "dont", "beep", "boop", "mode", "offline", "listening",
}  # fmt: skip

_SCHEMA = """
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
    updated_at REAL NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5
);
CREATE INDEX IF NOT EXISTS idx_memories_subject_active
    ON memories(subject, valid_to);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    subject,
    content,
    tokenize='porter unicode61'
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

# External-content FTS over episodes, kept in sync by triggers (pruning and compression included).
_EPISODE_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    summary,
    content='episodes',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS episodes_fts_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, summary) VALUES (new.rowid, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS episodes_fts_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, summary) VALUES ('delete', old.rowid, old.summary);
END;
CREATE TRIGGER IF NOT EXISTS episodes_fts_au AFTER UPDATE OF summary ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, summary) VALUES ('delete', old.rowid, old.summary);
    INSERT INTO episodes_fts(rowid, summary) VALUES (new.rowid, new.summary);
END;
"""

_MEMORY_COLUMNS = (
    "id, subject, content, kind, provenance, confidence, valid_from, valid_to, supersedes, sensitivity, "
    "updated_at, importance"
)


def _fold(text: str) -> str:
    """Lowercase and strip diacritics, mirroring the FTS5 unicode61 tokenizer."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _stem(word: str) -> str:
    word = word.removesuffix("'s")
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word[:-1] if len(word) > 3 and word.endswith("e") else word


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", _fold(text))


def _coverage(terms: list[str], text: str) -> float:
    """Share of query terms found in `text`: a whole word counts fully, a mere prefix ("cat" in "category")
    counts 0.5, and a document FTS matched never scores below half a term."""
    stems = {_stem(w) for w in _words(text)}
    hits = 0.0
    for term in terms:
        stem = _stem(term)
        if stem in stems:
            hits += 1.0
        elif any(s.startswith(stem) for s in stems):
            hits += 0.5
    return max(hits, 0.5) / len(terms)


def _recency(age_s: float, half_life_s: float) -> float:
    return 0.5 ** (max(0.0, age_s) / half_life_s)


def _keywords(text: str) -> list[str]:
    return [w for w in re.findall(r"[^\W\d_]{3,}", _fold(text)) if w not in _SUMMARY_STOPWORDS]


def _canon_sql(subject: object) -> str:
    return canonical_subject(str(subject or ""))


def _day_key(ts: float) -> str:
    """Local calendar day of `ts`; out-of-range timestamps fall back to a UTC day number."""
    try:
        return datetime.fromtimestamp(ts).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return f"day-{int(ts // 86400)}"


class MemoryStore:
    """Durable SQLite + FTS5 store for semantic facts, tombstones, and episodic logs."""

    def __init__(self, db_path: Path | str, clock: Clock | None = None, *, max_episodes: int = 1000) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.clock: Clock = clock or RealClock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Lets queries match subjects stored by older versions under non-canonical keys ("favourite colour").
        self._conn.create_function("canon_subject", 1, _canon_sql, deterministic=True)
        self._init_schema()
        self.max_episodes = max_episodes
        self.revision = 0
        # Tool names the agent loop registers here are never accepted as fact text.
        self.reserved_terms: set[str] = set(DEFAULT_RESERVED_TERMS)
        self._conn.execute("PRAGMA cache_size=-2048")
        self._conn.execute("PRAGMA mmap_size=0")
        self._episode_count = self._conn.execute("SELECT count(*) FROM episodes").fetchone()[0]

    # ------------------------------------------------------------------ schema / migration

    def _init_schema(self) -> None:
        had_episode_fts = self._table_exists("episodes_fts")
        with self._conn:
            self._conn.executescript(_SCHEMA)
        self._migrate_memories()
        with self._conn:
            self._conn.executescript(_EPISODE_FTS_SCHEMA)
        if not had_episode_fts or not self._episode_fts_consistent():
            with self._conn:
                self._conn.execute("INSERT INTO episodes_fts(episodes_fts) VALUES ('rebuild')")

    def _table_exists(self, name: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone()
        return row is not None

    def _migrate_memories(self) -> None:
        """Idempotently upgrade older databases: importance column and a porter-stemmed fact index."""
        columns = {str(r["name"]) for r in self._conn.execute("PRAGMA table_info(memories)")}
        if "importance" not in columns:
            with self._conn:
                self._conn.execute("ALTER TABLE memories ADD COLUMN importance REAL NOT NULL DEFAULT 0.5")
                self._conn.execute(
                    "UPDATE memories SET importance = CASE WHEN canon_subject(subject) = ? THEN 1.0 "
                    "WHEN provenance = ? THEN 0.8 ELSE 0.5 END",
                    (NAME_SUBJECT, str(MemoryProvenance.USER_EXPLICIT)),
                )
        fts_sql = self._conn.execute("SELECT sql FROM sqlite_master WHERE name='memories_fts'").fetchone()
        active = self._conn.execute("SELECT count(*) FROM memories WHERE valid_to IS NULL").fetchone()[0]
        indexed = self._conn.execute("SELECT count(*) FROM memories_fts").fetchone()[0]
        if "porter" in str(fts_sql[0] if fts_sql else "").lower() and active == indexed:
            return
        with self._conn:
            self._conn.execute("DROP TABLE IF EXISTS memories_fts")
            self._conn.execute(
                "CREATE VIRTUAL TABLE memories_fts USING fts5("
                "memory_id UNINDEXED, subject, content, tokenize='porter unicode61')"
            )
            self._conn.execute(
                "INSERT INTO memories_fts (memory_id, subject, content) "
                "SELECT id, subject, content FROM memories WHERE valid_to IS NULL"
            )

    def _episode_fts_consistent(self) -> bool:
        rows, max_rowid = self._conn.execute("SELECT count(*), max(rowid) FROM episodes").fetchone()
        docs, max_doc = self._conn.execute("SELECT count(*), max(id) FROM episodes_fts_docsize").fetchone()
        return rows == docs and max_rowid == max_doc

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ writes

    def apply(self, proposal: MemoryProposal) -> dict[str, Any]:
        """Apply a `remember` or `forget` proposal and return a structured result."""
        op = proposal.op.lower().strip()
        if op == "forget":
            count, tomb = self.forget(proposal.text, subject=proposal.subject)
            return {
                "op": "forget",
                "ok": True,
                "forgotten_count": count,
                "topic": tomb.topic,
                "tombstone_id": tomb.id,
            }
        if op == "remember":
            base = {"op": "remember", "ok": False, "text": proposal.text, "subject": proposal.subject, "id": None}
            if is_instruction_like(proposal.text, self.reserved_terms):
                return {**base, "status": "rejected", "reason": "instruction_like", "content": ""}
            rec = self.remember(
                proposal.text,
                subject=proposal.subject,
                provenance=proposal.provenance,
                confidence=proposal.confidence,
                importance=proposal.importance,
            )
            if rec is None:
                return {**base, "status": "blocked_by_tombstone", "content": ""}
            return {
                "op": "remember",
                "ok": True,
                "status": "stored",
                "subject": rec.subject,
                "content": rec.content,
                "id": rec.id,
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
        importance: float | None = None,
    ) -> MemoryRecord | None:
        """Store or update a semantic fact, superseding prior active facts on the same subject."""
        now = self.clock.now()
        norm_text = normalize_fact_text(text)[:2048]
        subj = canonical_subject(subject if subject and subject.strip() else infer_subject(text))[:160]
        if not norm_text or not subj:
            raise ValueError("Memory content and subject must not be empty")
        if importance is None:
            importance = default_importance(subj, provenance)
        importance = 1.0 if subj == NAME_SUBJECT else min(1.0, max(0.0, float(importance)))
        topics = self._matching_tombstones(subj)
        if topics and provenance != MemoryProvenance.USER_EXPLICIT:
            return None
        prior = self.get_active(subj)
        if prior and prior.content.casefold() == norm_text.casefold():
            self._refresh(prior.id, confidence=confidence, importance=importance, provenance=provenance, topics=topics)
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
            importance=importance,
        )
        # Superseding and insertion are one transaction: failure cannot lose the active fact.
        with self._conn:
            for topic in topics:
                self._conn.execute("DELETE FROM tombstones WHERE topic=?", (topic,))
            if prior:
                # Every active row on this subject, including legacy non-canonical keys, is superseded.
                stale = [
                    str(r[0])
                    for r in self._conn.execute(
                        "SELECT id FROM memories WHERE valid_to IS NULL AND (subject=? OR canon_subject(subject)=?)",
                        (subj, subj),
                    )
                ]
                for stale_id in stale:
                    self._conn.execute(
                        "UPDATE memories SET valid_to=?, updated_at=? WHERE id=?", (now, now, stale_id)
                    )
                    self._conn.execute("DELETE FROM memories_fts WHERE memory_id=?", (stale_id,))
            self._conn.execute(
                f"INSERT INTO memories ({_MEMORY_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
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
                    importance,
                ),
            )
            self._conn.execute(
                "INSERT INTO memories_fts (memory_id,subject,content) VALUES (?,?,?)",
                (rec.id, subj, norm_text),
            )
        self.revision += 1
        return rec

    def remember_fact(
        self,
        text: str,
        *,
        subject: str | None = None,
        provenance: MemoryProvenance = MemoryProvenance.MODEL_PROPOSED,
        importance: float = 0.5,
    ) -> dict[str, Any]:
        """Validate, normalize, dedupe, and store one durable fact; returns an outcome dict like `apply()`.

        Status is one of stored, updated (superseded an older fact on the same subject), duplicate,
        blocked_by_tombstone, or rejected (with `reason`).
        """
        raw = (text or "").strip()
        result: dict[str, Any] = {
            "op": "remember",
            "ok": False,
            "status": "rejected",
            "subject": None,
            "content": raw[:200],
            "id": None,
        }
        reason = fact_rejection_reason(raw, reserved_terms=self.reserved_terms)
        if reason:
            return {**result, "reason": reason}
        subj, content = canonical_fact(raw, subject)
        reason = fact_rejection_reason(content, reserved_terms=self.reserved_terms) or (
            None if subj and subj != "general" else "no_subject"
        )
        result.update(subject=subj, content=content)
        if reason:
            return {**result, "reason": reason}
        if self._matching_tombstones(subj) and provenance != MemoryProvenance.USER_EXPLICIT:
            return {**result, "status": "blocked_by_tombstone", "reason": "tombstoned"}
        importance = max(min(1.0, max(0.0, float(importance))), default_importance(subj, provenance))
        confidence = 1.0 if provenance == MemoryProvenance.USER_EXPLICIT else 0.85
        duplicate = self._find_active_content(content)
        if duplicate is not None:
            self._refresh(
                duplicate.id,
                confidence=confidence,
                importance=importance,
                provenance=provenance,
                topics=self._matching_tombstones(duplicate.subject),
            )
            refreshed = self._get_by_id(duplicate.id) or duplicate
            return {
                **result,
                "ok": True,
                "status": "duplicate",
                "subject": refreshed.subject,
                "content": refreshed.content,
                "id": refreshed.id,
                "record": refreshed.to_dict(),
            }
        rec = self.remember(
            content, subject=subj, provenance=provenance, confidence=confidence, importance=importance
        )
        if rec is None:
            return {**result, "status": "blocked_by_tombstone", "reason": "tombstoned"}
        return {
            **result,
            "ok": True,
            "status": "updated" if rec.supersedes else "stored",
            "subject": rec.subject,
            "content": rec.content,
            "id": rec.id,
            "supersedes": rec.supersedes,
            "importance": rec.importance,
            "record": rec.to_dict(),
        }

    def _refresh(
        self,
        memory_id: str,
        *,
        confidence: float,
        importance: float,
        provenance: MemoryProvenance,
        topics: list[str],
    ) -> None:
        """Re-confirm an identical active fact: bump recency, never lower confidence or importance.

        A user-explicit restatement upgrades provenance and lifts tombstones, like a fresh explicit write.
        """
        explicit = provenance == MemoryProvenance.USER_EXPLICIT
        with self._conn:
            self._conn.execute(
                "UPDATE memories SET updated_at=?, confidence=MAX(confidence,?), importance=MAX(importance,?), "
                "provenance=CASE WHEN ? THEN ? ELSE provenance END WHERE id=?",
                (self.clock.now(), confidence, importance, explicit, str(MemoryProvenance.USER_EXPLICIT), memory_id),
            )
            if explicit:
                for topic in topics:
                    self._conn.execute("DELETE FROM tombstones WHERE topic=?", (topic,))
        self.revision += 1

    def _get_by_id(self, memory_id: str) -> MemoryRecord | None:
        row = self._conn.execute(f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id=?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def _find_active_content(self, content: str) -> MemoryRecord | None:
        target = content.casefold().strip()
        rows = self._conn.execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE valid_to IS NULL AND content = ? COLLATE NOCASE",
            (content.strip(),),
        ).fetchall()
        for row in rows:
            if str(row["content"]).casefold().strip() == target:
                return self._row_to_memory(row)
        return None

    def get_active(self, subject: str) -> MemoryRecord | None:
        # Exact key first (indexed); then any key that canonicalizes alike, e.g. legacy "favourite colour".
        raw = subject.strip().lower()
        key = canonical_subject(subject)
        for sql, params in (
            ("subject IN (?, ?)", (key, raw)),
            ("canon_subject(subject) = ?", (key,)),
        ):
            row = self._conn.execute(
                f"SELECT * FROM memories WHERE valid_to IS NULL AND {sql} ORDER BY valid_from DESC LIMIT 1",
                params,
            ).fetchone()
            if row:
                return self._row_to_memory(row)
        return None

    def _matching_tombstones(self, subject: str) -> list[str]:
        # Boundary matching covers broad forget requests without matching 'cat' in 'education'.
        subject = subject.strip().lower()
        prefixes = {subject} | {subject[:i].rstrip(" '") for i, char in enumerate(subject) if char in " '" and i}
        prefixes |= {canonical_subject(p) for p in prefixes}
        prefixes.discard("")
        if not prefixes:
            return []
        placeholders = ",".join("?" for _ in prefixes)
        return [
            str(row[0])
            for row in self._conn.execute(
                f"SELECT DISTINCT topic FROM tombstones WHERE topic IN ({placeholders}) "
                f"OR canon_subject(topic) IN ({placeholders})",
                (*prefixes, *prefixes),
            )
        ]

    def forget(self, query_or_topic: str, *, subject: str | None = None) -> tuple[int, TombstoneRecord]:
        """Invalidate matching active memories and episodes, and record a tombstone."""
        if not query_or_topic.strip():
            raise ValueError("Forget topic must not be empty")
        now = self.clock.now()
        subj = canonical_subject(subject if subject and subject.strip() else infer_subject(query_or_topic))
        tokens = self._terms(query_or_topic)
        subj_pattern = re.compile(rf"(?<![\w']){re.escape(subj)}(?![\w])")

        active_rows = self._conn.execute("SELECT id, subject, content FROM memories WHERE valid_to IS NULL")
        matched: list[tuple[str, str]] = []
        for row in active_rows:
            row_subj = str(row["subject"]).lower()
            canon = canonical_subject(row_subj)
            text = f"{row_subj} {str(row['content']).lower()}"
            if canon == subj or canon.startswith((subj + " ", subj + "'")):
                matched.append((str(row["id"]), row_subj))
            elif canon.endswith("'s " + subj):
                # Someone else's attribute: "forget my name" must not erase "cat's name".
                continue
            elif subj_pattern.search(text) or (tokens and all(self._has_word(text, tok) for tok in tokens)):
                # Whole words only, so "forget my cat" spares "favorite category" and "education".
                matched.append((str(row["id"]), row_subj))
        matched_subjects = {row_subj for _, row_subj in matched}

        with self._conn:
            for mid, _ in matched:
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
            # Episodes that mention the forgotten topic would otherwise resurface it through recall().
            # Exact (stemmed) terms, not prefixes: forgetting "cat" must not delete talk about categories.
            if tokens:
                fts_all = " AND ".join(f'"{t}"' for t in tokens)
                self._conn.execute(
                    "DELETE FROM episodes WHERE rowid IN (SELECT rowid FROM episodes_fts WHERE episodes_fts MATCH ?)",
                    (fts_all,),
                )
        self._episode_count = self._conn.execute("SELECT count(*) FROM episodes").fetchone()[0]
        self.revision += 1
        return len(matched), tomb

    def is_tombstoned(self, subject_or_topic: str) -> bool:
        return bool(self._matching_tombstones(canonical_subject(subject_or_topic)))

    # ------------------------------------------------------------------ retrieval

    @staticmethod
    def _terms(query: str) -> list[str]:
        terms: list[str] = []
        for word in _words(query):
            if len(word) > 1 and word not in _QUERY_STOPWORDS and word not in terms:
                terms.append(word)
        return terms[:_MAX_QUERY_TERMS]

    @staticmethod
    def _fts_term(term: str) -> str:
        return f'"{term}"*' if len(term) >= 3 else f'"{term}"'

    @staticmethod
    def _has_word(text: str, term: str) -> bool:
        stem = _stem(term)
        return any(_stem(w) == stem for w in _words(text))

    def _fact_hits(self, terms: list[str], limit: int) -> list[tuple[MemoryRecord, float]]:
        """Active facts matching any term, ranked by relevance, importance, and recency."""
        if not terms or limit <= 0:
            return []
        rows = self._conn.execute(
            """
            SELECT m.*, bm25(memories_fts, 0.0, 2.0, 1.0) AS rank
            FROM memories_fts
            JOIN memories m ON m.id = memories_fts.memory_id
            WHERE memories_fts MATCH ? AND m.valid_to IS NULL
            ORDER BY rank ASC, m.updated_at DESC
            LIMIT ?
            """,
            (" OR ".join(self._fts_term(t) for t in terms), max(limit * 4, 20)),
        ).fetchall()
        if not rows:
            return []
        now = self.clock.now()
        best = max(-float(r["rank"]) for r in rows) or 1e-12
        term_stems = {_stem(t) for t in terms}
        scored: list[tuple[MemoryRecord, float]] = []
        for row in rows:
            rec = self._row_to_memory(row)
            relevance = 0.65 * _coverage(terms, f"{rec.subject} {rec.content}") + 0.35 * (-float(row["rank"]) / best)
            subject_stems = {_stem(w) for w in _words(rec.subject) if w not in _QUERY_STOPWORDS and w != "s"}
            bonus = 0.1 if subject_stems and subject_stems <= term_stems else 0.0
            score = (
                0.6 * relevance
                + 0.25 * rec.importance
                + 0.15 * _recency(now - rec.updated_at, _FACT_HALF_LIFE_S)
                + bonus
            )
            scored.append((rec, score))
        scored.sort(key=lambda item: (-item[1], -item[0].updated_at))
        return scored[:limit]

    def _episode_hits(self, terms: list[str], limit: int) -> list[tuple[EpisodeRecord, float]]:
        if not terms or limit <= 0:
            return []
        rows = self._conn.execute(
            """
            SELECT e.*, bm25(episodes_fts) AS rank
            FROM episodes_fts
            JOIN episodes e ON e.rowid = episodes_fts.rowid
            WHERE episodes_fts MATCH ?
            ORDER BY rank ASC, e.ts DESC
            LIMIT ?
            """,
            (" OR ".join(self._fts_term(t) for t in terms), max(limit * 4, 20)),
        ).fetchall()
        if not rows:
            return []
        now = self.clock.now()
        best = max(-float(r["rank"]) for r in rows) or 1e-12
        scored: list[tuple[EpisodeRecord, float]] = []
        for row in rows:
            ep = self._row_to_episode(row)
            relevance = 0.65 * _coverage(terms, ep.summary) + 0.35 * (-float(row["rank"]) / best)
            base = 0.4 if ep.kind == DAILY_SUMMARY_KIND else 0.3
            score = 0.6 * relevance + 0.15 * base + 0.25 * _recency(now - ep.ts, _EPISODE_HALF_LIFE_S)
            scored.append((ep, score))
        scored.sort(key=lambda item: (-item[1], -item[0].ts))
        return scored[:limit]

    def search(self, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        """Rank active facts by FTS5 relevance, importance, and recency; never pads with unrelated facts.

        A query with no meaningful terms ("what do you remember?") returns the most important facts.
        """
        terms = self._terms(query)
        if terms:
            return [rec for rec, _ in self._fact_hits(terms, limit)]
        if limit <= 0:
            return []
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE valid_to IS NULL ORDER BY importance DESC, updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def search_episodes(self, query: str, *, limit: int = 5) -> list[EpisodeRecord]:
        """Full-text search over episodes (including daily summaries), most relevant and recent first."""
        return [ep for ep, _ in self._episode_hits(self._terms(query), limit)]

    def recall(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Hybrid recall over active facts and episodes as compact dicts: kind, text, age_s, score."""
        terms = self._terms(query)
        if not terms or limit <= 0:
            return []
        now = self.clock.now()
        items: list[dict[str, Any]] = []
        fact_texts: list[str] = []
        for rec, score in self._fact_hits(terms, limit):
            fact_texts.append(rec.content.casefold())
            items.append(
                {
                    "kind": "fact",
                    "text": rec.content,
                    "age_s": round(max(0.0, now - rec.updated_at), 1),
                    "score": round(score, 4),
                    "id": rec.id,
                    "subject": rec.subject,
                    "importance": rec.importance,
                }
            )
        for ep, score in self._episode_hits(terms, limit):
            summary = ep.summary.casefold()
            if any(fact in summary for fact in fact_texts):
                continue  # The fact already carries this information more compactly.
            items.append(
                {
                    "kind": "episode",
                    "text": ep.summary,
                    "age_s": round(max(0.0, now - ep.ts), 1),
                    "score": round(score, 4),
                    "id": ep.id,
                    "episode_kind": ep.kind,
                }
            )
        items.sort(key=lambda item: (-item["score"], item["age_s"]))
        return items[:limit]

    def core_facts(self, limit: int = 6) -> list[MemoryRecord]:
        """Small always-in-context fact set: the user's name first, then important or user-stated facts."""
        if limit <= 0:
            return []
        rows = self._conn.execute(
            """
            SELECT *, canon_subject(subject) = ? AS is_name FROM memories
            WHERE valid_to IS NULL AND kind != ?
              AND (is_name OR importance >= 0.7 OR provenance = ?)
            ORDER BY is_name DESC, importance DESC, updated_at DESC
            LIMIT ?
            """,
            (NAME_SUBJECT, str(MemoryKind.EPISODIC), str(MemoryProvenance.USER_EXPLICIT), limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

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

    # ------------------------------------------------------------------ episodes

    @property
    def episode_count(self) -> int:
        return self._episode_count

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
        return [self._row_to_episode(r) for r in rows]

    def compress_episodes(self, *, older_than_s: float = 86400.0, min_group: int = 2) -> dict[str, int]:
        """Fold episodes older than `older_than_s` into one deterministic summary episode per local day."""
        cutoff = self.clock.now() - older_than_s
        rows = self._conn.execute("SELECT * FROM episodes WHERE ts < ? ORDER BY ts ASC, rowid ASC", (cutoff,))
        days: dict[str, list[EpisodeRecord]] = {}
        for row in rows.fetchall():
            ep = self._row_to_episode(row)
            days.setdefault(_day_key(ep.ts), []).append(ep)
        compressed = summaries = 0
        with self._conn:
            for day, episodes in days.items():
                raw = [e for e in episodes if e.kind != DAILY_SUMMARY_KIND]
                previous = [e for e in episodes if e.kind == DAILY_SUMMARY_KIND]
                if not raw or (not previous and len(raw) < min_group):
                    continue
                summary = self._daily_summary(day, raw, previous)
                self._conn.executemany("DELETE FROM episodes WHERE id=?", [(e.id,) for e in raw + previous])
                self._conn.execute(
                    "INSERT INTO episodes (id, ts, kind, summary, details_json) VALUES (?, ?, ?, ?, ?)",
                    (summary.id, summary.ts, summary.kind, summary.summary, json.dumps(summary.details)),
                )
                compressed += len(raw)
                summaries += 1
        if compressed:
            self._episode_count = self._conn.execute("SELECT count(*) FROM episodes").fetchone()[0]
            self.revision += 1
        return {"compressed": compressed, "summaries": summaries}

    @staticmethod
    def _daily_summary(day: str, raw: list[EpisodeRecord], previous: list[EpisodeRecord]) -> EpisodeRecord:
        kinds: Counter[str] = Counter()
        keywords: Counter[str] = Counter()
        highlights: list[tuple[float, str]] = []
        count = 0
        first_ts = min(e.ts for e in raw)
        last_ts = max(e.ts for e in raw)
        for prev in previous:
            details = prev.details
            kinds.update({str(k): int(v) for k, v in dict(details.get("kinds", {})).items()})
            keywords.update({str(k): int(v) for k, v in dict(details.get("keywords", {})).items()})
            highlights.extend((float(ts), str(text)) for ts, text in details.get("highlights", []))
            count += int(details.get("count", 0))
            first_ts = min(first_ts, float(details.get("first_ts", prev.ts)))
            last_ts = max(last_ts, float(details.get("last_ts", prev.ts)))
        for ep in raw:
            kinds[ep.kind] += 1
            keywords.update(_keywords(ep.summary))
            text = re.sub(r"\s+", " ", ep.summary).strip()
            highlights.append((ep.ts, text if len(text) <= 90 else text[:87].rstrip() + "..."))
            count += 1
        unique: dict[str, tuple[float, str]] = {}
        for ts, text in sorted(highlights):
            unique.setdefault(text.casefold(), (ts, text))

        def salience(item: tuple[float, str]) -> int:
            return sum(keywords[w] for w in set(_keywords(item[1])))

        chosen = sorted(sorted(unique.values(), key=lambda h: (-salience(h), h[0]))[:_MAX_HIGHLIGHTS])
        top_keywords = sorted(keywords.items(), key=lambda kv: (-kv[1], kv[0]))[:24]
        kinds_text = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0])))
        topics = ", ".join(word for word, _ in top_keywords[:8]) or "none"
        text = f"Day {day}: {count} episodes ({kinds_text}). Topics: {topics}. Highlights: " + " | ".join(
            h for _, h in chosen
        )
        return EpisodeRecord(
            id=uuid.uuid4().hex[:12],
            ts=last_ts,
            kind=DAILY_SUMMARY_KIND,
            summary=text[:_MAX_SUMMARY_CHARS],
            details={
                "day": day,
                "count": count,
                "kinds": dict(kinds),
                "keywords": dict(top_keywords),
                "highlights": [[ts, h] for ts, h in chosen],
                "first_ts": first_ts,
                "last_ts": last_ts,
            },
        )

    # ------------------------------------------------------------------ snapshots

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
    def _row_to_episode(row: sqlite3.Row) -> EpisodeRecord:
        return EpisodeRecord(
            id=str(row["id"]),
            ts=float(row["ts"]),
            kind=str(row["kind"]),
            summary=str(row["summary"]),
            details=json.loads(str(row["details_json"] or "{}")),
        )

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
            importance=float(row["importance"]) if "importance" in row.keys() else 0.5,
        )
