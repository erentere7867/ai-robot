"""Long-term memory for Sprocket: SQLite records that persist across restarts (the stack is still being decided).

Short-term memory is the agent's own conversation for the current task, its to-do list, and a summary of the last
few tasks (robot/agent/agent.py). This module keeps what lasts longer:
   - Objects with volatility half-lives ("blocker": 25m, "movable": 2h, "anchor": 7d).
   - Habit EMA linking movable items to nearby furniture anchors (e.g. keys on sofa vs desk).
   - Live visual overwrite ("Eyes beat old memory"): clearing blockers and absent movables.
   - People profiles with voiceprint embeddings and appearance tracking.
   - Semantic facts with deterministic fast vector search.
   - Route notes with expiration.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from robot.perception.tracker import Tracker

# Volatility half-lives in seconds
HALF_LIFE: dict[str, float] = {
    "blocker": 1500.0,    # ~25 minutes
    "movable": 7200.0,    # ~2 hours
    "anchor": 604800.0,   # 7 days
}

# Furniture / structural anchor labels that persist over long timescales
ANCHOR_LABELS: frozenset[str] = frozenset({
    "sofa", "couch", "table", "desk", "bed", "fridge", "cabinet", "shelf", "chair"
})

# Canonical mappings for household item synonyms to ensure high semantic similarity
CANONICAL_SYNONYMS: dict[str, str] = {
    "couch": "sofa",
    "sofa": "sofa",
    "backup": "spare",
    "spare": "spare",
    "mug": "cup",
    "cup": "cup",
    "bag": "backpack",
    "backpack": "backpack",
    "rucksack": "backpack",
    "tv": "television",
    "television": "television",
    "monitor": "television",
    "keys": "key",
    "key": "key",
    "refrigerator": "fridge",
    "fridge": "fridge",
    "trash can": "bin",
    "garbage can": "bin",
    "bin": "bin",
    "dining table": "table",
    "coffee table": "table",
    "table": "table",
    "office chair": "chair",
    "armchair": "chair",
    "chair": "chair",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY,
    label TEXT,
    color TEXT,
    x REAL,
    y REAL,
    radius REAL,
    var REAL,
    hits INTEGER,
    present INTEGER,
    verified INTEGER,
    first_seen REAL,
    last_seen REAL,
    kind TEXT,
    anchor TEXT,
    visual_vec BLOB,
    width REAL,
    depth_extent REAL,
    orientation_yaw REAL
);
CREATE TABLE IF NOT EXISTS anchors_habits (
    item_label TEXT,
    anchor_name TEXT,
    score REAL,
    last_updated REAL,
    PRIMARY KEY (item_label, anchor_name)
);
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    voice_vec BLOB,
    look_vec BLOB,
    room TEXT,
    last_seen REAL
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    room TEXT,
    vec BLOB,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS route_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor TEXT,
    note TEXT,
    x REAL,
    y REAL,
    kind TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS places (
    name TEXT PRIMARY KEY,
    x REAL,
    y REAL,
    yaw REAL
);
"""


def _vec_to_blob(vec: np.ndarray | None) -> bytes | None:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def _blob_to_vec(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "in", "on", "at", "to", "of", "for", "by", "with", "from", "into", "about",
    "my", "your", "our", "his", "her", "their", "its", "i", "me", "you", "we", "they", "it",
    "this", "that", "these", "those", "there", "here",
    "where", "what", "who", "when", "which", "how",
    "did", "do", "does", "have", "has", "had", "can", "could", "would", "should", "will",
    "and", "or", "but", "so", "if", "then",
})


def embed_text(text: str, dim: int = 128) -> np.ndarray:
    """Deterministic fast vector embedding: strips function words, maps synonyms to canonical stems,

    hashes word unigrams, bigrams, and character trigrams into a float32 unit-norm vector.
    """
    raw_tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if not raw_tokens:
        return np.zeros(dim, dtype=np.float32)

    content = [t for t in raw_tokens if t not in STOPWORDS] or raw_tokens
    tokens = [CANONICAL_SYNONYMS.get(t, t) for t in content]
    vec = np.zeros(dim, dtype=np.float32)

    def _hash_add(feature: str, weight: float) -> None:
        h = int.from_bytes(hashlib.md5(feature.encode("utf-8")).digest()[:8], "little")
        idx = (h >> 1) % dim
        sign = 1.0 if (h & 1) else -1.0
        vec[idx] += sign * weight

    # Word unigrams and character trigrams
    for t in tokens:
        _hash_add(f"w:{t}", 1.0)
        padded = f"_{t}_"
        for i in range(len(padded) - 2):
            _hash_add(f"c:{padded[i:i+3]}", 0.5)

    # Word bigrams
    for i in range(len(tokens) - 1):
        _hash_add(f"b:{tokens[i]}_{tokens[i+1]}", 1.5)

    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 1e-9 else np.zeros(dim, dtype=np.float32)


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Safe cosine similarity dot product over unit vectors; returns 0.0 for None or empty inputs."""
    if a is None or b is None:
        return 0.0
    va = np.asarray(a, dtype=np.float32).ravel()
    vb = np.asarray(b, dtype=np.float32).ravel()
    if va.size == 0 or vb.size == 0 or va.shape != vb.shape:
        return 0.0
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    dot = float(np.dot(va, vb) / (na * nb))
    return float(np.clip(dot, -1.0, 1.0))


def freshness(last_seen_wall: float, kind: str, now_wall: float | None = None) -> float:
    """Compute the temporal freshness factor 0.5 ** (dt / half_life) for a given record kind."""
    if now_wall is None:
        now_wall = time.time()
    hl = HALF_LIFE.get(kind, 7200.0)
    dt = max(0.0, now_wall - last_seen_wall)
    return float(0.5 ** (dt / hl))


# ---------------------------------------------------------------------------
# Long-Term Circulating Memory (SQLite)
# ---------------------------------------------------------------------------

@dataclass
class PersonProfile:
    id: int
    name: str
    voice_vec: np.ndarray | None = None
    look_vec: np.ndarray | None = None
    room: str = ""
    last_seen: float = 0.0


class CirculatingMemory:
    """Long-term memory in SQLite: objects, habits, people, facts, route notes and places."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()

        with self.lock, self.db:
            self.db.executescript(SCHEMA)
            self._migrate_schema()

    def close(self) -> None:
        """Close the SQLite connection (safe to call more than once)."""
        with self.lock:
            self.db.close()

    def _migrate_schema(self) -> None:
        """Add any missing columns to existing database tables if opened from earlier versions."""
        cur = self.db.execute("PRAGMA table_info(objects)")
        existing_cols = {row[1] for row in cur.fetchall()}
        additions = [
            ("kind", "TEXT DEFAULT 'movable'"),
            ("anchor", "TEXT DEFAULT ''"),
            ("visual_vec", "BLOB"),
            ("width", "REAL DEFAULT 0.0"),
            ("depth_extent", "REAL DEFAULT 0.0"),
            ("orientation_yaw", "REAL DEFAULT 0.0"),
        ]
        for col_name, col_def in additions:
            if col_name not in existing_cols:
                try:
                    self.db.execute(f"ALTER TABLE objects ADD COLUMN {col_name} {col_def}")
                except sqlite3.OperationalError:
                    pass

    def save(
        self,
        tracker: Tracker,
        notes: list[str] = (),
        places: dict[str, Any] | None = None,
        now_wall: float | None = None,
    ) -> None:
        """Upsert active confirmed non-rejected non-person tracks, update habit EMAs near anchors,

        persist new facts, save places, and run circulation eviction.
        """
        if now_wall is None:
            now_wall = time.time()
        wall, mono = now_wall, time.monotonic()
        active_tracks = [t for t in list(tracker.tracks.values()) if t.confirmed and not t.rejected and not t.person]

        # Identify anchor tracks to update habit EMAs for nearby movable items
        anchor_tracks = [
            t for t in active_tracks
            if CANONICAL_SYNONYMS.get(t.label.lower(), t.label.lower()) in ANCHOR_LABELS or t.label.lower() in ANCHOR_LABELS
        ]

        upsert_rows = []
        for t in active_tracks:
            c_label = CANONICAL_SYNONYMS.get(t.label.lower(), t.label.lower())
            is_anchor = c_label in ANCHOR_LABELS or t.label.lower() in ANCHOR_LABELS
            kind = "anchor" if is_anchor else "movable"

            anchor_name = ""
            if not is_anchor and anchor_tracks:
                best_anc, best_dist = None, math.inf
                for a in anchor_tracks:
                    d = math.hypot(t.x - a.x, t.y - a.y)
                    if d < best_dist:
                        best_anc, best_dist = a, d
                if best_anc is not None and best_dist <= (best_anc.radius + t.radius + 1.2):
                    anchor_name = best_anc.label
                    self.record_habit(t.label, anchor_name, alpha=0.3, now_wall=wall)

            color = t.color
            first_wall = wall - (mono - t.first_seen)
            last_wall = wall - (mono - t.last_seen)

            width = float(getattr(t, "width", 0.0))
            depth_extent = float(getattr(t, "depth_extent", 0.0))
            orientation_yaw = float(getattr(t, "orientation_yaw", 0.0))
            visual_vec = getattr(t, "visual_vec", None)
            vblob = _vec_to_blob(visual_vec)

            upsert_rows.append((
                t.id, t.label, color, t.x, t.y, t.radius, t.var, t.hits,
                int(t.present), int(t.verified), first_wall, last_wall,
                kind, anchor_name, vblob, width, depth_extent, orientation_yaw
            ))

        with self.lock, self.db:
            # Upsert into objects
            self.db.executemany(
                """
                INSERT INTO objects (
                    id, label, color, x, y, radius, var, hits, present, verified,
                    first_seen, last_seen, kind, anchor, visual_vec, width, depth_extent, orientation_yaw
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    label=excluded.label, color=excluded.color, x=excluded.x, y=excluded.y,
                    radius=excluded.radius, var=excluded.var, hits=excluded.hits, present=excluded.present,
                    verified=excluded.verified, first_seen=excluded.first_seen, last_seen=excluded.last_seen,
                    kind=excluded.kind, anchor=excluded.anchor, visual_vec=excluded.visual_vec,
                    width=excluded.width, depth_extent=excluded.depth_extent, orientation_yaw=excluded.orientation_yaw
                """,
                upsert_rows,
            )

            # Places update
            if places is not None:
                self.db.execute("DELETE FROM places")
                place_rows = []
                for name, p in places.items():
                    if name == "start":
                        continue
                    px = getattr(p, "x", p[0] if isinstance(p, (tuple, list)) else 0.0)
                    py = getattr(p, "y", p[1] if isinstance(p, (tuple, list)) else 0.0)
                    pyaw = getattr(p, "yaw", p[2] if isinstance(p, (tuple, list)) else 0.0)
                    place_rows.append((name, float(px), float(py), float(pyaw)))
                self.db.executemany("INSERT INTO places VALUES (?,?,?,?)", place_rows)

        # Record facts avoiding exact duplicates without overwriting existing timestamps
        for n in (notes or ()):
            self.remember_fact(n, now_wall=wall, refresh=False)

        # Run routine circulation
        self.circulate(now_wall=wall)

    def restore(self, tracker: Tracker, now_wall: float | None = None) -> int:
        """Run circulate eviction, then restore surviving objects into the tracker; returns restored count."""
        from robot.perception.tracker import Track

        self.circulate(now_wall)
        if now_wall is None:
            now_wall = time.time()
        wall, mono = now_wall, time.monotonic()

        with self.lock:
            rows = self.db.execute(
                """
                SELECT id, label, color, x, y, radius, var, hits, present, verified,
                       first_seen, last_seen, kind, anchor, visual_vec, width, depth_extent, orientation_yaw
                FROM objects
                """
            ).fetchall()

        for row in rows:
            (
                oid, label, color, x, y, radius, var, hits, present, verified,
                first, last, kind, anchor, vblob, width, depth_extent, orientation_yaw
            ) = row
            t = Track(
                oid, label, x, y, radius, mono - (wall - first), mono - (wall - last),
                hits=max(2, hits), present=bool(present), verified=bool(verified), var=var,
                colors=Counter({color: 1} if color else {}), votes=Counter({label: 1.0}),
                score=1.0,
            )
            t.kind = kind
            t.anchor = anchor
            t.width = width
            t.depth_extent = depth_extent
            t.orientation_yaw = orientation_yaw
            if vblob is not None:
                t.visual_vec = _blob_to_vec(vblob)
            tracker.tracks[oid] = t

        if rows:
            tracker._next = max([tracker._next, *(r[0] + 1 for r in rows)])
        return len(rows)

    def record_habit(
        self, item_label: str, anchor_name: str, alpha: float = 0.3, now_wall: float | None = None
    ) -> None:
        """EMA update for habit anchor score: new_score = (1 - alpha) * old + alpha * 1.0;

        decays competing anchors for this item_label by (1 - alpha).
        """
        item_label = item_label.strip()
        anchor_name = anchor_name.strip()
        if not item_label or not anchor_name:
            return
        if now_wall is None:
            now_wall = time.time()

        with self.lock, self.db:
            rows = self.db.execute(
                "SELECT anchor_name, score FROM anchors_habits WHERE item_label = ?",
                (item_label,),
            ).fetchall()
            existing = dict(rows)

            if anchor_name in existing:
                new_score = (1.0 - alpha) * existing[anchor_name] + alpha * 1.0
                for anc, old_s in existing.items():
                    s = new_score if anc == anchor_name else (1.0 - alpha) * old_s
                    self.db.execute(
                        "UPDATE anchors_habits SET score = ?, last_updated = ? WHERE item_label = ? AND anchor_name = ?",
                        (s, now_wall, item_label, anc),
                    )
            else:
                for anc, old_s in existing.items():
                    self.db.execute(
                        "UPDATE anchors_habits SET score = ?, last_updated = ? WHERE item_label = ? AND anchor_name = ?",
                        ((1.0 - alpha) * old_s, now_wall, item_label, anc),
                    )
                self.db.execute(
                    "INSERT INTO anchors_habits (item_label, anchor_name, score, last_updated) VALUES (?, ?, ?, ?)",
                    (item_label, anchor_name, alpha * 1.0, now_wall),
                )

    def usual_anchors(self, item_label: str, limit: int = 5) -> list[tuple[str, float]]:
        """Return the usual anchors for item_label sorted descending by habit score."""
        with self.lock:
            rows = self.db.execute(
                "SELECT anchor_name, score FROM anchors_habits WHERE item_label = ? ORDER BY score DESC LIMIT ?",
                (item_label, limit),
            ).fetchall()
            return [(str(name), float(score)) for name, score in rows]

    def observe_clear_space(self, x: float, y: float, radius: float = 0.6) -> None:
        """Immediately delete route_notes within radius and mark present=1 movables as present=0."""
        rad_sq = radius * radius
        with self.lock, self.db:
            notes = self.db.execute("SELECT id, x, y FROM route_notes").fetchall()
            for nid, nx, ny in notes:
                if (nx - x) ** 2 + (ny - y) ** 2 <= rad_sq:
                    self.db.execute("DELETE FROM route_notes WHERE id = ?", (nid,))

            objs = self.db.execute(
                "SELECT id, x, y FROM objects WHERE present = 1 AND kind = 'movable'"
            ).fetchall()
            for oid, ox, oy in objs:
                if (ox - x) ** 2 + (oy - y) ** 2 <= rad_sq:
                    self.db.execute("UPDATE objects SET present = 0 WHERE id = ?", (oid,))

    def circulate(self, now_wall: float | None = None) -> int:
        """Evict route_notes and movable objects whose freshness has dropped below 0.10."""
        if now_wall is None:
            now_wall = time.time()
        evicted = 0
        with self.lock, self.db:
            notes = self.db.execute("SELECT id, created_at, kind FROM route_notes").fetchall()
            for nid, cat, k in notes:
                kind = k or "blocker"
                if freshness(cat, kind, now_wall) < 0.10:
                    self.db.execute("DELETE FROM route_notes WHERE id = ?", (nid,))
                    evicted += 1

            objs = self.db.execute("SELECT id, last_seen, kind FROM objects WHERE kind = 'movable'").fetchall()
            for oid, ls, _ in objs:
                if freshness(ls, "movable", now_wall) < 0.10:
                    self.db.execute("DELETE FROM objects WHERE id = ?", (oid,))
                    evicted += 1
        return evicted

    def upsert_person(
        self,
        name: str,
        voice_vec: np.ndarray | None = None,
        look_vec: np.ndarray | None = None,
        room: str = "",
        now_wall: float | None = None,
    ) -> PersonProfile:
        """Blend/update voiceprint and replace/blend appearance vector for a recognized or new person."""
        if now_wall is None:
            now_wall = time.time()
        name = name.strip()

        with self.lock, self.db:
            row = self.db.execute(
                "SELECT id, name, voice_vec, look_vec, room, last_seen FROM people WHERE name = ?",
                (name,),
            ).fetchone()
            if row is not None:
                pid, _, old_v_blob, old_l_blob, old_room, _ = row
                old_voice = _blob_to_vec(old_v_blob)
                old_look = _blob_to_vec(old_l_blob)

                if voice_vec is not None:
                    if old_voice is not None:
                        blended_voice = 0.7 * old_voice + 0.3 * voice_vec
                        norm = float(np.linalg.norm(blended_voice))
                        blended_voice = (blended_voice / norm).astype(np.float32) if norm > 1e-9 else blended_voice
                    else:
                        blended_voice = voice_vec.astype(np.float32)
                else:
                    blended_voice = old_voice

                new_look = look_vec.astype(np.float32) if look_vec is not None else old_look
                new_room = room if room else old_room

                self.db.execute(
                    "UPDATE people SET voice_vec = ?, look_vec = ?, room = ?, last_seen = ? WHERE id = ?",
                    (_vec_to_blob(blended_voice), _vec_to_blob(new_look), new_room, now_wall, pid),
                )
                return PersonProfile(
                    id=pid,
                    name=name,
                    voice_vec=blended_voice,
                    look_vec=new_look,
                    room=new_room,
                    last_seen=now_wall,
                )
            else:
                v_blob = _vec_to_blob(voice_vec)
                l_blob = _vec_to_blob(look_vec)
                cur = self.db.execute(
                    "INSERT INTO people (name, voice_vec, look_vec, room, last_seen) VALUES (?, ?, ?, ?, ?)",
                    (name, v_blob, l_blob, room, now_wall),
                )
                pid = cur.lastrowid or 1
                return PersonProfile(
                    id=pid,
                    name=name,
                    voice_vec=voice_vec,
                    look_vec=look_vec,
                    room=room,
                    last_seen=now_wall,
                )

    def identify_speaker(
        self,
        voice_vec: np.ndarray | None,
        look_vec: np.ndarray | None = None,
        threshold: float = 0.65,
        now_wall: float | None = None,
    ) -> PersonProfile | None:
        """Match speaker by voice_vec similarity; update person's look_vec and last_seen when matched."""
        if voice_vec is None:
            return None
        if now_wall is None:
            now_wall = time.time()

        with self.lock, self.db:
            rows = self.db.execute("SELECT id, name, voice_vec, look_vec, room, last_seen FROM people").fetchall()
            best_row, best_sim = None, -1.0
            for row in rows:
                stored_voice = _blob_to_vec(row[2])
                sim = cosine_similarity(voice_vec, stored_voice)
                if sim > best_sim:
                    best_sim = sim
                    best_row = row

            if best_row is not None and best_sim >= threshold:
                pid, name, v_blob, l_blob, room, _ = best_row
                stored_voice = _blob_to_vec(v_blob)
                if look_vec is not None:
                    updated_look = look_vec.astype(np.float32)
                    self.db.execute(
                        "UPDATE people SET look_vec = ?, last_seen = ? WHERE id = ?",
                        (_vec_to_blob(updated_look), now_wall, pid),
                    )
                else:
                    updated_look = _blob_to_vec(l_blob)
                    self.db.execute("UPDATE people SET last_seen = ? WHERE id = ?", (now_wall, pid))
                return PersonProfile(
                    id=pid,
                    name=name,
                    voice_vec=stored_voice,
                    look_vec=updated_look,
                    room=room,
                    last_seen=now_wall,
                )
            return None

    def remember_fact(self, text: str, room: str = "", now_wall: float | None = None,
                      *, refresh: bool = True) -> None:
        """Store a semantic fact with text embedding, avoiding exact duplicate entries."""
        text = text.strip()
        if not text:
            return
        if now_wall is None:
            now_wall = time.time()
        with self.lock, self.db:
            exists = self.db.execute("SELECT 1 FROM facts WHERE text = ?", (text,)).fetchone()
            if exists is None:
                vec = embed_text(text)
                self.db.execute(
                    "INSERT INTO facts (text, room, vec, created_at) VALUES (?, ?, ?, ?)",
                    (text, room, _vec_to_blob(vec), now_wall),
                )
            elif refresh:
                self.db.execute(
                    "UPDATE facts SET created_at = ?, room = CASE WHEN ? != '' THEN ? ELSE room END WHERE text = ?",
                    (now_wall, room, room, text),
                )

    def notes(self) -> list[str]:
        """Return all remembered fact texts in chronological order."""
        with self.lock:
            rows = self.db.execute("SELECT text FROM facts ORDER BY id").fetchall()
            return [r[0] for r in rows]

    def query_facts(self, query: str, top_k: int = 3, min_score: float = 0.25) -> list[tuple[str, float]]:
        """Semantic vector similarity search over facts; returns (text, created_at), newest first on ties."""
        q_vec = embed_text(query)
        with self.lock:
            rows = self.db.execute("SELECT id, text, vec, created_at FROM facts").fetchall()
        scored = []
        for row_id, text, vec_blob, created_at in rows:
            f_vec = _blob_to_vec(vec_blob)
            sim = cosine_similarity(q_vec, f_vec)
            if sim >= min_score:
                scored.append((sim, float(created_at or 0.0), row_id, text))
        scored.sort(key=lambda x: (round(x[0], 4), x[1], x[2]), reverse=True)
        picked = scored[:top_k]
        if top_k > 1 and len(scored) > top_k:
            cutoff = 0.8 * picked[0][0]
            close = [item for item in scored if item[0] >= cutoff]
            newest = max(close, key=lambda x: (x[1], x[2]))
            if newest not in picked:
                picked[-1] = newest
        return [(text, created_at) for _, created_at, _, text in picked]

    def record_route_note(
        self,
        anchor: str,
        note: str,
        x: float,
        y: float,
        kind: str = "blocker",
        now_wall: float | None = None,
    ) -> None:
        """Record a route obstacle or waypoint note."""
        if now_wall is None:
            now_wall = time.time()
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO route_notes (anchor, note, x, y, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (anchor, note, x, y, kind, now_wall),
            )

    def route_notes(self, anchor: str = "", now_wall: float | None = None) -> list[str]:
        """Return active, non-expired route notes (optionally filtered by anchor)."""
        if now_wall is None:
            now_wall = time.time()
        with self.lock:
            if anchor:
                rows = self.db.execute(
                    "SELECT note, kind, created_at FROM route_notes WHERE anchor = ? ORDER BY id",
                    (anchor,),
                ).fetchall()
            else:
                rows = self.db.execute("SELECT note, kind, created_at FROM route_notes ORDER BY id").fetchall()
        valid = []
        for note, kind, created_at in rows:
            k = kind or "blocker"
            if freshness(created_at, k, now_wall) >= 0.10:
                valid.append(note)
        return valid

    def places(self) -> dict[str, tuple[float, float, float]]:
        """Return known place positions (x, y, yaw)."""
        with self.lock:
            return {name: (x, y, yaw) for name, x, y, yaw in self.db.execute("SELECT name, x, y, yaw FROM places")}


# Module-level alias for backward compatibility and clean public API
Memory = CirculatingMemory
