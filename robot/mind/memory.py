"""The robot's long-term memory: what it has seen, where and when; what it was told; the places it was shown.

One SQLite file (the records part of the eMEM design in docs/architecture.md), so the robot still knows after a restart
where it last saw the keys. Objects are written from the tracker every few seconds and when the robot stops; on start
they are read back into the tracker as things known but not yet seen this session, and the tracker's own rules then
keep them honest: a clear look at a remembered place that does not see the thing marks it gone. People are not
remembered: they move. Times are stored as wall-clock seconds, so "last seen" survives the restart.

    memory = Memory("~/sprocket-memory/living room.db")
    memory.restore(tracker)                    # at start
    memory.save(tracker, notes, places)        # every few seconds and at the end
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path

from robot.perception.tracker import Track, Tracker

SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (id INTEGER PRIMARY KEY, label TEXT, color TEXT, x REAL, y REAL, radius REAL,
    var REAL, hits INTEGER, present INTEGER, verified INTEGER, first_seen REAL, last_seen REAL);
CREATE TABLE IF NOT EXISTS notes (n INTEGER PRIMARY KEY, text TEXT);
CREATE TABLE IF NOT EXISTS places (name TEXT PRIMARY KEY, x REAL, y REAL, yaw REAL);
"""


class Memory:
    def __init__(self, path: str | Path) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def save(self, tracker: Tracker, notes: list[str], places: dict) -> None:
        """Write what the robot knows now, replacing what was there: merged and dropped objects go too."""
        wall, mono = time.time(), time.monotonic()
        rows = [(t.id, t.label, t.color, t.x, t.y, t.radius, t.var, t.hits, int(t.present), int(t.verified),
                 wall - (mono - t.first_seen), wall - (mono - t.last_seen))
                for t in list(tracker.tracks.values()) if t.confirmed and not t.rejected and not t.person]
        with self.lock, self.db:
            self.db.execute("DELETE FROM objects")
            self.db.executemany("INSERT INTO objects VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            self.db.execute("DELETE FROM notes")
            self.db.executemany("INSERT INTO notes (text) VALUES (?)", [(n,) for n in notes])
            self.db.execute("DELETE FROM places")
            self.db.executemany("INSERT INTO places VALUES (?,?,?,?)",
                                [(name, p.x, p.y, p.yaw) for name, p in places.items() if name != "start"])

    def restore(self, tracker: Tracker) -> int:
        """Put the remembered objects back into the tracker, with their ages; returns how many."""
        wall, mono = time.time(), time.monotonic()
        with self.lock:
            rows = self.db.execute("SELECT * FROM objects").fetchall()
        for oid, label, color, x, y, radius, var, hits, present, verified, first, last in rows:
            tracker.tracks[oid] = Track(oid, label, x, y, radius, mono - (wall - first), mono - (wall - last),
                                        hits=max(2, hits), present=bool(present), verified=bool(verified), var=var,
                                        colors=Counter({color: 1} if color else {}), votes=Counter({label: 1.0}),
                                        score=1.0)
        tracker._next = max([tracker._next, *(r[0] + 1 for r in rows)])
        return len(rows)

    def notes(self) -> list[str]:
        with self.lock:
            return [text for (text,) in self.db.execute("SELECT text FROM notes ORDER BY n")]

    def places(self) -> dict[str, tuple[float, float, float]]:
        with self.lock:
            return {name: (x, y, yaw) for name, x, y, yaw in self.db.execute("SELECT * FROM places")}
