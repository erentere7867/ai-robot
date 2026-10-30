"""Event stream JSONL recording and deterministic session replay."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

from robot.core.bus import EventBus
from robot.core.clock import ReplayClock
from robot.core.events import Event

logger = logging.getLogger(__name__)

MAX_RECORD_CHARS = 65536
FLUSH_INTERVAL_S = 0.5


class EventRecorder:
    """Appends every bus event as a single JSON line for replay and debugging."""

    def __init__(self, output_dir: Path | str, session_id: str | None = None) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.session_id = session_id or f"session_{stamp}"
        self.file_path = self.output_dir / f"{self.session_id}.jsonl"
        self._fp: TextIO | None = None
        self.recorded_count: int = 0
        self.dropped_count = 0
        self._queue: queue.Queue[Event | None] = queue.Queue(maxsize=256)
        self._worker: threading.Thread | None = None
        self._closed = False
        self.max_file_bytes = 8 * 1024 * 1024
        self._part = 0
        self.max_parts = 4
        self._saturated = False
        self.failed = False

    def open(self) -> Path:
        if self._closed:
            raise RuntimeError("Recorder is closed")
        if self._worker is None:
            self._fp = self.file_path.open("a", encoding="utf-8")
            self._worker = threading.Thread(target=self._write_loop, name="event-recorder", daemon=True)
            self._worker.start()
        return self.file_path

    def record(self, event: Event) -> None:
        if self._closed:
            return
        if self._saturated or self.failed:
            self.dropped_count += 1
            return
        if self._worker is None:
            self.open()
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Recording is diagnostic; disk congestion must never delay safety.
            self.dropped_count += 1

    def _write_loop(self) -> None:
        path = self.file_path
        written = 0
        last_flush = time.monotonic()
        try:
            while True:
                try:
                    event = self._queue.get(timeout=0.25)
                except queue.Empty:
                    self._fp.flush()
                    last_flush = time.monotonic()
                    if self._closed:
                        break
                    continue
                try:
                    if event is None:
                        break
                    if self._saturated:
                        self.dropped_count += 1
                        continue
                    try:
                        line = event.to_json() + "\n"
                    except (TypeError, ValueError):
                        # One unserializable payload must not end the whole session's recording.
                        self.dropped_count += 1
                        logger.warning("Skipping unserializable event %s", event.type)
                        continue
                    if len(line) > MAX_RECORD_CHARS:
                        # The reader refuses such lines, so writing one would make the session unreplayable.
                        self.dropped_count += 1
                        continue
                    self._fp.write(line)
                    # A busy stream never idles; flush periodically so a crash loses at most a moment.
                    if time.monotonic() - last_flush >= FLUSH_INTERVAL_S:
                        self._fp.flush()
                        last_flush = time.monotonic()
                    written += len(line.encode("utf-8"))
                    self.recorded_count += 1
                    if written >= self.max_file_bytes:
                        if self._part >= self.max_parts - 1:
                            self._saturated = True
                            continue
                        self._fp.close()
                        self._part += 1
                        path = self.file_path.with_name(f"{self.session_id}.{self._part}.jsonl")
                        self._fp = path.open("w", encoding="utf-8")
                        written = 0
                finally:
                    self._queue.task_done()
        except Exception:
            self.failed = True
            logger.exception("Recording writer failed")
        finally:
            self._fp.close()
            self._fp = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)

    @staticmethod
    def iter_events(file_path: Path | str):
        """Stream bounded JSONL records instead of loading a whole session into RAM."""
        initial = Path(file_path).expanduser().resolve()
        for part in range(4):
            path = initial if part == 0 else initial.with_name(f"{initial.stem}.{part}.jsonl")
            if part and not path.exists():
                break
            with path.open(encoding="utf-8", errors="replace") as fp:
                while True:
                    line = fp.readline(MAX_RECORD_CHARS + 1)
                    if not line:
                        break
                    if len(line) > MAX_RECORD_CHARS:
                        raise ValueError(f"Recording event exceeds the {MAX_RECORD_CHARS}-character limit")
                    if not line.strip():
                        continue
                    try:
                        event = Event.from_json(line)
                    except (ValueError, KeyError, TypeError):
                        if line.endswith("\n"):
                            raise
                        # An unterminated final record is a write torn by a crash or power loss.
                        logger.warning("Ignoring truncated final record in %s", path)
                        break
                    yield event

    @staticmethod
    def load_events(file_path: Path | str) -> list[Event]:
        """Load recorded events from a JSONL file."""
        return list(EventRecorder.iter_events(file_path))


class EventReplayer:
    """Replays recorded events onto an EventBus with deterministic clock stepping."""

    def __init__(self, bus: EventBus, clock: ReplayClock | None = None) -> None:
        self.bus = bus
        self.clock = (
            clock
            if isinstance(clock, ReplayClock)
            else (bus.clock if isinstance(bus.clock, ReplayClock) else None)
        )

    async def replay(
        self,
        events: Iterable[Event],
        *,
        include_types: set[str] | None = None,
        exclude_types: set[str] | None = None,
        collect: bool = True,
    ) -> list[Event]:
        """Publish recorded events in order and drain the bus after each step."""
        replayed: list[Event] = []
        for event in events:
            if include_types is not None and event.type not in include_types:
                continue
            if exclude_types is not None and event.type in exclude_types:
                continue
            if self.clock is not None and event.ts > 0:
                self.clock.set_time(event.ts)
            await self.bus.publish(event)
            await self.bus.drain()
            if collect:
                replayed.append(event)
        return replayed
