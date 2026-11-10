"""The runner: executes one program in a separate process that can only reach the robot through a pipe.

Every robot call is a checkpoint: the parent serves it on the real backend and can answer with Preempted instead, so a
spoken "stop", a new instruction or the time limit ends the program at its next call. The backend's `cancel()` also
interrupts a call already in progress, so motion stops within one control step.

In the child the program sees only `robot`, `geom`, `math`, the catchable exceptions and a short list of builtins, and
the gate has already rejected imports, private attributes and anything else outside that. On Linux the child also gets
a memory limit; running it as an unprivileged user with no network is part of deployment on the robot.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

from robot.agent.api import CATCHABLE, EXCEPTIONS, ROBOT_CALLS, SAFE_BUILTINS, geom

MEMORY_LIMIT = 512 * 1024 * 1024


@dataclass
class RunResult:
    status: str  # "done", "error" (the program raised) or "preempted"
    error: str = ""  # "PathBlocked: no free path to (3.0, 1.0)"
    line: int | None = None  # program line that raised
    calls: int = 0
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "done"


class _Proxy:
    """`robot` inside the child: every call goes to the parent and waits for its answer."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def __getattr__(self, name: str) -> Any:
        if name not in ROBOT_CALLS:
            raise AttributeError(f"robot has no {name}")

        def call(*args: Any, **kwargs: Any) -> Any:
            self._conn.send(("call", name, args, kwargs))
            kind, value = self._conn.recv()
            if kind == "ok":
                return value
            exc_name, message = value
            raise EXCEPTIONS.get(exc_name, TypeError)(message)

        return call


def _program_line(tb: Any) -> int | None:
    lines = [frame.lineno for frame in traceback.extract_tb(tb) if frame.filename == "<program>"]
    return lines[-1] if lines else None


def _child(source: str, conn: Any) -> None:
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    names = {"__builtins__": dict(SAFE_BUILTINS), "robot": _Proxy(conn), "geom": geom, "math": math,
             **{e.__name__: e for e in CATCHABLE}}
    try:
        exec(compile(source, "<program>", "exec"), names)  # noqa: S102 - gated program, restricted names
        conn.send(("done", None))
    except BaseException as exc:  # report everything, including Preempted and SafetyStop
        conn.send(("done", (type(exc).__name__, str(exc)[:300], _program_line(exc.__traceback__))))


class Runner:
    def __init__(self, backend: Any, *, time_limit_s: float = 600.0) -> None:
        self.backend = backend
        self.time_limit_s = time_limit_s
        self._stop = threading.Event()
        self._proc: Any = None

    def stop(self) -> None:
        """End the program now and stop the robot (safe from any thread)."""
        self._stop.set()
        self.backend.cancel()

    def run(self, source: str) -> RunResult:
        ctx = mp.get_context("spawn")
        here, there = ctx.Pipe()
        self._proc = proc = ctx.Process(target=_child, args=(source, there), daemon=True)
        started = time.monotonic()
        proc.start()
        calls, result, stopped_at = 0, None, None
        try:
            while result is None:
                now = time.monotonic()
                if now - started > self.time_limit_s and not self._stop.is_set():
                    self.stop()
                if self._stop.is_set() and stopped_at is None:
                    stopped_at = now
                if not here.poll(0.05):
                    if not proc.is_alive():
                        result = RunResult("error", "the program process died", None, calls)
                    elif stopped_at is not None and now - stopped_at > 1.0:  # computing, not calling robot
                        result = RunResult("preempted", "Preempted: stopped while computing", None, calls)
                    continue
                message = here.recv()
                if message[0] == "done":
                    result = self._finish(message[1], calls)
                    break
                _, name, args, kwargs = message
                calls += 1
                here.send(self._serve(name, args, kwargs))
        finally:
            if proc.is_alive():
                proc.kill()
            proc.join(1.0)
        result.seconds = round(time.monotonic() - started, 3)
        return result

    def _serve(self, name: str, args: tuple, kwargs: dict) -> tuple[str, Any]:
        if self._stop.is_set():
            return "raise", ("Preempted", "stopped")
        try:
            return "ok", getattr(self.backend, name)(*args, **kwargs)
        except Exception as exc:  # a catchable robot exception, or a bad argument
            kind = type(exc).__name__
            if kind in EXCEPTIONS or kind == "TypeError":
                return "raise", (kind if kind in EXCEPTIONS else "TypeError", str(exc))
            return "raise", ("TypeError", f"{kind}: {exc}")
        except BaseException as exc:  # Preempted, SafetyStop or the dry run's call limit: always ends the program
            kind = type(exc).__name__
            return "raise", (kind if kind in EXCEPTIONS else "Preempted", str(exc))

    @staticmethod
    def _finish(outcome: Any, calls: int) -> RunResult:
        if outcome is None:
            return RunResult("done", calls=calls)
        name, message, line = outcome
        status = "preempted" if name == "Preempted" else "error"
        return RunResult(status, f"{name}: {message}", line, calls)
