"""Clock abstractions for real-time execution and deterministic replay."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol


class Clock(Protocol):
    """Time provider protocol used across all harness subsystems."""

    def now(self) -> float:
        """Return current epoch timestamp in seconds."""
        ...

    def monotonic(self) -> float:
        """Return monotonic time in seconds."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend execution for the given duration."""
        ...


class RealClock:
    """Wall-clock and monotonic time provider for live execution."""

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class ReplayClock:
    """Deterministic controllable clock for testing and event replay."""

    def __init__(self, initial_time: float = 1_700_000_000.0) -> None:
        self._now: float = float(initial_time)
        self._monotonic: float = 0.0

    def now(self) -> float:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> None:
        """Advance simulated clock by `seconds`."""
        delta = max(0.0, float(seconds))
        self._now += delta
        self._monotonic += delta

    def set_time(self, timestamp: float) -> None:
        """Jump clock to `timestamp` while preserving monotonic progression."""
        if timestamp > self._now:
            self._monotonic += timestamp - self._now
        self._now = float(timestamp)

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)
        await asyncio.sleep(0)
