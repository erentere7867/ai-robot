"""How often the robot may say something nobody asked for.

Whatever it says unasked must not repeat what it said recently, and each unanswered remark doubles the wait
before the next.
"""

from __future__ import annotations

import math
import re
from collections import deque

from robot.core.clock import Clock


class SpeechPacer:
    def __init__(self, clock: Clock, interval_s: float) -> None:
        self.clock = clock
        self.interval_s = interval_s
        self.quiet = False
        self._spoke_at = -math.inf
        self._unanswered = 0  # remarks made on its own since the human last spoke
        self._said: deque[set[str]] = deque(maxlen=8)

    def human_spoke(self) -> None:
        self._unanswered = 0

    def due(self) -> bool:
        interval = self.interval_s * 2 ** min(self._unanswered, 5)
        return not self.quiet and self.clock.monotonic() - self._spoke_at >= interval

    def allow(self, text: str = "") -> bool:
        """May an autonomous turn say this now? Saying it counts, so call this once per reply."""
        if not self.due():
            return False
        words = set(re.findall(r"[a-z']+", text.lower()))
        if words and any(len(words & old) >= 0.6 * min(len(words), len(old)) for old in self._said):
            return False  # nearly the same as something it said unasked recently
        self._spoke_at = self.clock.monotonic()
        self._unanswered += 1
        if words:
            self._said.append(words)
        return True
