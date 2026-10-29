"""Interaction finite-state machine with follow-up window and idle tracking."""

from __future__ import annotations

from enum import StrEnum

from robot.core.clock import Clock, RealClock


class InteractionState(StrEnum):
    IDLE = "idle"
    ATTENDING = "attending"
    THINKING = "thinking"
    SPEAKING = "speaking"
    FOLLOW_UP = "follow_up"


class InteractionFSM:
    """Tracks conversational state, follow-up window expiration, and time since last interaction."""

    def __init__(
        self,
        clock: Clock | None = None,
        *,
        follow_up_window_s: float = 8.0,
        idle_sleepy_timeout_s: float = 45.0,
    ) -> None:
        self.clock: Clock = clock or RealClock()
        self.follow_up_window_s = follow_up_window_s
        self.idle_sleepy_timeout_s = idle_sleepy_timeout_s
        self.state: InteractionState = InteractionState.IDLE
        self.last_interaction_ts: float = self.clock.monotonic()
        self.follow_up_until_ts: float = 0.0
        self.sleepy_triggered: bool = False

    def touch(self) -> None:
        """Record user interaction activity and reset sleepy flag."""
        self.last_interaction_ts = self.clock.monotonic()
        self.sleepy_triggered = False

    def transition(self, new_state: InteractionState) -> InteractionState:
        """Transition to `new_state`, updating follow-up timers as appropriate."""
        now = self.clock.monotonic()
        self.state = new_state
        if new_state in (InteractionState.ATTENDING, InteractionState.THINKING, InteractionState.SPEAKING):
            self.last_interaction_ts = now
            self.sleepy_triggered = False
        elif new_state == InteractionState.FOLLOW_UP:
            self.last_interaction_ts = now
            self.follow_up_until_ts = now + self.follow_up_window_s
        return self.state

    def in_follow_up_window(self) -> bool:
        now = self.clock.monotonic()
        if self.state == InteractionState.FOLLOW_UP and now <= self.follow_up_until_ts:
            return True
        if self.state == InteractionState.FOLLOW_UP and now > self.follow_up_until_ts:
            self.state = InteractionState.IDLE
        return False

    def idle_seconds(self) -> float:
        return max(0.0, self.clock.monotonic() - self.last_interaction_ts)

    def check_timers(self) -> str | None:
        """Update time-driven transitions; returns `'follow_up_expired'` or `'became_sleepy'` when triggered."""
        now = self.clock.monotonic()
        if self.state == InteractionState.FOLLOW_UP and now > self.follow_up_until_ts:
            self.state = InteractionState.IDLE
            return "follow_up_expired"
        if (
            self.state == InteractionState.IDLE
            and not self.sleepy_triggered
            and (now - self.last_interaction_ts) >= self.idle_sleepy_timeout_s
        ):
            self.sleepy_triggered = True
            return "became_sleepy"
        return None
