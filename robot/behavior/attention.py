"""Attention manager: filters incoming events and routes safety stops vs cognitive requests."""

from __future__ import annotations

import re

from robot.behavior.fsm import InteractionFSM
from robot.core.bus import EventBus
from robot.core.events import Event, EventPriority

_STOP_RE = re.compile(
    r"^\s*(?:(?:hey\s+)?(?:sprocket|robot)\s*[,:]?\s*)?(?:stop|halt|freeze|cancel|shut\s+up|e-?stop)\s*[.!]*\s*$",
    re.IGNORECASE,
)


class AttentionManager:
    """Determines whether incoming messages require emergency stop, wake-word gating, or cognition."""

    def __init__(
        self,
        bus: EventBus,
        fsm: InteractionFSM,
        *,
        wake_words: list[str] | None = None,
        require_wake_word: bool = False,
    ) -> None:
        self.bus = bus
        self.fsm = fsm
        self.wake_words = [
            w.lower() for w in (wake_words or ["sprocket", "hey sprocket", "robot", "hey robot"])
        ]
        self.require_wake_word = require_wake_word
        self._sorted_wake_words = sorted(self.wake_words, key=len, reverse=True)
        self._sub_ids: list[str] = []

    def bind(self) -> None:
        self._sub_ids.append(
            self.bus.subscribe(
                "input.user_message", self._on_user_message, name="attention:user_msg", immediate=True
            )
        )

    def unbind(self) -> None:
        for sid in self._sub_ids:
            self.bus.unsubscribe(sid)
        self._sub_ids.clear()

    def should_attend(self, text: str, *, direct_address: bool = False) -> tuple[bool, str]:
        """Decide whether `text` should engage the robot and return `(engaged, cleaned_text)`."""
        cleaned = text.strip()
        if not cleaned:
            return False, ""

        lower = cleaned.lower()
        for ww in self._sorted_wake_words:
            if lower.startswith(ww) and (len(lower) == len(ww) or not lower[len(ww)].isalnum()):
                rest = cleaned[len(ww) :].lstrip(" ,:;-!?")
                return True, (rest or "Hello!")

        if direct_address or not self.require_wake_word or self.fsm.in_follow_up_window():
            return True, cleaned

        return False, cleaned

    async def _on_user_message(self, event: Event) -> None:
        if event.payload.get("handled"):
            return
        text = str(event.payload.get("text", "")).strip()
        direct = bool(event.payload.get("direct", not self.require_wake_word))
        if not text:
            return
        correlation_id = event.correlation_id or event.id
        if len(text) > 4096:
            await self.bus.emit(
                "attention.ignored",
                "attention",
                {"reason": "message_too_long"},
                correlation_id=correlation_id,
            )
            return

        # Reflexive/Behavioral immediate stop command bypasses LLM inference
        if _STOP_RE.match(text):
            self.fsm.touch()
            await self.bus.emit(
                "action.cancel",
                source="attention",
                payload={"reason": "user_stop_command"},
                priority=EventPriority.CRITICAL,
                correlation_id=correlation_id,
            )
            await self.bus.emit(
                "action.request",
                source="attention",
                payload={"tool": "stop", "args": {}},
                priority=EventPriority.CRITICAL,
                correlation_id=correlation_id,
            )
            await self.bus.emit(
                "robot.response",
                source="attention",
                payload={
                    "say": "Stopping right now!",
                    "sound": "ack_chirp",
                    "emotion": {"name": "concerned", "intensity": 0.6},
                    "actions": [{"tool": "stop", "args": {}}],
                    "memory": [],
                    "backend": "reflex_stop",
                },
                priority=EventPriority.HIGH,
                correlation_id=correlation_id,
            )
            return

        attend, cleaned_text = self.should_attend(text, direct_address=direct)
        if not attend:
            await self.bus.emit(
                "attention.ignored",
                source="attention",
                payload={"text": text, "reason": "wake_word_not_detected"},
                priority=EventPriority.LOW,
                correlation_id=correlation_id,
            )
            return

        self.fsm.touch()
        await self.bus.emit(
            "attention.engaged",
            source="attention",
            payload={"text": cleaned_text, "raw_text": text},
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
        )
