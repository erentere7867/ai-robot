"""Attention: the one place a human message is routed, in this order.

Where the voice came from is noted first (the speaker is placed before anything reacts to the words); a stop
halts at once, before the language model has read anything; autonomy mode commands are handled here; whatever is
left and addressed to the robot goes to cognition.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from robot.core.bus import EventBus
from robot.core.events import CancelReason, Event, EventPriority
from robot.interaction.fsm import InteractionFSM

# Said on its own: halt, and the model never gets it.
_HARD_STOP = re.compile(r"^\s*(?:stop|halt|freeze|cancel|shut\s+up|e-?stop)\s*[.!]*\s*$", re.I)
# Said on its own: halt now; the model still gets the message, to acknowledge it.
_SOFT_STOP = re.compile(r"^\W*(?:(?:please|hey|ok|okay|no)\W+)?(?:stop|halt|freeze|stay|wait|don'?t move)"
                        r"(?:\W+(?:it|now|there|moving|right now|please))*\W*$", re.I)


class AttentionManager:
    def __init__(self, bus: EventBus, fsm: InteractionFSM, autonomy: Any, on_voice: Callable[[dict], None], *,
                 wake_words: list[str], require_wake_word: bool = False) -> None:
        self.bus = bus
        self.fsm = fsm
        self.autonomy = autonomy
        self.on_voice = on_voice
        self.wake_words = sorted((w.lower() for w in wake_words), key=len, reverse=True)
        self.require_wake_word = require_wake_word
        self._sub_ids: list[str] = []

    def bind(self) -> None:
        self._sub_ids.append(self.bus.subscribe("input.user_message", self._on_user_message,
                                                name="attention:user_msg", immediate=True))

    def unbind(self) -> None:
        for sid in self._sub_ids:
            self.bus.unsubscribe(sid)
        self._sub_ids.clear()

    def _strip_wake_word(self, text: str) -> tuple[bool, str]:
        lower = text.lower()
        for word in self.wake_words:
            if lower.startswith(word) and (len(lower) == len(word) or not lower[len(word)].isalnum()):
                return True, text[len(word):].lstrip(" ,:;-!?")
        return False, text

    def should_attend(self, text: str, *, direct_address: bool = False) -> tuple[bool, str]:
        """Decide whether `text` should engage the robot and return `(engaged, cleaned_text)`."""
        cleaned = text.strip()
        if not cleaned:
            return False, ""
        woken, rest = self._strip_wake_word(cleaned)
        if woken:
            return True, rest or "Hello!"
        return direct_address or not self.require_wake_word or self.fsm.in_follow_up_window(), cleaned

    async def _on_user_message(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        if not text:
            return
        correlation_id = event.correlation_id or event.id
        if len(text) > 4096:
            await self.bus.emit("attention.ignored", "attention", {"reason": "message_too_long"},
                                correlation_id=correlation_id)
            return
        voice = event.payload.get("voice")
        if voice is not None:
            self.on_voice(voice)
        attend, cleaned = self.should_attend(text, direct_address=bool(event.payload.get("direct", False)))
        if attend:
            self.autonomy.note_human()
        if _HARD_STOP.match(self._strip_wake_word(text)[1]):
            await self._hard_stop(correlation_id)
            return
        if not attend:
            await self.bus.emit("attention.ignored", "attention", {"text": text, "reason": "wake_word_not_detected"},
                                priority=EventPriority.LOW, correlation_id=correlation_id)
            return
        if await self.autonomy.handle_mode_command(cleaned, correlation_id):
            return
        self.fsm.touch()
        if _SOFT_STOP.match(cleaned):
            await self.bus.emit("action.cancel", "attention", {"reason": CancelReason.USER_STOP},
                                priority=EventPriority.CRITICAL, correlation_id=correlation_id)
        await self.bus.emit("attention.engaged", "attention", {"text": cleaned, "raw_text": text},
                            priority=EventPriority.HIGH, correlation_id=correlation_id)

    async def _hard_stop(self, correlation_id: str) -> None:
        self.fsm.touch()
        await self.bus.emit("action.cancel", "attention", {"reason": CancelReason.STOP_COMMAND},
                            priority=EventPriority.CRITICAL, correlation_id=correlation_id)
        await self.bus.emit("robot.response", "attention", {
            "say": "Stopping right now!", "sound": "ack_chirp", "emotion": {"name": "concerned", "intensity": 0.6},
            "actions": [{"tool": "stop", "args": {}}], "memory": [], "backend": "reflex_stop",
        }, priority=EventPriority.HIGH, correlation_id=correlation_id)
