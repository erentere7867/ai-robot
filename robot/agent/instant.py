"""Instant commands: "stop" and "wait" halt the robot at once and never wait for a model."""

from __future__ import annotations

import re

COMMANDS = {
    "stop": ("stop", "stop it", "stop now", "halt", "freeze", "stop moving", "enough", "that's enough"),
    "wait": ("wait", "hold on", "hang on", "pause", "wait a second", "wait a moment", "one moment"),
}
FILLER = {"please", "ok", "okay", "hey", "sprocket", "robot", "now", "right", "just", "oh", "uh", "um", "no"}


def instant(text: str) -> str | None:
    """The instant command a transcript is, or None when it is an instruction for the model."""
    phrase = " ".join(w for w in re.findall(r"[a-z']+", text.lower()) if w not in FILLER)
    return next((command for command, phrases in COMMANDS.items() if phrase in phrases), None)
