"""Instant commands: a few spoken words act at once and never wait for a model."""

from __future__ import annotations

import re

from robot.agent.api import Intent

COMMANDS = {
    "stop": ("stop", "stop it", "stop now", "halt", "freeze", "stop moving", "enough", "that's enough"),
    "wait": ("wait", "hold on", "hang on", "pause", "wait a second", "wait a moment", "one moment"),
    "come back": ("come back", "come back here", "get back here", "return"),
}
FILLER = {"please", "ok", "okay", "hey", "sprocket", "robot", "now", "right", "just", "oh", "uh", "um", "no"}

COME_BACK_INTENT = Intent(
    say="Coming back to you.",
    chime="listen",
    posture="listening",
    action="pursue",
    target="user",
    params={"distance": 1.0},
)


def instant(text: str) -> str | None:
    """The instant command a transcript is, or None when it is an instruction for the planner."""
    words = [w for w in re.findall(r"[a-z']+", text.lower()) if w not in FILLER]
    phrase = " ".join(words)
    if not phrase:
        return None
    for command, phrases in COMMANDS.items():
        if phrase in phrases:
            return command
    return None
