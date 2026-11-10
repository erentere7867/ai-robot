"""Instant commands: a few spoken words act at once and never wait for a model.

"Stop" must halt the robot within half a second whatever the planner is doing, so it is matched by keyword on the
transcript, before any model sees it. "Wait" stops the running program and holds still. "Come back" stops and
returns to the speaker. Anything longer than a short phrase is an instruction for the planner, so "stop at the door"
is not a stop: only an utterance that is the command alone (with filler words and politeness) counts.
"""

from __future__ import annotations

import re

COMMANDS = {
    "stop": ("stop", "stop it", "stop now", "halt", "freeze", "stop moving", "enough", "that's enough"),
    "wait": ("wait", "hold on", "hang on", "pause", "wait a second", "wait a moment", "one moment"),
    "come back": ("come back", "come back here", "get back here", "return"),
}
FILLER = {"please", "ok", "okay", "hey", "sprocket", "robot", "now", "right", "just", "oh", "uh", "um", "no"}
COME_BACK = "robot.go_to(robot.user(), distance=1.0)\nrobot.face(robot.user())\n"


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
