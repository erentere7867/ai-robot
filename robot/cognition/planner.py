"""A to-do list for requests with several parts: "go to that ball and come back".

The 2B model picks the right tool for one clear step most of the time, and it naturally answers once that
step is done. What it does badly is the rest of a request: after "go to the ball" it says "I'm at the
ball!" and forgets "and come back". So a request is cut into its parts, in the human's own words, and
the model is handed one part at a time; its answer to a part is the signal that the part is done, and the
next part follows until the list is empty. Only the last answer is spoken.

The parts come from the sentence itself (sequencing words: "then", "after that", "and" + an action verb),
not from the model. Asked to write the plan (as tool steps or as parts), the 2B model dropped parts,
paraphrased them wrongly ("back up a little" -> "stop") and invented new ones ("hi!" -> two parts), and
writing a plan before its first call made that call worse (measured: 23/27 -> 18/27 right first calls).
"""

from __future__ import annotations

import re

# Verbs a person starts a new instruction with ("... and come back", "... and park next to it").
_VERBS = ("go|come|turn|look|move|find|drive|tell|show|say|do|make|bring|check|spin|back|park|stop|wait|wave|"
          "dance|circle|follow|face|return|fetch|push|sit|nod|take|put|give|explore|search|head|roll|get|"
          "describe|count|remember|answer|smile|celebrate|rest|charge|try|walk|run|let")
_SPLIT = re.compile(rf"\s*,?\s*(?:\band then\b|\bthen\b|\bafter that\b|\bafterwards\b|\band (?=(?:{_VERBS})\b))\s*",
                    re.I)
_LEAD = re.compile(r"^(?:and|also|please|,)\s+", re.I)


def split_parts(request: str) -> list[str]:
    """["go to that ball", "come back"] for "go to that ball and come back"; [] for a one-part request."""
    parts = []
    for raw in _SPLIT.split(request or ""):
        part = _LEAD.sub("", raw.strip(" ,.;!?")).strip()
        if part:
            parts.append(part)
    return parts if len(parts) >= 2 else []


class TaskList:
    """The parts of one request and which one is being done."""

    def __init__(self, request: str, parts: list[str]) -> None:
        self.request = request
        self.parts = parts
        self.index = 0

    @property
    def current(self) -> str:
        return self.parts[self.index]

    @property
    def has_next(self) -> bool:
        return self.index + 1 < len(self.parts)

    def advance(self) -> str:
        self.index += 1
        return self.current

    def hint(self) -> str:
        """Rides on each tool result: where the model is in the list."""
        return (f"Request: \"{self.request}\". Now doing part {self.index + 1} of {len(self.parts)}: "
                f"\"{self.current}\". When that part is done, answer.")

    def next_part_message(self, world: str = "") -> str:
        now = f"\nWhere things are now:\n{world}" if world else ""
        return (f"(Next part of \"{self.request}\": \"{self.current}\". Do it now; if it is already done, "
                f"answer the human.){now}")
