"""Checks on what the model says before the human hears it, and cleaning it up for speech."""

from __future__ import annotations

import re
from typing import Any

from robot.cognition.repair import COMMAND

_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
_ID = re.compile(r"\b([a-z]+)_\d+\b")
_CLAIM = re.compile(
    r"\b(?:i've|i have|i'm|i am|i just|i'll|i will|let me|let's|now i|i turned|i moved|i went|i drove|i spun)\b"
    r"[^.!?]{0,40}?\b(?:turn|spin|spun|mov|go|went|head|driv|drove|circl|look|explor|danc|search|find|found|"
    r"approach|park|rotat|sweep|com(?:e|ing)\b|follow|fac(?:e|ing)\b)", re.I)
_REFUSAL = re.compile(
    r"\bi (?:can't|cannot|can not|am not able to|am unable to|don't know how to)\b[^.!?]{0,40}?\b(?:face|danc|turn|"
    r"spin|mov|go|look|smile|nod|wiggle|express|show|circl|drive|explor|find|search|eyes?)", re.I)
_FOLLOW_CLAIM = re.compile(r"\bi(?:'m| am) following\b|\bi(?:'ll| will) (?:keep )?follow", re.I)


def ignores_command(request: str, tool_log: list[dict[str, Any]]) -> bool:
    """Answered a plain command with no tool at all (measured: "look at me" -> "I see you!")."""
    return bool(request and COMMAND.match(request)) and not tool_log


def claims_unperformed_action(say: str, tool_log: list[dict[str, Any]], acting: set[str], persistent: set[str]) -> bool:
    """True when the reply claims an action no tool performed, or refuses one the robot has a tool for.

    Following is checked on its own: it is a promise the human relies on, and a search before it is not it
    (measured: look_around, then "I'm following the red ball!" with the robot standing still)."""
    if _FOLLOW_CLAIM.search(say or "") and not any(e["tool"] in persistent and e["ok"] for e in tool_log):
        return True
    if any(e["tool"] in acting for e in tool_log):
        return False
    return bool(_CLAIM.search(say or "") or _REFUSAL.search(say or ""))


def contradicts_success(say: str, tool_log: list[dict[str, Any]], acting: set[str]) -> bool:
    """True when the reply says "I can't ..." right after an action tool succeeded."""
    return bool(_REFUSAL.search(say or "")) and any(e["ok"] and e["tool"] in acting for e in tool_log)


def speakable(text: str, spatial_ids: Any, robot_name: str = "") -> str:
    """Remove emoji and internal ids ("box_1" -> "the box") from text meant for the human.

    Also drops the robot's own name used as a vocative ("Hi, Sprocket!"): a small model sometimes
    addresses the human by the robot's name. "I'm Sprocket" is left alone.
    """
    def repl(match: re.Match[str]) -> str:
        return f"the {match.group(1)}" if match.group(0) in spatial_ids else match.group(0)
    text = _EMOJI.sub("", text)
    text = re.sub(r"\bthe the\b", "the", _ID.sub(repl, text), flags=re.I)
    if robot_name:
        text = re.sub(rf",\s*{re.escape(robot_name)}\s*(?=[!?.,])", "", text, flags=re.I)
    # The prompt calls the person "the human"; the robot speaks *to* them.
    text = re.sub(r"\b[Tt]he human's\b", lambda m: "Your" if m.group(0)[0] == "T" else "your", text)
    return re.sub(r"\s{2,}", " ", text).strip()
