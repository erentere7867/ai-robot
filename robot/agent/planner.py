"""The closed-loop multimodal planner: receives scene summary + annotated images and emits structured Intent updates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from robot.agent.api import Intent
from robot.mind.brain import Brain, Reply, _with_image

SYSTEM_PROMPT = """You are Sprocket v2's closed-loop embodied mind.
At each step, emit a single compact Intent following the exact grammar.

Allowed postures: neutral, listening, curious, happy, uh_oh, yielding
Allowed actions: go_to, pursue, orbit, search_frontier, peek, step_closer, look_under, face, move, turn, spin, motion_sequence, ask, remember, done, refuse

Format:
say: <spoken words to announce now, or empty>
chime: <wake | listen | curious | happy | uh_oh | yield | empty>
posture: <posture>
action: <action>
target: <object ID, person, anchor name, or none>
params: <JSON dict of arguments, e.g. {"distance": 0.5}>
subgoal: <one short line describing current subgoal>"""

GRAMMAR = r"""root ::= "say: " [^\n]* "\nchime: " [^\n]* "\nposture: " [a-z_]+ "\naction: " [a-z_]+ "\ntarget: " [^\n]* "\nparams: " [^\n]* "\nsubgoal: " [^\n]*"""


class FormatError(ValueError):
    """The model's answer did not have the required Intent fields."""


@dataclass
class PlanResult:
    intent: Intent
    reply: Reply
    say_first: str = ""
    plan: str = ""

    def __post_init__(self) -> None:
        if not self.say_first:
            self.say_first = self.intent.say
        if not self.plan:
            self.plan = self.intent.subgoal or self.intent.action


def parse_intent(text: str) -> Intent:
    """Parses line-structured or JSON output into an Intent dataclass."""
    data: dict[str, Any] = {
        "say": "",
        "chime": "",
        "posture": "neutral",
        "action": "done",
        "target": None,
        "params": {},
        "subgoal": "",
    }

    # First attempt: JSON parse if wrapped in JSON
    trimmed = text.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, dict):
                data.update(parsed)
                return Intent(
                    say=str(data.get("say", "")),
                    chime=str(data.get("chime", "")),
                    posture=str(data.get("posture", "neutral")),
                    action=str(data.get("action", "done")),
                    target=data.get("target"),
                    params=data.get("params", {}) if isinstance(data.get("params"), dict) else {},
                    subgoal=str(data.get("subgoal", "")),
                )
        except Exception:
            pass

    # Line-by-line field parsing
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        k = key.strip().lower()
        v = val.strip()
        if k == "say":
            data["say"] = v
        elif k == "chime":
            data["chime"] = v
        elif k == "posture":
            data["posture"] = v or "neutral"
        elif k == "action":
            data["action"] = v or "done"
        elif k == "target":
            if not v or v.lower() in ("none", "null"):
                data["target"] = None
            elif re.fullmatch(r"-?\d+", v):
                data["target"] = int(v)
            else:
                data["target"] = v
        elif k == "params":
            if v and v.startswith("{") and v.endswith("}"):
                try:
                    data["params"] = json.loads(v)
                except Exception:
                    data["params"] = {}
            else:
                data["params"] = {}
        elif k == "subgoal":
            data["subgoal"] = v

    return Intent(
        say=str(data["say"]),
        chime=str(data["chime"]),
        posture=str(data["posture"]),
        action=str(data["action"]),
        target=data["target"],
        params=data["params"] if isinstance(data["params"], dict) else {},
        subgoal=str(data["subgoal"]),
    )


class LocalPlanner:
    """The on-board closed-loop planner constraining LFM3-VL-2B to compact Intent outputs."""

    def __init__(
        self,
        brain: Brain | None = None,
        max_tokens: int = 120,
        think: int = 0,
        multimodal: bool = True,
    ) -> None:
        self.brain = brain or Brain()
        self.max_tokens = max_tokens
        self.think = think
        self.multimodal = multimodal
        self.messages: list[dict[str, Any]] = []

    def plan_step(
        self,
        context: str,
        instruction: str,
        image_url: str | None = None,
        scratchpad: str = "",
        earlier: tuple[str, ...] | list[str] = (),
    ) -> Intent:
        """Plans the next immediate Intent step given multimodal observation and memory."""
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Scene State:\n{context}\n"
        )
        if scratchpad:
            prompt += f"Working Memory Scratchpad:\n{scratchpad}\n"
        if earlier:
            prompt += "Earlier:\n" + "\n".join(f"- {line}" for line in earlier) + "\n"
        prompt += f"\nInstruction: {instruction}\nNext Intent:"

        if image_url:
            messages = _with_image(image_url, prompt)
        else:
            messages = [{"role": "user", "content": prompt}]

        self.messages = messages
        reply = self.brain.complete(messages, GRAMMAR, self.max_tokens, self.think)
        return parse_intent(reply.text)

    def decide_step(
        self,
        context: str,
        instruction: str,
        image_url: str | None = None,
        scratchpad: str = "",
        earlier: tuple[str, ...] | list[str] = (),
    ) -> Intent:
        """Alias for plan_step."""
        return self.plan_step(context, instruction, image_url=image_url, scratchpad=scratchpad, earlier=earlier)

    def plan(
        self,
        context: str,
        instruction: str,
        earlier: tuple[str, ...] | list[str] = (),
    ) -> PlanResult:
        """Backwards-compatible plan interface returning PlanResult."""
        intent = self.plan_step(context, instruction, earlier=earlier)
        reply = Reply(text=f"action: {intent.action}")
        return PlanResult(intent=intent, reply=reply)

    def repair(self, problem: str, context: str | None = None) -> PlanResult:
        """Attempts an intent repair when an action fails."""
        prompt = f"The previous action encountered a problem: {problem}.\n"
        if context:
            prompt += f"Current State:\n{context}\n"
        prompt += "Provide an alternative Intent to recover or resolve:"
        self.messages.append({"role": "user", "content": prompt})
        reply = self.brain.complete(self.messages, GRAMMAR, self.max_tokens, self.think)
        intent = parse_intent(reply.text)
        return PlanResult(intent=intent, reply=reply)
