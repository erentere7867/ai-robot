"""Expression, memory and pacing skills."""

from __future__ import annotations

from robot.body.hal import BodyCapabilities
from robot.control.behaviors import Gesture
from robot.core.events import EventPriority
from robot.skills.base import Skill, SkillContext, SkillError, SkillResult


async def express(ctx: SkillContext, emotion: str, gesture: str = "none") -> SkillResult:
    ctx.expression = {"emotion": emotion, "sound": ctx.capabilities.emotion_sounds.get(emotion, "none")}
    if gesture != "none":
        done = await ctx.behave(Gesture(ctx.capabilities.gestures[gesture]))
        if not done.ok:
            return SkillResult(False, f"I showed {emotion} but the {gesture} was stopped: {done.message}")
    return SkillResult(True, f"Showing {emotion}" + (f" with a {gesture}." if gesture != "none" else "."))


async def set_eye_color(ctx: SkillContext, color: str) -> SkillResult:
    await ctx.bus.emit("face.theme", "skill", {"theme": color}, priority=EventPriority.HIGH,
                       correlation_id=ctx.correlation_id)
    return SkillResult(True, f"Set eye color theme to '{color}'.")


async def remember(ctx: SkillContext, fact: str) -> SkillResult:
    fact = fact.strip()
    if len(fact) < 3 or len(fact) > 240:
        raise SkillError("A fact must be 3..240 characters.")
    outcome = ctx.memory.remember_fact(fact)
    await ctx.emit("memory.updated", outcome)
    if outcome["ok"]:
        return SkillResult(True, "Saved to long-term memory.")
    if outcome.get("reason") == "instruction_like":
        return SkillResult(False, "I won't store instructions as memories.", status="rejected")
    return SkillResult(False, f"Not saved: {outcome['status']}.")


async def recall(ctx: SkillContext, query: str) -> SkillResult:
    items = [item["text"] for item in ctx.memory.recall(query, limit=5)]
    if not items:
        return SkillResult(True, f"I have no memories about '{query}'.")
    return SkillResult(True, "Memories found.", {"memories": items})


async def wait(ctx: SkillContext, seconds: float) -> SkillResult:
    ctx.wake_request_s = seconds
    return SkillResult(True, f"I'll check again in {seconds:.0f} seconds.")


def skills(capabilities: BodyCapabilities, eye_colors: list[str]) -> list[Skill]:
    return [
        Skill("express", "Show a feeling with your face and a sound, optionally with a gesture such as a dance, "
              "nod, wiggle or spin.",
              {"emotion": {"type": "string", "enum": [e for e in capabilities.emotion_sounds if e != "neutral"]},
               "gesture": {"type": "string", "enum": ["none", *capabilities.gestures]}},
              express, ["emotion"], acts=True, ends_part=True),
        Skill("set_eye_color", "Change the colour of your eyes.",
              {"color": {"type": "string", "enum": eye_colors}}, set_eye_color, ["color"],
              ends_part=True, report_is_answer=True, idempotent=True),
        Skill("remember", "Save a lasting fact the user told you (their name, likes, things about home) "
              "to long-term memory. Write it as a short sentence about the user.",
              {"fact": {"type": "string"}}, remember, ["fact"], ends_part=True, idempotent=True),
        Skill("recall", "Search your long-term memory of facts and past events.",
              {"query": {"type": "string"}}, recall, ["query"], idempotent=True),
        Skill("wait", "Do nothing for now and think again after some seconds (for when you are on your own).",
              {"seconds": {"type": "number", "minimum": 5, "maximum": 600}}, wait, ["seconds"]),
    ]
