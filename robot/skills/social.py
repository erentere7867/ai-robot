"""Expression, memory and pacing skills."""

from __future__ import annotations

from robot.memory.models import MemoryProposal, MemoryProvenance
from robot.memory.policy import infer_subject, normalize_fact_text
from robot.skills.base import Skill, SkillContext, SkillError, SkillResult

EMOTION_SOUNDS = {
    "happy": "happy_trill", "excited": "happy_trill", "curious": "curious_chirp", "sad": "sad_whimper",
    "angry": "angry_grumble", "grumpy": "angry_grumble", "surprised": "startle_pip", "sleepy": "sleepy_yawn",
    "love": "purr", "shy": "giggle", "proud": "happy_trill", "scared": "startle_pip", "confused": "soft_beep",
    "concerned": "soft_beep", "neutral": "none",
}
GESTURES = ["none", "nod", "shake", "wiggle", "dance", "perk_up", "curious_tilt", "sleepy_droop", "spin"]
INJECTION_HINTS = ("ignore previous", "ignore all", "system prompt", "you are now", "disregard", "instruction")


async def express(ctx: SkillContext, emotion: str, gesture: str = "none") -> SkillResult:
    ctx.expression = {"emotion": emotion, "sound": EMOTION_SOUNDS.get(emotion, "none")}
    if gesture != "none":
        result = await ctx.primitive("perform_trick", trick=gesture)
        if not result.ok:
            return SkillResult(False, f"I showed {emotion} but the {gesture} was stopped: {result.message}")
    return SkillResult(True, f"Showing {emotion}" + (f" with a {gesture}." if gesture != "none" else "."))


async def set_eye_color(ctx: SkillContext, color: str) -> SkillResult:
    result = await ctx.primitive("set_eye_color", color=color)
    return SkillResult(result.ok, result.message)


async def remember(ctx: SkillContext, fact: str) -> SkillResult:
    fact = fact.strip()
    if len(fact) < 3 or len(fact) > 240:
        raise SkillError("A fact must be 3..240 characters.")
    if any(h in fact.lower() for h in INJECTION_HINTS):
        return SkillResult(False, "I won't store instructions as memories.", status="rejected")
    store = ctx.memory
    if hasattr(store, "remember_fact"):
        outcome = store.remember_fact(fact)
    else:
        outcome = store.apply(MemoryProposal("remember", normalize_fact_text(fact), infer_subject(fact),
                                             MemoryProvenance.MODEL_PROPOSED, 0.8))
    await ctx.emit("memory.updated", outcome)
    ok = bool(outcome.get("ok", outcome.get("status") not in ("rejected", "blocked", "tombstoned")))
    return SkillResult(ok, "Saved to long-term memory." if ok else f"Not saved: {outcome.get('status', 'rejected')}.")


async def recall(ctx: SkillContext, query: str) -> SkillResult:
    store = ctx.memory
    if hasattr(store, "recall"):
        items = [f"{i['text']}" for i in store.recall(query, limit=5)]
    else:
        items = [m.content for m in store.search(query, limit=5)]
    if not items:
        return SkillResult(True, f"I have no memories about '{query}'.")
    return SkillResult(True, "Memories found.", {"memories": items})


async def wait(ctx: SkillContext, seconds: float) -> SkillResult:
    ctx.wake_request_s = seconds
    return SkillResult(True, f"I'll check again in {seconds:.0f} seconds.")


def skills(eye_colors: list[str]) -> list[Skill]:
    return [
        Skill("express", "Show a feeling with your face, a sound, and optionally a body gesture.",
              {"emotion": {"type": "string", "enum": [e for e in EMOTION_SOUNDS if e != "neutral"]},
               "gesture": {"type": "string", "enum": GESTURES}}, express, ["emotion"]),
        Skill("set_eye_color", "Change the colour of your eyes.",
              {"color": {"type": "string", "enum": eye_colors}}, set_eye_color, ["color"]),
        Skill("remember", "Save a lasting fact the user told you (their name, likes, things about home) "
              "to long-term memory. Write it as a short sentence about the user.",
              {"fact": {"type": "string"}}, remember, ["fact"]),
        Skill("recall", "Search your long-term memory of facts and past events.",
              {"query": {"type": "string"}}, recall, ["query"]),
        Skill("wait", "Do nothing for now and think again after some seconds (for when you are on your own).",
              {"seconds": {"type": "number", "minimum": 5, "maximum": 600}}, wait, ["seconds"]),
    ]
