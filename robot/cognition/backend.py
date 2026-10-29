"""Inference backend interface and offline pet-mode NullBackend."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from robot.memory.policy import extract_explicit_memory_proposals


@dataclass
class InferenceResult:
    """Raw output and telemetry returned by an InferenceBackend."""

    raw_text: str
    backend_name: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceBackend(Protocol):
    """Model-independent inference adapter."""

    name: str

    async def is_available(self) -> bool: ...

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> InferenceResult: ...


class UnavailableBackend:
    """A missing model never substitutes a scripted physical behavior."""

    name = "inference_unavailable"

    async def generate(self, messages, *, json_schema=None) -> InferenceResult:
        autonomous = any(header in str(messages[-1].get("content", ""))
                         for header in ("[Background Event]", "[Autonomous Objective]")) if messages else False
        return InferenceResult(json.dumps({
            "say": "" if autonomous else "The model is unavailable, so I cannot decide how to carry out this request.",
            "sound": "none", "emotion": {"name": "neutral", "intensity": 0.3},
            "actions": [], "memory": [], "status": "blocked", "next_wake_s": 60,
        }), self.name)


class NullBackend:
    """Deterministic test/demo adapter, selected explicitly with --backend null."""

    name = "null_offline"

    async def is_available(self) -> bool:
        return True

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> InferenceResult:
        raw_content = messages[-1]["content"] if messages else ""
        if isinstance(raw_content, list):
            user_content = "\n".join(
                str(part.get("text", ""))
                for part in raw_content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            user_content = str(raw_content)
        if "[Background Event]" in user_content or "[Autonomous Objective]" in user_content:
            # The test adapter has no self-directed behavior.
            return InferenceResult(json.dumps({
                "say": "", "sound": "none", "emotion": {"name": "curious", "intensity": 0.5},
                "actions": [], "memory": [], "status": "completed",
            }), self.name)
        user_msg = self._extract_section(user_content, "[User Message]") or user_content
        mem_section = self._extract_section(user_content, "[Retrieved Memories]")
        world_section = self._extract_section(user_content, "[World State]")

        response_obj = self._synthesize_response(user_msg, mem_section, world_section)
        return InferenceResult(
            raw_text=json.dumps(response_obj, ensure_ascii=False),
            backend_name=self.name,
            latency_ms=1.0,
        )

    @staticmethod
    def _extract_section(text: str, header: str) -> str:
        idx = text.find(header)
        if idx == -1:
            return ""
        sub = text[idx + len(header) :].lstrip()
        next_idx = sub.find("\n[")
        return sub[:next_idx].strip() if next_idx != -1 else sub.strip()

    def _synthesize_response(
        self,
        user_msg: str,
        mem_section: str,
        world_section: str,
    ) -> dict[str, Any]:
        lower = user_msg.lower().strip()

        # 1. Explicit remember / forget
        proposals = extract_explicit_memory_proposals(user_msg)
        if proposals:
            p = proposals[0]
            if p.op == "remember":
                return {
                    "say": f"Got it! I'll remember that: {p.text}.",
                    "sound": "happy_trill",
                    "emotion": {"name": "happy", "intensity": 0.75},
                    "actions": [{"tool": "perform_trick", "args": {"trick": "nod"}}],
                    "memory": [{"op": "remember", "text": p.text}],
                }
            return {
                "say": f"Okay, I've wiped '{p.text}' from my memory.",
                "sound": "soft_beep",
                "emotion": {"name": "neutral", "intensity": 0.4},
                "actions": [{"tool": "perform_trick", "args": {"trick": "shake"}}],
                "memory": [{"op": "forget", "text": p.text}],
            }

        # 2. Explicit emotion / face requests ("try making a sad face", "make an angry face")
        if any(w in lower for w in ("face", "expression", "try making", "act ")):
            for emo_kw, snd, trick, reply in [
                (
                    "sad",
                    "sad_whimper",
                    "sleepy_droop",
                    "*Droops head and whimpers softly*... Here is my sad face.",
                ),
                (
                    "angry",
                    "angry_grumble",
                    "stomp",
                    "Grrrr! *Narrows eyebrows and stomps*... Here is my angry face!",
                ),
                (
                    "mad",
                    "angry_grumble",
                    "stomp",
                    "Grrrr! *Narrows eyebrows and stomps*... Here is my angry face!",
                ),
                ("grumpy", "angry_grumble", "shake", "Hmph! Feeling grumpy!"),
                ("surprised", "startle_pip", "perk_up", "Whoa!! *Eyes go wide!*"),
                ("scared", "startle_pip", "shake", "Eep! That startled me!"),
                ("proud", "happy_trill", "perk_up", "*Tilts head high with pride!*"),
                ("shy", "soft_beep", "sleepy_droop", "*Looks down shyly and beeps.*"),
                ("love", "purr", "wiggle", "*Purrs warmly!*"),
            ]:
                if emo_kw in lower:
                    emo_name = "angry" if emo_kw == "mad" else emo_kw
                    return {
                        "say": reply,
                        "sound": snd,
                        "emotion": {"name": emo_name, "intensity": 0.9},
                        "actions": [{"tool": "perform_trick", "args": {"trick": trick}}],
                        "memory": [],
                    }

        # 3. Head tilt requests (e.g. "Tilt your head up 200 degrees")
        if "tilt" in lower or ("look" in lower and ("up" in lower or "down" in lower)):
            num_match = re.search(r"([+-]?\d+(?:\.\d+)?)", lower)
            angle = float(num_match.group(1)) if num_match else 20.0
            if "down" in lower and angle > 0:
                angle = -angle
            return {
                "say": f"Tilting my head toward {angle:.0f} degrees!",
                "sound": "curious_chirp",
                "emotion": {"name": "curious", "intensity": 0.7},
                "actions": [{"tool": "tilt_head", "args": {"angle_deg": angle}}],
                "memory": [],
            }

        # 3. Named tricks
        for trick, emo, snd, line in [
            ("dance", "excited", "happy_trill", "Ooh, a dance? Watch my moves!"),
            ("wiggle", "happy", "happy_trill", "Wiggle wiggle!"),
            ("nod", "happy", "ack_chirp", "Yep, nodding along!"),
            ("shake", "curious", "soft_beep", "Nope, shaking it off!"),
            ("sleepy", "sleepy", "sleepy_yawn", "*Yaaawn*... getting a little drowsy."),
        ]:
            if trick in lower:
                trick_name = "sleepy_droop" if trick == "sleepy" else trick
                return {
                    "say": line,
                    "sound": snd,
                    "emotion": {"name": emo, "intensity": 0.8},
                    "actions": [{"tool": "perform_trick", "args": {"trick": trick_name}}],
                    "memory": [],
                }

        # 4. Spin / turn
        if any(w in lower for w in ("spin", "turn around", "rotate")):
            num_match = re.search(r"([+-]?\d+(?:\.\d+)?)", lower)
            angle = float(num_match.group(1)) if num_match else (180.0 if "around" in lower else 360.0)
            if "right" in lower and angle > 0:
                angle = -angle
            return {
                "say": f"Spinning {angle:.0f} degrees!",
                "sound": "happy_trill",
                "emotion": {"name": "excited", "intensity": 0.75},
                "actions": [{"tool": "spin", "args": {"angle_deg": angle}}],
                "memory": [],
            }

        # 5. Drive forward / backward
        if any(w in lower for w in ("drive", "move forward", "move back", "go forward", "back up")):
            direction = "backward" if ("back" in lower or "reverse" in lower) else "forward"
            num_match = re.search(r"(\d+(?:\.\d+)?)", lower)
            dist = float(num_match.group(1)) if num_match else 250.0
            return {
                "say": f"Rolling {direction} {dist:.0f} millimeters!",
                "sound": "soft_beep",
                "emotion": {"name": "happy", "intensity": 0.65},
                "actions": [{"tool": "drive", "args": {"direction": direction, "distance_mm": dist}}],
                "memory": [],
            }

        # 6. Spatial & Camera query ("behind", "around you", "what do you see")
        if any(w in lower for w in ("what do you see", "in front of you")):
            for line in world_section.splitlines():
                if line.strip().startswith("Camera:"):
                    return {
                        "say": f"In my head camera view: {line.strip()[7:].strip()}.",
                        "sound": "curious_chirp",
                        "emotion": {"name": "curious", "intensity": 0.75},
                        "actions": [{"tool": "perform_trick", "args": {"trick": "nod"}}],
                        "memory": [],
                    }
        if any(w in lower for w in ("behind", "around you", "what did you see", "what do you see")):
            for line in world_section.splitlines():
                if line.strip().startswith("- "):
                    return {
                        "say": f"From my spatial memory: {line.strip()[2:]}.",
                        "sound": "curious_chirp",
                        "emotion": {"name": "curious", "intensity": 0.7},
                        "actions": [],
                        "memory": [],
                    }
            return {
                "say": "I don't have any recent objects in my short-term spatial memory right now.",
                "sound": "curious_chirp",
                "emotion": {"name": "curious", "intensity": 0.6},
                "actions": [{"tool": "perform_trick", "args": {"trick": "curious_tilt"}}],
                "memory": [],
            }

        # 7. Memory recall question ("what's my ...", "do you remember ...")
        if "?" in user_msg or lower.startswith(("what", "who", "where", "do you remember")):
            mem_lines = [
                ln.strip()[2:]
                for ln in mem_section.splitlines()
                if ln.strip().startswith("- ") and "(none)" not in ln
            ]
            if mem_lines:
                best = mem_lines[0]
                # Strip leading `(subject) `
                best_clean = re.sub(r"^\([^)]+\)\s*", "", best)
                return {
                    "say": f"I remember! {best_clean}.",
                    "sound": "happy_trill",
                    "emotion": {"name": "happy", "intensity": 0.75},
                    "actions": [{"tool": "perform_trick", "args": {"trick": "nod"}}],
                    "memory": [],
                }
            return {
                "say": "Hmm, I don't have a memory stored about that yet!",
                "sound": "curious_chirp",
                "emotion": {"name": "curious", "intensity": 0.6},
                "actions": [{"tool": "perform_trick", "args": {"trick": "curious_tilt"}}],
                "memory": [],
            }

        # 8. Default friendly pet greeting
        return {
            "say": "Beep-boop! I'm here and listening (running in offline pet mode).",
            "sound": "curious_chirp",
            "emotion": {"name": "happy", "intensity": 0.65},
            "actions": [{"tool": "perform_trick", "args": {"trick": "curious_tilt"}}],
            "memory": [],
        }
