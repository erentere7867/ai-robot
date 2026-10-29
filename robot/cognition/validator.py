"""Response validator: sanitizes and validates raw model output against capability and policy limits."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from robot.actions.registry import CapabilityRegistry
from robot.actions.tools import ToolCallRequest
from robot.cognition.schema import EmotionSpec, ValidatedResponse
from robot.memory.models import MemoryProposal, MemoryProvenance
from robot.memory.policy import infer_subject, normalize_fact_text

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_VOCATIVE_COMMA_SPROCKET_RE = re.compile(r",\s*sprocket(?=\s*[!.,?]|$)", re.IGNORECASE)
_GREETING_SPROCKET_RE = re.compile(
    r"\b(hey|hi|hello)\s+sprocket(?=\s*[!.,?]|$)",
    re.IGNORECASE,
)


def _strip_self_name_vocative(text: str) -> str:
    """Remove cases where the small LLM mistakenly addresses the human user by the robot's own name ('Sprocket')."""
    cleaned = _VOCATIVE_COMMA_SPROCKET_RE.sub("", text)
    cleaned = _GREETING_SPROCKET_RE.sub(r"\1 there", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _extract_json_dict(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Attempt to extract a JSON object from `raw_text`, returning `(dict, error)`."""
    if not raw_text or not raw_text.strip():
        return None, "Empty model response"

    stripped = raw_text.strip()
    depth = 0
    quoted = escaped = False
    for char in stripped:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[":
            depth += 1
            if depth > 16:
                return None, "Model JSON exceeds nesting limit"
        elif char in "}]":
            depth -= 1
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed, None
    except (json.JSONDecodeError, RecursionError):
        pass

    fence_match = _CODE_FENCE_RE.search(stripped)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed, None
        except (json.JSONDecodeError, RecursionError):
            pass

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last > first:
        candidate = stripped[first : last + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
        except (json.JSONDecodeError, RecursionError) as exc:
            return None, f"Malformed JSON object: {exc}"

    return None, "No JSON object found in model output"


class ResponseValidator:
    """Validates model output against CapabilityRegistry without ever raising on malformed output."""

    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    def validate(
        self,
        raw_output: str | dict[str, Any],
        *,
        backend_name: str = "unknown",
        latency_ms: float = 0.0,
    ) -> ValidatedResponse:
        warnings: list[str] = []
        rejected_actions: list[dict[str, Any]] = []
        clamped_notes: list[str] = []

        if isinstance(raw_output, dict):
            data: dict[str, Any] | None = raw_output
            raw_text = json.dumps(raw_output, ensure_ascii=False)
            parse_err = None
        else:
            raw_text = str(raw_output or "")
            if len(raw_text) > 32768:
                raw_text = raw_text[:32768]
                data, parse_err = None, "Model response exceeds size limit"
            else:
                data, parse_err = _extract_json_dict(raw_text)

        if data is None:
            warnings.append(parse_err or "Invalid model output")
            clean_fallback = re.sub(r"[{}<>`]", "", raw_text).strip()
            if not clean_fallback or len(clean_fallback) > 220:
                clean_fallback = "Bzzzt—my thoughts got tangled for a second! Could you say that again?"
            return ValidatedResponse(
                say=clean_fallback,
                sound="curious_chirp",
                emotion=EmotionSpec(name="confused", intensity=0.5),
                actions=[],
                memory=[],
                warnings=warnings,
                raw_text=raw_text,
                backend_name=backend_name,
                latency_ms=latency_ms,
                fallback_used=True,
            )

        # 1. Validate `say`
        raw_say = data.get("say")
        say = _strip_self_name_vocative(str(raw_say).strip()) if raw_say is not None else ""
        if raw_say is None:
            say = "*chirp!*"
            warnings.append("Missing or empty 'say' field; substituted default chirp.")
        say = say[:400]

        # 2. Validate `sound`
        raw_sound = str(data.get("sound", "none") or "none").strip().lower()
        if raw_sound not in self.registry.allowed_sounds:
            warnings.append(f"Unsupported sound '{raw_sound}' replaced with 'none'.")
            sound = "none"
        else:
            sound = raw_sound

        # 3. Validate `emotion`
        emo_raw = data.get("emotion")
        emo_name = "neutral"
        emo_intensity = 0.5
        if isinstance(emo_raw, dict):
            cand_name = str(emo_raw.get("name", "neutral")).strip().lower()
            if cand_name in self.registry.allowed_emotions:
                emo_name = cand_name
            else:
                warnings.append(f"Unsupported emotion '{cand_name}' replaced with 'neutral'.")
            try:
                value = float(emo_raw.get("intensity", 0.5))
                if not math.isfinite(value):
                    raise ValueError("Non-finite intensity")
                emo_intensity = max(0.0, min(1.0, value))
            except (TypeError, ValueError):
                warnings.append("Invalid emotion.intensity; defaulted to 0.5.")
                emo_intensity = 0.5
        elif isinstance(emo_raw, str):
            cand_name = emo_raw.strip().lower()
            if cand_name in self.registry.allowed_emotions:
                emo_name = cand_name

        # 4. Validate `actions`
        valid_actions: list[ToolCallRequest] = []
        raw_actions = data.get("actions")
        if isinstance(raw_actions, list):
            for item in raw_actions[:3]:
                if not isinstance(item, dict):
                    warnings.append(f"Ignored non-object action item: {item!r}")
                    continue
                tool_name = str(item.get("tool", "")).strip()
                raw_args = item.get("args")
                args = dict(raw_args) if isinstance(raw_args, dict) else {}
                call_id = str(item["call_id"])[:160] if item.get("call_id") else None
                req = ToolCallRequest(tool=tool_name, args=args)
                outcome = self.registry.validate_tool_call(req)
                if outcome.valid:
                    # Pass original args so SimBody can also report clamping if needed,
                    # while recording clamped_notes for transparent user feedback
                    call_args = dict(outcome.normalized_args)
                    # Preserve safe normalized values through every layer.
                    valid_actions.append(ToolCallRequest(tool=outcome.tool, args=call_args, call_id=call_id))
                    if outcome.clamped:
                        clamped_notes.extend(outcome.warnings)
                        warnings.extend(outcome.warnings)
                else:
                    err_msg = outcome.error or f"Invalid tool call '{tool_name}'"
                    warnings.append(err_msg)
                    rejected_actions.append({"tool": tool_name, "args": args, "error": err_msg, "tool_call_id": call_id})

        # 5. Validate `memory`
        valid_memories: list[MemoryProposal] = []
        raw_memories = data.get("memory")
        if isinstance(raw_memories, list):
            for mitem in raw_memories[:2]:
                if not isinstance(mitem, dict):
                    continue
                op = str(mitem.get("op", "")).strip().lower()
                mtext = str(mitem.get("text", "")).strip()[:2048]
                if op in ("remember", "forget") and mtext:
                    norm_txt = normalize_fact_text(mtext) if op == "remember" else mtext
                    valid_memories.append(
                        MemoryProposal(
                            op=op,
                            text=norm_txt,
                            subject=infer_subject(mtext),
                            provenance=MemoryProvenance.MODEL_PROPOSED,
                            confidence=0.85,
                        )
                    )
                else:
                    warnings.append(f"Ignored invalid memory entry: {mitem!r}")

        if any(action.tool == "stop" for action in valid_actions):
            valid_actions = [next(action for action in valid_actions if action.tool == "stop")]
        status = data.get("status", "completed")
        if status not in ("continue", "completed", "blocked"):
            status = "blocked"
            warnings.append("Invalid task status; autonomous continuation stopped.")
        next_wake_s = data.get("next_wake_s")
        if next_wake_s is not None:
            try:
                if isinstance(next_wake_s, bool):
                    raise ValueError("Boolean wake interval")
                next_wake_s = float(next_wake_s)
                if not math.isfinite(next_wake_s) or not 1 <= next_wake_s <= 300:
                    raise ValueError("Wake interval out of range")
            except (TypeError, ValueError):
                next_wake_s = None
                warnings.append("Invalid idle wake interval ignored.")
        return ValidatedResponse(
            say=say,
            sound=sound,
            emotion=EmotionSpec(name=emo_name, intensity=emo_intensity),
            actions=valid_actions,
            memory=valid_memories,
            warnings=warnings,
            rejected_actions=rejected_actions,
            clamped_notes=clamped_notes,
            raw_text=raw_text,
            backend_name=backend_name,
            latency_ms=latency_ms,
            fallback_used=False,
            status=status,
            next_wake_s=next_wake_s,
        )
