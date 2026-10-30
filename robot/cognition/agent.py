"""The model-driven agent loop: think -> call skills -> read results -> repeat until it answers."""

from __future__ import annotations

import ast
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from robot.cognition.backend import BackendUnavailable, ChatResult, InferenceBackend, ToolCall
from robot.cognition.prompts import estimate_tokens
from robot.core.config import AgentConfig
from robot.skills.base import SkillContext, SkillRegistry, SkillResult

_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
_ID = re.compile(r"\b([a-z]+)_\d+\b")
IMAGE_NOTE = "(camera view after {tool})"


@dataclass
class TurnResult:
    say: str
    status: str = "completed"  # completed | blocked | cancelled | unavailable | error
    steps: int = 0
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    next_wake_s: float | None = None
    backend_name: str = ""
    latency_ms: float = 0.0
    expression: dict[str, Any] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)  # this turn's calls and results

    @property
    def backend(self) -> str:
        return self.backend_name

    def action_notes(self) -> list[str]:
        notes = []
        for entry in self.tool_log:
            args = ", ".join(f"{v}" for v in entry["args"].values())
            notes.append(f"{entry['tool']}({args}) {'ok' if entry['ok'] else 'failed'}")
        return notes

    def to_dict(self) -> dict[str, Any]:
        return {"say": self.say, "status": self.status, "steps": self.steps, "tools": self.tool_log,
                "next_wake_s": self.next_wake_s, "backend": self.backend_name,
                "latency_ms": round(self.latency_ms, 1), "emotion": self.expression.get("emotion"),
                "sound": self.expression.get("sound", "none")}


_CALL_TOKENS = re.compile(r"<\|tool_call_(?:start|end)\|>|</?think>|```(?:python|json)?")


def recover_text_tool_calls(content: str, tool_names: set[str]) -> list[ToolCall]:
    """Small models sometimes write a call as text ("observe()", or JSON) instead of a real call.

    Only a reply that is *entirely* calls to known tools is recovered; prose is left alone.
    """
    text = _CALL_TOKENS.sub("", content or "").strip()
    if not text or len(text) > 400:
        return []
    if text.startswith("{"):
        try:
            raw = json.loads(text)
        except ValueError:
            return []
        name, args = raw.get("name"), raw.get("arguments", raw.get("parameters", {}))
        if name in tool_names and isinstance(args, dict):
            return [ToolCall("recovered_0", name, args)]
        return []
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError:
        return []
    nodes = tree.elts if isinstance(tree, ast.List) else [tree]
    calls = []
    for i, node in enumerate(nodes[:3]):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in tool_names
                and not node.args):
            return []
        try:
            args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        except ValueError:
            return []
        calls.append(ToolCall(f"recovered_{i}", node.func.id, args))
    return calls


def speakable(text: str, spatial_ids: dict[str, str]) -> str:
    """Remove emoji and internal ids ("box_1" -> "the box") from text meant for the human."""
    def repl(match: re.Match[str]) -> str:
        return f"the {match.group(1)}" if match.group(0) in spatial_ids else match.group(0)
    text = _EMOJI.sub("", text)
    text = re.sub(r"\bthe the\b", "the", _ID.sub(repl, text), flags=re.I)
    return re.sub(r"\s{2,}", " ", text).strip()


class AgentLoop:
    def __init__(self, backend: InferenceBackend, skills: SkillRegistry, config: AgentConfig,
                 *, prompt_limit_tokens: int = 3300, on_event: Callable[[str, dict[str, Any]], Awaitable[None]]
                 | None = None) -> None:
        self.backend = backend
        self.skills = skills
        self.config = config
        self.prompt_limit_tokens = prompt_limit_tokens
        self.on_event = on_event
        self._tools = skills.tool_schemas()
        self._tool_names = set(skills.names)

    async def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.on_event is not None:
            await self.on_event(event_type, payload)

    async def run(self, messages: list[dict[str, Any]], ctx: SkillContext, *, background: bool) -> TurnResult:
        result = TurnResult(say="")
        started = time.monotonic()
        first_new = len(messages)
        attempts: dict[str, int] = {}
        failures = 0
        for step in range(self.config.max_steps):
            ctx.check()
            if time.monotonic() - started > self.config.timeout_s:
                result.status = "blocked"
                result.say = "That took too long, so I stopped."
                break
            self._fit(messages)
            try:
                reply: ChatResult = await self.backend.chat(messages, tools=self._tools)
            except BackendUnavailable as exc:
                result.status = "unavailable"
                result.say = "" if background else "My thinking model is offline right now, so I can't do that."
                await self._emit("cognition.unavailable", {"error": str(exc)[:200]})
                break
            ctx.check()
            result.steps = step + 1
            result.backend_name = reply.backend_name
            result.latency_ms += reply.latency_ms
            await self._emit("cognition.model_call", {
                "step": step + 1, "latency_ms": round(reply.latency_ms, 1), "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens, "content": reply.content[:300],
                "tool_calls": [{"name": c.name, "args": c.arguments} for c in reply.tool_calls],
            })
            if not reply.tool_calls:
                reply.tool_calls = recover_text_tool_calls(reply.content, self._tool_names)
                if reply.tool_calls:
                    reply.content = ""
                    await self._emit("cognition.recovered_calls", {"calls": [c.name for c in reply.tool_calls]})
            if not reply.tool_calls:
                result.say = reply.content
                break
            messages.append(reply.assistant_message())
            stop_batch = False
            for call in reply.tool_calls:
                if stop_batch:
                    outcome = SkillResult(False, "Not executed because an earlier step failed.", status="skipped")
                elif call.invalid_arguments:
                    outcome = SkillResult(False, "The arguments were not valid JSON.", status="rejected")
                else:
                    if background and not getattr(self.skills.get(call.name), "background_ok", True):
                        outcome = SkillResult(False, "Not available on your own.", status="rejected")
                    else:
                        await self._emit("skill.started", {"tool": call.name, "args": call.arguments})
                        outcome = await self.skills.run(call.name, call.arguments, ctx)
                        await self._emit("skill.completed", {"tool": call.name, "args": call.arguments,
                                                             "ok": outcome.ok, "status": outcome.status,
                                                             "message": outcome.message, "data": outcome.data})
                    result.tool_log.append({"tool": call.name, "args": call.arguments, "ok": outcome.ok,
                                            "message": outcome.message})
                    stop_batch = not outcome.ok
                    failures += 0 if outcome.ok else 1
                    key = json.dumps([call.name, call.arguments], sort_keys=True)
                    attempts[key] = attempts.get(key, 0) + 1
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(outcome.to_model(), ensure_ascii=False)})
            if ctx.attached_image is not None:
                self._attach_image(messages, ctx.attached_image.data_url, reply.tool_calls[-1].name)
                ctx.attached_image = None
            if max(attempts.values(), default=0) >= 3 or failures >= 4:
                # Going round in circles: stop and let the human (or a later wake) decide.
                result.status = "blocked"
                result.say = "" if background else ("I keep trying the same thing without getting anywhere, "
                                                     "so I stopped.")
                break
        else:
            result.status = "blocked"
            result.say = "" if background else "I ran out of steps before finishing."
        result.next_wake_s = ctx.wake_request_s
        result.expression = dict(ctx.expression)
        result.transcript = [m for m in messages[first_new:] if m.get("role") in ("assistant", "tool")]
        result.say = speakable(result.say, ctx.spatial_map.objects if ctx.spatial_map else {})
        return result

    @staticmethod
    def _attach_image(messages: list[dict[str, Any]], data_url: str, tool: str) -> None:
        """Keep only the newest camera image in context; older views become a text note."""
        for message in messages:
            if message.get("role") == "user" and isinstance(message.get("content"), list):
                texts = [p.get("text", "") for p in message["content"] if p.get("type") == "text"]
                message["content"] = " ".join(texts) + " (image no longer shown)"
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": IMAGE_NOTE.format(tool=tool)},
        ]})

    def _fit(self, messages: list[dict[str, Any]]) -> None:
        """Drop the oldest history and tool exchanges until the prompt fits; never split a call/result pair."""
        def cost(m: dict[str, Any]) -> int:
            content = m.get("content")
            if isinstance(content, list):
                return 300 + sum(estimate_tokens(p.get("text", "")) for p in content if isinstance(p, dict))
            return estimate_tokens(str(content or "")) + estimate_tokens(json.dumps(m.get("tool_calls", ""))) + 6

        tool_budget = 1100  # the tool schemas are sent alongside the messages
        while sum(map(cost, messages)) + tool_budget > self.prompt_limit_tokens and len(messages) > 2:
            request_idx = next((i for i, m in enumerate(messages) if m.get("role") == "user"
                                and "[Situation]" in str(m.get("content"))), None)
            if request_idx is not None and request_idx > 1:
                del messages[1]  # oldest conversation turn, then any of its orphaned calls/results
                while len(messages) > 2 and messages[1].get("role") != "user":
                    del messages[1]
                continue
            # Oldest assistant tool-call message after the request, with its tool results.
            start = next((i for i in range((request_idx or 0) + 1, len(messages))
                          if messages[i].get("role") == "assistant" and messages[i].get("tool_calls")), None)
            if start is None or start >= len(messages) - 2:
                break
            end = start + 1
            while end < len(messages) and messages[end].get("role") == "tool":
                end += 1
            if end - start < 2:
                break
            summary = "; ".join(json.loads(messages[k]["content"]).get("result", "")[:60]
                                for k in range(start + 1, end))
            messages[start:end] = [{"role": "user", "content": f"(earlier steps, shortened: {summary})"}]
