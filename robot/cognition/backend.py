"""Model-independent chat interface with native tool calls."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


class BackendUnavailable(RuntimeError):
    """The language model cannot be reached; callers must not invent a decision."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    invalid_arguments: bool = False

    def to_message(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)}}


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    backend_name: str = "unknown"
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None

    def assistant_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [c.to_message() for c in self.tool_calls]
        return message


class InferenceBackend(Protocol):
    name: str

    async def is_available(self) -> bool: ...

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult: ...


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content") or ""
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content)


class UnavailableBackend:
    name = "inference_unavailable"

    async def is_available(self) -> bool:
        return False

    async def chat(self, messages, *, tools=None, max_tokens=None) -> ChatResult:
        raise BackendUnavailable("No language model is available")


class NullBackend:
    """Explicit `--backend null` adapter: talks, never moves. For wiring tests and demos only."""

    name = "null_offline"

    async def is_available(self) -> bool:
        return True

    async def chat(self, messages, *, tools=None, max_tokens=None) -> ChatResult:
        if tools is None:  # perception requests (detector) get an empty detection list
            return ChatResult("[]", backend_name=self.name, latency_ms=1.0)
        last = messages[-1] if messages else {}
        if last.get("role") == "tool":
            return ChatResult("Done.", backend_name=self.name, latency_ms=1.0)
        text = message_text(last)
        if "[Background event]" in text:
            return ChatResult("", backend_name=self.name, latency_ms=1.0)
        return ChatResult("I'm running without my language model, so I can only chat a little.",
                          backend_name=self.name, latency_ms=1.0)


Responder = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], ChatResult | dict[str, Any] | str]


class ScriptedBackend:
    """Deterministic backend for tests: each entry is a reply, or a callable receiving the messages.

    Reply shorthand: a string is plain text; a dict {"call": name, "args": {...}} is one tool call;
    {"calls": [[name, args], ...]} is several; {"say": text} is plain text.
    """

    name = "scripted"

    def __init__(self, script: list[Any] | None = None, *, default: Any = "") -> None:
        self.script = list(script or [])
        self.default = default
        self.requests: list[dict[str, Any]] = []
        self._counter = 0

    async def is_available(self) -> bool:
        return True

    async def chat(self, messages, *, tools=None, max_tokens=None) -> ChatResult:
        self.requests.append({"messages": json.loads(json.dumps(messages, default=str)), "tools": tools})
        entry = self.script.pop(0) if self.script else self.default
        if callable(entry):
            entry = entry(messages, tools)
        return self._to_result(entry)

    def _to_result(self, entry: Any) -> ChatResult:
        if isinstance(entry, ChatResult):
            return entry
        if isinstance(entry, str):
            return ChatResult(entry, backend_name=self.name)
        calls = []
        pairs = entry.get("calls") or ([[entry["call"], entry.get("args", {})]] if "call" in entry else [])
        for name, args in pairs:
            self._counter += 1
            calls.append(ToolCall(f"call_{self._counter}", name, dict(args)))
        return ChatResult(str(entry.get("say", "")), calls, backend_name=self.name)
