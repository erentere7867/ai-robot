"""Model-independent chat interface with native tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from robot.cognition.prompts import BACKGROUND_HEADER


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
    reasoning: str = ""  # the model's thinking, for logs and the dashboard; never spoken

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
        tool_choice: str = "auto",  # "required" forces a tool call
    ) -> ChatResult: ...


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content") or ""
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content)


class NullBackend:
    """Explicit `--backend null` adapter: talks, never moves. For wiring tests and demos only."""

    name = "null_offline"

    async def is_available(self) -> bool:
        return True

    async def chat(self, messages, *, tools=None, max_tokens=None, tool_choice="auto") -> ChatResult:
        if tools is None:  # perception requests (detector) get an empty detection list
            return ChatResult("[]", backend_name=self.name, latency_ms=1.0)
        last = messages[-1] if messages else {}
        if last.get("role") == "tool":
            return ChatResult("Done.", backend_name=self.name, latency_ms=1.0)
        text = message_text(last)
        if BACKGROUND_HEADER in text:
            return ChatResult("", backend_name=self.name, latency_ms=1.0)
        return ChatResult("I'm running without my language model, so I can only chat a little.",
                          backend_name=self.name, latency_ms=1.0)
