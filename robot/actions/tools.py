"""Typed action request structures and validation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCallRequest:
    """High-level tool invocation proposed by the LLM or behavior layer."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass(frozen=True)
class ToolValidationOutcome:
    """Result of checking a proposed tool call against the CapabilityRegistry."""

    valid: bool
    tool: str
    normalized_args: dict[str, Any] = field(default_factory=dict)
    clamped: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
