"""llama.cpp `llama-server` client (OpenAI-compatible, native tool calls) with local server lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx

from robot.cognition.backend import BackendUnavailable, ChatResult, ToolCall
from robot.core.config import LLMConfig

logger = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 65536
MAX_TOOL_CALLS = 3


class LlamaCppBackend:
    name = "llamacpp"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(config.timeout_s, connect=2.5),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
            trust_env=False,
        )
        self._proc: asyncio.subprocess.Process | None = None
        # One request at a time: llama-server runs a single slot and queued requests only add latency.
        self._lock = asyncio.Lock()

    async def ensure_server_started(self) -> bool:
        """Start a local `llama-server` with the vision projector if none is answering."""
        if await self.is_available():
            return True
        if not self.config.auto_start_server:
            return False
        bin_path, model_path = self.config.llama_server_bin, self.config.model_path
        if not bin_path.exists() or not model_path.exists():
            logger.info("llama-server (%s) or model (%s) not found; model offline", bin_path, model_path)
            return False
        parsed = urlparse(self.base_url)
        host = parsed.hostname or "127.0.0.1"
        if host not in {"127.0.0.1", "localhost", "::1"}:
            return False
        cmd = [
            str(bin_path), "-m", str(model_path), "--host", host, "--port", str(parsed.port or 8080),
            "-ngl", str(self.config.gpu_layers), "-c", str(self.config.ctx_size), "-np", "2",
            "--jinja", "--reasoning", "auto", "--reasoning-format", "deepseek", "--no-reasoning-preserve",
            "--reasoning-budget", str(self.config.thinking_budget_tokens), "--temp", str(self.config.temperature),
            # LFM3 mixes convolutions and sliding-window attention, so a cached prompt can only be resumed
            # from a checkpoint. The default spacing (8192 tokens) is longer than our prompts; with these two
            # flags a decision re-reads ~5-150 tokens instead of ~400 (measured: 1.9 s -> 0.7 s per call).
            "--swa-full", "--checkpoint-min-step", "256",
        ]
        if self.config.mmproj_path and self.config.mmproj_path.exists():
            cmd.extend(["--mmproj", str(self.config.mmproj_path)])
        else:
            logger.warning("Vision projector not found (%s); the robot will be blind", self.config.mmproj_path)
        print(f"[llm] Starting local model server ({self.config.model_name}) on {self.base_url} ...")
        self._proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            if self._proc.returncode is not None:
                break
            if await self.is_available():
                print(f"[llm] {self.config.model_name} is ready.")
                return True
            await asyncio.sleep(0.5)
        print("[llm] Warning: llama-server did not become ready; the model is offline.")
        return False

    async def close(self) -> None:
        await self._client.aclose()
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            except TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None

    async def is_available(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=1.5)
            return resp.status_code == 200 and str(resp.json().get("status", "ok")).lower() == "ok"
        except Exception:
            return False

    async def chat(self, messages, *, tools=None, max_tokens=None, tool_choice="auto") -> ChatResult:
        # Decisions (requests with tools) may think first; detection and yes/no checks answer directly.
        thinking = self.config.thinking and bool(tools)
        budget = (max_tokens or self.config.max_tokens) + (self.config.thinking_budget_tokens if thinking else 0)
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": budget,
            "stream": False,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if tools:
            payload.update(tools=tools, tool_choice=tool_choice)
        t0 = time.perf_counter()
        async with self._lock:
            for attempt in range(2):
                try:
                    async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                        resp.raise_for_status()
                        raw = bytearray()
                        async for chunk in resp.aiter_bytes():
                            raw.extend(chunk)
                            if len(raw) > MAX_RESPONSE_BYTES:
                                raise ValueError("Inference response exceeds 64 KiB")
                    break
                except httpx.TransportError as exc:
                    # A pooled keep-alive connection the server already closed fails at once with
                    # "Server disconnected without sending a response"; a fresh connection works.
                    if attempt == 0 and not isinstance(exc, httpx.TimeoutException):
                        logger.info("Retrying model request after transport error: %s", exc)
                        continue
                    raise BackendUnavailable(str(exc)) from exc
                except httpx.HTTPStatusError as exc:
                    raise BackendUnavailable(str(exc)) from exc
        data = json.loads(raw)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish = choice.get("finish_reason")
        calls = self.parse_tool_calls(message.get("tool_calls") or [])
        if calls and finish == "length":
            # A truncated call could carry half an argument list; never execute it.
            calls = []
        usage = data.get("usage") or {}
        return ChatResult(
            content=str(message.get("content") or "").strip(),
            reasoning=str(message.get("reasoning_content") or "").strip(),
            tool_calls=calls,
            backend_name=f"llamacpp:{self.config.model_name}",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=finish,
        )

    @staticmethod
    def parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
        calls = []
        for raw in raw_calls[:MAX_TOOL_CALLS]:
            function = raw.get("function") or {}
            arguments: Any = function.get("arguments", {})
            invalid = False
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                if arguments is None:
                    arguments = {}
                if not isinstance(arguments, dict):
                    raise ValueError
            except ValueError:
                arguments, invalid = {}, True
            calls.append(ToolCall(str(raw.get("id") or uuid.uuid4().hex)[:64],
                                  str(function.get("name", ""))[:64], arguments, invalid))
        return calls


class AutoBackend:
    """Primary model with exponential backoff; while it is down every request is BackendUnavailable."""

    name = "auto"

    def __init__(self, primary: LlamaCppBackend) -> None:
        self.primary = primary
        self.last_used_backend = "none"
        self._retry_at = 0.0
        self._failures = 0

    async def is_available(self) -> bool:
        return await self.primary.is_available()

    async def chat(self, messages, *, tools=None, max_tokens=None, tool_choice="auto") -> ChatResult:
        if time.monotonic() < self._retry_at:
            raise BackendUnavailable("Model recently failed; waiting before retry")
        try:
            result = await self.primary.chat(messages, tools=tools, max_tokens=max_tokens, tool_choice=tool_choice)
        except BackendUnavailable:
            self._failures += 1
            self._retry_at = time.monotonic() + min(30.0, 2.0 ** min(self._failures, 5))
            raise
        self._failures = 0
        self.last_used_backend = result.backend_name
        return result
