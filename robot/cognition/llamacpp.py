"""llama.cpp HTTP server backend and AutoBackend with automatic server lifecycle management."""

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

from robot.cognition.backend import InferenceBackend, InferenceResult, UnavailableBackend
from robot.core.config import LLMConfig

logger = logging.getLogger(__name__)


class LlamaCppBackend:
    """OpenAI-compatible client for `llama-server` with automatic local process startup."""

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

    async def ensure_server_started(self) -> bool:
        """Start local `llama-server.exe` if not already running and `auto_start_server` is enabled."""
        if await self.is_available():
            return True

        if not self.config.auto_start_server:
            return False

        bin_path = self.config.llama_server_bin
        model_path = self.config.model_path
        if not bin_path.exists() or not model_path.exists():
            logger.info(
                "llama-server binary (%s) or model (%s) not found; staying in offline mode",
                bin_path,
                model_path,
            )
            return False

        parsed = urlparse(self.base_url)
        host = parsed.hostname or "127.0.0.1"
        if host not in {"127.0.0.1", "localhost", "::1"}:
            return False
        port = parsed.port or 8080

        cmd = [
            str(bin_path),
            "-m",
            str(model_path),
            "--host",
            host,
            "--port",
            str(port),
            "-ngl",
            str(self.config.gpu_layers),
            "-c",
            str(self.config.ctx_size),
            "-np",
            "1",
            "--reasoning",
            "off",
            "--temp",
            str(self.config.temperature),
        ]
        if self.config.mmproj_path and self.config.mmproj_path.exists():
            cmd.extend(["--mmproj", str(self.config.mmproj_path)])
        print(f"[llm] Starting local model server ({self.config.model_name}) on {self.base_url} ...")
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        # Wait up to 15 seconds for server health check
        for _ in range(50):
            if self._proc.returncode is not None:
                break
            if await self.is_available():
                print(f"[llm] {self.config.model_name} is ready on {self.base_url}!")
                return True
            await asyncio.sleep(0.3)

        print("[llm] Warning: llama-server did not become ready in time; using offline fallback.")
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
            if resp.status_code == 200:
                data = resp.json()
                return str(data.get("status", "ok")).lower() == "ok"
            return False
        except Exception:
            return False

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> InferenceResult:
        t0 = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        functions = self._function_tools(json_schema) if self.config.native_tools else []
        if functions:
            payload.update(tools=functions, tool_choice="auto", parallel_tool_calls=False)
        elif json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "robot_response",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        # Bound even an unexpected/misconfigured server response before parsing JSON.
        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            raw = bytearray()
            async for chunk in resp.aiter_bytes():
                raw.extend(chunk)
                if len(raw) > 65536:
                    raise ValueError("Inference response exceeds 64 KiB")
        data = json.loads(raw)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        choices = data.get("choices") or []
        content = ""
        metadata = {"timings": data.get("timings") or {}}
        if choices:
            msg = choices[0].get("message") or {}
            content = str(msg.get("content") or "")
            if functions:
                content, assistant = self._normalize_native(
                    msg, choices[0].get("finish_reason"), {t["function"]["name"] for t in functions},
                )
                metadata.update(native_tools=True, assistant_message=assistant)

        usage = data.get("usage") or {}
        return InferenceResult(
            raw_text=content,
            backend_name=f"llamacpp:{self.config.model_name}",
            latency_ms=latency_ms,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            metadata=metadata,
        )

    @staticmethod
    def _function_tools(schema: dict | None) -> list[dict]:
        variants = (schema or {}).get("properties", {}).get("actions", {}).get("items", {}).get("oneOf", [])
        descriptions = {
            "spin": "Rotate the mobile robot base in place by a relative angle in degrees. 180 faces the opposite direction; 360 returns to the original heading. Positive is left, negative is right.",
            "drive": "Move the mobile robot forward or backward by distance_mm. Sensors can stop motion; inspect the returned result.",
            "tilt_head": "Tilt the head camera to an absolute angle in degrees. Positive looks upward, negative downward.",
            "perform_trick": "Perform a named expressive gesture. For camera observation and navigation use spin, drive and tilt_head directly instead.",
            "set_gaze": "Change the display's eye gaze. This does not rotate the camera or robot base.",
            "set_eye_color": "Change the display's eye colour.",
            "stop": "Stop the robot's motors immediately.",
        }
        result = []
        for variant in variants:
            properties = variant.get("properties", {})
            names = properties.get("tool", {}).get("enum", [])
            if len(names) != 1:
                continue
            name = names[0]
            if name == "perform_trick":
                continue  # The model composes movement from primitives instead of canned routines.
            result.append({"type": "function", "function": {
                "name": name, "description": descriptions.get(name, name),
                "parameters": properties["args"],
            }})
        if result:
            result.append({"type": "function", "function": {
                "name": "observe", "description": "Read the current head camera when visual information is needed for the current request. Does not move the robot. Perception is not a request to narrate or rearrange the scene.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            }})
            result.append({"type": "function", "function": {
                "name": "schedule_wake", "description": "Request one future background decision after this request ends. Otherwise unchanged idle observations cause no model calls.",
                "parameters": {"type": "object", "properties": {
                    "delay_s": {"type": "number", "minimum": 1, "maximum": 300}},
                    "required": ["delay_s"], "additionalProperties": False},
            }})
        return result

    @staticmethod
    def _normalize_native(message: dict, finish_reason: str | None, allowed_tools: set[str]) -> tuple[str, dict]:
        """Adapt actual function calls to the existing validated execution path."""
        if finish_reason == "length":
            raise ValueError("Model output was truncated; no partial tool call will execute")
        calls = message.get("tool_calls") or []
        if len(calls) > 1:
            raise ValueError("Parallel calls are disabled: the robot executes one tool per decision")
        assistant = {"role": "assistant", "content": str(message.get("content") or "")}
        actions, normalized = [], []
        wake = None
        for call in calls:
            function = call.get("function") or {}
            name = str(function.get("name", ""))[:80]
            arguments = function.get("arguments", {})
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                if not isinstance(arguments, dict):
                    raise ValueError("Arguments must be an object")
            except (ValueError, TypeError):
                arguments = {"_invalid_arguments": True}
            call_id = str(call.get("id") or uuid.uuid4().hex)[:160]
            normalized.append({"id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
            }})
            if name == "schedule_wake":
                wake = arguments.get("delay_s", 0)
            elif name == "observe" and name in allowed_tools:
                pass  # Camera access is handled by the arbiter, outside the motor HAL.
            else:
                actions.append({"tool": name if name in allowed_tools else f"unavailable:{name}",
                                "args": arguments, "call_id": call_id})
        if normalized:
            assistant["tool_calls"] = normalized
        if message.get("reasoning_content"):
            assistant["reasoning_content"] = message["reasoning_content"]
        return json.dumps({
            "say": assistant["content"], "actions": actions,
            "status": "continue" if normalized else "completed", "next_wake_s": wake,
        }, ensure_ascii=False), assistant


class AutoBackend:
    """Routes inference requests with backoff; an unavailable model cannot issue motion."""

    name = "auto"

    def __init__(
        self,
        primary: InferenceBackend,
        fallback: InferenceBackend | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or UnavailableBackend()
        self.last_used_backend: str = "none"
        self._retry_at = 0.0
        self._failures = 0

    async def is_available(self) -> bool:
        return True

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> InferenceResult:
        if time.monotonic() >= self._retry_at:
            try:
                res = await self.primary.generate(messages, json_schema=json_schema)
                self.last_used_backend = res.backend_name
                self._failures = 0
                return res
            except Exception as exc:
                self._failures += 1
                self._retry_at = time.monotonic() + min(30.0, 2.0 ** min(self._failures, 5))
                logger.warning("Primary LLM backend failed (%s); using offline fallback", exc)

        res = await self.fallback.generate(messages, json_schema=json_schema)
        self.last_used_backend = res.backend_name
        return res
