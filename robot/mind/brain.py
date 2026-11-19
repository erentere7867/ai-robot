"""The brain: the vision-language model (LFM3-VL-2B for now) behind llama-server."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Reply:
    text: str
    read_ms: float = 0.0
    write_ms: float = 0.0
    read_tokens: int = 0
    write_tokens: int = 0
    thought: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    tool_calls: list[ToolCall] = field(default_factory=list)


# The model card's sampling for LFM3-VL-2B (with temperature 0.1); llama-server's own defaults differ (top_p 0.95,
# min_p 0.05, no repeat penalty).
CARD_SAMPLING = {"top_k": 50, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.1}


def _with_image(url: str, text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": text},
            ],
        }
    ]


class Brain:
    def __init__(self, url: str = "http://127.0.0.1:8081", timeout_s: float = 180.0) -> None:
        self.http = httpx.Client(base_url=url, timeout=timeout_s, trust_env=False)

    def healthy(self) -> bool:
        try:
            return self.http.get("/health", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 400,
        temperature: float = 0.1,
        think: int = 0,
    ) -> Reply:
        """One assistant turn with the model's native tool calling: spoken text, tool calls, or both.

        llama-server renders `tools` with the model's own chat template and parses the calls it writes back into
        OpenAI-style `tool_calls`, so the model acts in the format it was trained on. With `think` > 0 the model
        reasons first, for at most that many tokens (llama-server closes the thought when the budget runs out).
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            **CARD_SAMPLING,
            "max_tokens": max_tokens + max(0, think),
            "cache_prompt": True,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": think > 0},
        }
        if think > 0:
            payload["thinking_budget_tokens"] = think
        r = self.http.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        message = data["choices"][0]["message"]
        calls = []
        for i, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or {}
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                args = {"_unparsed": raw_args}
            calls.append(ToolCall(tc.get("id") or f"call_{i}", str(fn.get("name", "")), args if isinstance(args, dict)
                                  else {"_unparsed": raw_args}))
        t = data.get("timings") or {}
        return Reply(
            (message.get("content") or "").strip(),
            t.get("prompt_ms", 0.0),
            t.get("predicted_ms", 0.0),
            t.get("prompt_n", 0),
            t.get("predicted_n", 0),
            thought=message.get("reasoning_content") or "",
            raw=data,
            tool_calls=calls,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        rules: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Reply:
        """One answer held to the grammar `rules` (short yes/no, one-word and number answers)."""
        payload: dict[str, Any] = {
            "messages": messages,
            "grammar": rules,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cache_prompt": True,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_format": "none",
        }
        r = self.http.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        t = data.get("timings") or {}
        text, thought = data["choices"][0]["message"]["content"] or "", ""
        if "</think>" in text:
            thought, text = text.split("</think>", 1)
            thought, text = thought.removeprefix("<think>"), text.lstrip()
        return Reply(
            text,
            t.get("prompt_ms", 0.0),
            t.get("predicted_ms", 0.0),
            t.get("prompt_n", 0),
            t.get("predicted_n", 0),
            thought=thought,
            raw=data,
        )

    def decide(
        self,
        messages: list[dict[str, Any]],
        choices: tuple[str, ...] = ("yes", "no"),
    ) -> dict[str, float]:
        """One token from the model, its log-probabilities matched to `choices`,

        and returns normalized probability dict summing to 1.0 (uniform if no matching tokens).
        """
        r = self.http.post(
            "/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": 20,
                "cache_prompt": True,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        r.raise_for_status()
        content = (r.json()["choices"][0].get("logprobs") or {}).get("content") or []
        probs = {c: 0.0 for c in choices}
        for item in content[0]["top_logprobs"] if content else []:
            tok = item["token"].strip().lower()
            if not tok:
                continue
            p = math.exp(item["logprob"])
            for c in choices:
                cl = c.lower()
                if tok == cl:
                    probs[c] += p
                    break
        total = sum(probs.values())
        if total > 0.0:
            return {c: probs[c] / total for c in choices}
        return {c: 1.0 / len(choices) for c in choices}

    def yes_probability(self, messages: list[dict[str, Any]]) -> float:
        """P(yes) / (P(yes) + P(no)) for the first word of the answer."""
        res = self.decide(messages, choices=("yes", "no"))
        return res["yes"]

    def check(self, image_url: str, question: str) -> tuple[bool, Reply]:
        """A yes or no question about one image."""
        reply = self.complete(_with_image(image_url, f"{question} Answer yes or no."), 'root ::= "yes" | "no"', 2)
        return reply.text.strip() == "yes", reply

    def ground(self, image_url: str, marked_url: str, description: str, marks: int) -> tuple[int | None, Reply]:
        """Which numbered box is this? None when it is not there."""
        seen, first = self.check(image_url, f"Is {description} visible in this picture?")
        if not seen:
            return None, first
        reply = self.complete(
            _with_image(
                marked_url,
                f"Numbered boxes mark objects in the image. Which number is {description}? Answer with the number.",
            ),
            "root ::= " + " | ".join(f'"{i}"' for i in range(1, marks + 1)),
            4,
        )
        reply.read_ms += first.read_ms
        reply.write_ms += first.write_ms
        return int(reply.text), reply

    def room_kind(self, image_url: str) -> str:
        """What kind of room a picture from inside it shows, in one or two words ("kitchen", "living room")."""
        reply = self.complete(
            _with_image(image_url, "This picture is from a small robot's camera inside a home. What kind of room is "
                                   "it? Answer with the room's usual name, in one or two words."),
            'root ::= [a-z]+ (" " [a-z]+)?', 6)
        return reply.text.strip()

    def describe(self, image_url: str, question: str) -> Reply:
        """A short answer in words about one image."""
        return self.complete(
            _with_image(image_url, f"{question} Answer in one short sentence."),
            r"root ::= [a-zA-Z0-9 ,.:;'!?-]{1,160}",
            40,
        )
