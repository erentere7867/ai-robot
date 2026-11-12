"""The brain: LFM3-VL-2B behind llama-server.

Every query is built from current state with no chat history, and a grammar restricts each answer to valid output.
Planning (writing programs against robot.agent.api) is added in phase 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class Reply:
    text: str
    read_ms: float = 0.0
    write_ms: float = 0.0
    read_tokens: int = 0
    write_tokens: int = 0
    thought: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


NOT_A_THING = ("wall", "floor", "nothing")  # answers to "what object is this?" that mean there is none


def _with_image(url: str, text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": text}]}]


class Brain:
    def __init__(self, url: str = "http://127.0.0.1:8080", timeout_s: float = 180.0) -> None:
        self.http = httpx.Client(base_url=url, timeout=timeout_s, trust_env=False)

    def healthy(self) -> bool:
        try:
            return self.http.get("/health", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def complete(self, messages: list[dict[str, Any]], rules: str, max_tokens: int, think: int = 0) -> Reply:
        """One answer held to the grammar `rules`. With `think` > 0 the model first reasons in up to that many
        characters, then answers under the grammar: a grammar alone would forbid the reasoning, because it applies
        from the first token."""
        if think:
            head = f'root ::= [^<]{{1,{think}}} "</think>" [ \\n]* answer\n'  # the template already opened <think>
            rules = head + rules.replace("root ::=", "answer ::=", 1)
            max_tokens += think // 3
        r = self.http.post("/v1/chat/completions", json={
            "messages": messages, "grammar": rules, "temperature": 0.0, "max_tokens": max_tokens,
            "cache_prompt": True, "stream": False, "chat_template_kwargs": {"enable_thinking": bool(think)},
            "reasoning_format": "none",
        })
        r.raise_for_status()
        data = r.json()
        t = data.get("timings") or {}
        text, thought = data["choices"][0]["message"]["content"] or "", ""
        if "</think>" in text:  # with thinking off some templates still open and close an empty <think> block
            thought, text = text.split("</think>", 1)
            thought, text = thought.removeprefix("<think>"), text.lstrip()
        return Reply(text, t.get("prompt_ms", 0.0), t.get("predicted_ms", 0.0), t.get("prompt_n", 0),
                     t.get("predicted_n", 0), data, thought)

    def yes_probability(self, messages: list[dict[str, Any]]) -> float:
        """P(yes) / (P(yes) + P(no)) for the first word of the answer. A small model's bare yes/no can be one-sided
        (the readback check said "no" to all 92 programs it saw), while the probability still ranks them, so callers
        compare it with a threshold measured on labelled examples."""
        r = self.http.post("/v1/chat/completions", json={
            "messages": messages, "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 20,
            "cache_prompt": True, "stream": False, "chat_template_kwargs": {"enable_thinking": False},
        })
        r.raise_for_status()
        content = (r.json()["choices"][0].get("logprobs") or {}).get("content") or []
        yes = no = 0.0
        for item in content[0]["top_logprobs"] if content else []:
            word = item["token"].strip().lower()
            if word in ("yes", "y"):
                yes += math.exp(item["logprob"])
            elif word in ("no", "n"):
                no += math.exp(item["logprob"])
        return yes / (yes + no) if yes + no else 0.5

    def check(self, image_url: str, question: str) -> tuple[bool, Reply]:
        """A yes or no question about one image."""
        reply = self.complete(_with_image(image_url, f"{question} Answer yes or no."), 'root ::= "yes" | "no"', 2)
        return reply.text.strip() == "yes", reply

    def name(self, image_url: str, words: list[str]) -> str | None:
        """What the object in the middle of a crop is, chosen from `words`; None when the model sees no object.

        A choice beats a free answer: answering freely, the model called a red ball a "button" and a cardboard box a
        "wall"; choosing from the robot's words it named 8 of 10 simulator crops right (both misses called a low
        coffee table a desk). The house itself is offered too ("wall", "floor"): without those, the end of a wall
        beside a doorway was named a cabinet, the nearest word on offer. They mean no object, like "nothing"."""
        words = sorted(set(words) - set(NOT_A_THING)) + list(NOT_A_THING)
        reply = self.complete(_with_image(image_url, "What is the object in the middle of this picture? Answer with "
                                          "one word from: " + ", ".join(words) + "."),
                              "root ::= " + " | ".join(f'"{w}"' for w in words), 8)
        answer = reply.text.strip()
        return None if answer in NOT_A_THING else answer

    def ground(self, image_url: str, marked_url: str, description: str, marks: int) -> tuple[int | None, Reply]:
        """Which numbered box is this? None when it is not there.

        Presence is checked first. Offered "none" as an answer, the model picked a box 49 times out of 50 for things that
        were not in the picture, while it answered "is there a ..." correctly 50 out of 50 (M0).
        """
        seen, first = self.check(image_url, f"Is {description} visible in this picture?")
        if not seen:
            return None, first
        reply = self.complete(_with_image(marked_url, f"Numbered boxes mark objects in the image. Which number is {description}? "
                                                  "Answer with the number."),
                          "root ::= " + " | ".join(f'"{i}"' for i in range(1, marks + 1)), 4)
        reply.read_ms += first.read_ms
        reply.write_ms += first.write_ms
        return int(reply.text), reply

    def describe(self, image_url: str, question: str) -> Reply:
        """A short answer in words about one image."""
        return self.complete(_with_image(image_url, f"{question} Answer in one short sentence."),
                         r"root ::= [a-zA-Z0-9 ,.:;'!?-]{1,160}", 40)
