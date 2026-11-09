"""The model-driven agent loop: think -> call skills -> read results -> repeat until it answers."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from robot.cognition.backend import BackendUnavailable, ChatResult, InferenceBackend
from robot.cognition.guards import claims_unperformed_action, contradicts_success, ignores_command, speakable
from robot.cognition.planner import TaskList, split_parts
from robot.cognition.prompts import SITUATION_HEADER, estimate_tokens
from robot.cognition.repair import repair_call
from robot.cognition.textcalls import recover_text_tool_calls, strip_reasoning, strip_written_calls
from robot.core.angles import wrap_deg
from robot.core.config import AgentConfig, ContextBudgetConfig
from robot.skills.base import SkillContext, SkillRegistry, SkillResult

logger = logging.getLogger(__name__)

IMAGE_NOTE = "(camera view after {tool})"
NUDGE = ("(No tool was called, so nothing happened. The request was: \"{request}\". Every request is new, even "
         "if you did the same before. Call the tool that does it now.)")
FIX_REPLY = "(That tool succeeded: {result} Tell the human in one short sentence what you just did.)"
DUPLICATE = "Already done: {result} Nothing more to do for this; tell the human."
# Appended to the last tool result of each step of a human request. With several steps ("go to the ball
# and then to the cube") a small model tends to stop after the first; with one ("go to the ball") it
# tends to keep acting unless told that this may be the end.
_SEVERAL_PARTS = re.compile(r"\b(?:then|after|afterwards|before|also|next)\b|\band (?:go|come|turn|look|move|"
                            r"find|drive|tell|show|say|do|make|bring|check|spin|back)\b", re.I)
DONE_HINT = "If that finishes \"{request}\", answer the human now."
PROGRESS = ("Request: \"{request}\". Answer the human now, unless the request asks for another step "
            "that is not done yet.")


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

    def action_notes(self) -> list[str]:
        notes = []
        for entry in self.tool_log:
            args = ", ".join(f"{v}" for v in entry["args"].values())
            notes.append(f"{entry['tool']}({args}) {'ok' if entry['ok'] else 'failed'}")
        return notes


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]


def _statements(text: str) -> str:
    return " ".join(s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s and not s.endswith("?"))


def _same_spot(a: Any, b: Any) -> bool:
    return (math.hypot(a.x_mm - b.x_mm, a.y_mm - b.y_mm) < 100.0
            and abs(wrap_deg(a.heading_deg - b.heading_deg)) < 10.0)


def _shorten(payload: dict[str, Any]) -> str:
    """A tool result squeezed for an over-long prompt; what was seen survives (the answer may depend on it)."""
    text = str(payload.get("result", ""))[:60]
    objects = payload.get("objects")
    if isinstance(objects, list) and objects:
        text += " Seen: " + ", ".join(str(o).split(":")[0] for o in objects[:8])
    return text


class AgentLoop:
    def __init__(self, backend: InferenceBackend, skills: SkillRegistry, config: AgentConfig,
                 *, budget: ContextBudgetConfig | None = None,
                 on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
                 robot_name: str = "") -> None:
        self.backend = backend
        self.robot_name = robot_name
        self.skills = skills
        self.config = config
        budget = budget or ContextBudgetConfig()
        self.message_limit_tokens = budget.total_tokens - budget.response_reserve_tokens - budget.tools_tokens
        self.on_event = on_event
        self.tools = skills.tool_schemas()
        self.tool_names = set(skills.names)
        self.acting = skills.names_where("acts")
        self.persistent = skills.names_where("persistent")

    async def warm_up(self, system_prompt: str) -> None:
        """Have the model read the system prompt and tool list once, so the server caches them.

        Measured: the first decision after the server starts took 6.9 s (2.5k prompt tokens read cold on the
        laptop GPU), the next ones 0.5-1.6 s. Without this the human's first request pays that, and the
        robot set off late enough to lose a rolling ball. On the robot's CPU the cold read is longer still.
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Hello."}]
        try:
            await self.backend.chat(messages, tools=self.tools, max_tokens=1)
        except BackendUnavailable:
            logger.info("Model warm-up skipped", exc_info=True)

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.on_event is not None:
            await self.on_event(event_type, payload)

    async def run(self, messages: list[dict[str, Any]], ctx: SkillContext, *, request: str) -> TurnResult:
        """One turn. `request` is what the human said; empty on a background turn."""
        return await _Turn(self, messages, ctx, request.strip()[:200]).run()

    @staticmethod
    def attach_image(messages: list[dict[str, Any]], data_url: str, tool: str) -> None:
        """Keep only the newest camera image in context; older views become a text note."""
        for message in messages:
            if message.get("role") == "user" and isinstance(message.get("content"), list):
                texts = [p.get("text", "") for p in message["content"] if p.get("type") == "text"]
                message["content"] = " ".join(texts) + " (image no longer shown)"
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": IMAGE_NOTE.format(tool=tool)},
        ]})

    def fit(self, messages: list[dict[str, Any]]) -> None:
        """Drop the oldest history and tool exchanges until the prompt fits; never split a call/result pair."""
        def cost(m: dict[str, Any]) -> int:
            content = m.get("content")
            if isinstance(content, list):
                return 300 + sum(estimate_tokens(p.get("text", "")) for p in content if isinstance(p, dict))
            return estimate_tokens(str(content or "")) + estimate_tokens(json.dumps(m.get("tool_calls", ""))) + 6

        while sum(map(cost, messages)) > self.message_limit_tokens and len(messages) > 2:
            request_idx = next((i for i, m in enumerate(messages) if m.get("role") == "user"
                                and SITUATION_HEADER in str(m.get("content"))), None)
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
            summary = "; ".join(_shorten(json.loads(messages[k]["content"])) for k in range(start + 1, end))
            messages[start:end] = [{"role": "user", "content": f"(earlier steps, shortened: {summary})"}]


class _Turn:
    """The state of one turn of the agent loop."""

    def __init__(self, loop: AgentLoop, messages: list[dict[str, Any]], ctx: SkillContext, request: str) -> None:
        self.loop = loop
        self.messages = messages
        self.ctx = ctx
        self.request = request
        self.background = not request
        self.result = TurnResult(say="")
        self.started = time.monotonic()
        # This turn's calls and results, kept apart from `messages`: fit may delete or merge entries there,
        # so slicing `messages` by a start index could cut a call from its result.
        self.transcript: list[dict[str, Any]] = []
        self.attempts: dict[str, int] = {}
        self.failures = 0
        self.nudged = False
        self.force_tool = self.force_reply = False
        self.succeeded: dict[str, tuple[Any, SkillResult]] = {}  # idempotent call -> (pose after it, its result)
        self.hint = PROGRESS if _SEVERAL_PARTS.search(request) else DONE_HINT
        parts = split_parts(request) if request else []
        self.todo = TaskList(request, parts) if parts else None
        self.words = self.todo.current if self.todo else request  # what the current part says
        self.said: list[str] = []  # answers to the parts before the last one
        self.part_ok = True  # no tool of the current part failed
        self.part_start = 0  # tool_log index where the current part began

    @property
    def last_part(self) -> bool:
        return self.todo is None or not self.todo.has_next

    async def run(self) -> TurnResult:
        if self.todo is not None:
            await self.loop.emit("cognition.plan", {"request": self.request, "steps": self.todo.parts})
        for step in range(self.loop.config.max_steps):
            self.ctx.check()
            if time.monotonic() - self.started > self.loop.config.timeout_s:
                self.result.status = "blocked"
                self.result.say = "That took too long, so I stopped."
                break
            reply = await self._ask(step)
            if reply is None:
                break
            if not reply.tool_calls:
                reply.tool_calls = recover_text_tool_calls(reply.content, self.loop.tool_names)
                if reply.tool_calls:
                    reply.content = ""
                    await self.loop.emit("cognition.recovered_calls", {"calls": [c.name for c in reply.tool_calls]})
            finished = await self._act(reply) if reply.tool_calls else await self._answer(reply)
            if finished:
                break
        else:
            self.result.status = "blocked"
            self.result.say = "" if self.background else "I ran out of steps before finishing."
        return self._finish()

    async def _ask(self, step: int) -> ChatResult | None:
        loop, result = self.loop, self.result
        loop.fit(self.messages)
        try:
            # The tools stay in the request either way, so the cached prompt prefix is reused.
            choice = "required" if self.force_tool else "none" if self.force_reply else "auto"
            reply = await loop.backend.chat(self.messages, tools=loop.tools, tool_choice=choice)
            self.force_tool = self.force_reply = False
        except BackendUnavailable as exc:
            result.status = "unavailable"
            result.say = "" if self.background else "My thinking model is offline right now, so I can't do that."
            await loop.emit("cognition.unavailable", {"error": str(exc)[:200]})
            return None
        self.ctx.check()
        result.steps = step + 1
        result.backend_name = reply.backend_name
        result.latency_ms += reply.latency_ms
        await loop.emit("cognition.model_call", {
            "step": step + 1, "latency_ms": round(reply.latency_ms, 1), "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens, "content": reply.content[:300],
            "reasoning": reply.reasoning[:600],
            "tool_calls": [{"name": c.name, "args": c.arguments} for c in reply.tool_calls],
        })
        return reply

    async def _answer(self, reply: ChatResult) -> bool:
        """The model replied in words. True when that ends the turn."""
        loop, result, messages = self.loop, self.result, self.messages
        result.say, leaked = strip_reasoning(strip_written_calls(reply.content))
        if leaked and not self.background:
            messages.append({"role": "user", "content": "(Reply to the human now in one short sentence.)"})
            try:
                retry = await loop.backend.chat(messages, max_tokens=80)
                result.say = strip_reasoning(retry.content)[0]
            except BackendUnavailable:
                pass
            messages.pop()
        if not self.last_part and self.part_ok and result.say:
            # The answer to one part ("I'm at the ball!") ends that part; hand over the next one.
            self.said.append(result.say)
            messages.append({"role": "assistant", "content": reply.content})
            await self._next_part()
            self.part_ok, self.part_start = True, len(result.tool_log)
            result.say = ""
            return False
        # Only this part's tools count: "now I'll head back" after part 1's go_to is still a claim.
        part_log = result.tool_log[self.part_start:]
        if not self.nudged and (claims_unperformed_action(result.say, part_log, loop.acting, loop.persistent)
                                or ignores_command(self.words, part_log)):
            # The model narrated (or refused) an action instead of doing it, often by copying an earlier reply.
            self.nudged = self.force_tool = True
            messages.append({"role": "assistant", "content": reply.content})
            messages.append({"role": "user", "content": NUDGE.format(request=self.request)})
            await loop.emit("cognition.nudged", {"reply": result.say[:200]})
            result.say = ""
            return False
        if not self.nudged and contradicts_success(result.say, result.tool_log, loop.acting):
            self.nudged = True  # e.g. shows a sad face, then says it can't make one
            done = next(e for e in reversed(result.tool_log) if e["ok"] and e["tool"] in loop.acting)
            messages.append({"role": "assistant", "content": reply.content})
            messages.append({"role": "user", "content": FIX_REPLY.format(result=done["message"])})
            await loop.emit("cognition.nudged", {"reply": result.say[:200], "reason": "contradiction"})
            result.say = ""
            return False
        return True

    async def _act(self, reply: ChatResult) -> bool:
        """Run the model's tool calls. True when that ends the turn."""
        loop, ctx, result, messages = self.loop, self.ctx, self.result, self.messages
        messages.append(reply.assistant_message())
        self.transcript.append(messages[-1])
        stop_batch = False
        finished_part = False
        persistent: SkillResult | None = None  # a behaviour that keeps running started in this batch
        batch: list[tuple[str, SkillResult]] = []
        status = ""
        for call in reply.tool_calls:
            name = call.name
            if stop_batch and persistent:
                outcome = SkillResult(False, "Not executed: it would stop what you just started.", status="skipped")
            elif stop_batch:
                outcome = SkillResult(False, "Not executed because an earlier step failed.", status="skipped")
            elif call.invalid_arguments:
                outcome = SkillResult(False, "The arguments were not valid JSON.", status="rejected")
            else:
                fixed = repair_call(call.name, call.arguments, self.words, ctx)
                name = fixed.tool
                skill = loop.skills.get(name)
                key = json.dumps([name, fixed.args], sort_keys=True)
                if key in self.succeeded and _same_spot(self.succeeded[key][0], ctx.pose()):
                    # Keep the first call's data (an observe's object list): the model answers from it.
                    first = self.succeeded[key][1]
                    outcome = SkillResult(True, DUPLICATE.format(result=first.message), dict(first.data),
                                          status="duplicate")
                    self.force_reply = self.last_part
                    finished_part = finished_part or bool(skill and skill.ends_part)
                else:
                    outcome = await self._run_skill(name, fixed)
                    result.tool_log.append({"tool": name, "args": fixed.args, "ok": outcome.ok,
                                            "message": outcome.message})
                    stop_batch = not outcome.ok
                    if outcome.ok and skill is not None and skill.persistent:
                        persistent, stop_batch = outcome, True
                    self.part_ok = self.part_ok and outcome.ok
                    finished_part = finished_part or (outcome.ok and bool(skill and skill.ends_part))
                    self.failures += 0 if outcome.ok else 1
                    self.attempts[key] = self.attempts.get(key, 0) + 1
                    if outcome.ok and skill is not None and skill.idempotent:
                        self.succeeded[key] = (ctx.pose(), outcome)
            batch.append((name, outcome))
            status = outcome.status
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(outcome.to_model(), ensure_ascii=False)})
            self.transcript.append(messages[-1])
        advance = not self.last_part and self.part_ok and finished_part
        if persistent is not None and self.last_part:
            result.say = _first_sentence(persistent.message)  # it keeps going; the turn is over
            await loop.emit("cognition.fast_finish", {"tool": batch[-1][0], "say": result.say})
            return True
        last_name, last_outcome = batch[-1]
        last_skill = loop.skills.get(last_name)
        if (self.request and self.part_ok and self.last_part and last_skill is not None and last_skill.report_is_answer
                and last_outcome.ok and last_outcome.status != "duplicate" and all(o.ok for _, o in batch)):
            # Measured: the closing call was ~0.75 s of a 2-4 s request on the laptop GPU, and its reply was
            # usually the report reworded plus "What would you like to do next?".
            result.say = _first_sentence(last_outcome.message)
            await loop.emit("cognition.fast_finish", {"tool": last_name, "say": result.say})
            return True
        if self.request and (status != "duplicate" or self.todo is not None):
            # A field of its own: merged into "result" it made the model act on it as an instruction
            # (measured: "two_stops" went to the wrong object).
            last = json.loads(messages[-1]["content"])
            last["next"] = (f"Part {self.todo.index + 1} (\"{self.todo.current}\") is done." if advance and self.todo
                            else self.todo.hint() if self.todo is not None else self.hint.format(request=self.request))
            messages[-1]["content"] = json.dumps(last, ensure_ascii=False)
        if ctx.attached_image is not None:
            loop.attach_image(messages, ctx.attached_image.data_url, reply.tool_calls[-1].name)
            ctx.attached_image = None
        if advance:
            await self._next_part()
            self.part_ok, self.force_reply, self.part_start = True, False, len(result.tool_log)
        if max(self.attempts.values(), default=0) >= 3 or self.failures >= 4:
            # Going round in circles: stop and let the human (or a later wake) decide.
            result.status = "blocked"
            result.say = "" if self.background else "I keep trying the same thing without getting anywhere, so I stopped."
            return True
        return False

    async def _run_skill(self, name: str, fixed: Any) -> SkillResult:
        if fixed.refusal:
            return SkillResult(False, fixed.refusal, status="rejected")
        loop = self.loop
        await loop.emit("skill.started", {"tool": name, "args": fixed.args})
        outcome = await loop.skills.run(name, fixed.args, self.ctx)
        outcome.message += fixed.note
        await loop.emit("skill.completed", {"tool": name, "args": fixed.args, "ok": outcome.ok,
                                            "status": outcome.status, "message": outcome.message,
                                            "data": outcome.data})
        return outcome

    async def _next_part(self) -> None:
        self.words = self.todo.advance()
        # The robot has moved since the situation was written ("start: here" is stale).
        world = self.ctx.spatial_map.render_for_prompt(self.ctx.pose())
        self.messages.append({"role": "user", "content": self.todo.next_part_message(world)})
        await self.loop.emit("cognition.next_part", {"part": self.todo.current, "index": self.todo.index})

    def _finish(self) -> TurnResult:
        result, ctx = self.result, self.ctx
        if self.said and result.say:
            # "I'm at the ball! I'm back where I started." Questions to the human in between are dropped:
            # the robot carried on without waiting for an answer.
            kept = [s for s in (_statements(x) for x in self.said) if s]
            result.say = " ".join([*kept, result.say])
        result.next_wake_s = ctx.wake_request_s
        result.expression = dict(ctx.expression)
        result.transcript = self.transcript
        result.say = speakable(result.say, ctx.spatial_map.objects, self.loop.robot_name)
        return result
