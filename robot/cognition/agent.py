"""The model-driven agent loop: think -> call skills -> read results -> repeat until it answers."""

from __future__ import annotations

import ast
import json
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from robot.cognition.backend import BackendUnavailable, ChatResult, InferenceBackend, ToolCall
from robot.cognition.planner import TaskList, split_parts
from robot.cognition.prompts import estimate_tokens
from robot.core.config import AgentConfig
from robot.skills.base import SkillContext, SkillRegistry, SkillResult

logger = logging.getLogger(__name__)

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


_THINK_TAIL = re.compile(r"</t(?:h(?:i(?:n(?:k(?:>)?)?)?)?)?\s*$")


def strip_reasoning(content: str) -> tuple[str, bool]:
    """Return (reply, leaked): drop <think> blocks; a reply that is only reasoning counts as leaked."""
    text = content or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"<think>.*", "", text, flags=re.S)
    if _THINK_TAIL.search(text):
        return "", True  # reasoning cut off right where the answer would have started
    leaked = bool(re.match(r"\s*(?:The user|User (?:said|asked|wants|introduced)|I should|Let me think)", text))
    return ("" if leaked else text.strip()), leaked


_CLAIM = re.compile(
    r"\b(?:i've|i have|i'm|i am|i just|i'll|i will|let me|let's|now i|i turned|i moved|i went|i drove|i spun)\b"
    r"[^.!?]{0,40}?\b(?:turn|spin|spun|mov|go|went|head|driv|drove|circl|look|explor|danc|search|find|found|"
    r"approach|park|rotat|sweep)", re.I)
_ACTION_TOOLS = {"go_to", "turn_to", "turn", "move", "explore", "look_around", "observe", "express", "tilt_head",
                 "circle_around"}
# Actions that are the point of a part rather than a step towards it (looking, turning first). When one
# succeeds, that part of a several-part request is done and the next part is handed over. The model is
# not trusted to say so: it kept calling tools ("I'm at the ball. Let me come back now." + go_to the ball).
FINISHING_TOOLS = {"go_to", "circle_around", "move", "follow", "stop", "express", "set_eye_color", "remember",
                   "explore"}

NUDGE = ("(No tool was called, so nothing happened. The request was: \"{request}\". Every request is new, even "
         "if you did the same before. Call the tool that does it now.)")


def request_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str) and "[Situation]" in content:
            return content.rsplit("]\n", 1)[-1].strip()[:200]
    return ""


_REFUSAL = re.compile(
    r"\bi (?:can't|cannot|can not|am not able to|am unable to|don't know how to)\b[^.!?]{0,40}?\b(?:face|danc|turn|"
    r"spin|mov|go|look|smile|nod|wiggle|express|show|circl|drive|explor|find|search|eyes?)", re.I)


def claims_unperformed_action(say: str, tool_log: list[dict[str, Any]]) -> bool:
    """True when the reply claims an action no tool performed, or refuses one the robot has a tool for."""
    if any(e["tool"] in _ACTION_TOOLS for e in tool_log):
        return False
    return bool(_CLAIM.search(say or "") or _REFUSAL.search(say or ""))


def contradicts_success(say: str, tool_log: list[dict[str, Any]]) -> bool:
    """True when the reply says "I can't ..." right after an action tool succeeded."""
    return bool(_REFUSAL.search(say or "")) and any(e["ok"] and e["tool"] in _ACTION_TOOLS for e in tool_log)


FIX_REPLY = "(That tool succeeded: {result} Tell the human in one short sentence what you just did.)"
# Calls whose second run from the same spot can only repeat the first: going where I already am, facing what
# I already face. A small model often re-issues them after success instead of answering.
IDEMPOTENT_TOOLS = {"go_to", "turn_to", "look_around", "observe", "set_eye_color", "tilt_head", "remember",
                    "recall"}
REQUEST_GROUNDED_TOOLS = {"go_to"}
DUPLICATE = "Already done: {result} Nothing more to do for this; tell the human."


def _call_key(call: ToolCall, ctx: SkillContext) -> str:
    """What makes two calls the same. go_to reads the words of the current part (its relation, "come back"),
    so go_to(ball_1) for "come back" is a different call from go_to(ball_1) for "go to the ball"."""
    part = ctx.request if call.name in REQUEST_GROUNDED_TOOLS else ""
    return json.dumps([call.name, call.arguments, part], sort_keys=True)


def _pose(ctx: SkillContext) -> Any:
    return ctx.pose() if ctx.body is not None else None


def _same_spot(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False  # pose unknown: never assume nothing changed
    turn = abs((a.heading_deg - b.heading_deg + 180.0) % 360.0 - 180.0)
    return math.hypot(a.x_mm - b.x_mm, a.y_mm - b.y_mm) < 100.0 and turn < 10.0


# Appended to the last tool result of each step of a human request. With several steps ("go to the ball
# and then to the cube") a small model tends to stop after the first; with one ("go to the ball") it
# tends to keep acting unless told that this may be the end.
_SEVERAL_PARTS = re.compile(r"\b(?:then|after|afterwards|before|also|next)\b|\band (?:go|come|turn|look|move|"
                            r"find|drive|tell|show|say|do|make|bring|check|spin|back)\b", re.I)
DONE_HINT = "If that finishes \"{request}\", answer the human now."
PROGRESS = ("Request: \"{request}\". Answer the human now, unless the request asks for another step "
            "that is not done yet.")


_CALL_TOKENS = re.compile(r"<\|tool_call_(?:start|end)\|>|</?think>|```(?:python|json)?")
# <tool_call><function=go_to><parameter=target>ball_1</parameter></function></tool_call>
_XML_CALL = re.compile(r"<function=(\w+)>(.*?)</function>", re.S)
_XML_PARAM = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.S)
_XML_BLOCK = re.compile(r"<tool_call>.*?(?:</tool_call>|$)|<function=\w+>.*?(?:</function>|$)", re.S)


def _xml_value(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def strip_written_calls(text: str) -> str:
    """Remove tool calls written out as markup: they must never be spoken."""
    return _XML_BLOCK.sub("", text or "").strip()


def recover_text_tool_calls(content: str, tool_names: set[str]) -> list[ToolCall]:
    """Small models sometimes write a call as text ("observe()", or JSON) instead of a real call.

    Only a reply that is *entirely* calls to known tools is recovered; prose is left alone.
    """
    xml = _XML_CALL.findall(content or "")
    if xml and not strip_written_calls(re.sub(r"</?tool_call>", "", content or "")):
        return [ToolCall(f"recovered_{i}", name, {k: _xml_value(v) for k, v in _XML_PARAM.findall(body)})
                for i, (name, body) in enumerate(xml[:3]) if name in tool_names]
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
        except (ValueError, TypeError):  # TypeError: unhashable literal keys such as {[1]: 2}
            return []
        calls.append(ToolCall(f"recovered_{i}", node.func.id, args))
    return calls


def speakable(text: str, spatial_ids: dict[str, str], robot_name: str = "") -> str:
    """Remove emoji and internal ids ("box_1" -> "the box") from text meant for the human.

    Also drops the robot's own name used as a vocative ("Hi, Sprocket!"): a small model sometimes
    addresses the human by the robot's name. "I'm Sprocket" is left alone.
    """
    def repl(match: re.Match[str]) -> str:
        return f"the {match.group(1)}" if match.group(0) in spatial_ids else match.group(0)
    text = _EMOJI.sub("", text)
    text = re.sub(r"\bthe the\b", "the", _ID.sub(repl, text), flags=re.I)
    if robot_name:
        text = re.sub(rf",\s*{re.escape(robot_name)}\s*(?=[!?.,])", "", text, flags=re.I)
    # The prompt calls the person "the human"; the robot speaks *to* them.
    text = re.sub(r"\b[Tt]he human's\b", lambda m: "Your" if m.group(0)[0] == "T" else "your", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _statements(text: str) -> str:
    return " ".join(s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s and not s.endswith("?"))


def _shorten(payload: dict[str, Any]) -> str:
    """A tool result squeezed for an over-long prompt; what was seen survives (the answer may depend on it)."""
    text = str(payload.get("result", ""))[:60]
    objects = payload.get("objects")
    if isinstance(objects, list) and objects:
        text += " Seen: " + ", ".join(str(o).split(":")[0] for o in objects[:8])
    return text


class AgentLoop:
    def __init__(self, backend: InferenceBackend, skills: SkillRegistry, config: AgentConfig,
                 *, prompt_limit_tokens: int = 3300, on_event: Callable[[str, dict[str, Any]], Awaitable[None]]
                 | None = None, robot_name: str = "") -> None:
        self.backend = backend
        self.robot_name = robot_name
        self.skills = skills
        self.config = config
        self.prompt_limit_tokens = prompt_limit_tokens
        self.on_event = on_event
        self._tools = skills.tool_schemas()
        self._tool_names = set(skills.names)

    async def warm_up(self, system_prompt: str) -> None:
        """Have the model read the system prompt and tool list once, so the server caches them.

        Measured: the first decision after the server starts took 6.9 s (2.5k prompt tokens read cold on the
        laptop GPU), the next ones 0.5-1.6 s. Without this the human's first request pays that, and the
        robot set off late enough to lose a rolling ball. On the Pi the cold read is longer still.
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Hello."}]
        try:
            await self.backend.chat(messages, tools=self._tools, max_tokens=1)
        except Exception:  # an absent model is handled per request; warming is only an optimisation
            logger.info("Model warm-up skipped", exc_info=True)

    async def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.on_event is not None:
            await self.on_event(event_type, payload)

    async def run(self, messages: list[dict[str, Any]], ctx: SkillContext, *, background: bool) -> TurnResult:
        result = TurnResult(say="")
        started = time.monotonic()
        # This turn's calls and results, kept apart from `messages`: _fit may delete or merge entries
        # there, so slicing `messages` by a start index could cut a call from its result.
        transcript: list[dict[str, Any]] = []
        attempts: dict[str, int] = {}
        failures = 0
        nudged = False
        force_tool = force_reply = False
        succeeded: dict[str, tuple[Any, SkillResult]] = {}  # idempotent call -> (pose after it, its result)
        request = "" if background else request_text(messages)
        hint = PROGRESS if _SEVERAL_PARTS.search(request) else DONE_HINT
        ctx.request = request
        parts = split_parts(request) if request and self.config.plan else []
        todo = TaskList(request, parts) if parts else None
        if todo is not None:
            ctx.request = todo.current  # what go_to checks its relation against: this part's words
            await self._emit("cognition.plan", {"request": request, "steps": parts})
        said: list[str] = []  # answers to the parts before the last one
        part_ok = True  # no tool of the current part failed
        part_start = 0  # tool_log index where the current part began
        for step in range(self.config.max_steps):
            ctx.check()
            if time.monotonic() - started > self.config.timeout_s:
                result.status = "blocked"
                result.say = "That took too long, so I stopped."
                break
            self._fit(messages)
            try:
                # The tools stay in the request either way, so the cached prompt prefix is reused.
                choice = "required" if force_tool else "none" if force_reply else "auto"
                reply: ChatResult = await self.backend.chat(messages, tools=self._tools, tool_choice=choice)
                force_tool = force_reply = False
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
                "reasoning": reply.reasoning[:600],
                "tool_calls": [{"name": c.name, "args": c.arguments} for c in reply.tool_calls],
            })
            if not reply.tool_calls:
                reply.tool_calls = recover_text_tool_calls(reply.content, self._tool_names)
                if reply.tool_calls:
                    reply.content = ""
                    await self._emit("cognition.recovered_calls", {"calls": [c.name for c in reply.tool_calls]})
            if not reply.tool_calls:
                result.say, leaked = strip_reasoning(strip_written_calls(reply.content))
                if leaked and not background:
                    # One retry without tools: just say something to the human.
                    messages.append({"role": "user", "content": "(Reply to the human now in one short sentence.)"})
                    try:
                        retry = await self.backend.chat(messages, max_tokens=80)
                        result.say = strip_reasoning(retry.content)[0]
                    except BackendUnavailable:
                        pass
                    messages.pop()
                if todo is not None and todo.has_next and part_ok and result.say:
                    # The answer to one part ("I'm at the ball!") ends that part; hand over the next one.
                    said.append(result.say)
                    messages.append({"role": "assistant", "content": reply.content})
                    await self._next_part(todo, ctx, messages)
                    part_ok, part_start = True, len(result.tool_log)
                    result.say = ""
                    continue
                # Only this part's tools count: "now I'll head back" after part 1's go_to is still a claim.
                if not nudged and claims_unperformed_action(result.say, result.tool_log[part_start:]):
                    # The model narrated (or refused) an action instead of doing it, often by copying an
                    # earlier reply. Ask once more and make the next answer a tool call.
                    nudged = force_tool = True
                    messages.append({"role": "assistant", "content": reply.content})
                    messages.append({"role": "user", "content": NUDGE.format(request=request_text(messages))})
                    await self._emit("cognition.nudged", {"reply": result.say[:200]})
                    result.say = ""
                    continue
                if not nudged and contradicts_success(result.say, result.tool_log):
                    nudged = True  # e.g. shows a sad face, then says it can't make one
                    done = next(e for e in reversed(result.tool_log) if e["ok"] and e["tool"] in _ACTION_TOOLS)
                    messages.append({"role": "assistant", "content": reply.content})
                    messages.append({"role": "user", "content": FIX_REPLY.format(result=done["message"])})
                    await self._emit("cognition.nudged", {"reply": result.say[:200], "reason": "contradiction"})
                    result.say = ""
                    continue
                break
            messages.append(reply.assistant_message())
            transcript.append(messages[-1])
            stop_batch = False
            finished_part = False
            for call in reply.tool_calls:
                if stop_batch:
                    outcome = SkillResult(False, "Not executed because an earlier step failed.", status="skipped")
                elif call.invalid_arguments:
                    outcome = SkillResult(False, "The arguments were not valid JSON.", status="rejected")
                elif (key := _call_key(call, ctx)) in succeeded and \
                        _same_spot(succeeded[key][0], _pose(ctx)):
                    # Keep the first call's data (an observe's object list): the model answers from it.
                    first = succeeded[key][1]
                    outcome = SkillResult(True, DUPLICATE.format(result=first.message), dict(first.data),
                                          status="duplicate")
                    force_reply = todo is None or not todo.has_next
                    finished_part = finished_part or call.name in FINISHING_TOOLS
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
                    part_ok = part_ok and outcome.ok
                    finished_part = finished_part or (outcome.ok and call.name in FINISHING_TOOLS)
                    failures += 0 if outcome.ok else 1
                    attempts[key] = attempts.get(key, 0) + 1
                    if outcome.ok and call.name in IDEMPOTENT_TOOLS:
                        succeeded[key] = (_pose(ctx), outcome)
                payload = outcome.to_model()
                last_status = outcome.status
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(payload, ensure_ascii=False)})
                transcript.append(messages[-1])
            advance = todo is not None and todo.has_next and part_ok and finished_part
            if request and (last_status != "duplicate" or todo is not None):
                # A field of its own: merged into "result" it made the model act on it as an instruction
                # (measured: "two_stops" went to the wrong object).
                last = json.loads(messages[-1]["content"])
                last["next"] = (f"Part {todo.index + 1} (\"{todo.current}\") is done." if advance and todo
                                else todo.hint() if todo is not None else hint.format(request=request))
                messages[-1]["content"] = json.dumps(last, ensure_ascii=False)
            if ctx.attached_image is not None:
                self._attach_image(messages, ctx.attached_image.data_url, reply.tool_calls[-1].name)
                ctx.attached_image = None
            if advance and todo is not None:
                await self._next_part(todo, ctx, messages)
                part_ok, force_reply, part_start = True, False, len(result.tool_log)
            if max(attempts.values(), default=0) >= 3 or failures >= 4:
                # Going round in circles: stop and let the human (or a later wake) decide.
                result.status = "blocked"
                result.say = "" if background else ("I keep trying the same thing without getting anywhere, "
                                                     "so I stopped.")
                break
        else:
            result.status = "blocked"
            result.say = "" if background else "I ran out of steps before finishing."
        if said and result.say:
            # "I'm at the ball! I'm back where I started." Questions to the human in between are dropped:
            # the robot carried on without waiting for an answer.
            kept = [s for s in (_statements(x) for x in said) if s]
            result.say = " ".join([*kept, result.say])
        result.next_wake_s = ctx.wake_request_s
        result.expression = dict(ctx.expression)
        result.transcript = transcript
        result.say = speakable(result.say, ctx.spatial_map.objects if ctx.spatial_map else {}, self.robot_name)
        return result

    async def _next_part(self, todo: TaskList, ctx: SkillContext, messages: list[dict[str, Any]]) -> None:
        ctx.request = todo.advance()
        # The robot has moved since the situation was written ("start: here" is stale).
        pose = _pose(ctx)
        world = ctx.spatial_map.render_for_prompt(pose) if ctx.spatial_map is not None and pose else ""
        messages.append({"role": "user", "content": todo.next_part_message(world)})
        await self._emit("cognition.next_part", {"part": todo.current, "index": todo.index})

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
            summary = "; ".join(_shorten(json.loads(messages[k]["content"])) for k in range(start + 1, end))
            messages[start:end] = [{"role": "user", "content": f"(earlier steps, shortened: {summary})"}]
