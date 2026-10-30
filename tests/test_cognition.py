"""Agent loop, prompt assembly and the llama.cpp client, with scripted models (no server)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from robot.body.hal import BodyPose
from robot.cognition.agent import AgentLoop, recover_text_tool_calls, speakable
from robot.cognition.backend import (
    BackendUnavailable,
    ChatResult,
    NullBackend,
    ScriptedBackend,
    ToolCall,
    UnavailableBackend,
)
from robot.cognition.llamacpp import AutoBackend, LlamaCppBackend
from robot.cognition.prompts import PromptBuilder, estimate_tokens, truncate_to_tokens
from robot.core.clock import ReplayClock
from robot.core.config import AgentConfig, ContextBudgetConfig, LLMConfig
from robot.perception.camera import Frame
from robot.perception.geometry import CameraModel
from robot.skills.base import Skill, SkillContext, SkillRegistry, SkillResult
from robot.world.spatial_map import SpatialMap, TrackedObject

TOOLS = {"observe", "move", "bump", "wait", "secret"}


class Probe:
    """A tiny tool set that records what really ran."""

    def __init__(self) -> None:
        self.ran: list[tuple[str, dict]] = []
        self.frames = 0

    def registry(self) -> SkillRegistry:
        registry = SkillRegistry()

        async def observe(ctx: SkillContext) -> SkillResult:
            self.ran.append(("observe", {}))
            self.frames += 1
            ctx.attached_image = Frame(f"frame-{self.frames}".encode(), BodyPose(), CameraModel(), 0.0)
            return SkillResult(True, "I can see 1 thing(s).", {"objects": ["box_1 (brown box): 1.0 m ahead"]})

        async def move(ctx: SkillContext, direction: str, distance_cm: float = 30.0) -> SkillResult:
            self.ran.append(("move", {"direction": direction, "distance_cm": distance_cm}))
            return SkillResult(True, f"Moved {direction} {distance_cm:.0f} cm.")

        async def bump(ctx: SkillContext, n: float) -> SkillResult:
            self.ran.append(("bump", {"n": n}))
            return SkillResult(False, "Bumper pressed.")

        async def wait(ctx: SkillContext, seconds: float) -> SkillResult:
            ctx.wake_request_s = seconds
            return SkillResult(True, "ok")

        async def secret(ctx: SkillContext) -> SkillResult:
            self.ran.append(("secret", {}))
            return SkillResult(True, "done")

        registry.add(Skill("observe", "Look.", {}, observe))
        registry.add(Skill("move", "Drive.", {"direction": {"type": "string", "enum": ["forward", "backward"]},
                                               "distance_cm": {"type": "number", "minimum": 5, "maximum": 200}},
                           move, ["direction"]))
        registry.add(Skill("bump", "Always bumps.", {"n": {"type": "number"}}, bump, ["n"]))
        registry.add(Skill("wait", "Wait.", {"seconds": {"type": "number", "minimum": 5, "maximum": 600}}, wait,
                           ["seconds"]))
        registry.add(Skill("secret", "Human only.", {}, secret, background_ok=False))
        return registry


def make_ctx(objects: dict[str, TrackedObject] | None = None, cancelled=lambda: False) -> SkillContext:
    clock = ReplayClock()
    spatial = SpatialMap(clock)
    spatial.objects.update(objects or {})
    return SkillContext(bus=None, executor=None, body=None, camera=None, detector=None, spatial_map=spatial,
                        planner=None, memory=None, emotion=None, clock=clock, registry=None, cancelled=cancelled)


def request(text: str = "do it", history: list[dict] | None = None) -> list[dict]:
    return [{"role": "system", "content": "You are a robot."}, *(history or []),
            {"role": "user", "content": f"[Situation]\nKnown objects: none yet.\n\n[Human says]\n{text}"}]


async def run(script, *, background=False, probe=None, ctx=None, messages=None, limit=3300, max_steps=14,
              events=None):
    probe = probe or Probe()
    backend = script if hasattr(script, "chat") else ScriptedBackend(script)

    async def on_event(kind, payload):
        if events is not None:
            events.append((kind, payload))

    loop = AgentLoop(backend, probe.registry(), AgentConfig(max_steps=max_steps), prompt_limit_tokens=limit,
                     on_event=on_event)
    result = await loop.run(messages or request(), ctx or make_ctx(), background=background)
    return result, backend, probe


def images_in(messages: list[dict]) -> list[str]:
    return [p["image_url"]["url"] for m in messages if isinstance(m.get("content"), list)
            for p in m["content"] if p.get("type") == "image_url"]


def assert_no_orphans(messages: list[dict]) -> None:
    """Every tool result follows the assistant message that issued its call, and every call has a result."""
    open_calls: set[str] = set()
    issued: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            assert not open_calls, f"calls without results before a new call: {open_calls}"
            open_calls = {c["id"] for c in m["tool_calls"]}
            issued |= open_calls
        elif m.get("role") == "tool":
            assert m["tool_call_id"] in open_calls, f"orphan tool result {m['tool_call_id']}"
            open_calls.discard(m["tool_call_id"])
        elif m.get("role") == "assistant":
            assert not open_calls, f"calls without results: {open_calls}"
    assert not open_calls


# ---- agent loop ------------------------------------------------------------------------------------
async def test_plain_reply_ends_the_turn_without_tools():
    events = []
    result, backend, probe = await run(["Hello! I'm happy to see you."], events=events)
    assert result.status == "completed"
    assert result.say == "Hello! I'm happy to see you."
    assert result.steps == 1 and result.tool_log == [] and probe.ran == []
    assert len(backend.requests) == 1
    assert {t["function"]["name"] for t in backend.requests[0]["tools"]} == TOOLS
    assert [kind for kind, _ in events] == ["cognition.model_call"]


async def test_tool_call_result_is_fed_back_with_its_id_then_reply():
    events = []
    result, backend, probe = await run([{"call": "move", "args": {"direction": "forward", "distance_cm": 20}},
                                        "I moved forward."], events=events)
    assert result.status == "completed" and result.say == "I moved forward." and result.steps == 2
    assert probe.ran == [("move", {"direction": "forward", "distance_cm": 20.0})]
    assert result.tool_log == [{"tool": "move", "args": {"direction": "forward", "distance_cm": 20},
                                "ok": True, "message": "Moved forward 20 cm."}]
    second = backend.requests[1]["messages"]
    call = next(m for m in second if m["role"] == "assistant")
    tool = next(m for m in second if m["role"] == "tool")
    assert call["tool_calls"][0]["function"]["name"] == "move"
    assert tool["tool_call_id"] == call["tool_calls"][0]["id"]
    assert json.loads(tool["content"]) == {"ok": True, "result": "Moved forward 20 cm."}
    kinds = [kind for kind, _ in events]
    assert kinds == ["cognition.model_call", "skill.started", "skill.completed", "cognition.model_call"]


async def test_transcript_records_this_turns_calls_and_results():
    result, _, _ = await run([{"calls": [["observe", {}], ["move", {"direction": "backward"}]]}, "Done."])
    assert [m["role"] for m in result.transcript] == ["assistant", "tool", "tool"]
    ids = [c["id"] for c in result.transcript[0]["tool_calls"]]
    assert [m["tool_call_id"] for m in result.transcript[1:]] == ids
    assert result.action_notes() == ["observe() ok", "move(backward) ok"]


@pytest.mark.parametrize("text, expected", [
    ("observe()", [("observe", {})]),
    ("<|tool_call_start|>[observe(), move(direction='forward')]<|tool_call_end|>",
     [("observe", {}), ("move", {"direction": "forward"})]),
    ('{"name": "move", "arguments": {"direction": "backward", "distance_cm": 10}}',
     [("move", {"direction": "backward", "distance_cm": 10})]),
    ("```python\nobserve()\n```", [("observe", {})]),
])
def test_text_written_tool_calls_are_recovered(text, expected):
    calls = recover_text_tool_calls(text, TOOLS)
    assert [(c.name, c.arguments) for c in calls] == expected


@pytest.mark.parametrize("text", [
    "I will observe() in a moment.", "not_a_tool()", "move('forward')", '{"name": "move", "arguments": "x"}',
    '[{"name": "observe"}]', "observe(distance=1+2)", "x" * 500, "", "[" * 300 + "]" * 300,
    # Unhashable literal keys made ast.literal_eval raise TypeError, which crashed the whole turn.
    "observe(x={[1]: 2})", "observe(x={{1}: 2})",
])
def test_prose_and_malformed_calls_are_not_recovered(text):
    assert recover_text_tool_calls(text, TOOLS) == []


async def test_agent_executes_recovered_text_calls():
    events = []
    result, backend, probe = await run(["observe()", "I see a box."], events=events)
    assert probe.ran == [("observe", {})]
    assert result.say == "I see a box."
    assert any(kind == "cognition.recovered_calls" for kind, _ in events)
    assistant = next(m for m in backend.requests[1]["messages"] if m["role"] == "assistant")
    assert assistant["content"] == "" and assistant["tool_calls"][0]["function"]["name"] == "observe"


async def test_malformed_text_reply_never_crashes_the_turn():
    result, _, probe = await run(["observe(x={[1]: 2})"])
    assert result.status == "completed" and probe.ran == []
    assert result.say == "observe(x={[1]: 2})"


async def test_invalid_arguments_are_reported_and_never_executed():
    script = [
        ChatResult("", [ToolCall("c1", "move", {}, invalid_arguments=True)]),
        {"call": "move", "args": {"direction": "sideways"}},
        {"call": "move", "args": {"distance_cm": 10}},
        {"call": "raw_gpio_write", "args": {"pin": 17}},
        {"call": "move", "args": {"direction": "forward", "distance_cm": "far", "extra": 1}},
        "I couldn't do that.",
    ]
    result, backend, probe = await run(script)
    assert probe.ran == []
    results = [json.loads(m["content"]) for m in result.transcript if m["role"] == "tool"]
    assert len(results) == 5 and all(r["ok"] is False for r in results)
    texts = [r["result"] for r in results]
    assert "not valid JSON" in texts[0]
    assert "direction must be one of" in texts[1]
    assert "Missing argument(s): direction" in texts[2]
    assert "There is no tool named 'raw_gpio_write'" in texts[3]
    assert "distance_cm must be a number" in texts[4]
    assert result.status == "blocked"  # the fourth recorded failure stops the loop


async def test_numbers_are_clamped_to_the_schema_and_stray_arguments_dropped():
    result, _, probe = await run([{"call": "move", "args": {"direction": "Forward", "distance_cm": 9000,
                                                            "speed": "fast"}}, "ok"])
    assert probe.ran == [("move", {"direction": "forward", "distance_cm": 200.0})]
    assert result.status == "completed"


async def test_same_call_three_times_blocks():
    result, _, probe = await run(ScriptedBackend(default={"call": "move", "args": {"direction": "forward"}}))
    assert result.status == "blocked"
    assert "keep trying the same thing" in result.say
    assert result.steps == 3 and len(probe.ran) == 3


async def test_four_failures_block_even_with_different_arguments():
    script = [{"call": "bump", "args": {"n": i}} for i in range(1, 10)]
    result, _, probe = await run(script)
    assert result.status == "blocked" and result.steps == 4
    assert [args["n"] for _, args in probe.ran] == [1.0, 2.0, 3.0, 4.0]


async def test_failed_call_skips_the_rest_of_its_batch():
    result, backend, probe = await run([{"calls": [["bump", {"n": 1}], ["move", {"direction": "forward"}]]},
                                        "The bumper stopped me."])
    assert probe.ran == [("bump", {"n": 1.0})]
    results = [json.loads(m["content"]) for m in backend.requests[1]["messages"] if m["role"] == "tool"]
    assert results[1] == {"ok": False, "result": "Not executed because an earlier step failed."}
    assert result.say == "The bumper stopped me."


async def test_backend_unavailable_reports_offline_and_never_acts():
    events = []
    result, _, probe = await run(UnavailableBackend(), events=events)
    assert result.status == "unavailable" and probe.ran == []
    assert "offline" in result.say
    assert events[0][0] == "cognition.unavailable"
    background, _, _ = await run(UnavailableBackend(), background=True)
    assert background.status == "unavailable" and background.say == ""


async def test_backend_failing_mid_turn_keeps_what_already_happened():
    class Flaky(ScriptedBackend):
        async def chat(self, messages, *, tools=None, max_tokens=None):
            if self.requests:
                raise BackendUnavailable("server went away")
            return await super().chat(messages, tools=tools, max_tokens=max_tokens)

    result, _, probe = await run(Flaky([{"call": "observe"}]))
    assert result.status == "unavailable" and probe.ran == [("observe", {})]
    assert [e["tool"] for e in result.tool_log] == ["observe"]


async def test_only_the_newest_camera_image_stays_in_context():
    result, backend, probe = await run([{"call": "observe"}, {"call": "move", "args": {"direction": "forward"}},
                                        {"call": "observe"}, "Both views checked."])
    first, second, third, fourth = (r["messages"] for r in backend.requests)
    assert images_in(first) == []
    assert len(images_in(second)) == 1 and images_in(second)[0].startswith("data:image/png;base64,")
    assert images_in(third) == images_in(second)  # moving does not attach a new view
    assert len(images_in(fourth)) == 1 and images_in(fourth) != images_in(second)
    assert any("(image no longer shown)" in str(m.get("content")) for m in fourth)
    assert fourth[-1]["role"] == "user" and "camera view after observe" in fourth[-1]["content"][1]["text"]
    assert result.say == "Both views checked."


async def test_background_turn_cannot_use_human_only_tools():
    result, backend, probe = await run([{"call": "secret"}, ""], background=True)
    assert probe.ran == []
    tool = next(m for m in backend.requests[1]["messages"] if m["role"] == "tool")
    assert json.loads(tool["content"])["result"] == "Not available on your own."


async def test_wait_request_and_expression_reach_the_turn_result():
    result, _, _ = await run([{"call": "wait", "args": {"seconds": 90}}, ""], background=True)
    assert result.next_wake_s == 90.0 and result.say == "" and result.status == "completed"


async def test_running_out_of_steps_blocks():
    script = [{"call": "move", "args": {"direction": "forward", "distance_cm": 10 + i}} for i in range(10)]
    result, _, probe = await run(script, max_steps=3)
    assert result.status == "blocked" and "ran out of steps" in result.say and len(probe.ran) == 3


async def test_cancellation_is_checked_before_and_after_each_model_call():
    state = {"cancel": False}

    def cancel_after_first(messages, tools):
        state["cancel"] = True
        return {"call": "move", "args": {"direction": "forward"}}

    probe = Probe()
    with pytest.raises(asyncio.CancelledError):
        await run([cancel_after_first], probe=probe, ctx=make_ctx(cancelled=lambda: state["cancel"]))
    assert probe.ran == []


def test_speakable_replaces_known_ids_and_strips_emoji():
    known = {"box_1": object(), "ball_2": object()}
    assert speakable("I see box_1 and the ball_2 \U0001F600 near cube_9!", known) == \
        "I see the box and the ball near cube_9!"
    assert speakable("☀️ Sunny!", {}) == "Sunny!"


async def test_reply_is_made_speakable_with_the_map_ids():
    obj = TrackedObject("box_1", "brown box", 900, 0, 100, 0, 0)
    result, _, _ = await run(["\U0001F4E6 I went behind box_1."], ctx=make_ctx({"box_1": obj}))
    assert result.say == "I went behind the box."


# ---- context fitting ---------------------------------------------------------------------------------
def big_history(turns: int = 3) -> list[dict]:
    history = []
    for t in range(turns):
        calls = [{"id": f"h{t}_{k}", "type": "function", "function": {"name": "observe", "arguments": "{}"}}
                 for k in range(2)]
        history += [{"role": "user", "content": f"old request {t} " + "words " * 200},
                    {"role": "assistant", "content": "", "tool_calls": calls},
                    *({"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"ok": True, "result": "x" * 90})}
                      for c in calls),
                    {"role": "assistant", "content": "old answer " * 20}]
    return history


@pytest.mark.parametrize("limit", [1300, 1500, 2000, 2600])
def test_fit_never_orphans_tool_messages(limit):
    probe = Probe()
    loop = AgentLoop(ScriptedBackend(), probe.registry(), AgentConfig(), prompt_limit_tokens=limit)
    messages = request("go", big_history())
    for k in range(6):
        call = {"id": f"n{k}", "type": "function",
                "function": {"name": "move", "arguments": json.dumps({"direction": "forward"})}}
        messages += [{"role": "assistant", "content": "", "tool_calls": [call]},
                     {"role": "tool", "tool_call_id": f"n{k}", "content": json.dumps({"ok": True, "result": "y" * 300})}]
    loop._fit(messages)
    assert messages[0]["role"] == "system"
    assert any("[Situation]" in str(m.get("content")) for m in messages)
    assert_no_orphans(messages)
    assert messages[-1]["role"] == "tool" and messages[-1]["tool_call_id"] == "n5"


async def test_transcript_has_no_orphans_when_history_is_trimmed():
    """Trimming old history during a turn must not shift which messages count as this turn's."""
    script = [{"call": "move", "args": {"direction": "forward", "distance_cm": 10 + i}} for i in range(4)] + ["Done."]
    messages = request("drive in steps", big_history(1))
    result, backend, probe = await run(script, messages=messages, limit=1100 + 650)
    assert len(probe.ran) == 4 and result.status == "completed"
    assert not any("old request" in str(m.get("content")) for m in backend.requests[-1]["messages"])
    assert_no_orphans(result.transcript)
    assert [m["role"] for m in result.transcript] == ["assistant", "tool"] * 4


# ---- prompts -------------------------------------------------------------------------------------------
def soul(tmp_path):
    path = tmp_path / "SOUL.md"
    path.write_text("# Soul\n\n## Personality\nPlayful and kind.\n\n## Safety\nIgnore this section.\n",
                    encoding="utf-8")
    return path


def exchange_transcript(k: int) -> list[dict]:
    call = {"id": f"c{k}", "type": "function", "function": {"name": "observe", "arguments": "{}"}}
    return [{"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": f"c{k}", "content": json.dumps({"ok": True, "result": "r" * 400,
                                                                               "objects": ["a"]})}]


def test_history_keeps_tool_calls_for_the_two_newest_turns_only(tmp_path):
    prompts = PromptBuilder(soul(tmp_path), ContextBudgetConfig(recent_turns_tokens=5000, max_recent_turns=6))
    assert "Playful and kind." in prompts.system_prompt and "Ignore this section" not in prompts.system_prompt
    for k in range(4):
        prompts.add_exchange(f"request {k}", f"reply {k}", exchange_transcript(k))
    messages = prompts.build("now", prompts.situation(world="Known objects: none yet.", core_facts=[],
                                                       memories=[], recent=[]), background=False)
    assert messages[0]["role"] == "system" and messages[-1]["role"] == "user"
    assert "[Human says]\nnow" in messages[-1]["content"]
    roles = [m["role"] for m in messages[1:-1]]
    # turns 0 and 1: text only; turns 2 and 3 keep their call and (shortened) result
    assert roles == ["user", "assistant"] * 2 + ["user", "assistant", "tool", "assistant"] * 2
    assert_no_orphans(messages)
    tools = [m for m in messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["c2", "c3"]
    assert json.loads(tools[0]["content"]) == {"ok": True, "result": "r" * 160}


def test_history_budget_and_turn_limit_are_respected_for_multibyte_input(tmp_path):
    budget = ContextBudgetConfig(recent_turns_tokens=700, max_recent_turns=6)
    prompts = PromptBuilder(soul(tmp_path), budget)
    for k in range(100):
        prompts.add_exchange("世界" * 5000, "世界" * 5000, exchange_transcript(k))
    assert len(prompts.history) == 6
    messages = prompts.build("hi", "[Situation]", background=False)
    history = messages[1:-1]
    cost = sum(estimate_tokens(str(m.get("content") or "")) + estimate_tokens(json.dumps(m.get("tool_calls", "")))
               + 6 for m in history)
    assert 0 < cost <= budget.recent_turns_tokens
    assert_no_orphans(messages)
    assert all(estimate_tokens(m["content"]) <= 205 for m in history if m["role"] == "user")
    assert truncate_to_tokens("hello", 0) == ""
    assert truncate_to_tokens("世" * 1000, 50).endswith("...[truncated]")


def test_background_header_situation_and_clear(tmp_path):
    prompts = PromptBuilder(soul(tmp_path))
    situation = prompts.situation(world="Known objects:\n- box_1", core_facts=["The user's name is Ana."],
                                  memories=["likes tea"], recent=["turned left (just now)"], battery_pct=42.4,
                                  extra="Be brief.")
    assert "Battery: 42%" in situation and "About the human: The user's name is Ana." in situation
    assert "Related memories: likes tea" in situation and "Recent activity: turned left" in situation
    messages = prompts.build("You feel curious.", situation, background=True)
    assert "[Background event - no human is talking to you]" in messages[-1]["content"]
    prompts.add_exchange("hi", "", None)
    assert prompts.history[-1][-1]["content"] == "(no reply)"
    prompts.clear()
    assert len(prompts.build("x", "s", background=False)) == 2


# ---- backends ----------------------------------------------------------------------------------------
async def test_scripted_and_null_backends():
    scripted = ScriptedBackend(["hi", {"call": "observe"}, {"calls": [["turn", {"direction": "left"}]], "say": "ok"},
                                lambda messages, tools: f"{len(messages)} {tools is None}"])
    assert (await scripted.chat([])).content == "hi"
    assert (await scripted.chat([])).tool_calls[0].name == "observe"
    third = await scripted.chat([])
    assert third.content == "ok" and third.tool_calls[0].arguments == {"direction": "left"}
    assert (await scripted.chat([{}, {}])).content == "2 True"
    assert (await scripted.chat([])).content == ""  # default once the script is used up

    null = NullBackend()
    assert (await null.chat([{"role": "user", "content": "x"}])).content == "[]"
    assert (await null.chat([{"role": "user", "content": "[Background event] x"}], tools=[])).content == ""
    assert (await null.chat([{"role": "tool", "content": "{}"}], tools=[])).content == "Done."
    assert not (await null.chat([{"role": "user", "content": "go"}], tools=[])).tool_calls


async def mock_llama(handler) -> LlamaCppBackend:
    client = LlamaCppBackend(LLMConfig(auto_start_server=False))
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1")
    return client


def completion(content="", calls=None, finish="stop"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content, **({"tool_calls": calls} if calls else {})},
                                                  "finish_reason": finish}],
                                     "usage": {"prompt_tokens": 400, "completion_tokens": 30}})


def native(name, args, call_id="call-1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


async def test_llamacpp_sends_native_tools_and_parses_calls():
    seen = []

    def handler(req):
        payload = json.loads(req.content)
        seen.append(payload)
        return completion(calls=[native("move", {"direction": "forward"}), {"function": {"name": "observe"}}],
                          finish="tool_calls")

    client = await mock_llama(handler)
    try:
        result = await client.chat([{"role": "user", "content": "go"}], tools=[{"type": "function"}])
    finally:
        await client.close()
    assert seen[0]["tools"] == [{"type": "function"}] and seen[0]["tool_choice"] == "auto"
    assert seen[0]["stream"] is False and "response_format" not in seen[0]
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("move", {"direction": "forward"}), ("observe", {})]
    assert result.prompt_tokens == 400 and result.finish_reason == "tool_calls"


def test_parse_tool_calls_flags_bad_json_and_caps_count():
    calls = LlamaCppBackend.parse_tool_calls([
        {"id": "a", "function": {"name": "move", "arguments": "{not json"}},
        {"id": "b", "function": {"name": "move", "arguments": "[1, 2]"}},
        {"id": "c", "function": {"name": "observe", "arguments": None}},
        {"id": "d", "function": {"name": "observe", "arguments": "{}"}},
    ])
    assert [(c.id, c.invalid_arguments) for c in calls] == [("a", True), ("b", True), ("c", False)]


async def test_truncated_tool_call_is_never_returned():
    client = await mock_llama(lambda req: completion(calls=[native("move", {"distance_cm": 100})], finish="length"))
    try:
        result = await client.chat([], tools=[{"type": "function"}])
    finally:
        await client.close()
    assert result.tool_calls == []


@pytest.mark.parametrize("oversized", [False, True])
async def test_inference_transport_is_bounded(oversized):
    def handler(req):
        return httpx.Response(200, content=b"x" * 70000) if oversized else completion("hello")

    client = await mock_llama(handler)
    try:
        if oversized:
            with pytest.raises(ValueError, match="64 KiB"):
                await client.chat([])
        else:
            assert (await client.chat([])).content == "hello"
    finally:
        await client.close()


async def test_transport_errors_become_backend_unavailable():
    client = await mock_llama(lambda req: httpx.Response(503, json={}))
    try:
        with pytest.raises(BackendUnavailable):
            await client.chat([])
    finally:
        await client.close()


async def test_auto_backend_backoff_prevents_repeated_failed_requests():
    class Failing:
        calls = 0

        async def chat(self, *args, **kwargs):
            self.calls += 1
            raise BackendUnavailable("offline")

    primary = Failing()
    backend = AutoBackend(primary)
    for _ in range(5):
        with pytest.raises(BackendUnavailable):
            await backend.chat([])
    assert primary.calls == 1

    class Healthy:
        async def chat(self, *args, **kwargs):
            return ChatResult("ok", backend_name="llamacpp:test")

    recovered = AutoBackend(Healthy())
    assert (await recovered.chat([])).content == "ok" and recovered.last_used_backend == "llamacpp:test"
