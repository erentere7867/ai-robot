"""Native function calls must reach motors, return matched results, and remain bounded."""

from __future__ import annotations

import json

import httpx
import pytest

from robot.actions.tools import ToolCallRequest
from robot.cognition.llamacpp import LlamaCppBackend
from robot.core.clock import ReplayClock
from robot.core.config import RobotConfig
from robot.main import RobotHarness


def reply(content="", calls=None, finish_reason=None):
    return httpx.Response(200, json={"choices": [{"message": {
        "role": "assistant", "content": content, **({"tool_calls": calls} if calls else {}),
    }, "finish_reason": finish_reason or ("tool_calls" if calls else "stop")}],
        "usage": {"prompt_tokens": 400, "completion_tokens": 30}})


def call(name, arguments, call_id="native-call"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def text_content(message):
    content = message["content"]
    return content if isinstance(content, str) else next(part["text"] for part in content if part["type"] == "text")


def images(payload):
    return [part["image_url"]["url"] for m in payload["messages"] if isinstance(m.get("content"), list)
            for part in m["content"] if part.get("type") == "image_url"]


async def make_harness(tmp_path, handler):
    config = RobotConfig.from_yaml()
    config.database_path = tmp_path / "native.db"
    config.record_events = False
    config.llm.auto_start_server = False
    backend = LlamaCppBackend(config.llm)
    await backend._client.aclose()
    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1")
    harness = RobotHarness(config, backend=backend, clock=ReplayClock(), tts_mode="warble", realtime_audio_sync=False)
    harness.face_bridge._send_udp = lambda _: None
    return harness, backend


async def test_native_call_executes_motor_and_returns_matching_tool_result(tmp_path):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert "response_format" not in payload
        assert "spin" in [tool["function"]["name"] for tool in payload["tools"]]
        assert "perform_trick" not in [tool["function"]["name"] for tool in payload["tools"]]
        assert "Return JSON" not in payload["messages"][0]["content"]
        if len(requests) == 1:
            return reply(calls=[call("spin", {"angle_deg": 180})])
        assert harness.body.pose().heading_deg == pytest.approx(180)
        tool = next(message for message in payload["messages"] if message["role"] == "tool")
        assert tool["tool_call_id"] == "native-call"
        text = text_content(tool)
        assert json.loads(text)["ok"]
        assert "Spun 180" in text
        assert len([m for m in payload["messages"] if m["role"] == "user"]) == 1
        assert payload["messages"][-1]["role"] == "tool"
        assert not images(payload)
        assert "Charging Dock" not in str(payload["messages"])
        initial = text_content(requests[0]["messages"][-1])
        original = next(m["content"] for m in payload["messages"] if m["role"] == "user")
        assert original == initial
        return reply("I have turned around.")

    harness, backend = await make_harness(tmp_path, handler)
    try:
        async with harness:
            await harness.submit_message("turn around")
            await harness.arbiter.wait_until_done()
            assert len(requests) == 2
            assert harness.executor.completed_count == 1
            assert harness.arbiter.active_goal["status"] == "completed"
            assert len(harness.arbiter._tool_exchanges) == 1
    finally:
        await backend.close()


async def test_invalid_native_motor_arguments_are_rejected_and_reported(tmp_path):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return reply(calls=[call("spin", {"angle_deg": "NaN"})])
        tool = next(m for m in payload["messages"] if m["role"] == "tool")
        text = text_content(tool)
        assert json.loads(text)["status"] == "rejected"
        return reply("That angle was invalid.")

    harness, backend = await make_harness(tmp_path, handler)
    try:
        async with harness:
            await harness.submit_message("turn around")
            await harness.arbiter.wait_until_done()
            assert harness.body.pose().heading_deg == 0
            assert harness.executor.completed_count == 0
            assert len(requests) == 2
    finally:
        await backend.close()


async def test_camera_is_only_supplied_after_the_model_requests_observation(tmp_path):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        step = len(requests)
        if step == 1:
            assert not images(payload)
            assert "Red Ball" not in str(payload["messages"])
            return reply(calls=[call("observe", {}, "look-front")])
        if step == 2:
            assert len(images(payload)) == 1
            assert payload["messages"][-1]["tool_call_id"] == "look-front"
            return reply(calls=[call("spin", {"angle_deg": 180}, "turn")])
        if step == 3:
            assert not images(payload)
            return reply(calls=[call("observe", {}, "look-back")])
        assert len(images(payload)) == 1
        assert images(payload)[0] != images(requests[1])[0]
        return reply("I've checked both views.")

    harness, backend = await make_harness(tmp_path, handler)
    try:
        async with harness:
            await harness.submit_message("Compare the views in front and behind")
            await harness.arbiter.wait_until_done()
            assert len(requests) == 4
            assert harness.executor.completed_count == 1
            assert harness.body.pose().heading_deg == pytest.approx(180)
    finally:
        await backend.close()


async def test_native_schedule_survives_the_final_reply(tmp_path):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return reply(calls=[call("schedule_wake", {"delay_s": 60})]) if count == 1 else reply("I'll check later.")

    harness, backend = await make_harness(tmp_path, handler)
    try:
        async with harness:
            await harness.submit_message("check again later")
            await harness.arbiter.wait_until_done()
            assert harness.executor.completed_count == 0
            assert harness.autonomy.snapshot()["next_decision_in_s"] == pytest.approx(60)
    finally:
        await backend.close()


async def test_truncated_tool_call_cannot_execute(tmp_path):
    harness, backend = await make_harness(tmp_path, lambda request: reply(
        calls=[call("drive", {"distance_mm": 1000})], finish_reason="length"))
    try:
        async with harness:
            response = await harness.submit_message("move forward")
            assert response["status"] == "blocked"
            assert harness.executor.completed_count == 0
            assert harness.body.pose().x_mm == 0
    finally:
        await backend.close()


async def test_full_rotation_reports_objects_seen_during_motion(tmp_path):
    harness, backend = await make_harness(tmp_path, lambda request: reply("Done."))
    try:
        async with harness:
            assert "Charging Dock" not in [d.label for d in harness.sandbox.detect_objects(harness.body.pose())]
            result = await harness.executor.execute_call(ToolCallRequest("spin", {"angle_deg": 360}))
            assert result.ok
            assert harness.body.pose().heading_deg == pytest.approx(0)
            assert "Charging Dock" in result.details["observed_objects"]
            assert result.details["rotation_deg"] == pytest.approx(360)
    finally:
        await backend.close()
