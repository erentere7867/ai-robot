from __future__ import annotations

import httpx
import pytest

from robot.mind.brain import Brain


@pytest.mark.parametrize("arguments", ["{", "null", 3, [], [["degrees", 90]], '[["degrees", 90]]'])
def test_chat_keeps_malformed_arguments_as_a_repairable_tool_call(arguments: object) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [
            {"id": "bad", "function": {"name": "turn", "arguments": arguments}},
            {"id": "good", "function": {"name": "turn", "arguments": '{"degrees": 90, "direction": "right"}'}},
        ]}}]})

    brain = Brain()
    brain.http.close()
    with httpx.Client(base_url="http://model", transport=httpx.MockTransport(respond)) as brain.http:
        bad, good = brain.chat([{"role": "user", "content": "Turn right."}], []).tool_calls
    assert (bad.id, bad.name, bad.args) == ("bad", "turn", {"_unparsed": arguments})
    assert (good.id, good.name, good.args) == ("good", "turn", {"degrees": 90, "direction": "right"})


@pytest.mark.parametrize(("token", "expected_yes"), [
    ("yes", 1.0), ("no", 0.0), (" YES ", 1.0), (" No ", 0.0),
    ("yes.", 1.0), ("Yes.", 1.0), ("no!", 0.0), ("YES", 1.0), (" “\u00a0No!\u00a0” ", 0.0),
    ("Not", 0.5), ("Nothing", 0.5), ("Yesterday", 0.5), ("Ye", 0.5), ("y.es", 0.5),
])
def test_decide_counts_only_exact_answers(token: str, expected_yes: float) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"logprobs": {"content": [
            {"top_logprobs": [{"token": token, "logprob": -0.5}]},
        ]}}]})

    brain = Brain()
    brain.http.close()
    with httpx.Client(base_url="http://model", transport=httpx.MockTransport(respond)) as brain.http:
        assert brain.decide([{"role": "user", "content": "Answer yes or no."}]) == {
            "yes": expected_yes, "no": 1.0 - expected_yes,
        }
