from __future__ import annotations

import httpx
import pytest

from robot.mind.brain import Brain


@pytest.mark.parametrize(("token", "expected_yes"), [
    ("yes", 1.0), ("no", 0.0), (" YES ", 1.0), (" No ", 0.0),
    ("Not", 0.5), ("Nothing", 0.5), ("Yesterday", 0.5), ("Ye", 0.5), ("yes.", 0.5),
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
