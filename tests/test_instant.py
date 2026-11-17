"""Instant commands are matched without a model, and only when the utterance is the command alone."""

import pytest

from robot.agent.instant import instant


@pytest.mark.parametrize(("text", "command"), [
    ("Stop!", "stop"), ("stop it please", "stop"), ("Sprocket, stop.", "stop"), ("okay, halt", "stop"),
    ("wait", "wait"), ("hold on", "wait"), ("Hey robot, wait a second", "wait"),
    ("come back", "come back"), ("Come back here please!", "come back"),
    ("come here", "come back"), ("Hey Sprocket, come here please!", "come back"),
    ("come to me", "come back"), ("come over here", "come back"), ("over here", "come back"),
])
def test_short_commands_are_instant(text, command):
    assert instant(text) == command


@pytest.mark.parametrize("text", [
    "stop at the door", "wait until you see the dog", "come back to me after you find the ball",
    "come over here after checking the kitchen", "go to the kitchen",
    "", "please",
])
def test_instructions_go_to_the_planner(text):
    assert instant(text) is None
