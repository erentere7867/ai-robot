"""Tool-argument parsing: every common way a model writes a number is read as that number."""

from __future__ import annotations

import pytest

from robot.agent.tools import _num


@pytest.mark.parametrize(("text", "expected"), [
    ("2", 2.0), ("-2", -2.0), ("1.5", 1.5), (" 3 m ", 3.0), ("90 degrees", 90.0), ("2s", 2.0),
    (".5", 0.5), ("-.5", -0.5), ("+1", 1.0), ("1.", 1.0), ("1e2", 100.0), ("2.5E-1", 0.25), ("+.5 m", 0.5),
])
def test_every_common_number_form_is_read(text: str, expected: float) -> None:
    assert _num(text) == expected


@pytest.mark.parametrize("text", ["", ".", "-", "e2", "1e", "1,5", "1.2.3", "inf", "nan", "1e999", "two", "5 km"])
def test_anything_else_is_refused(text: str) -> None:
    with pytest.raises(ValueError):
        _num(text)


def test_true_and_false_are_not_numbers() -> None:
    with pytest.raises(ValueError):
        _num(True)
