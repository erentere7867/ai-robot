"""Degrees on a circle: the one definition of "the short way round"."""

from __future__ import annotations

import numpy as np


def wrap_deg(deg: float) -> float:
    """Into [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0


def wrap_deg_array(deg: np.ndarray) -> np.ndarray:
    return (deg + 180.0) % 360.0 - 180.0
