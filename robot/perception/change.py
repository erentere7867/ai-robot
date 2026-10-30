"""Cheap scene-change detection on tiny grayscale thumbnails (no model inference)."""

from __future__ import annotations

import numpy as np


class ChangeDetector:
    """Flags a change when enough thumbnail blocks differ from the reference view.

    The reference is only meaningful for one camera pose, so callers reset it whenever
    the robot or its head moves.
    """

    def __init__(self, *, pixel_threshold: float = 18.0, fraction_threshold: float = 0.02) -> None:
        self.pixel_threshold = pixel_threshold
        self.fraction_threshold = fraction_threshold
        self._reference: np.ndarray | None = None

    def reset(self, thumbnail: np.ndarray | None = None) -> None:
        self._reference = None if thumbnail is None else thumbnail.astype(np.float32)

    def changed(self, thumbnail: np.ndarray) -> float:
        """Return the fraction of changed blocks (0 when there is no reference yet)."""
        current = thumbnail.astype(np.float32)
        if self._reference is None or self._reference.shape != current.shape:
            self._reference = current
            return 0.0
        diff = np.abs(current - self._reference) > self.pixel_threshold
        return float(diff.mean())

    def is_changed(self, thumbnail: np.ndarray) -> bool:
        return self.changed(thumbnail) >= self.fraction_threshold
