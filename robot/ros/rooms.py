"""Rooms from the robot's own map, named by a guided tour or by the vision model (docs/architecture.md: "the grid
is segmented into rooms; named places come from a guided tour").

Segmenting: the free floor is shrunk by half a door's width, which closes every doorway, so what is left falls apart
into one core per room; each free cell then joins the core it can reach soonest. Naming: a place saved inside a room
during the tour ("this is the kitchen") names it; otherwise what the vision model answered, from pictures taken
inside it, to "what kind of room is this?" (it saw tables 96% and kitchen counters 90% of the time from the robot's
height, where YOLOE saw them under 8%); a room with neither is "room 1", "room 2" and so on. A room is kept as the
rectangle around it, the form the rest of the robot uses.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

MARGIN = 2  # a named room takes another name only when another answer leads by this many


def _distance_to_blocked(free: np.ndarray) -> np.ndarray:
    """Cells to the nearest blocked cell (4-connected steps), for every cell."""
    if free.all():
        return np.full(free.shape, free.size, dtype=np.int32)
    import cv2

    return cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L1, 3).astype(np.int32)


def segment(free: np.ndarray, cell: float, door: float = 1.0, smallest: float = 1.5) -> list[np.ndarray]:
    """The rooms of a free-floor grid (indexed [x, y], `cell` metres a side), as masks, largest first. `door`: the
    widest doorway to split at; `smallest`: the least floor, square metres, a room's core must have."""
    import cv2

    core = _distance_to_blocked(free) * cell > door / 2
    n_labels, labels = cv2.connectedComponents(core.astype(np.uint8), connectivity=4)
    counts = np.bincount(labels.ravel(), minlength=n_labels)
    keep = [k for k in range(1, n_labels) if counts[k] * cell * cell >= smallest]
    if not keep:
        return []
    keep_lut = np.zeros(n_labels, dtype=np.int32)
    for k in keep:
        keep_lut[k] = k
    labels = keep_lut[labels]
    # Grow every core through the free floor one 4-connected wavefront at a time.
    while True:
        unlabeled = free & (labels == 0)
        if not unlabeled.any():
            break
        grown = np.zeros_like(labels)
        grown[1:, :] = np.where(grown[1:, :], grown[1:, :], labels[:-1, :])
        grown[:-1, :] = np.where(grown[:-1, :], grown[:-1, :], labels[1:, :])
        grown[:, 1:] = np.where(grown[:, 1:], grown[:, 1:], labels[:, :-1])
        grown[:, :-1] = np.where(grown[:, :-1], grown[:, :-1], labels[:, 1:])
        step = unlabeled & (grown > 0)
        if not step.any():
            break
        labels[step] = grown[step]
    masks = [labels == k for k in keep]
    return sorted(masks, key=lambda m: -int(m.sum()))


def name(masks: list[np.ndarray], cell: float, size: tuple[float, float], votes: list[Counter], places: dict,
         before: list[str] | None = None) -> dict[str, tuple[float, float, float, float]]:
    """A name and rectangle (x0, y0, x1, y1) for each room, in the masks' order: from the tour's places inside it,
    else the vision model's answers about it (`votes[i]`, room i's answers counted). `before`: each room's last name;
    it stays until another answer leads by MARGIN, so one odd answer does not rename the living room."""
    def index(x: float, y: float) -> tuple[int, int]:
        return int((x + size[0] / 2) / cell), int((y + size[1] / 2) / cell)

    rooms: dict[str, tuple[float, float, float, float]] = {}
    unnamed = 0
    for k, mask in enumerate(masks):
        def inside(x: float, y: float, mask: np.ndarray = mask) -> bool:
            i, j = index(x, y)
            return 0 <= i < mask.shape[0] and 0 <= j < mask.shape[1] and bool(mask[i, j])

        told = [n for n, p in places.items() if n != "start" and inside(p.x, p.y) and n not in rooms]
        said = votes[k] if k < len(votes) else Counter()
        choice = told[0] if told else next((n for n, _ in said.most_common() if n not in rooms), None)
        last = before[len(rooms)] if before and len(rooms) < len(before) else None
        if (not told and last in said and last not in rooms and choice is not None
                and said[choice] < said[last] + MARGIN):
            choice = last
        if choice is None:
            unnamed += 1
            choice = f"room {unnamed}"
        xs, ys = np.nonzero(mask)
        rooms[choice] = (round(-size[0] / 2 + xs.min() * cell, 2), round(-size[1] / 2 + ys.min() * cell, 2),
                         round(-size[0] / 2 + (xs.max() + 1) * cell, 2), round(-size[1] / 2 + (ys.max() + 1) * cell, 2))
    return rooms


def room_of(masks: list[np.ndarray], cell: float, size: tuple[float, float], x: float, y: float) -> int | None:
    """Which room mask a map point is in."""
    i, j = int((x + size[0] / 2) / cell), int((y + size[1] / 2) / cell)
    return next((k for k, m in enumerate(masks) if 0 <= i < m.shape[0] and 0 <= j < m.shape[1] and m[i, j]), None)
