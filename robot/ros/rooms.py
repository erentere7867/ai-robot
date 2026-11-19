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

from collections import Counter, deque

import numpy as np

STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
MARGIN = 2  # a named room takes another name only when another answer leads by this many


def _distance_to_blocked(free: np.ndarray) -> np.ndarray:
    """Cells to the nearest blocked cell (4-connected steps), for every cell."""
    dist = np.where(free, -1, 0)
    todo = deque(zip(*np.nonzero(~free), strict=True))
    while todo:
        i, j = todo.popleft()
        for di, dj in STEPS:
            a, b = i + di, j + dj
            if 0 <= a < free.shape[0] and 0 <= b < free.shape[1] and dist[a, b] < 0:
                dist[a, b] = dist[i, j] + 1
                todo.append((a, b))
    return np.where(dist < 0, free.size, dist)


def segment(free: np.ndarray, cell: float, door: float = 1.0, smallest: float = 1.5) -> list[np.ndarray]:
    """The rooms of a free-floor grid (indexed [x, y], `cell` metres a side), as masks, largest first. `door`: the
    widest doorway to split at; `smallest`: the least floor, square metres, a room's core must have."""
    core = _distance_to_blocked(free) * cell > door / 2
    labels = np.zeros(free.shape, dtype=np.int32)
    seeds: list[list[tuple[int, int]]] = []
    for i, j in zip(*np.nonzero(core), strict=True):
        if labels[i, j]:
            continue
        group, todo = [], deque([(i, j)])
        labels[i, j] = len(seeds) + 1
        while todo:
            a, b = todo.popleft()
            group.append((a, b))
            for di, dj in STEPS:
                c, d = a + di, b + dj
                if 0 <= c < free.shape[0] and 0 <= d < free.shape[1] and core[c, d] and not labels[c, d]:
                    labels[c, d] = labels[i, j]
                    todo.append((c, d))
        seeds.append(group)
    keep = [k + 1 for k, g in enumerate(seeds) if len(g) * cell * cell >= smallest]
    labels = np.where(np.isin(labels, keep), labels, 0)
    todo = deque(zip(*np.nonzero(labels), strict=True))  # grow every core through the free floor, all at once
    while todo:
        i, j = todo.popleft()
        for di, dj in STEPS:
            a, b = i + di, j + dj
            if 0 <= a < free.shape[0] and 0 <= b < free.shape[1] and free[a, b] and not labels[a, b]:
                labels[a, b] = labels[i, j]
                todo.append((a, b))
    masks = [labels == k for k in keep]
    return sorted(masks, key=lambda m: -m.sum())


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
