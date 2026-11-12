"""Rooms from the robot's own map, named by a guided tour or by the furniture in them (docs/architecture.md: "the grid
is segmented into rooms; named places come from a guided tour").

Segmenting: the free floor is shrunk by half a door's width, which closes every doorway, so what is left falls apart
into one core per room; each free cell then joins the core it can reach soonest. Naming: a place saved inside a room
during the tour ("this is the kitchen") names it; otherwise its furniture votes (a fridge or an oven says kitchen, a
sofa or a television says living room, a bed says bedroom); a room with neither is "room 1", "room 2" and so on.
A room is kept as the rectangle around it, the form the rest of the robot uses.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

KIND = {"fridge": "kitchen", "refrigerator": "kitchen", "oven": "kitchen", "microwave": "kitchen", "sink": "kitchen",
        "sofa": "living room", "couch": "living room", "tv": "living room", "bed": "bedroom", "toilet": "bathroom",
        "bathtub": "bathroom", "desk": "office", "laptop": "office", "printer": "office"}
STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


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


def name(masks: list[np.ndarray], cell: float, size: tuple[float, float], things: list, places: dict
         ) -> dict[str, tuple[float, float, float, float]]:
    """A name and rectangle (x0, y0, x1, y1) for each room: from the tour's places inside it, else its furniture."""
    def index(x: float, y: float) -> tuple[int, int]:
        return int((x + size[0] / 2) / cell), int((y + size[1] / 2) / cell)

    rooms: dict[str, tuple[float, float, float, float]] = {}
    unnamed = 0
    for mask in masks:
        def inside(x: float, y: float, mask: np.ndarray = mask) -> bool:
            i, j = index(x, y)
            return 0 <= i < mask.shape[0] and 0 <= j < mask.shape[1] and bool(mask[i, j])

        told = [n for n, p in places.items() if n != "start" and inside(p.x, p.y) and n not in rooms]
        votes = Counter(KIND[t.label] for t in things if t.label in KIND and inside(t.x, t.y))
        choice = told[0] if told else next((k for k, _ in votes.most_common() if k not in rooms), None)
        if choice is None:
            unnamed += 1
            choice = f"room {unnamed}"
        xs, ys = np.nonzero(mask)
        rooms[choice] = (round(-size[0] / 2 + xs.min() * cell, 2), round(-size[1] / 2 + ys.min() * cell, 2),
                         round(-size[0] / 2 + (xs.max() + 1) * cell, 2), round(-size[1] / 2 + (ys.max() + 1) * cell, 2))
    return rooms
