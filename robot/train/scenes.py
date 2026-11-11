"""Random rooms for training data: the suite's building blocks (Scene, Thing), a new layout every time.

One room or two joined by a door, 6 to 11 pieces of furniture and smaller things from a household catalogue with
random colours, the robot and the user at random free spots. The 50-task suite's rooms are never produced, so its
scores stay a held-out test.
"""

from __future__ import annotations

import math
import random

from robot.agent.api import Pose
from robot.sim.world import Scene, Thing

SEAT = ("seat", "something to sit on")
# label: (radius range, colours, other names, solid)
CATALOG: dict[str, tuple[tuple[float, float], tuple[str, ...], tuple[str, ...], bool]] = {
    "chair": ((0.25, 0.32), ("black", "brown", "white", "grey"), SEAT, True),
    "sofa": ((0.5, 0.6), ("grey", "blue", "green", "brown"), ("couch", *SEAT), True),
    "table": ((0.4, 0.5), ("brown", "white", "black"), (), True),
    "box": ((0.2, 0.35), ("brown", "blue", "yellow", "red", "green"), (), True),
    "ball": ((0.1, 0.15), ("red", "blue", "yellow", "green", "orange"), ("toy",), True),
    "backpack": ((0.18, 0.22), ("blue", "black", "red", "green"), ("bag",), True),
    "plant": ((0.2, 0.3), ("green",), (), True),
    "lamp": ((0.15, 0.2), ("white", "black"), (), True),
    "bin": ((0.15, 0.2), ("grey", "black", "blue"), (), True),
    "cabinet": ((0.35, 0.45), ("white", "brown"), (), True),
    "basket": ((0.18, 0.22), ("brown", "white"), (), True),
    "shoe": ((0.1, 0.12), ("red", "black", "white"), (), True),
    "bottle": ((0.05, 0.06), ("green", "blue", "white"), (), False),
    "cup": ((0.05, 0.06), ("white", "red", "blue"), ("mug",), False),
    "laptop": ((0.15, 0.18), ("grey", "black"), (), True),
    "pillow": ((0.15, 0.2), ("white", "blue", "red"), (), True),
    "desk": ((0.4, 0.5), ("white", "brown"), (), True),
    "fridge": ((0.4, 0.45), ("white", "grey"), (), True),
    "tv": ((0.25, 0.3), ("black",), (), True),
}
REPEATABLE = ("chair", "box", "ball", "cup", "pillow", "bottle")  # things a room often has more than one of
ROOM_NAMES = ("living", "kitchen", "office", "bedroom", "hall", "study", "playroom", "garage")


def random_scene(rng: random.Random, two_rooms: bool | None = None) -> Scene | None:
    """A furnished room or pair of rooms; None when the random placement failed (the caller tries again)."""
    two = rng.random() < 0.35 if two_rooms is None else two_rooms
    w, h = (rng.uniform(8.0, 10.5), rng.uniform(4.5, 6.0)) if two else (rng.uniform(5.5, 8.0), rng.uniform(4.0, 6.0))
    names = rng.sample(ROOM_NAMES, 2)
    things: list[Thing] = []
    walls: list[tuple[tuple[float, float], tuple[float, float]]] = []
    door = None
    if two:
        rooms = {names[0]: (-w / 2, -h / 2, 0.0, h / 2), names[1]: (0.0, -h / 2, w / 2, h / 2)}
        d = rng.uniform(-h / 2 + 0.9, h / 2 - 0.9)
        walls = [((0.0, -h / 2), (0.0, d - 0.45)), ((0.0, d + 0.45), (0.0, h / 2))]
        door = (0.0, d)
        things.append(Thing("door", 0.0, d, 0.45, "white", facts={"open": True}, solid=False))
    else:
        rooms = {names[0]: (-w / 2, -h / 2, w / 2, h / 2)}

    def free(x: float, y: float, r: float, spacing: float) -> bool:
        if not (-w / 2 + r + 0.3 < x < w / 2 - r - 0.3 and -h / 2 + r + 0.3 < y < h / 2 - r - 0.3):
            return False
        if two and abs(x) < r + 0.35:  # not on the dividing wall
            return False
        if door and math.dist((x, y), door) < r + 0.9:  # keep the doorway clear
            return False
        return all(math.dist((x, y), (t.x, t.y)) >= r + t.radius + spacing for t in things if t.label != "door")

    labels = rng.sample(sorted(CATALOG), rng.randint(5, 8))
    labels += [rng.choice(REPEATABLE) for _ in range(rng.randint(1, 3))]
    for label in labels:
        (r0, r1), colors, other, solid = CATALOG[label]
        r = round(rng.uniform(r0, r1), 2)
        for _ in range(60):
            x, y = rng.uniform(-w / 2, w / 2), rng.uniform(-h / 2, h / 2)
            if free(x, y, r, 0.55 if solid else 0.25):
                things.append(Thing(label, round(x, 2), round(y, 2), r, rng.choice(colors), names=other, solid=solid))
                break
    start = next(((x, y) for x, y in ((rng.uniform(-w / 2, w / 2), rng.uniform(-h / 2, h / 2)) for _ in range(200))
                  if free(x, y, 0.15, 0.6)), None)
    if start is None:
        return None
    things_now = list(things)
    user = next(((x, y) for x, y in ((rng.uniform(-w / 2, w / 2), rng.uniform(-h / 2, h / 2)) for _ in range(200))
                 if free(x, y, 0.25, 0.4) and math.dist((x, y), start) > 1.2), None)
    if user is None:
        return None
    things = things_now + [Thing("person", round(user[0], 2), round(user[1], 2), 0.25, solid=False, user=True)]
    return Scene(round(w, 2), round(h, 2), things, rooms, walls=walls,
                 start=Pose(round(start[0], 2), round(start[1], 2), float(rng.choice(range(-180, 180, 15)))))
