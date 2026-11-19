"""Rooms from the robot's own map (robot/ros/rooms.py): split at doorways, named by the tour or the vision model."""

from collections import Counter

import numpy as np

from robot.agent.api import Pose
from robot.ros.rooms import name, room_of, segment

CELL = 0.1


def _two_rooms() -> np.ndarray:
    free = np.ones((100, 50), dtype=bool)  # 10 x 5 m, indexed [x, y]
    free[[0, -1], :] = free[:, [0, -1]] = False  # outer walls
    free[49:51, :] = False  # a wall at x = 0...
    free[49:51, 20:29] = True  # ...with a 0.9 m door in it
    return free


def test_the_floor_splits_at_the_door_and_every_free_cell_has_a_room():
    free = _two_rooms()
    rooms = segment(free, CELL)
    assert len(rooms) == 2
    assert all(m[10:40, 5:45].sum() > 0 or m[60:90, 5:45].sum() > 0 for m in rooms)
    assert not (rooms[0] & rooms[1]).any() and ((rooms[0] | rooms[1]) == free).all()


def _votes(rooms: list[np.ndarray], **said: Counter) -> list[Counter]:
    """The model's answers for the room holding each named point: west=(x, y), east=(x, y)."""
    out = [Counter() for _ in rooms]
    for (x, y), answers in said.values():
        out[room_of(rooms, CELL, (10.0, 5.0), x, y)].update(answers)
    return out


def test_rooms_are_named_by_the_tour_first_then_by_what_the_model_saw():
    rooms = segment(_two_rooms(), CELL)
    votes = _votes(rooms, east=((4.0, 2.0), {"kitchen": 3}), west=((-3.0, 1.5), {"living room": 2, "office": 1}))
    named = name(rooms, CELL, (10.0, 5.0), votes, {"start": Pose(-2.5, 0, 0)})
    assert set(named) == {"kitchen", "living room"}
    x0, _, x1, _ = named["kitchen"]
    assert x0 >= -0.2 and x1 <= 5.0
    toured = name(rooms, CELL, (10.0, 5.0), votes, {"study": Pose(-2.0, 0.0, 0.0)})
    assert set(toured) == {"study", "kitchen"}  # the tour's name beats the model's answers
    assert set(name(rooms, CELL, (10.0, 5.0), [Counter(), Counter()], {})) == {"room 1", "room 2"}


def test_a_room_keeps_its_name_until_another_answer_clearly_leads():
    rooms = segment(_two_rooms(), CELL)
    first = name(rooms, CELL, (10.0, 5.0), _votes(rooms, west=((-3.0, 1.5), {"living room": 1})), {})
    again = name(rooms, CELL, (10.0, 5.0), _votes(rooms, west=((-3.0, 1.5), {"living room": 1, "office": 2})), {},
                 before=list(first))
    assert list(again) == list(first)  # office 2, living room 1: not a clear lead
    later = name(rooms, CELL, (10.0, 5.0), _votes(rooms, west=((-3.0, 1.5), {"living room": 1, "office": 3})), {},
                 before=list(first))
    assert "office" in later  # 3 against 1
