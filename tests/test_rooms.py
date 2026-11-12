"""Rooms from the robot's own map (robot/ros/rooms.py): split at doorways, named by the tour or the furniture."""

import numpy as np

from robot.agent.api import Pose
from robot.ros.rooms import name, segment
from robot.sim.world import Thing

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


def test_rooms_are_named_by_the_tour_first_then_by_their_furniture():
    rooms = segment(_two_rooms(), CELL)
    things = [Thing("fridge", 4.0, 2.0, 0.4, "white"), Thing("sofa", -3.0, 1.5, 0.5, "grey")]
    named = name(rooms, CELL, (10.0, 5.0), things, {"start": Pose(-2.5, 0, 0)})
    assert set(named) == {"kitchen", "living room"}
    x0, _, x1, _ = named["kitchen"]
    assert x0 >= -0.2 and x1 <= 5.0
    toured = name(rooms, CELL, (10.0, 5.0), things, {"study": Pose(-2.0, 0.0, 0.0)})
    assert set(toured) == {"study", "kitchen"}  # the tour's name beats the sofa's vote
    assert set(name(rooms, CELL, (10.0, 5.0), [], {})) == {"room 1", "room 2"}
