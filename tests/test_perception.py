"""The perception pipeline's geometry and bookkeeping, without a detector or ROS: rig, locate, colour, tracker."""

import math

import numpy as np

from robot.perception import colour
from robot.perception.locate import locate
from robot.perception.rig import RIG, to_map, to_robot
from robot.perception.tracker import Detection, Tracker


def test_a_pixel_and_its_depth_give_back_the_point():
    cam = RIG["front_camera"]
    point = (2.0, 0.5, 0.3)  # robot frame
    # project by hand: forward and left of the camera, then into pixels
    fwd, left, up = point[0] - cam.x, point[1] - cam.y, point[2] - cam.z
    u, v = cam.width / 2 - left / fwd * cam.focal, cam.height / 2 - up / fwd * cam.focal
    x, y, z = cam.point(u, v, fwd)
    assert math.dist((x, y, z), point) < 1e-6


def test_the_rear_cameras_cover_what_the_front_one_does_not():
    for deg in range(-180, 180, 10):
        b = math.radians(deg)
        assert any(c.sees(3 * math.cos(b), 3 * math.sin(b)) for c in RIG.values()), deg


def test_a_box_on_the_floor_is_located_from_its_bottom_edge():
    cam = RIG["rear_left_camera"]
    # a 0.3 m wide box whose near face is 2 m from the camera, straight along the camera's axis
    d, half = 2.0, 0.15
    u1, u2 = cam.width / 2 - half / d * cam.focal, cam.width / 2 + half / d * cam.focal
    v2 = cam.height / 2 + cam.z / d * cam.focal  # the floor line under the box's face
    x, y, r = locate(cam, (u1, v2 - 60, u2, v2))
    expected = (cam.x + (d + r) * math.cos(cam.yaw), cam.y + (d + r) * math.sin(cam.yaw))
    assert math.dist((x, y), expected) < 0.05 and abs(r - half) < 0.03


def test_depth_wins_on_the_front_camera():
    cam = RIG["front_camera"]
    depth = np.full((cam.height, cam.width), 3.0, dtype=np.float32)
    x, y, r = locate(cam, (300, 200, 340, 300), depth=depth)
    assert abs(math.hypot(x - cam.x, y - cam.y) - (3.0 + r)) < 0.1


def test_colour_names():
    for rgb, word in [((200, 30, 30), "red"), ((30, 60, 200), "blue"), ((240, 240, 235), "white"),
                      ((20, 20, 20), "black"), ((120, 80, 40), "brown"), ((40, 160, 60), "green"),
                      ((230, 200, 30), "yellow"), ((128, 128, 128), "grey")]:
        assert colour.name(np.array([rgb] * 4)) == word, (rgb, word)


def test_map_and_robot_frames_round_trip():
    pose = (1.0, -2.0, math.radians(30))
    assert math.dist(to_robot(pose, *to_map(pose, 0.7, -0.4)), (0.7, -0.4)) < 1e-9


def test_the_tracker_keeps_ids_merges_views_and_forgets_on_evidence():
    tr = Tracker()
    tr.update([Detection("couch", 1.0, 1.0, 0.5, "grey")], 0.0)
    tr.update([Detection("sofa", 1.1, 0.95, 0.5, "grey", camera="rear_left_camera")], 0.5)  # same thing, another view
    tr.update([Detection("chair", 3.0, 0.0, 0.3)], 0.5)
    objects = tr.objects()
    assert [t.label for t in objects] == ["sofa"] and objects[0].hits == 2  # the chair needs a second sighting
    sofa = objects[0]
    for _ in range(3):
        tr.missed(sofa)
    assert not sofa.present
    tr.update([Detection("sofa", 1.05, 1.0, 0.5)], 2.0)
    assert sofa.present and len(tr.tracks) == 2


def test_one_thing_named_two_ways_is_one_object_and_the_vote_names_it():
    tr = Tracker()
    tr.update([Detection("coffee table", 2.0, 0.0, 0.5, score=0.6)], 0.0)
    tr.update([Detection("tv", 2.1, 0.05, 0.45, score=0.3)], 0.5)  # the low table seen edge-on
    tr.update([Detection("cup", 2.05, 0.0, 0.05, score=0.7)], 0.5)  # a cup on it is not the table
    tr.update([Detection("cup", 2.05, 0.02, 0.05, score=0.7)], 1.0)
    assert sorted(t.label for t in tr.objects()) == ["cup", "table"]
    tr.update([Detection("person", 0.0, 3.0, 0.3, score=0.2)], 1.0)  # a faint person needs a second look
    assert "person" not in [t.label for t in tr.objects()]
    tr.update([Detection("person", 4.0, 3.0, 0.3, score=0.8)], 1.0)  # a clear one counts at once
    assert [t.label for t in tr.objects()].count("person") == 1


def test_the_vision_model_has_the_last_word_on_a_new_object():
    tr = Tracker()
    for t in (0.0, 0.5, 1.0):
        tr.update([Detection("chair", 2.0, 0.0, 0.5, score=0.5)], t)  # a low table the detector calls a chair
        tr.update([Detection("tv", 0.0, 2.0, 0.3, score=0.3)], t)  # a reflection
    table, phantom = sorted(tr.objects(), key=lambda t: t.x, reverse=True)
    tr.verify(table, "table")
    tr.verify(phantom, None)
    tr.update([Detection("chair", 2.0, 0.0, 0.5, score=0.5)], 1.5)
    assert [(t.label, t.id) for t in tr.objects()] == [("table", table.id)]


def test_size_and_place_hold_near_the_edge_of_the_wide_lens():
    cam = RIG["rear_left_camera"]
    off, d, half = math.radians(60), 2.0, 0.15  # a 0.3 m wide thing 60 degrees off the lens axis, near face 2 m away

    def column(angle: float) -> float:  # image column of a direction from the camera (pinhole)
        return cam.width / 2 - math.tan(angle) * cam.focal

    u1, u2 = column(off + half / d), column(off - half / d)
    v2 = cam.height / 2 + cam.z / (d * math.cos(off)) * cam.focal  # the floor line under its face
    x, y, r = locate(cam, (u1, v2 - 40, u2, v2))
    assert abs(r - half) < 0.03
    expected = (cam.x + (d + r) * math.cos(cam.yaw + off), cam.y + (d + r) * math.sin(cam.yaw + off))
    assert math.dist((x, y), expected) < 0.1


def test_depth_reads_the_near_face_of_an_open_thing():
    cam = RIG["front_camera"]
    depth = np.full((cam.height, cam.width), 4.0, dtype=np.float32)  # the wall, seen through a table's frame
    depth[200:215, 280:400] = 2.0  # the table top
    depth[200:300, 280:290] = depth[200:300, 390:400] = 2.0  # its legs
    x, y, r = locate(cam, (280, 200, 400, 300), depth=depth)
    assert abs(math.hypot(x - cam.x, y - cam.y) - (2.0 + r)) < 0.15


def test_a_part_seen_past_the_picture_edge_joins_its_thing_and_neighbours_stay_apart():
    tr = Tracker()
    for t in (0.0, 0.5):
        tr.update([Detection("table", 2.0, -1.0, 0.5, score=0.6)], t)
        tr.update([Detection("desk", 2.2, -1.2, 0.07, score=0.3, cut=True)], t)  # the part past a picture edge
        tr.update([Detection("chair", 0.0, 2.0, 0.39, score=0.8)], t)
        tr.update([Detection("backpack", 0.0, 2.45, 0.22, score=0.7)], t)  # leaning on the chair's side
    assert sorted(t.label for t in tr.objects()) == ["backpack", "chair", "table"]


def test_a_thing_is_named_again_when_seen_twice_as_large():
    tr = Tracker()
    for t in (0.0, 0.5):
        tr.update([Detection("book", 4.0, 1.0, 0.2, score=0.4)], t)
    far, near = np.zeros((20, 20, 3)), np.zeros((40, 40, 3))
    track = tr.to_name({1: far})
    tr.verify(track, "book", 400)
    assert tr.to_name({1: far}) is None  # named from this view already
    assert tr.to_name({1: near}) is track
    tr.verify(track, "backpack", 1600)
    assert tr.objects()[0].label == "backpack"


def test_a_thing_first_seen_cut_takes_its_size_from_the_first_whole_view():
    tr = Tracker()
    tr.update([Detection("desk", 1.3, -1.1, 0.07, score=0.3, cut=True)], 0.0)  # only its edge, past the picture
    tr.update([Detection("table", 1.6, -1.0, 0.5, score=0.6)], 0.5)  # then the whole table from another camera
    tr.update([Detection("table", 1.6, -1.0, 0.5, score=0.6)], 1.0)
    (table,) = tr.objects()
    assert table.label == "table" and abs(table.radius - 0.5) < 0.05


def test_a_cut_view_joins_the_thing_whose_middle_it_misses():
    tr = Tracker()
    for t in (0.0, 0.5):
        tr.update([Detection("table", 1.6, -1.0, 0.25, score=0.6)], t)
        tr.update([Detection("desk", 1.3, -1.1, 0.07, score=0.3, cut=True)], t)  # 0.32 m off, past the edge
    assert [t.label for t in tr.objects()] == ["table"]


def test_a_thing_under_the_lidar_beam_ignores_the_wall_the_beam_hits_behind_it():
    cam = RIG["rear_left_camera"]
    d, half, height = 2.0, 0.12, 0.24  # a ball's near face 2 m away on the lens axis; the wall 2.8 m behind
    u1, u2 = cam.width / 2 - half / d * cam.focal, cam.width / 2 + half / d * cam.focal
    v2 = cam.height / 2 + cam.z / d * cam.focal
    v1 = cam.height / 2 + (cam.z - height) / d * cam.focal
    ranges = np.full(360, 2.8 + math.hypot(cam.x, cam.y), dtype=np.float32)
    x, y, r = locate(cam, (u1, v1, u2, v2), scan=(ranges, -math.pi, 2 * math.pi / 360))
    assert abs(math.hypot(x - cam.x, y - cam.y) - (d + r)) < 0.1
