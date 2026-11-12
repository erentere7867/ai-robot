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
    place = locate(cam, (u1, v2 - 60, u2, v2))
    x, y, r = place.x, place.y, place.radius
    expected = (cam.x + (d + r) * math.cos(cam.yaw), cam.y + (d + r) * math.sin(cam.yaw))
    assert math.dist((x, y), expected) < 0.05 and abs(r - half) < 0.03


def test_depth_wins_on_the_front_camera():
    cam = RIG["front_camera"]
    depth = np.full((cam.height, cam.width), 3.0, dtype=np.float32)
    place = locate(cam, (300, 200, 340, 300), depth=depth)
    x, y, r = place.x, place.y, place.radius
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
    place = locate(cam, (u1, v2 - 40, u2, v2))
    x, y, r = place.x, place.y, place.radius
    assert abs(r - half) < 0.03
    expected = (cam.x + (d + r) * math.cos(cam.yaw + off), cam.y + (d + r) * math.sin(cam.yaw + off))
    assert math.dist((x, y), expected) < 0.1


def test_depth_reads_the_near_face_of_an_open_thing():
    cam = RIG["front_camera"]
    depth = np.full((cam.height, cam.width), 4.0, dtype=np.float32)  # the wall, seen through a table's frame
    depth[200:215, 280:400] = 2.0  # the table top
    depth[200:300, 280:290] = depth[200:300, 390:400] = 2.0  # its legs
    place = locate(cam, (280, 200, 400, 300), depth=depth)
    x, y, r = place.x, place.y, place.radius
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
    place = locate(cam, (u1, v1, u2, v2), scan=(ranges, -math.pi, 2 * math.pi / 360))
    x, y, r = place.x, place.y, place.radius
    assert abs(math.hypot(x - cam.x, y - cam.y) - (d + r)) < 0.1


def test_a_still_thing_settles_on_the_precise_sightings_instead_of_twitching():
    tr = Tracker()
    for k in range(40):  # depth says 1.53 m, a far floor estimate says 1.27 m, in turn
        x, sigma = (1.53, 0.08) if k % 2 == 0 else (1.27, 0.2)
        tr.update([Detection("table", x, -1.0, 0.5, sigma=sigma)], k * 0.1)
    table = tr.objects()[0]
    before = table.x
    tr.update([Detection("table", 1.27, -1.0, 0.5, sigma=0.2)], 4.0)
    assert abs(table.x - 1.53) < 0.06 and abs(table.x - before) < 0.01  # near the depth, and still


def test_a_box_that_holds_only_floor_is_no_thing():
    cam = RIG["front_camera"]
    rows = np.arange(cam.height, dtype=np.float32)[:, None]
    down = np.maximum((rows - cam.height / 2) / cam.focal, 1e-3)
    depth = np.broadcast_to(cam.z / down, (cam.height, cam.width)).astype(np.float32).copy()  # the floor everywhere
    assert locate(cam, (0, 300, 640, 480), depth=depth) is None


def test_a_lidar_return_far_behind_the_floor_estimate_is_not_believed():
    cam = RIG["rear_left_camera"]
    d, half, height = 2.0, 0.2, 0.6  # a chair-height thing, 2 m away; the beam passes its legs to a wall at 2.5 m
    u1, u2 = cam.width / 2 - half / d * cam.focal, cam.width / 2 + half / d * cam.focal
    v2 = cam.height / 2 + cam.z / d * cam.focal
    v1 = cam.height / 2 + (cam.z - height) / d * cam.focal
    ranges = np.full(360, 2.5 + math.hypot(cam.x, cam.y), dtype=np.float32)
    place = locate(cam, (u1, v1, u2, v2), scan=(ranges, -math.pi, 2 * math.pi / 360))
    assert place.how == "floor" and abs(math.hypot(place.x - cam.x, place.y - cam.y) - (d + place.radius)) < 0.1


def test_depth_finds_a_thin_thing_that_fills_little_of_its_box():
    cam = RIG["front_camera"]
    depth = np.full((cam.height, cam.width), 4.0, dtype=np.float32)  # the wall behind
    depth[150:300, 300:306] = depth[150:300, 394:400] = 2.0  # two thin chair legs: 6% of the box
    place = locate(cam, (290, 150, 410, 300), depth=depth)
    assert abs(math.hypot(place.x - cam.x, place.y - cam.y) - (2.0 + place.radius)) < 0.15


def test_a_thing_whose_foot_is_below_the_picture_gets_no_floor_estimate():
    cam = RIG["rear_left_camera"]
    assert locate(cam, (600, 400, 700, cam.height)) is None  # runs off the bottom: no floor contact in view


def test_twins_of_one_name_are_merged_into_the_older():
    tr = Tracker()
    for t in (0.0, 0.5):
        tr.update([Detection("ball", 1.30, 1.90, 0.13, sigma=0.1)], t)
    first = tr.objects()[0]
    tr.verify(first, "toy")  # renamed for a while: a ball sighting now starts a twin
    for t in (1.0, 1.5):
        tr.update([Detection("ball", 1.35, 1.85, 0.13, sigma=0.1)], t)
    tr.verify(first, "ball")  # named back
    tr.update([Detection("ball", 1.32, 1.88, 0.13, sigma=0.1)], 2.0)
    assert [(t.id, t.label) for t in tr.objects()] == [(first.id, "ball")]


def test_a_standing_person_stops_jumping_and_a_walking_one_is_followed():
    tr = Tracker()
    for k in range(30):  # standing still; sightings scatter by 0.4 m, ten a second
        tr.update([Detection("person", -1.4 + (0.4 if k % 2 else -0.4), 0.9, 0.3, score=0.9, sigma=0.3)], k * 0.1)
    person = tr.objects()[0]
    assert abs(person.x + 1.4) < 0.15
    for k in range(30):  # then walks 1.5 m in 3 s
        tr.update([Detection("person", -1.4 + 0.05 * k, 0.9, 0.3, score=0.9, sigma=0.1)], 3.0 + k * 0.1)
    assert abs(person.x - 0.05) < 0.3


def test_a_person_boxed_twice_is_one_detection_and_a_cup_on_a_table_is_two():
    from robot.perception.detector import Seen, _same_thing

    body, upper = Seen("person", 0.9, (500, 100, 560, 380)), Seen("person", 0.3, (505, 100, 555, 220))
    table, cup = Seen("table", 0.8, (300, 300, 600, 420)), Seen("cup", 0.6, (420, 280, 450, 320))
    assert _same_thing(body, upper) and not _same_thing(table, cup)


def test_frontiers_are_open_edges_of_the_map_with_room_for_the_robot():
    from robot.ros.explore import frontiers

    grid = np.full((60, 60), -1, dtype=np.int8)  # unknown
    grid[10:50, 5:30] = 0  # a mapped patch of floor
    grid[10:50, 4] = 100  # with a wall on its left
    found = frontiers(grid, 0.05, (0.0, 0.0))
    assert found and all(x > 0.5 for x, _, _ in found)  # the open right side, not by the wall
    assert all(0.75 <= y <= 2.25 for _, y, _ in found) or any(y < 0.75 or y > 2.25 for _, y, _ in found)
    ring = np.full((80, 80), -1, dtype=np.int8)  # early in mapping: known floor all round the robot
    yy, xx = np.mgrid[:80, :80]
    ring[(yy - 40) ** 2 + (xx - 40) ** 2 < 15 ** 2] = 0
    goals = frontiers(ring, 0.05, (-2.0, -2.0))
    assert min(math.hypot(x, y) for x, y, _ in goals) > 0.5  # on the ring, not in its middle where the robot is


def test_a_saved_map_becomes_the_planners_grid_in_the_house_frame(tmp_path):
    from robot.ros.explore import load_grid

    image = np.full((100, 140), 254, dtype=np.uint8)  # a 7 x 5 m map at 5 cm, all free...
    image[:, 70] = 0  # ...with a wall along x = 0 (the map starts at x = -3.5)
    image[:10, :] = 205  # and unknown along its top (largest y) edge
    (tmp_path / "m.pgm").write_bytes(b"P5\n140 100\n255\n" + image.tobytes())
    (tmp_path / "m.yaml").write_text("image: m.pgm\nresolution: 0.050\norigin: [-3.5, -2.5, 0.0]\n")
    grid = load_grid(tmp_path / "m.yaml", (70, 50), (7.0, 5.0))  # 10 cm cells
    assert grid[35, 25] and not grid[20, 25] and not grid[50, 25]  # the wall at x = 0, floor either side
    assert grid[20, 48] and not grid[20, 40]  # unknown counts as blocked


def test_a_person_lost_from_view_is_known_again_by_their_clothes():
    from robot.perception.tracker import look

    red_shirt = np.zeros((40, 20, 3), dtype=np.uint8)
    red_shirt[..., 0] = 200
    blue_shirt = np.zeros((40, 20, 3), dtype=np.uint8)
    blue_shirt[..., 2] = 200
    tr = Tracker()
    for k in range(5):
        tr.update([Detection("person", 1.0 + 0.1 * k, 0.0, 0.3, score=0.9, look=look(red_shirt))], k * 0.1)
    user = tr.objects()[0]
    # behind the sofa for two seconds; then seen again 1.5 m on, and a stranger in blue appears too
    tr.update([Detection("person", 2.9, 0.4, 0.3, score=0.9, look=look(red_shirt)),
               Detection("person", 2.6, -0.6, 0.3, score=0.8, look=look(blue_shirt))], 2.5)
    people = {t.id: t for t in tr.objects() if t.person}
    assert user.id in people and abs(people[user.id].x - 2.9) < 0.3  # the user kept their id
    assert len(people) == 2  # and the stranger is someone new
    swapped = Tracker()  # the stranger seen first, right beside where the user was: still not the user
    for k in range(5):
        swapped.update([Detection("person", 1.0, 0.0, 0.3, score=0.9, look=look(red_shirt))], k * 0.1)
    swapped.update([Detection("person", 1.3, 0.2, 0.3, score=0.95, look=look(blue_shirt))], 0.6)
    assert len([t for t in swapped.objects() if t.person]) == 2


def test_the_vision_model_can_tell_a_backpack_the_detector_took_for_a_person():
    tr = Tracker()
    for t in (0.0, 0.5):
        tr.update([Detection("person", -0.9, -1.8, 0.2, score=0.6)], t)
    (track,) = tr.objects()
    tr.verify(track, "backpack")
    assert track.label == "backpack" and not track.person
