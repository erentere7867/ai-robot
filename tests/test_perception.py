"""The perception pipeline's geometry and bookkeeping, without a detector or ROS: rig, locate, colour, tracker."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from robot.perception import colour
from robot.perception.locate import locate
from robot.perception.rig import RIG, to_map, to_robot
from robot.perception.tracker import GONE_AFTER, Detection, Track, Tracker
from robot.ros.perceive import SyntheticScanPublisher


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
    place = locate(cam, (620, 300, 660, 450), depth=depth)
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
    assert [tr.missed(sofa) for _ in range(GONE_AFTER)] == [False] * (GONE_AFTER - 1) + [True]
    sofa.present = False  # perception's to decide, once the vision model has looked at its place too
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


def _pixel(cam, bx: float, by: float, up: float = 0.0) -> tuple[float, float]:
    """The pixel showing robot-frame point (bx, by) at height `up` (pinhole, as the simulator renders)."""
    dx, dy = bx - cam.x, by - cam.y
    fwd = dx * math.cos(cam.yaw) + dy * math.sin(cam.yaw)
    left = -dx * math.sin(cam.yaw) + dy * math.cos(cam.yaw)
    return cam.width / 2 - left / fwd * cam.focal, cam.height / 2 + (cam.z - up) / fwd * cam.focal


def _edges(cam, bx: float, by: float, r: float) -> tuple[float, float]:
    """The image columns of a round thing's two sides."""
    bearing, half = math.atan2(by - cam.y, bx - cam.x) - cam.yaw, math.asin(r / math.dist((bx, by), (cam.x, cam.y)))
    return tuple(cam.width / 2 - math.tan(bearing + s * half) * cam.focal for s in (1, -1))


def test_size_and_place_hold_near_the_edge_of_the_wide_lens():
    cam = RIG["rear_left_camera"]
    off, far, r = math.radians(60), 2.2, 0.15  # a round thing 0.3 m across, 2.2 m away, 60 degrees off the lens axis
    middle = cam.to_base(far * math.cos(off), far * math.sin(off), 0.0)[:2]
    # the lowest point of its outline: its nearest point along the lens axis
    foot = _pixel(cam, middle[0] - r * math.cos(cam.yaw), middle[1] - r * math.sin(cam.yaw))
    u1, u2 = _edges(cam, *middle, r)
    place = locate(cam, (u1, foot[1] - 40, u2, foot[1]), foot=foot)
    assert abs(place.radius - r) < 0.03 and math.dist((place.x, place.y), middle) < 0.05


def test_a_person_straight_behind_is_put_in_one_place_by_both_rear_cameras():
    """66-73 degrees off both rear lenses, half hidden by a box in front, cut by one picture's edge: the two cameras
    put them 1 m apart from their boxes' bottom middles. From their visible foot and the lidar's view of their legs,
    both find the same place."""
    person, r = (-2.91, 0.2), 0.3
    leg = (-2.787, 0.27)  # the leg both cameras see the foot of, and the lidar the front of
    angles = -math.pi + np.arange(500) * 2 * math.pi / 500
    deg = np.degrees(np.mod(angles, 2 * math.pi))
    ranges = np.full(500, 4.5, dtype=np.float32)  # the wall
    ranges[(deg >= 174) & (deg < 175.6)] = 2.80  # the legs
    ranges[(deg >= 175.6) & (deg < 177)] = 3.54  # the sofa behind, between them
    ranges[(deg >= 177) & (deg < 195)] = 1.43  # a box in front of the person, hiding their other leg
    places = []
    for name in ("rear_left_camera", "rear_right_camera"):
        cam = RIG[name]
        u1, u2 = sorted(_edges(cam, *person, r))
        foot = _pixel(cam, *leg)
        place = locate(cam, (max(u1, 0), 100, min(u2, cam.width), foot[1]), scan=(ranges, -math.pi, 2 * math.pi / 500),
                       foot=foot, person=True)
        assert place.how == "lidar" and math.dist((place.x, place.y), person) < 0.15, (name, place)
        places.append((place.x, place.y))
    assert math.dist(*places) < 0.05


def test_depth_reads_the_near_face_of_an_open_thing():
    cam = RIG["front_camera"]
    depth = np.full((cam.height, cam.width), 4.0, dtype=np.float32)  # the wall, seen through a table's frame
    depth[300:325, 580:700] = 2.0  # the table top
    depth[300:450, 580:595] = depth[300:450, 685:700] = 2.0  # its legs
    place = locate(cam, (580, 300, 700, 450), depth=depth)
    x, y, r = place.x, place.y, place.radius
    assert abs(math.hypot(x - cam.x, y - cam.y) - (2.0 + r)) < 0.15


def test_visible_floor_contact_refines_depth_but_an_elevated_outline_does_not():
    cam = RIG["front_camera"]
    u, horizon = cam.width // 2, cam.height // 2
    v = horizon + cam.focal * cam.z / 2.0  # the floor at 2 metres
    depth = np.full((cam.height, cam.width), 1.8, dtype=np.float32)
    depth[horizon, u] = np.inf  # no return right on the horizon
    for bottom, supported in ((v, True), (horizon + 3, False)):
        box = (u - 22, horizon - 35, u + 22, bottom)
        with np.errstate(invalid="raise"):
            raw = locate(cam, box, depth)
            placed = locate(cam, box, depth, foot=(u, bottom))
        if supported:
            assert raw.x < placed.x < locate(cam, box, foot=(u, bottom)).x
        else:
            assert placed == raw


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
    assert locate(cam, (0, 360, 1280, 720), depth=depth) is None


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
    depth[250:450, 620:628] = depth[250:450, 652:660] = 2.0  # two thin chair legs
    place = locate(cam, (610, 250, 670, 450), depth=depth)
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


def test_an_id_merged_into_another_still_finds_the_thing():
    tr = Tracker()
    tr.update([Detection("sofa", 0.0, 0.0, 0.5, sigma=0.3), Detection("sofa", 0.45, 0.0, 0.5, sigma=0.3)], 0.0)
    (kept,) = tr.tracks  # two sightings of one sofa: two tracks, merged into the older
    assert tr.current(kept + 1) == kept and tr.current(kept) == kept


def test_separate_boxes_stay_separate_across_a_duplicate_merge_and_an_empty_frame():
    tr = Tracker()
    # The wide proposal overlaps both; the other two cannot be duplicate boxes in this picture.
    ds = [Detection("bag", x, 1.0, 0.3, camera="front", box=box) for x, box in (
        (1.0, (0, 0, 100, 60)), (1.05, (0, 0, 40, 60)), (1.2, (65, 10, 95, 60)))]
    first, duplicate, other = [t.id for _, t in tr.update(ds, 0.0)]
    assert first == duplicate == tr.current(2) and other != first
    tr.update([], 0.1)
    assert set(tr.tracks) == {first, other}
    assert tr.tracks[first].apart == {other} and tr.tracks[other].apart == {first}


def test_parts_keep_a_known_footprint_but_a_small_neighbour_has_its_own_id():
    tr = Tracker()
    big = Detection("bag", 1.0, 0.0, 0.30, box=(10, 10, 60, 60), score=0.7)
    tr.update([big], 0)
    (_, parent), = tr.update([big], 0.2)
    parts = [Detection("bag", x, 0, 0.05, box=b) for x, b in
             ((0.82, (10, 15, 20, 58)), (1.15, (45, 15, 55, 57)))]
    assert {t.id for _, t in tr.update(parts, 0.4)} == {parent.id}
    assert parent.x == 1.0 and parent.radius == 0.30
    small = Detection("bag", 1.35, 0, 0.07, box=(64, 33, 76, 47), score=0.95)
    for now in (0.6, 0.8):
        ids = {id(d): t.id for d, t in tr.update([small, big], now)}
        assert ids[id(big)] == parent.id and ids[id(small)] != parent.id


def test_cross_camera_range_error_can_move_a_part_along_its_ray_but_not_across_it():
    full = Detection("rack", 0.0, 2.0, 0.23, camera="front", box=(10, 15, 70, 85))
    for x, y, same in ((0.0, 1.5, True), (0.5, 2.0, False)):
        leg = Detection("rack", x, y, 0.05, camera="rear", cut=True, box=(0, 20, 10, 85),
                        sigma=0.4, normal=(1.0, 0.0))
        for first, second in ((full, leg), (leg, full)):
            tr = Tracker()
            for k, d in enumerate((first, second, first, second)):
                tr.update([d], k * 0.2)
            assert len(tr.objects()) == (1 if same else 2)
            if not same:
                # After the turn the leg has a whole outline in front. Its noisy location alone would merge them.
                reacquired = Detection("rack", 0.36, 2.0, 0.23, camera="front", sigma=0.4, box=(50, 15, 110, 85))
                ids = {t.id for t in tr.objects()}
                tr.update([reacquired], 1.0)
                assert {t.id for t in tr.objects()} == ids


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


def test_sightings_made_while_driving_join_their_thing_instead_of_making_twins():
    rng = np.random.default_rng(3)
    tr = Tracker()
    for k in range(40):  # the sofa at (-1.4, 1.8), seen while driving: 0.25 m of scatter, sometimes called a chair
        x, y = -1.4 + rng.normal(0, 0.25), 1.8 + rng.normal(0, 0.25)
        tr.update([Detection("sofa" if k % 4 else "chair", x, y, 0.5, score=0.6, sigma=0.3)], k * 0.2)
    assert len(tr.objects()) == 1 and math.dist((tr.objects()[0].x, tr.objects()[0].y), (-1.4, 1.8)) < 0.2
    still = Tracker()  # seen precisely while still, a backpack 0.45 m from a chair is a backpack
    for k in range(5):
        still.update([Detection("chair", 2.2, 1.2, 0.35, score=0.8, sigma=0.03),
                      Detection("backpack", 2.2, 1.65, 0.2, score=0.7, sigma=0.03)], k * 0.2)
    assert sorted(t.label for t in still.objects()) == ["backpack", "chair"]


def test_a_crop_for_naming_shows_only_its_thing():
    from robot.ros.perceive import _only

    picture = np.zeros((200, 300, 3), dtype=np.uint8)
    picture[50:150, 100:200] = (30, 60, 200)  # a blue chair
    picture[110:140, 120:150] = (220, 30, 30)  # a red ball in front of it
    chair = _only(picture, (100, 50, 200, 150), [(120, 110, 150, 140)])
    assert not ((chair[..., 0] > 200) & (chair[..., 1] < 50)).any()  # no red left in the chair's picture
    ball = _only(picture, (120, 110, 150, 140), [(100, 50, 200, 150)])
    assert ((ball[..., 0] > 200) & (ball[..., 1] < 50)).any()  # the bigger chair box is not painted over the ball


def test_camera_project_and_point_round_trip():
    for cam in RIG.values():
        forward = 2.0
        bx = cam.x + forward * math.cos(cam.yaw)
        by = cam.y + forward * math.sin(cam.yaw)
        bz = cam.z
        uv = cam.project(bx, by, bz)
        assert uv is not None
        u, v = uv
        assert abs(u - cam.width / 2) < 1.0 and abs(v - cam.height / 2) < 1.0
        pt = cam.point(u, v, forward)
        assert math.dist(pt, (bx, by, bz)) < 1e-4

        bx_behind = cam.x - 1.0 * math.cos(cam.yaw)
        by_behind = cam.y - 1.0 * math.sin(cam.yaw)
        assert cam.project(bx_behind, by_behind, bz) is None


def test_classify_tof_grid_hazards():
    from robot.perception.depth import classify_tof_grid

    expected = np.full((8, 8), 0.50, dtype=np.float32)

    # 1. Clear floor (abs(dz) < 0.004)
    h_clear = classify_tof_grid(expected.copy(), expected)
    assert h_clear.kind == "clear" and not h_clear.torque_boost

    # 2. Traversable step up (1.5 cm rug, dz = +0.015) -> torque_boost True
    grid_rug = expected.copy()
    grid_rug[4, 4] = 0.485
    h_rug = classify_tof_grid(grid_rug, expected)
    assert h_rug.kind == "traversable" and h_rug.torque_boost
    assert abs(h_rug.delta_z - 0.015) < 1e-3

    # 3. Traversable step down (1.5 cm step-down, dz = -0.015) -> torque_boost False
    grid_down = expected.copy()
    grid_down[4, 4] = 0.515
    h_down = classify_tof_grid(grid_down, expected)
    assert h_down.kind == "traversable" and not h_down.torque_boost
    assert abs(h_down.delta_z - (-0.015)) < 1e-3

    # 4. Cliff drop (6 cm drop, dz = -0.06 < -0.04)
    grid_cliff = expected.copy()
    grid_cliff[6, 3] = 0.56
    h_cliff = classify_tof_grid(grid_cliff, expected)
    assert h_cliff.kind == "cliff" and h_cliff.delta_z <= -0.04

    # 5. Obstacle (8 cm obstacle, dz = +0.08 > 0.04)
    grid_obs = expected.copy()
    grid_obs[2, 3] = 0.42
    h_obs = classify_tof_grid(grid_obs, expected)
    assert h_obs.kind == "obstacle" and h_obs.delta_z >= 0.04


def _floor_with_box(cam, box_at=None, box_top=0.2, box_bottom=0.0, cols=None):  # noqa: ANN001, ANN202
    """True depth (along the lens axis) of a level floor seen by `cam` at 640x360, far wall at 6 m, and optionally a
    flat face `box_at` metres ahead spanning box_bottom..box_top above the floor in columns `cols`."""
    c = cam.at(640, 360)
    v = (np.arange(360, dtype=np.float32)[:, None] + 0.5 - 180) / c.focal * np.ones((1, 640), np.float32)
    depth = np.where(v > 1e-3, c.z / np.maximum(v, 1e-3), np.inf).astype(np.float32)
    depth = np.minimum(depth, 6.0)
    if box_at is not None:
        z = c.z - v * box_at  # height of the face's point in each row
        face = (z >= box_bottom) & (z <= box_top)
        face[:, :] &= np.isin(np.arange(640), cols)[None, :]
        depth[face & (depth > box_at)] = box_at
    return depth


def test_fit_to_floor_recovers_metres_from_relative_inverse_depth():
    from robot.perception.depth import fit_to_floor

    cam = RIG["front_camera"]
    true = _floor_with_box(cam, box_at=1.0, box_top=0.3, cols=np.arange(300, 340))
    relative = 0.37 / true + 0.05  # a network's output: inverse depth, up to scale and shift
    metres = fit_to_floor(relative.astype(np.float32), cam)
    near = true < 3
    assert np.median(np.abs(metres[near] - true[near]) / true[near]) < 0.02
    assert abs(float(metres[200, 320]) - 1.0) < 0.05  # the box's face, not only the floor


def test_scan_ranges_the_foot_of_a_thing_on_the_floor_and_sees_clear_floor_elsewhere():
    from robot.perception.depth import synthetic_scan

    cam = RIG["front_camera"]
    depth = _floor_with_box(cam, box_at=1.5, cols=np.arange(300, 340))
    ranges, angle_min, step = synthetic_scan({"front_camera": depth})
    ahead = int(round(-angle_min / step))
    assert abs(float(np.nanmin(ranges[ahead - 2:ahead + 3])) - (1.5 + cam.x)) < 0.05  # from the robot's middle
    assert ranges[ahead + 30] > 5 and ranges[ahead - 30] > 5  # beside the box: clear floor up to the far wall


def test_scan_finds_the_foot_of_a_face_whose_depth_is_misjudged_and_ignores_a_rug_edge():
    from robot.perception.depth import synthetic_scan

    cam = RIG["front_camera"]
    cols = np.arange(300, 340)
    true = _floor_with_box(cam, box_at=1.5, box_top=0.3, cols=cols)
    face = np.zeros_like(true, bool)
    face[:, cols] = np.abs(true[:, cols] - 1.5) < 1e-6
    for scale in (0.7, 2.0):  # a network that puts the whole face nearer, or twice as far, than it is
        ranges, angle_min, step = synthetic_scan({"front_camera": np.where(face, true * scale, true)})
        ahead = int(round(-angle_min / step))
        # within two rows (0.09 m each at 1.5 m): the rate is measured over two
        assert abs(float(np.nanmin(ranges[ahead - 2:ahead + 3])) - (1.5 + cam.x)) < 0.2
    rug = _floor_with_box(cam, box_at=1.0, box_top=0.02, cols=cols)  # 2 cm: the wheels roll over it
    ranges, _, _ = synthetic_scan({"front_camera": rug})
    assert ranges[ahead] > 5


def test_scan_sees_under_a_table_top_and_ranges_a_low_shelf_by_its_depth():
    from robot.perception.depth import synthetic_scan

    cam = RIG["front_camera"]
    ahead_cols = np.arange(300, 340)
    table = _floor_with_box(cam, box_at=1.0, box_bottom=0.35, box_top=0.6, cols=ahead_cols)
    ranges, angle_min, step = synthetic_scan({"front_camera": table})
    ahead = int(round(-angle_min / step))
    assert ranges[ahead] > 5  # a table top 35 cm up: the robot passes under it, and sees the wall beyond
    shelf = _floor_with_box(cam, box_at=1.0, box_bottom=0.12, box_top=0.25, cols=ahead_cols)
    ranges, _, _ = synthetic_scan({"front_camera": shelf})
    assert abs(float(ranges[ahead]) - (1.0 + cam.x)) < 0.05  # a shelf's edge 12 cm up: its depth, not the floor's


def test_estimate_pose_and_resolve_ray():
    from robot.perception.pose import estimate_pose_from_keypoints, resolve_ray_target

    cam = RIG["front_camera"]
    robot_pose = (0.0, 0.0, 0.0)
    depth_m = 2.0

    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 1.0

    focal = cam.focal
    v_nose = cam.height / 2 - (1.38 / depth_m) * focal
    kpts[0] = [cam.width / 2, v_nose, 1.0]

    v_sh = cam.height / 2 - (1.18 / depth_m) * focal
    u_l_sh = cam.width / 2 - (0.20 / depth_m) * focal
    u_r_sh = cam.width / 2 + (0.20 / depth_m) * focal
    kpts[5] = [u_l_sh, v_sh, 1.0]
    kpts[6] = [u_r_sh, v_sh, 1.0]

    v_hip = cam.height / 2 - (0.68 / depth_m) * focal
    kpts[11] = [u_l_sh, v_hip, 1.0]
    kpts[12] = [u_r_sh, v_hip, 1.0]

    # 1. Standing posture
    kpts[9] = [u_l_sh, v_hip, 1.0]
    kpts[10] = [u_r_sh, v_hip, 1.0]
    pose_stand = estimate_pose_from_keypoints(kpts, cam, depth_m, robot_pose)
    assert pose_stand.posture == "standing"

    # 2. Pointing posture
    u_point = u_r_sh + (0.45 / depth_m) * focal
    kpts[10] = [u_point, v_sh, 1.0]
    pose_point = estimate_pose_from_keypoints(kpts, cam, depth_m, robot_pose)
    assert pose_point.posture == "pointing"
    assert pose_point.pointing_ray is not None

    target_track = Track(2, "chair", 2.0, -1.0, 0.3, 0.0, 0.0)
    other_track = Track(3, "table", 2.0, 2.0, 0.4, 0.0, 0.0)
    found = resolve_ray_target(pose_point.pointing_ray, [target_track, other_track])
    assert found is target_track

    # 3. Floor pointing
    down_ray = ((2.0, 0.0, 1.2), (0.707, 0.0, -0.707))
    floor_hit = resolve_ray_target(down_ray, [])
    assert floor_hit is not None
    assert abs(floor_hit[0] - 3.2) < 0.1 and abs(floor_hit[1] - 0.0) < 0.1

    # 4. Waving posture
    kpts[10] = [u_r_sh, v_sh - (0.25 / depth_m) * focal, 1.0]
    pose_wave = estimate_pose_from_keypoints(kpts, cam, depth_m, robot_pose)
    assert pose_wave.posture == "waving"

    # 5. Crouching posture
    v_crouch = cam.height / 2 - (0.80 / depth_m) * focal
    kpts[0, 1] = v_crouch
    kpts[10] = [u_r_sh, v_hip, 1.0]
    pose_crouch = estimate_pose_from_keypoints(kpts, cam, depth_m, robot_pose)
    assert pose_crouch.posture == "crouching"


def test_draw_set_of_mark_badges():
    from robot.perception.overlay import draw_set_of_mark

    cam = RIG["front_camera"]
    rgb = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)

    boxes_and_ids = [((100, 100, 250, 300), 1), ((500, 200, 700, 450), 2)]

    annotated = draw_set_of_mark(rgb, boxes_and_ids, cam=cam)
    assert annotated.shape == (cam.height, cam.width, 3)
    assert (annotated > 0).any()
    assert (annotated[100:120, 100:150] > 0).any()


def test_a_moving_track_predicts_ahead():
    tr = Tracker()
    tr.update([Detection("person", 1.0, 0.0, 0.3)], 0.0)
    tr.update([Detection("person", 1.1, 0.0, 0.3)], 0.1)
    person = tr.objects()[0]
    assert person.vx > 0.2
    pred_x, pred_y = person.predict(1.0)
    assert pred_x > person.x


def test_locate_mask_guided_localization():
    from robot.perception.locate import locate

    cam = RIG["front_camera"]
    h, w = cam.height, cam.width
    depth = np.full((h, w), np.nan, dtype=np.float32)

    # Box region [300:450, 500:650]
    box = (500.0, 300.0, 650.0, 450.0)
    r0, r1 = int(box[1]), int(box[3])
    c0, c1 = int(box[0]), int(box[2])

    # Unmasked region has close clutter/noise at 0.5 m
    depth[r0:r1, c0:c1] = 0.5

    # True object isolated by mask is at 2.0 m
    depth[350:410, 540:600] = 2.0
    mask_patch = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    mask_patch[350 - r0:410 - r0, 540 - c0:600 - c0] = True

    # Without mask, locate picks up the 0.5 m clutter
    place_unmasked = locate(cam, box, depth=depth)
    assert place_unmasked is not None
    assert place_unmasked.how == "depth"
    dist_unmasked = math.hypot(place_unmasked.x, place_unmasked.y)
    assert dist_unmasked < 1.3

    # With mask, locate isolates the 2.0 m object
    place_masked = locate(cam, box, depth=depth, mask=mask_patch)
    assert place_masked is not None
    assert place_masked.how == "depth"
    dist_masked = math.hypot(place_masked.x, place_masked.y)
    assert dist_masked > 2.0


def test_draw_set_of_mark_with_labels_and_letters():
    from robot.perception.overlay import draw_set_of_mark

    cam = RIG["front_camera"]
    rgb = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)

    # Pass (box, id, label) tuples with uppercase/lowercase letters beyond G
    boxes_and_ids = [
        ((100, 100, 250, 300), 1, "table"),
        ((400, 150, 600, 350), 2, "couch"),
        ((700, 200, 850, 400), 3, "lamp:01"),
    ]

    annotated = draw_set_of_mark(rgb, boxes_and_ids, cam=cam)
    assert annotated.shape == rgb.shape
    # Ensure badges are drawn
    assert (annotated[100:130, 100:250] > 0).any()
    assert (annotated[150:180, 400:550] > 0).any()


def test_tracker_person_merge_distance_and_cut():
    tr = Tracker()
    # Two person detections separated by 0.32 m (< 0.35 m threshold)
    d1 = Detection("person", 1.0, 0.0, 0.25)
    d2 = Detection("person", 1.32, 0.0, 0.25)
    tr.update([d1, d2], 0.0)
    # Should merge into a single person track
    persons = [o for o in tr.objects() if o.person]
    assert len(persons) == 1

    # Two person detections separated by 0.60 m with one cut by edge (< 0.75 m cut threshold)
    tr2 = Tracker()
    d3 = Detection("person", 2.0, 0.0, 0.25, cut=True)
    d4 = Detection("person", 2.60, 0.0, 0.25, cut=False)
    tr2.update([d3, d4], 0.0)
    persons2 = [o for o in tr2.objects() if o.person]
    assert len(persons2) == 1


def test_the_vision_models_no_drops_a_detection():
    import threading
    from unittest.mock import MagicMock

    from robot.perception.tracker import Track, Tracker
    from robot.ros.perceive import Perception

    perc = object.__new__(Perception)
    perc._stop = False
    perc.lock = threading.Lock()
    perc.tracker = Tracker()
    perc.crops, perc.doubts = {}, {}
    perc.detector = MagicMock()
    perc.detector.vocabulary = ["chair", "table", "cup"]

    # Track with hits=2 so confirmed=True and to_name picks it up
    track = Track(1, "chair", 1.0, 0.0, 0.3, 0.0, 0.0, hits=2)
    perc.tracker.tracks[1] = track
    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    perc.crops[1] = crop

    def says_no_then_stop(*args, **kwargs):  # "Is there a chair in the middle of this picture?" mostly no
        perc._stop = True
        return 0.3

    brain = MagicMock()
    brain.yes_probability.side_effect = says_no_then_stop
    perc.brain = brain

    perc._verify_loop()

    assert track.verified is True
    assert track.rejected is True
    assert 1 not in [o.id for o in perc.tracker.objects()]


def test_locate_wide_angle_depth_overlap_consistency():
    from robot.perception.locate import locate

    cam = RIG["front_camera"]
    h, w = cam.height, cam.width

    # Object at bearing ~ +50 deg (off-axis)
    # Target pixel u near the edge
    u_edge = int(w * 0.15)
    box = (float(u_edge - 40), 300.0, float(u_edge + 40), 420.0)
    r0, r1 = int(box[1]), int(box[3])
    c0, c1 = int(box[0]), int(box[2])

    depth = np.full((h, w), np.nan, dtype=np.float32)
    # Fill object with true pinhole depth 1.5 m
    depth[r0:r1, c0:c1] = 1.5

    place = locate(cam, box, depth=depth)
    assert place is not None
    assert place.how == "depth"

    # Radial distance should not blow up: seen is radial range, reach is radial distance
    # With secant correction and radial projection, distance to base frame is ~3.2-3.3 m (not inflated 2x to 7.0m)
    reach_actual = math.hypot(place.x - cam.x, place.y - cam.y)
    assert 2.8 <= reach_actual <= 3.6


def test_ranging_uses_the_same_camera_and_exposure_even_if_another_frame_is_newer():
    scan = SyntheticScanPublisher(publish=False)

    def stamp(ns):
        return SimpleNamespace(sec=1_800_000_000, nanosec=ns)

    front, rear = np.full((2, 2), 1.0), np.full((2, 2), 3.0)
    scan.ingest_depth("front_camera", stamp(1), front)
    scan.ingest_depth("rear_right_camera", stamp(2), rear)
    scan.ingest_depth("front_camera", stamp(200_000_001), rear)
    assert scan.depth_for("front_camera", stamp(1))[0] is front
    assert scan.depth_for("rear_right_camera", stamp(2))[0] is rear
    assert scan.depth_for("front_camera", stamp(2))[0] is None


def test_a_thing_the_detector_stops_finding_is_gone_only_when_the_vision_model_agrees():
    import threading
    import time
    from unittest.mock import MagicMock

    from robot.ros.perceive import Perception

    def perception(answer: float) -> tuple:
        perc = object.__new__(Perception)
        perc.lock, perc.tracker, perc._stop = threading.Lock(), Tracker(), False
        perc.pose_sigma, perc.scan_bridge = (0.05, 0.05), MagicMock(turn_rate=0.0)
        perc.scan_bridge.depth_for.return_value = (None, 0.0)
        perc.boxes, perc.detected, perc.sightings, perc.crops, perc.doubts = {}, {}, [], {}, {}
        perc.timings, perc.looks, perc.depth_pairs, perc.scans, perc.scan = [], {}, [], [], None
        perc.pose_at, perc.tilt_at = (lambda s: (0.0, 0.0, 0.0)), (lambda s: (0.0, 0.0))
        perc.detector = MagicMock(return_value=[], vocabulary=["person", "bike"])  # it finds no bike any more
        cam = RIG["front_camera"]
        perc.frames = {"front_camera": (SimpleNamespace(sec=1, nanosec=0), np.zeros((cam.height, cam.width, 3),
                                                                                   np.uint8))}
        bike = Track(1, "bike", 2.0, 0.0, 0.4, 0.0, time.monotonic() - 10.0, hits=5)  # 2 m ahead, last seen 10 s ago
        perc.tracker.tracks[1] = bike
        asked = []

        def yes_probability(messages: list) -> float:
            asked.append(messages[0]["content"][-1]["text"])
            perc._stop = True  # one look, then the loop ends
            return answer

        perc.brain = SimpleNamespace(yes_probability=yes_probability)
        return perc, bike, asked

    for answer, present in ((0.9, True), (0.1, False)):
        perc, bike, asked = perception(answer)
        for _ in range(GONE_AFTER):
            perc.process("front_camera")
        assert bike.present and list(perc.doubts) == [1]  # the detector's misses alone do not make it gone
        assert perc.doubts[1][1].shape[0] >= 16  # the model is shown the part of the picture where it should be
        perc._verify_loop()
        assert asked == ["Is there a bike in the middle of this picture? Answer yes or no."]
        assert bike.present is present and not perc.doubts


def test_perception_scan_surface_consistency():
    import threading
    from unittest.mock import MagicMock

    from robot.perception.detector import Seen
    from robot.perception.rig import RIG
    from robot.perception.tracker import Tracker
    from robot.ros.perceive import Perception

    perc = object.__new__(Perception)
    perc.lock = threading.Lock()
    perc.tracker = Tracker()
    perc.pose = (0.0, 0.0, 0.0)
    perc.pose_sigma = (0.05, 0.05)
    perc.scan_bridge = MagicMock()
    perc.scan_bridge.turn_rate = 0.0
    perc.boxes = {}
    perc.detected = {}
    perc.sightings = []
    perc.crops = {}
    perc.tof_grids = {}

    cam = RIG["front_camera"]
    h, w = cam.height, cam.width
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 3.5, dtype=np.float32)  # depth camera sees object at 3.5m

    # Scan has a solid obstacle/wall at 1.5m straight ahead
    scan_ranges = np.full(360, 1.5, dtype=np.float32)
    scan = (scan_ranges, -math.pi, 2 * math.pi / 360)

    stamp = SimpleNamespace(sec=1, nanosec=0)
    perc.frames = {"front_camera": (stamp, rgb)}
    perc.scans = [(1.0, scan)]
    perc.scan = scan

    # Seen object in middle of frame at 3.5m, which lies behind the 1.5m scan surface
    seen = [Seen("chair", 0.9, (float(w // 2 - 30), float(h // 2 - 30), float(w // 2 + 30), float(h // 2 + 30)))]
    perc.pose_at = lambda s: (0.0, 0.0, 0.0)
    perc.tilt_at = lambda s: (0.0, 0.0)
    perc.timings = []
    perc.looks = {}
    perc.depth_pairs = []
    perc.scan_bridge.depth_for.return_value = (depth, 0.0)
    perc.detector = MagicMock(return_value=seen)
    perc.detector.estimate_depth.return_value = None

    perc.process("front_camera")

    # Detection behind the scan obstacle should have been discarded
    assert len(perc.tracker.objects()) == 0


def test_tracker_cut_sighting_downweighting_and_furniture_merge():
    tr = Tracker()

    # Sighting 1: whole-view chair
    d1 = Detection("chair", 2.0, 0.0, 0.3, sigma=0.05, cut=False)
    tr.update([d1], 0.0)
    tr.update([d1], 0.1)
    track1 = tr.objects()[0]
    orig_x = track1.x
    assert track1.hits == 2
    assert track1.sized is True and not track1.cut

    # Sighting 2: edge-cut sighting shifted by 0.2m
    d2 = Detection("chair", 2.2, 0.0, 0.3, sigma=0.05, cut=True)
    tr.update([d2], 0.2)

    # Position change should be small (gain capped at 0.15)
    assert abs(track1.x - orig_x) < 0.06

    # Test cross-camera furniture merge: table and desk within radius overlap
    tr2 = Tracker()
    d_table = Detection("table", 3.0, 1.0, 0.45, sigma=0.05)
    d_box = Detection("box", 3.2, 1.0, 0.40, sigma=0.05)
    tr2.update([d_table], 0.0)
    tr2.update([d_box], 0.0)

    # _merge is called at end of update; box and table should merge into 1 track
    furniture = [o for o in tr2.objects() if not o.person]
    assert len(furniture) == 1


def test_tof_sees_a_low_box_the_cameras_take_for_floor():
    import math

    from robot.perception.depth import tof_scan
    from robot.ros.model import TOF

    x0, _, z0, pitch, hfov, _ = TOF["front"]
    t = math.tan(hfov / 2)
    grid = np.zeros((8, 8), np.float32)
    for row in range(8):  # a level floor in every zone
        down = ((row + 0.5) - 4) / 4 * t
        fall = math.sin(pitch) + down * math.cos(pitch)
        grid[row, :] = z0 / fall if fall > 1e-3 else np.inf
    assert np.all(np.isnan(tof_scan(grid, "front")))  # floor only: nothing to report
    grid[2, 3:5] = 0.30  # something 5 cm tall, about 0.45 m ahead, in the zones that looked past it
    ranges = tof_scan(grid, "front")
    ahead = 180
    assert np.isfinite(ranges[ahead - 3:ahead + 4]).any()
    assert 0.35 < float(np.nanmin(ranges)) < 0.5


def test_locate_crop_ignores_side_edge_clutter_ranges_floor_and_falls_back_to_scan():
    from robot.perception.locate import locate_crop

    cam = RIG["front_camera"]
    h, w = cam.height, cam.width
    # Standing target at 3.0 m in middle columns, with a 1.0 m foreground clutter strip on the crop's right edge
    depth = np.full((h, w), 3.0, dtype=np.float32)
    depth[:, 700:770] = 1.0
    place = locate_crop(cam, (512, 0, 768, 430), depth=depth, on_floor=False)
    assert place is not None and place.how == "depth" and place.radius >= 0.15 and place.z > 0.5
    assert 2.8 <= math.hypot(place.x - cam.x, place.y - cam.y) <= 3.3

    # Floor-flat target (rug): r^2 weighting over [0.4, 3.5] m places it at the geometric midpoint (~1.95 m)
    rows = np.arange(h, dtype=np.float32)[:, None]
    down = np.maximum((rows - h / 2) / cam.focal, 1e-3)
    floor_depth = np.broadcast_to(cam.z / down, (h, w)).astype(np.float32).copy()
    rug = locate_crop(cam, (512, int(0.45 * h), 768, h), depth=floor_depth, on_floor=True)
    assert rug is not None and rug.how == "floor" and rug.radius == 0.0 and rug.z == 0.0
    assert 1.7 <= math.hypot(rug.x - cam.x, rug.y - cam.y) <= 2.2

    # Scan fallback when depth is missing
    ranges = np.full(360, 2.4, dtype=np.float32)
    via_scan = locate_crop(cam, (560, 100, 720, 400), depth=None, scan=(ranges, -math.pi, 2 * math.pi / 360))
    assert via_scan is not None and via_scan.how == "lidar"
    assert math.hypot(via_scan.x, via_scan.y) - via_scan.radius == pytest.approx(2.4)

    # Floor-ray fallback when both depth and scan are missing
    via_ray = locate_crop(cam, (560, int(0.45 * h), 720, h), depth=None, scan=None, on_floor=True)
    assert via_ray is not None and via_ray.how == "floor" and via_ray.radius == 0.0


def test_locate_crop_places_the_lidar_surface_on_the_returning_beam():
    from robot.perception.locate import locate_crop

    ranges = np.full(360, np.inf, dtype=np.float32)
    ranges[241:248] = [2.706, 2.709, 2.778, 1.235, 1.247, 1.241, 1.232]
    place = locate_crop(RIG["rear_left_camera"], (894, 0, 1150, 432),
                        scan=(ranges, -math.pi, math.tau / 360))
    assert place is not None and place.how == "lidar"
    assert math.degrees(math.atan2(place.y, place.x)) == pytest.approx(67)
    assert math.hypot(place.x, place.y) - place.radius == pytest.approx(1.232)


def test_locate_in_picture_refines_the_strongest_third_when_every_third_is_positive():
    from robot.ros.perceive import locate_in_picture

    rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    scores = iter([0.6792, 0.8221, 0.9941, 0.9999, 0.9759, 0.9919, 0.7078])
    hit = locate_in_picture(SimpleNamespace(yes_probability=lambda _: next(scores)), rgb, "table")
    assert hit == ((894, 0, 1150, 432), False)


def test_locate_in_picture_localises_by_crops_centres_straddling_objects_and_rejects_false_positives():
    from robot.ros.perceive import locate_in_picture

    # Paint a distinctive bright patch in the upper-left ([x=150..350, y=50..250]) of a dark 1280x720 frame
    rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    rgb[50:250, 150:350] = 255

    def make_brain(predicate):
        import base64

        import cv2

        def yes_prob(messages):
            url = messages[0]["content"][0]["image_url"]["url"]
            raw = base64.b64decode(url.split(",", 1)[1])
            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            return predicate(arr)

        return SimpleNamespace(yes_probability=yes_prob)

    # Object straddles the boundary between [0, 0.2] and [0.2, 0.4] -> weighted centre near x=250, upper half wins
    brain = make_brain(lambda arr: 0.95 if (arr > 200).mean() > 0.05 else 0.05)
    hit = locate_in_picture(brain, rgb, "computer monitor")
    assert hit is not None
    (x0, y0, x1, y1), on_floor = hit
    assert not on_floor and y0 == 0 and y1 < 500
    assert 150 <= (x0 + x1) / 2 <= 350

    # Floor-flat object in lower-middle -> on_floor is True
    rug_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    rug_rgb[450:680, 480:800] = 255
    hit_rug = locate_in_picture(brain, rug_rgb, "rug")
    assert hit_rug is not None and hit_rug[1] is True and hit_rug[0][1] >= 300

    # Narrow standing object centred at u=0.50 (both overlapping halves see it whole; non-overlapping would slice it)
    tall_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    tall_rgb[180:640, 615:665] = 255
    tall_brain = make_brain(lambda arr: 0.95 if (arr > 200).any(axis=2).any(axis=0).sum() >= 20 else 0.05)
    hit_tall = locate_in_picture(tall_brain, tall_rgb, "umbrella")
    assert hit_tall is not None and hit_tall[1] is False and 580 <= (hit_tall[0][0] + hit_tall[0][2]) / 2 <= 700

    # Wide floor surface spanning left + middle thirds -> weighted centre near u=0.35 (x=448), skips halves
    wide_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    wide_rgb[450:680, 40:820] = 255
    hit_wide = locate_in_picture(brain, wide_rgb, "rug")
    assert hit_wide is not None and hit_wide[1] is True and 380 <= (hit_wide[0][0] + hit_wide[0][2]) / 2 <= 520

    # Blank wall: one third and both halves say yes, vertical split says no -> rejected
    wall_scores = iter([0.8, 0.05, 0.05, 0.8, 0.8, 0.05, 0.05])
    assert locate_in_picture(SimpleNamespace(yes_probability=lambda _m: next(wall_scores)), rgb, "umbrella") is None


def test_visual_tracks_are_not_evicted_by_detector_misses():
    import threading
    import time
    from unittest.mock import MagicMock

    from robot.ros.perceive import Perception

    perc = object.__new__(Perception)
    perc.lock, perc.tracker = threading.Lock(), Tracker()
    perc.pose_sigma, perc.scan_bridge = (0.05, 0.05), MagicMock(turn_rate=0.0)
    perc.scan_bridge.depth_for.return_value = (None, 0.0)
    perc.boxes, perc.detected, perc.sightings, perc.crops, perc.doubts = {}, {}, [], {}, {}
    perc.timings, perc.looks, perc.depth_pairs, perc.scans, perc.scan = [], {}, [], [], None
    perc.pose_at, perc.tilt_at = (lambda s: (0.0, 0.0, 0.0)), (lambda s: (0.0, 0.0))
    perc.detector = MagicMock(return_value=[], vocabulary=["person", "kitchen"])
    perc.brain = MagicMock()
    cam = RIG["front_camera"]
    perc.frames = {"front_camera": (SimpleNamespace(sec=1, nanosec=0), np.zeros((cam.height, cam.width, 3), np.uint8))}
    kitchen = Track(1, "kitchen", 2.0, 0.0, 0.3, 0.0, time.monotonic() - 10.0, hits=2, verified=True, sure=True,
                    visual=True)
    perc.tracker.tracks[1] = kitchen
    for _ in range(GONE_AFTER + 2):
        perc.process("front_camera")
    assert kitchen.present and not perc.doubts

    # A new live visual track near an earlier rejected or absent ghost must not merge into the dead track
    now = time.monotonic()
    perc.tracker.tracks[10] = Track(10, "umbrella", 2.0, 0.0, 0.2, now - 10, now - 10, hits=2, verified=True,
                                    rejected=True)
    perc.tracker.tracks[11] = Track(11, "umbrella", 2.05, 0.0, 0.2, now, now, hits=2, verified=True, sure=True,
                                    visual=True)
    perc.tracker.tracks[12] = Track(12, "umbrella", 2.0, 0.0, 0.2, now - 5, now - 5, hits=2, verified=True,
                                    present=False, visual=True)
    perc.tracker._merge([])
    assert perc.tracker.current(11) == 11 and perc.tracker.tracks[11].present and not perc.tracker.tracks[11].rejected


def test_badge_deoverlap_and_detector_max_asked():
    from robot.perception.detector import MAX_ASKED, Detector
    from robot.perception.overlay import _place_badge, draw_set_of_mark

    placed: list[tuple[int, int, int, int]] = []
    x1, y1 = _place_badge(100, 100, 80, 20, 640, 360, placed)
    r1 = (x1, y1, x1 + 80, y1 + 20)
    placed.append(r1)
    x2, y2 = _place_badge(105, 100, 80, 20, 640, 360, placed)
    r2 = (x2, y2, x2 + 80, y2 + 20)
    assert r2[2] <= r1[0] or r2[0] >= r1[2] or r2[3] <= r1[1] or r2[1] >= r1[3]
    assert 0 <= r2[0] and r2[2] <= 640 and 0 <= r2[1] and r2[3] <= 360

    rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    out = draw_set_of_mark(rgb, [((100, 100, 180, 220), 19, "shoes"), ((105, 100, 190, 220), 20, "shoes")])
    assert out.shape == rgb.shape and (out > 0).any()

    # Even with plenty of GPU prompt slots, Detector keeps only MAX_ASKED recent asked words so stale tasks don't leak.
    fake_server = SimpleNamespace(slots=12)
    det = Detector(server=fake_server, always=("person",))
    for w in ("umbrella", "shoes", "stool", "mug", "bicycle"):
        det.watch(w)
    assert len(det.asked) == MAX_ASKED == 3
    assert det.asked == ["stool", "mug", "bicycle"]
    assert det.vocabulary == ["person", "stool", "mug", "bicycle"]


