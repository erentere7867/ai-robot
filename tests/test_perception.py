"""Camera geometry, detection parsing and the spatial map, against simulator ground truth."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from robot.body.hal import BodyPose
from robot.cognition.backend import ScriptedBackend
from robot.core.clock import ReplayClock
from robot.perception.camera import Frame
from robot.perception.change import ChangeDetector
from robot.perception.detector import Detection, VLMDetector, parse_detections
from robot.perception.geometry import CameraModel
from robot.world.sandbox import SimulationSandbox
from robot.world.spatial_map import SpatialMap, TrackedObject, head_noun, label_score, labels_match

MODEL = CameraModel(512, 384)


# ---- geometry ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("pose", [BodyPose(), BodyPose(120, -80, 37, -12), BodyPose(-400, 300, -150, 20)])
def test_project_and_floor_point_round_trip(pose):
    yaw = math.radians(pose.heading_deg)
    for dist, side in [(500, 0), (900, 200), (1500, -400), (2500, 300)]:
        x = pose.x_mm + dist * math.cos(yaw) - side * math.sin(yaw)
        y = pose.y_mm + dist * math.sin(yaw) + side * math.cos(yaw)
        uv = MODEL.project(pose, x, y, 0.0)
        assert uv is not None
        back = MODEL.floor_point(pose, *uv)
        assert back is not None
        assert math.hypot(back[0] - x, back[1] - y) < 1e-6
    behind = (pose.x_mm - 800 * math.cos(yaw), pose.y_mm - 800 * math.sin(yaw))
    assert MODEL.project(pose, *behind, 0.0) is None
    assert MODEL.floor_point(pose, 0.5, 0.05) is None  # above the horizon


def test_bearing_and_view_helpers_agree():
    pose = BodyPose(0, 0, 0)
    assert MODEL.bearing_deg(0.5) == pytest.approx(0)
    assert MODEL.bearing_deg(0.0) == pytest.approx(MODEL.hfov_deg / 2, abs=0.01)
    assert MODEL.bearing_deg(1.0) < 0  # right of centre is negative (CCW positive)
    assert MODEL.in_view(pose, 800, 0)
    assert not MODEL.in_view(pose, -800, 0) and not MODEL.in_view(pose, 800, 900)
    assert MODEL.scaled(256, 192).focal_px == pytest.approx(MODEL.focal_px / 2)


def gt_boxes(sandbox: SimulationSandbox, pose: BodyPose):
    w, h = sandbox.cam_width, sandbox.cam_height
    for d in sandbox.detect_objects(pose):
        x1, y1, x2, y2 = d.bbox_px
        yield d, (x1 / w, y1 / h, x2 / w, y2 / h)


@pytest.mark.parametrize("pose", [BodyPose(), BodyPose(0, 0, 30), BodyPose(200, 100, 20), BodyPose(0, 0, 180),
                                  BodyPose(0, 0, 0, -15), BodyPose(-300, -200, 45, 10)])
def test_locate_box_of_ground_truth_boxes_lands_on_true_centres(pose):
    sandbox = SimulationSandbox()
    checked = 0
    for det, bbox in gt_boxes(sandbox, pose):
        if bbox[0] <= 0.0 or bbox[2] >= 0.995 or bbox[3] >= 0.995:
            continue  # cut off by the image border: the box no longer shows the object's extent
        obj = sandbox.get_object(det.id)
        located = MODEL.locate_box(pose, bbox)
        assert located is not None, det.id
        x, y, radius = located
        assert math.hypot(x - obj.x_mm, y - obj.y_mm) < 60.0, (det.id, located, (obj.x_mm, obj.y_mm))
        assert 0.5 * obj.radius_mm < radius < 1.6 * math.hypot(obj.width_mm, obj.depth_mm) / 2, det.id
        checked += 1
    assert checked >= 1


def test_ground_truth_box_matches_the_projected_object_extent():
    """The cardboard box's ground-truth bottom edge is its near side, as in the rendered image."""
    sandbox = SimulationSandbox()
    pose = BodyPose(0, 0, 30)
    det = next(d for d in sandbox.detect_objects(pose) if d.id == "cardboard_box")
    x1, y1, x2, y2 = det.bbox_px
    obj = sandbox.get_object("cardboard_box")
    corners_v = [MODEL.project(pose, obj.x_mm + sx * 110, obj.y_mm + sy * 110, 0.0)[1] * 384
                 for sx in (-1, 1) for sy in (-1, 1)]
    assert abs(y2 - max(corners_v)) < 8  # sandbox and CameraModel mounts differ by a few mm
    assert x2 - x1 > 110  # seen corner-on, wider than its 220 mm face


# ---- detection parsing ---------------------------------------------------------------------------------
def test_parse_detections_handles_grids_bare_boxes_and_garbage():
    grid = parse_detections('[{"label": "Red Ball!", "bbox": [100, 200, 300, 400]}]')
    assert grid == [Detection("red ball", (0.1, 0.2, 0.3, 0.4))]
    fraction = parse_detections('Sure: [{"name": "cup", "bbox_2d": [0.1, 0.1, 0.2, 0.3]}]')
    assert fraction == [Detection("cup", (0.1, 0.1, 0.2, 0.3))]
    assert parse_detections("[100, 200, 300, 400]") == [Detection("object", (0.1, 0.2, 0.3, 0.4))]
    assert parse_detections("[0.1, 0.2, 0.3, 0.4]") == [Detection("object", (0.1, 0.2, 0.3, 0.4))]
    swapped = parse_detections('[{"label": "x", "bbox": [300, 400, 100, 200]}]')
    assert swapped == [Detection("x", (0.1, 0.2, 0.3, 0.4))]
    clamped = parse_detections('[{"label": "x", "bbox": [-50, 900, 400, 1300]}]')
    assert clamped == [Detection("x", (0.0, 0.9, 0.4, 1.0))]
    for garbage in ["", "no objects", "[not json]", "{}", "[1, 2, 3, 4, 5]", "[[1, 2, 3, 4]]", "[null]",
                    '[{"label": "x", "bbox": [1, 2, 3]}]', '[{"label": "x", "bbox": ["a", 1, 2, 3]}]',
                    '[{"label": "", "bbox": [100, 100, 200, 200]}]', '[{"bbox": [100, 100, 200, 200]}]',
                    '[{"label": "x", "bbox": [NaN, 1, 2, 3]}]', "[" * 5000 + "]" * 5000]:
        assert parse_detections(garbage) == [], garbage


def test_parse_detections_drops_tiny_and_oversized_boxes_and_caps_count():
    assert parse_detections('[{"label": "wall", "bbox": [0, 0, 1000, 1000]}]') == []
    assert parse_detections('[{"label": "speck", "bbox": [500, 500, 503, 600]}]') == []
    many = json.dumps([{"label": f"thing {i}", "bbox": [i * 30, 0, i * 30 + 20, 50]} for i in range(30)])
    assert len(parse_detections(many)) == 16
    assert len(parse_detections(many, max_items=4)) == 4


def test_parse_detections_suppresses_duplicate_boxes():
    text = json.dumps([{"label": "box", "bbox": [100, 100, 300, 300]},
                       {"label": "carton", "bbox": [105, 102, 302, 305]},  # IoU > 0.7: same object
                       {"label": "ball", "bbox": [200, 200, 400, 400]}])  # overlaps, but a different object
    assert [d.label for d in parse_detections(text)] == ["box", "ball"]


async def test_vlm_detector_relabels_generic_answers_with_the_query():
    backend = ScriptedBackend(["[100, 200, 300, 400]", '[{"label": "object", "bbox": [1, 2, 300, 400]},'
                                                      ' {"label": "red ball", "bbox": [500, 500, 700, 700]}]',
                               '[{"label": "box", "bbox": [100, 100, 300, 300]}]'])
    detector = VLMDetector(backend)
    frame = Frame(b"png", BodyPose(), MODEL, 0.0)
    assert [d.label for d in await detector.detect(frame, "Keys!")] == ["keys"]
    assert [d.label for d in await detector.detect(frame, "red ball")] == ["red ball", "red ball"]
    assert [d.label for d in await detector.detect(frame)] == ["box"]
    assert detector.calls == 3
    request = backend.requests[0]
    assert request["tools"] is None
    content = request["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Detect every keys" in content[1]["text"]
    assert "Detect every distinct physical object" in backend.requests[2]["messages"][0]["content"][1]["text"]


def test_change_detector_needs_a_reference_and_a_real_change():
    change = ChangeDetector()
    base = np.full((24, 32), 100.0, dtype=np.float32)
    assert change.changed(base) == 0.0  # first view is the reference
    assert not change.is_changed(base + 5)
    moved = base.copy()
    moved[5:12, 5:12] = 200
    assert change.is_changed(moved)
    change.reset()
    assert change.changed(moved) == 0.0


# ---- labels --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("a, b, score", [
    ("brown cardboard box", "box", 2), ("box", "cube", 1), ("cardboard box", "carton", 1),
    ("blue cube", "box", 1), ("red ball", "ball", 2), ("red ball", "blue ball", 0), ("brown box", "tan box", 2),
    ("cardboard box", "brown box", 2), ("cardboard box", "blue box", 0), ("green mug", "cup", 1),
    ("yellow toy duck", "rubber duck", 2), ("red ball", "box", 0), ("purple ball", "violet sphere", 1),
])
def test_label_score(a, b, score):
    assert label_score(a, b) == score
    assert label_score(b, a) == score
    assert labels_match(a, b) is (score > 0)


def test_head_noun():
    assert head_noun("brown cardboard box") == "box"
    assert head_noun("the small red") == "object"


# ---- spatial map ---------------------------------------------------------------------------------------
def det_at(x: float, y: float, label: str, pose: BodyPose | None = None, half_width_mm: float = 60.0,
           height_mm: float = 120.0) -> Detection:
    """A box as a perfect detector would draw it for an object centred at (x, y)."""
    pose = pose or BodyPose()
    yaw = math.radians(pose.heading_deg)
    rx = (x - pose.x_mm) * math.cos(yaw) + (y - pose.y_mm) * math.sin(yaw)
    ry = -(x - pose.x_mm) * math.sin(yaw) + (y - pose.y_mm) * math.cos(yaw)
    pts = []
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        px, py = rx + dx * half_width_mm, ry + dy * half_width_mm
        wx = pose.x_mm + px * math.cos(yaw) - py * math.sin(yaw)
        wy = pose.y_mm + px * math.sin(yaw) + py * math.cos(yaw)
        for z in (0.0, height_mm):
            pts.append(MODEL.project(pose, wx, wy, z))
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    return Detection(label, (min(us), min(vs), max(us), max(vs)))


def fresh_map() -> tuple[SpatialMap, ReplayClock]:
    clock = ReplayClock()
    return SpatialMap(clock), clock


def test_same_object_seen_twice_is_one_track():
    spatial, clock = fresh_map()
    first = spatial.integrate_view(BodyPose(), MODEL, [det_at(1000, 100, "brown box")])
    assert [(o.id, new) for o, _, new in first.seen] == [("box_1", True)]
    obj = spatial.objects["box_1"]
    assert math.hypot(obj.x_mm - 1000, obj.y_mm - 100) < 60
    clock.advance(5)
    pose = BodyPose(200, -100, 10)
    second = spatial.integrate_view(pose, MODEL, [det_at(1000, 100, "cardboard box", pose)])
    assert [(o.id, new) for o, _, new in second.seen] == [("box_1", False)]
    assert list(spatial.objects) == ["box_1"]
    assert obj.seen_count == 2 and obj.last_seen == 5 and obj.confidence == pytest.approx(1.0)


def test_new_ids_are_numbered_per_noun():
    spatial, _ = fresh_map()
    update = spatial.integrate_view(BodyPose(), MODEL, [det_at(900, 300, "red ball"), det_at(900, -300, "blue ball"),
                                                        det_at(1400, 0, "brown cardboard box")])
    assert sorted(o.id for o, _, _ in update.seen) == ["ball_1", "ball_2", "box_1"]
    assert spatial.objects["ball_1"].label != spatial.objects["ball_2"].label


def test_object_that_should_be_visible_but_is_not_fades_away():
    spatial, _ = fresh_map()
    spatial.integrate_view(BodyPose(), MODEL, [det_at(1000, 0, "red ball"), det_at(1100, 400, "blue cube")])
    first = spatial.integrate_view(BodyPose(), MODEL, [det_at(1100, 400, "blue cube")])
    assert first.lost == [] and spatial.objects["ball_1"].confidence < 0.5
    assert "ball_1" not in {o.label for o in spatial.obstacles()}  # doubtful objects do not block paths
    second = spatial.integrate_view(BodyPose(), MODEL, [det_at(1100, 400, "blue cube")])
    assert [o.id for o in second.lost] == ["ball_1"] and "ball_1" not in spatial.objects
    assert "cube_1" in spatial.objects


def test_objects_out_of_view_or_too_far_keep_their_confidence():
    spatial, _ = fresh_map()
    spatial.objects["ball_1"] = TrackedObject("ball_1", "red ball", -900, 0, 50, 0, 0)
    spatial.objects["box_1"] = TrackedObject("box_1", "box", 3000, 0, 100, 0, 0)
    for _ in range(3):
        assert spatial.integrate_view(BodyPose(), MODEL, []).lost == []
    assert spatial.objects["ball_1"].confidence == 0.7 and spatial.objects["box_1"].confidence == 0.7


def test_query_view_only_counts_against_matching_objects():
    spatial, _ = fresh_map()
    spatial.integrate_view(BodyPose(), MODEL, [det_at(1000, 0, "red ball"), det_at(1000, 400, "brown box")])
    for _ in range(2):
        update = spatial.integrate_view(BodyPose(), MODEL, [], query="box")
    assert [o.id for o in update.lost] == ["box_1"]
    assert spatial.objects["ball_1"].confidence == 0.7


def test_occluded_object_is_not_forgotten():
    spatial, _ = fresh_map()
    spatial.objects["ball_1"] = TrackedObject("ball_1", "red ball", 1600, 0, 50, 0, 0)
    for _ in range(3):
        update = spatial.integrate_view(BodyPose(), MODEL, [det_at(800, 0, "brown box", half_width_mm=150)])
    assert update.lost == [] and spatial.objects["ball_1"].confidence == 0.7
    assert "box_1" in spatial.objects


def test_close_object_cut_by_the_image_bottom_is_matched_by_bearing():
    spatial, _ = fresh_map()
    spatial.objects["box_1"] = TrackedObject("box_1", "brown box", 420, 60, 110, 0, 0)
    cut = Detection("cardboard box", (0.35, 0.55, 0.62, 1.0))
    update = spatial.integrate_view(BodyPose(), MODEL, [cut])
    assert [(o.id, new) for o, _, new in update.seen] == [("box_1", False)]
    assert (spatial.objects["box_1"].x_mm, spatial.objects["box_1"].y_mm) == (420, 60)  # not re-ranged
    assert spatial.objects["box_1"].seen_count == 2


def test_far_or_raised_detections_are_reported_unplaced():
    spatial, _ = fresh_map()
    update = spatial.integrate_view(BodyPose(), MODEL, [Detection("lamp", (0.4, 0.05, 0.5, 0.3))])
    assert update.seen == [] and [d.label for d in update.unplaced] == ["lamp"] and not spatial.objects


def test_resolve_prefers_ids_then_exact_nouns_over_synonyms():
    spatial, _ = fresh_map()
    near_cube = TrackedObject("cube_1", "blue cube", 400, 0, 50, 0, 0)
    far_box = TrackedObject("box_1", "brown cardboard box", 2000, 0, 110, 0, 0)
    spatial.objects.update({"cube_1": near_cube, "box_1": far_box})
    pose = BodyPose()
    assert spatial.resolve("cube_1", pose) is near_cube
    assert spatial.resolve("Box 1", pose) is far_box
    assert spatial.resolve("the box", pose) is far_box  # exact noun beats the nearer synonym
    assert spatial.resolve("block", pose) in (near_cube, far_box)
    assert spatial.resolve("red ball", pose) is None
    assert spatial.resolve("the", pose) is None and spatial.resolve("", pose) is None
    other = TrackedObject("box_2", "brown box", 600, 0, 100, 0, 0)
    spatial.objects["box_2"] = other
    assert spatial.resolve("brown box", pose) is other  # nearest of equally good matches


def test_range_readings_build_occupancy_and_clear_free_space():
    spatial, _ = fresh_map()
    pose = BodyPose()
    spatial.update_range(pose, 800.0)
    hit = spatial.to_cell(141 + 800, 0)
    assert spatial.occupied_mask()[hit]
    assert spatial.occupancy[spatial.to_cell(400, 0)] < 0
    for _ in range(4):
        spatial.update_range(pose, 2500.0)  # beam passes through: the old hit is cleared
    assert not spatial.occupied_mask()[hit]
    spatial.update_range(pose, float("nan"))
    spatial.update_range(pose, -5.0)
    spatial.add_obstacle_point(99999, 0)  # outside the map: ignored
    assert spatial.occupied_mask().sum() == 0


def test_coverage_marks_the_view_cone():
    spatial, _ = fresh_map()
    before = spatial.unseen_fraction_near(1000, 0, 800)
    spatial.mark_coverage(BodyPose(), MODEL)
    assert spatial.unseen_fraction_near(1000, 0, 800) < before
    assert spatial.unseen_fraction_near(-1500, 0, 600) == 1.0


def test_save_and_load_marks_objects_stale(tmp_path):
    spatial, clock = fresh_map()
    spatial.integrate_view(BodyPose(), MODEL, [det_at(1000, 0, "red ball")])
    spatial.objects["ghost_1"] = TrackedObject("ghost_1", "ghost", 0, 0, 10, 0, 0, confidence=0.1)
    spatial.places.update({"start": (0.0, 0.0), "kitchen": (1200.0, 300.0)})
    clock.advance(30)
    path = tmp_path / "map.json"
    spatial.save(path)

    restored, clock2 = fresh_map()
    clock2.advance(100)
    assert restored.load(path) == 1
    ball = restored.objects["ball_1"]
    assert ball.stale and ball.confidence <= 0.5 and 100 - ball.last_seen >= 30
    assert "ghost_1" not in restored.objects  # low-confidence tracks are not persisted
    assert restored.places == {"kitchen": (1200.0, 300.0)}  # "start" belongs to the new session
    assert "from an earlier session" in restored.render_for_prompt(BodyPose())
    update = restored.integrate_view(BodyPose(), MODEL, [det_at(950, 30, "blue ball")])
    assert [o.id for o, _, new in update.seen if new] == ["ball_2"]  # ids continue after restored ones
    assert restored.load(tmp_path / "missing.json") == 0
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert restored.load(tmp_path / "bad.json") == 0


def test_render_for_prompt_lists_nearest_first_and_places():
    spatial, _ = fresh_map()
    assert spatial.render_for_prompt(BodyPose()).startswith("Known objects: none yet.")
    for i in range(10):
        spatial.objects[f"ball_{i}"] = TrackedObject(f"ball_{i}", "ball", 300 + 100 * i, 0, 40, 0, 0)
    spatial.places["start"] = (0.0, 0.0)
    text = spatial.render_for_prompt(BodyPose(0, 0, 180))
    lines = text.splitlines()
    assert lines[1].startswith("- ball_0 (ball): 0.3 m behind me")
    assert "...and 2 more farther away" in text and "Places: start (where I was switched on): here." in text


async def test_targeted_detections_are_confirmed_on_a_crop() -> None:
    import numpy as np

    from robot.cognition.backend import ScriptedBackend
    from robot.perception.camera import Frame, encode_png
    from robot.perception.detector import VLMDetector

    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    frame = Frame(encode_png(rgb), BodyPose(), CameraModel(64, 48), 0.0, rgb=rgb)
    boxes = '[{"label": "yellow duck", "bbox": [100, 100, 300, 300]}, {"label": "yellow duck", "bbox": [600, 500, 800, 700]}]'
    backend = ScriptedBackend([boxes, "No.", "Yes."])
    found = await VLMDetector(backend).detect(frame, "yellow duck")
    assert [d.bbox for d in found] == [(0.6, 0.5, 0.8, 0.7)]  # the first box was a leading-question false hit
    assert backend.requests[1]["messages"][0]["content"][1]["text"] == "Is this a yellow duck? Answer only yes or no."
    # A general inventory is not double-checked.
    backend = ScriptedBackend([boxes])
    assert len(await VLMDetector(backend).detect(frame)) == 2


def test_room_surfaces_are_not_objects() -> None:
    from robot.perception.detector import parse_detections

    text = ('[{"label": "blue wall", "bbox": [0, 0, 400, 300]}, {"label": "blue floor", "bbox": [0, 600, 999, 999]},'
            ' {"label": "red ball", "bbox": [100, 600, 200, 700]}]')
    assert [d.label for d in parse_detections(text)] == ["red ball"]


def test_a_close_sighting_outweighs_a_distant_one():
    spatial, clock = fresh_map()
    spatial.integrate_view(BodyPose(), MODEL, [det_at(2000, 0, "brown box")])  # placed 30 cm short from afar
    clock.advance(1)
    near = BodyPose(1300, 0, 0)
    spatial.integrate_view(near, MODEL, [det_at(2300, 0, "brown box", near)])
    box = spatial.objects["box_1"]
    assert abs(box.x_mm - 2300) < 100  # not the 2150 midpoint an equal-weight average would give


def test_a_well_known_object_is_not_dragged_by_one_far_sighting():
    spatial, clock = fresh_map()
    near = BodyPose(400, 0, 0)
    for _ in range(3):
        spatial.integrate_view(near, MODEL, [det_at(1200, 0, "red ball", near)])
        clock.advance(1)
    far = BodyPose(-1000, 0, 0)
    spatial.integrate_view(far, MODEL, [det_at(1500, 0, "red ball", far)])
    assert abs(spatial.objects["ball_1"].x_mm - 1200) < 100
