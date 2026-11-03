"""Referring to things the way people do: by kind, colour, where they are, or how they compare."""

from robot.body.hal import BodyPose
from robot.core.clock import RealClock
from robot.world.spatial_map import SpatialMap, TrackedObject, label_score, parse_reference


def world() -> SpatialMap:
    m = SpatialMap(RealClock())
    for oid, label, x, y, r in [("box_1", "brown cardboard box", -750, 0, 120), ("ball_1", "red ball", 650, 450, 48),
                                ("cube_1", "blue cube", 650, -450, 60), ("mug_1", "green mug", 1400, 0, 43)]:
        m.objects[oid] = TrackedObject(oid, label, x, y, r, 0.0, 0.0)
    return m


def test_egocentric_direction_and_generic_nouns():
    m, pose = world(), BodyPose()
    assert m.resolve("the thing behind you", pose).id == "box_1"
    assert m.resolve("the object on your left", pose).id == "ball_1"
    assert m.resolve("whatever is to your right", pose).id == "cube_1"
    # The same words after turning round name something else: directions are relative to the robot.
    assert m.resolve("the thing behind you", BodyPose(heading_deg=180.0)).id in {"ball_1", "cube_1", "mug_1"}


def test_superlatives_colour_only_and_typos():
    m, pose = world(), BodyPose()
    assert m.resolve("whichever one is nearest", pose).id == "box_1"
    assert m.resolve("the farthest thing", pose).id == "mug_1"
    assert m.resolve("the biggest one", pose).id == "box_1"
    assert m.resolve("the red one", pose).id == "ball_1"
    assert m.resolve("cardboard boz", pose).id == "box_1"
    assert label_score("cardboard box", "boz") == 1 and label_score("box", "fox") == 0


def test_nothing_to_go_on_or_nothing_there():
    m, pose = world(), BodyPose()
    assert m.resolve("it", pose) is None
    assert m.resolve("the duck", pose) is None
    assert m.resolve("the ball behind you", pose) is None  # a ball, but not behind
    assert parse_reference("the closest box behind you") == ("box", (180.0, 65.0), "near")
