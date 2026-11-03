"""People: hearing where a voice comes from, knowing who spoke, "come here", "follow me", seeing all around."""

from __future__ import annotations

import math

import numpy as np
import pytest
from robot_fakes import make_harness

from robot.audio.doa import MicArray, simulate_voice, srp_phat
from robot.body.hal import BodyPose
from robot.core.clock import ReplayClock
from robot.perception.lidar import SimLidar
from robot.world.sandbox import SimulationSandbox
from robot.world.spatial_map import SpatialMap, TrackedObject

ARRAY = MicArray(((32.0, 0.0), (0.0, 32.0), (-32.0, 0.0), (0.0, -32.0)))


def true_bearing(pose: BodyPose, x: float, y: float) -> float:
    h = math.radians(pose.heading_deg)
    cx, cy = pose.x_mm + 25.0 * math.cos(h), pose.y_mm + 25.0 * math.sin(h)
    return (math.degrees(math.atan2(y - cy, x - cx)) - pose.heading_deg + 180.0) % 360.0 - 180.0


def test_voice_direction_from_the_microphone_array():
    """SRP-PHAT finds a voice's direction through room echoes: within a few degrees nearly every time."""
    rng = np.random.default_rng(3)
    errors = []
    for _ in range(30):
        pose = BodyPose(rng.uniform(-800, 800), rng.uniform(-800, 800), rng.uniform(-180, 180))
        source = (rng.uniform(-1400, 1400), rng.uniform(-1400, 1400), 1580.0)
        bearing, confidence = srp_phat(simulate_voice(source, pose, ARRAY, rng=rng), 16000, ARRAY)
        errors.append(abs((bearing - true_bearing(pose, *source[:2]) + 180.0) % 360.0 - 180.0))
        assert 0.0 <= confidence <= 1.0
    assert np.median(errors) < 4.0 and np.percentile(errors, 90) < 15.0


def test_mic_array_comes_from_the_body_config():
    array = MicArray.from_body_config("config/body.yaml")
    assert array is not None and array.n == 4 and array.center_x_mm == 25.0


def test_a_voice_matches_the_person_seen_in_that_direction():
    clock = ReplayClock()
    world = SpatialMap(clock)
    now = clock.monotonic()
    world.objects["person_1"] = TrackedObject("person_1", "person", -1000.0, 600.0, 200.0, now, now)
    world.objects["person_2"] = TrackedObject("person_2", "person", 900.0, -900.0, 200.0, now, now)
    pose = BodyPose(0, 0, 30)
    speaker = world.hear_voice(pose, true_bearing(pose, -1000.0, 600.0) + 6.0)  # a few degrees off
    assert speaker.track == "person_1" and speaker.how == "seen"
    assert world.resolve("me", pose).id == "person_1" and world.resolve("you", pose).id == "person_1"
    assert "the human talking to me" in world.render_for_prompt(pose)


def test_an_unseen_voice_is_placed_by_the_lidar_then_by_a_guess():
    sandbox = SimulationSandbox()
    for obj in list(sandbox.list_objects()):
        if obj["id"] != "person":
            sandbox.remove_object(obj["id"])
    sandbox.move_object("person", -900.0, 700.0)
    pose = BodyPose(0, 0, 0)

    class Body:
        def pose(self):
            return pose

    scan = SimLidar(sandbox, Body(), ReplayClock(), noise_mm=0.0).latest()
    world = SpatialMap(ReplayClock())
    speaker = world.hear_voice(pose, true_bearing(pose, -900.0, 700.0), scan=scan)
    assert speaker.how == "lidar" and speaker.track is None  # no made-up track: the cameras must place them
    assert math.hypot(speaker.point[0] + 900.0, speaker.point[1] - 700.0) < 150.0  # the person's legs, not the wall
    now = world.clock.monotonic()
    world.objects["person_1"] = TrackedObject("person_1", "person", -880.0, 690.0, 200.0, now, now, seen_count=3)
    assert world.speaker_track().id == "person_1" and world.speaker.how == "seen"
    blind = SpatialMap(ReplayClock())
    guess = blind.hear_voice(pose, 90.0)
    assert guess.how == "voice" and "voice came from to my left" in blind.render_for_prompt(pose)


@pytest.fixture
async def room(tmp_path):
    harness = make_harness(tmp_path, person=True)
    async with harness:
        for obj in list(harness.sandbox.list_objects()):
            if obj["id"] not in ("person", "cardboard_box"):
                harness.sandbox.remove_object(obj["id"])
        harness.spatial_map.forget_range_hits()
        yield harness


async def run(h, name, **args):
    return await h.skills.run(name, args, h.skill_context)


def heard(h) -> None:
    """The person spoke: the simulated array hears them, the map places them."""
    voice = h.voice.locate()
    h.spatial_map.hear_voice(h.body.pose(), voice["bearing_deg"], scan=h.lidar.latest() if h.lidar else None)


async def test_come_here_goes_to_the_person_behind_and_faces_them(room):
    room.sandbox.move_object("person", -1000.0, -600.0, yaw_deg=30.0)
    room.sandbox.move_object("cardboard_box", 700.0, 400.0)
    heard(room)
    room.skill_context.request = "come here"
    result = await run(room, "go_to", target="the box")  # the model's wrong guess, repaired from the words
    assert result.ok and result.message == "I'm here with you.", result.message
    pose, person = room.body.pose(), room.sandbox.get_object("person")
    gap = math.hypot(person.x_mm - pose.x_mm, person.y_mm - pose.y_mm)
    assert 380.0 < gap < 950.0
    assert abs(true_bearing(pose, person.x_mm, person.y_mm)) < 25.0
    assert not room.sandbox.check_collision(pose)


async def test_come_here_with_nobody_heard_says_so(room):
    room.sandbox.remove_object("person")
    result = await run(room, "go_to", target="me")
    assert result.status == "rejected" and "don't know where the human is" in result.message
    assert room.body.pose().x_mm == 0.0


async def test_follow_me_starts_following_the_speaker(room):
    room.sandbox.move_object("person", 900.0, -700.0)
    heard(room)
    room.skill_context.request = "follow me"
    result = await run(room, "follow", target="me")
    assert result.ok and "following you" in result.message
    assert room.fast.behavior is not None and room.fast.behavior.obj_id == room.spatial_map.speaker.track
    room.fast.cancel_behavior()


async def test_what_is_behind_needs_no_turn(room):
    room.sandbox.move_object("cardboard_box", -800.0, 100.0)
    room.sandbox.remove_object("person")
    before = room.body.pose()
    result = await run(room, "observe", direction="behind")
    assert result.ok and any("box" in o and "behind" in o for o in result.data["objects"]), result.data
    assert room.body.pose().heading_deg == pytest.approx(before.heading_deg)
    assert room.skill_context.attached_image is not None


async def test_circle_the_box_behind_you(room):
    """'Do a full circle around the box behind you': resolved from where it was asked, driven all the way."""
    room.sandbox.move_object("cardboard_box", -700.0, 0.0)
    room.sandbox.remove_object("person")
    room.skill_context.request = "do a full circle around the box behind you"
    result = await run(room, "circle_around", target="the box behind you")
    assert result.ok, result.message
    assert not room.sandbox.check_collision(room.body.pose())
