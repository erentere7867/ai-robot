"""The fast layer in the kinematic simulator: continuous perception + closed-loop behaviours."""

from __future__ import annotations

import asyncio
import math

import pytest
from robot_fakes import GroundTruthDetector, clear_scene

from robot.body.sim_body import SimBody
from robot.control.behaviors import Follow, GoToObject, Move, Orbit, Relation, Search, Turn
from robot.control.fast_loop import FastLoop
from robot.core.bus import EventBus
from robot.core.clock import ReplayClock
from robot.navigation.planner import GridPlanner
from robot.perception.camera import SimCamera
from robot.perception.geometry import CameraModel
from robot.world.sandbox import SimulationSandbox
from robot.world.spatial_map import SpatialMap


@pytest.fixture
async def rig():
    clock = ReplayClock()
    bus = EventBus(clock)
    body = SimBody(bus)
    sandbox = SimulationSandbox(bus)
    sandbox.bind(body)
    clear_scene(sandbox, keep=("cardboard_box",))
    world = SpatialMap(clock)
    camera = SimCamera(sandbox, body, clock, CameraModel(sandbox.cam_width, sandbox.cam_height))
    loop = FastLoop(bus=bus, clock=clock, body=body, camera=camera, spatial_map=world,
                    planner=GridPlanner(room_half_extent_mm=world.half, resolution_mm=50.0),
                    detector=GroundTruthDetector(sandbox), continuous=False)
    await bus.start()
    await body.start()
    await loop.start()
    try:
        yield clock, body, sandbox, world, loop
    finally:
        await loop.stop()
        await body.stop()
        sandbox.unbind()
        await bus.stop()


def dist(pose, obj) -> float:
    return math.hypot(pose.x_mm - obj.x_mm, pose.y_mm - obj.y_mm)


def facing(pose, obj) -> float:
    b = math.degrees(math.atan2(obj.y_mm - pose.y_mm, obj.x_mm - pose.x_mm)) - pose.heading_deg
    return abs((b + 180) % 360 - 180)


async def test_go_to_near_an_object_ends_in_front_of_it_facing_it(rig):
    clock, body, sandbox, world, loop = rig
    sandbox.move_object("cardboard_box", 1100.0, 300.0)
    await loop.perceive_now()
    done = await loop.run(GoToObject("box_1", Relation("near", view=(0.96, 0.26))))
    assert done.ok, done.message
    box = sandbox.get_object("cardboard_box")
    assert 110 + 150 < dist(body.pose(), box) < 110 + 450 and facing(body.pose(), box) < 10
    assert clock.monotonic() < 15.0  # smooth motion, no stop-and-look


async def test_go_behind_plans_around_the_object(rig):
    clock, body, sandbox, world, loop = rig
    sandbox.move_object("cardboard_box", 900.0, 0.0)
    await loop.perceive_now()
    done = await loop.run(GoToObject("box_1", Relation("behind", view=(1.0, 0.0))))
    assert done.ok, done.message
    pose = body.pose()
    assert pose.x_mm > 900 + 110 and abs(pose.y_mm) < 250  # far side
    assert not any(sandbox.check_collision(body.pose()) for _ in [0])


async def test_follow_keeps_up_with_a_moving_object_and_reports_losing_it(rig):
    clock, body, sandbox, world, loop = rig
    sandbox.move_object("cardboard_box", 900.0, 0.0)
    await loop.perceive_now()
    future = loop.start_behavior(Follow("box_1", distance_mm=500.0))
    gaps = []
    t0 = clock.monotonic()
    while clock.monotonic() - t0 < 12.0:  # the box drifts left at 80 mm/s
        await asyncio.sleep(0)
        t = clock.monotonic() - t0
        sandbox.move_object("cardboard_box", 900.0, min(900.0, 80.0 * t))
        if t > 6.0:
            gaps.append(dist(body.pose(), sandbox.get_object("cardboard_box")))
    assert not future.done()
    assert max(gaps) < 110 + 500 + 350 and min(gaps) > 110 + 150  # kept its distance while it moved
    assert facing(body.pose(), sandbox.get_object("cardboard_box")) < 20
    sandbox.remove_object("cardboard_box")
    done = await asyncio.wait_for(future, timeout=5.0)
    assert not done.ok and "lost sight" in done.message


async def test_search_turns_smoothly_until_the_detector_sees_it(rig):
    clock, body, sandbox, world, loop = rig
    sandbox.move_object("cardboard_box", -900.0, 400.0)  # behind and to the left
    done = await loop.run(Search("box"))
    assert done.ok, done.message
    assert facing(body.pose(), sandbox.get_object("cardboard_box")) < 10
    assert clock.monotonic() < 6.0  # one smooth turn, not six stop-and-look views


async def test_turn_and_move_are_accurate(rig):
    clock, body, sandbox, world, loop = rig
    clear_scene(sandbox)
    assert (await loop.run(Turn(90.0))).ok
    assert abs(body.pose().heading_deg - 90.0) < 4.0
    assert (await loop.run(Move(300.0))).ok
    assert abs(body.pose().y_mm - 300.0) < 20 and abs(body.pose().x_mm) < 20


async def test_orbit_circles_the_object(rig):
    clock, body, sandbox, world, loop = rig
    sandbox.move_object("cardboard_box", 700.0, 0.0)
    await loop.perceive_now()
    box = world.objects["box_1"]
    swept, last = 0.0, None
    future = loop.start_behavior(Orbit((box.x_mm, box.y_mm), 520.0, 1.0, math.pi, "box_1"))
    while not future.done():
        await asyncio.sleep(0)
        p = body.pose()
        a = math.degrees(math.atan2(p.y_mm - 0.0, p.x_mm - 700.0))
        if last is not None:
            swept += (a - last + 180) % 360 - 180
        last = a
    assert future.result().ok, future.result().message
    assert abs(swept) > 300 and not sandbox.check_collision(body.pose())


async def test_cancelling_the_wait_stops_the_behaviour(rig):
    clock, body, sandbox, world, loop = rig
    clear_scene(sandbox)
    task = asyncio.ensure_future(loop.run(Move(1000.0)))
    for _ in range(40):
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert loop.behavior is None
    x = body.pose().x_mm
    clock.advance(2.0)
    await body.update_motion()
    assert body.pose().x_mm - x < 150  # coasts to a stop on the watchdog, does not keep driving
