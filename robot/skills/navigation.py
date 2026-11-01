"""Moving: spatial goals resolved against the map, planned around obstacles, driven closed-loop."""

from __future__ import annotations

import math
import re

import numpy as np

from robot.skills.base import Skill, SkillContext, SkillError, SkillResult
from robot.skills.perception import look_around, perceive
from robot.world.spatial_map import TrackedObject

# From an object's surface to my centre: half my length (141 mm) plus more than the 120 mm range-sensor
# stop distance, so approaching an object does not trip the obstacle reflex.
ROBOT_CLEARANCE_MM = 300.0
ARRIVE_TOLERANCE_MM = 120.0
MAX_SEGMENT_MM = 600.0
# A box seen from 2 m is placed with a few decimetres of error. Beyond this distance go_to stops part way,
# looks again from closer and re-plans (look-move-look), instead of trusting the first sighting.
REFINE_BEYOND_MM = 1000.0
REFINE_AT_MM = 650.0
START_WORDS = {"start", "start point", "starting point", "home", "where i started", "where you started", "origin"}
RELATIONS = ["near", "behind", "in_front_of", "left_of", "right_of", "between", "on"]
# Words a small model reaches for instead (often borrowed from the map's direction text).
RELATION_ALIASES = {
    **{w: "near" for w in ("next_to", "beside", "by", "close_to", "near_to", "at", "to", "front_left",
                           "front_right", "ahead", "toward", "towards", "approach", "nearby")},
    **{w: "in_front_of" for w in ("in_front", "front", "front_of", "before")},
    **{w: "behind" for w in ("behind_of", "back", "back_of", "far_side", "beyond", "past")},
    "left": "left_of", "to_the_left_of": "left_of", "right": "right_of", "to_the_right_of": "right_of",
    **{w: "on" for w in ("onto", "on_top", "on_top_of", "into", "inside")},
}


# How people name each relation. Used to catch a call that drops the relation the human asked for, e.g.
# go_to(ball_1, relation=near) for "go between the ball and the cube".
RELATION_WORDS = {"between": re.compile(r"\bbetween\b"), "behind": re.compile(r"\bbehind\b|\bfar side\b"),
                  "left_of": re.compile(r"\bleft (?:of|side)\b|\bto the left of\b"),
                  "right_of": re.compile(r"\bright (?:of|side)\b|\bto the right of\b")}


def _asked_relation(request: str) -> str | None:
    """The one relation the request names, if it names exactly one."""
    named = [rel for rel, pattern in RELATION_WORDS.items() if pattern.search(request.lower())]
    return named[0] if len(named) == 1 else None


_BETWEEN = re.compile(r"\bbetween (?:the )?(.+?) and (?:the )?(.+?)(?:[.,!?]|$)")


def _other_named(ctx: SkillContext, request: str, target_id: str) -> str | None:
    """For "between the ball and the cube": the known object named in the request that is not the target."""
    match = _BETWEEN.search(request.lower())
    if not match:
        return None
    for phrase in match.groups():
        found = ctx.spatial_map.resolve(phrase.strip(), ctx.pose())
        if found is not None and found.id != target_id:
            return found.id
    return None


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


async def _resolve(ctx: SkillContext, target: str) -> TrackedObject:
    """Find a target in the map, searching all around for it first if it is not known yet."""
    obj = ctx.spatial_map.resolve(target, ctx.pose())
    if obj is not None:
        return obj
    query = re.sub(r"_\d+$", "", target.strip().lower()).replace("_", " ")
    await look_around(ctx, query)
    obj = ctx.spatial_map.resolve(target, ctx.pose()) or ctx.spatial_map.resolve(query, ctx.pose())
    if obj is None:
        known = ", ".join(sorted(ctx.spatial_map.objects)) or "nothing"
        raise SkillError(f"I looked all around and could not find '{target}'. Objects I know: {known}.")
    return obj


async def navigate_to(ctx: SkillContext, goal: tuple[float, float], *, exclude: set[str] | None = None,
                      max_iterations: int = 30) -> tuple[bool, str]:
    """Replan from the current pose every segment so new range readings and bumps are honoured."""
    blocked_count = 0
    for _ in range(max_iterations):
        ctx.check()
        pose = ctx.pose()
        here = (pose.x_mm, pose.y_mm)
        if _dist(here, goal) <= ARRIVE_TOLERANCE_MM:
            return True, "arrived"
        points = _occupied_points(ctx, exclude)
        plan = ctx.planner.plan(here, goal, ctx.spatial_map.obstacles(exclude=exclude), points=points)
        if not plan.ok:
            if "outside" in plan.message:
                return False, "that spot is outside the area I can map"
            return False, ("every way there is blocked by walls or objects I have sensed; "
                           "there may not be enough room for me")
        waypoint = plan.waypoints[0]
        if _dist(here, plan.waypoints[-1]) <= ARRIVE_TOLERANCE_MM:
            # The exact spot is blocked; the caller decides whether this is good enough.
            return False, f"the exact spot is blocked, so I stopped {_dist(here, goal) / 10:.0f} cm from it"
        turn = await ctx.face_point(*waypoint)
        if turn is not None and not turn.ok:
            return False, turn.message
        step = min(MAX_SEGMENT_MM, _dist((ctx.pose().x_mm, ctx.pose().y_mm), waypoint))
        result = await ctx.drive(step)
        if result is None or result.ok:
            continue
        if result.status in ("reflex_stopped", "failed") and blocked_count < 4 and "estop" not in result.message.lower():
            blocked_count += 1
            p = ctx.pose()
            yaw = math.radians(p.heading_deg)
            # Something is right in front: remember it so the next plan goes around.
            for lateral in (-60.0, 0.0, 60.0):
                ctx.spatial_map.add_obstacle_point(p.x_mm + 260.0 * math.cos(yaw) - lateral * math.sin(yaw),
                                                   p.y_mm + 260.0 * math.sin(yaw) + lateral * math.cos(yaw))
            backoff = await ctx.drive(-120.0)
            if backoff is not None and not backoff.ok and backoff.status == "cancelled":
                return False, backoff.message
            continue
        return False, result.message
    return False, "gave up after too many steps"


def _occupied_points(ctx: SkillContext, exclude: set[str] | None = None) -> np.ndarray:
    mask = ctx.spatial_map.occupied_mask()
    idx = np.nonzero(mask)
    if not len(idx[0]):
        return np.zeros((0, 2))
    half, cell = ctx.spatial_map.half, 50.0
    points = np.stack([-half + (idx[0] + 0.5) * cell, -half + (idx[1] + 0.5) * cell], axis=1)
    # Range hits on an object we are allowed to approach (between / on) must not block it either.
    for obj_id in exclude or ():
        obj = ctx.spatial_map.objects.get(obj_id)
        if obj is not None:
            keep = np.hypot(points[:, 0] - obj.x_mm, points[:, 1] - obj.y_mm) > obj.radius_mm + 120.0
            points = points[keep]
    return points


async def _verify_target(ctx: SkillContext, obj_id: str) -> str:
    """Look at the target after arriving so its position in memory is refreshed."""
    obj = ctx.spatial_map.objects.get(obj_id)
    if obj is None:
        return ""
    dist, bearing = obj.relative(ctx.pose())
    if dist - obj.radius_mm < 280.0 and abs(bearing) < 25.0:
        # Too close for the camera to see the whole object; the range sensor still measures it.
        tof = ctx.body.sensors().front_tof_mm
        return f" {obj_id} is right in front of me (range sensor: {tof / 10:.0f} cm)."
    tilted = dist < 700.0
    if tilted:
        await ctx.primitive("tilt_head", angle_deg=-20.0)
    try:
        view = await perceive(ctx, None)
    finally:
        if tilted:
            await ctx.primitive("tilt_head", angle_deg=0.0)
    ctx.attached_image = None  # a check for the map, not a view for the model to describe
    obj = ctx.spatial_map.objects.get(obj_id)
    if obj_id in view["seen"] and obj is not None:
        return f" I can see {obj_id} {obj.relative(ctx.pose())[0] / 1000:.1f} m away."
    return f" I can't see {obj_id} from here."


async def _refresh(ctx: SkillContext, obj: TrackedObject) -> None:
    """Face the object and look at it again; the map then holds a nearer, more accurate sighting."""
    current = ctx.spatial_map.objects.get(obj.id, obj)
    turned = await ctx.face_point(current.x_mm, current.y_mm)
    if turned is None or turned.ok:
        await perceive(ctx, None)


async def go_to(ctx: SkillContext, target: str, relation: str = "near", other: str = "",
                distance_cm: float | None = None) -> SkillResult:
    pose = ctx.pose()
    start = (pose.x_mm, pose.y_mm)
    if target.strip().lower() in START_WORDS:
        goal = ctx.spatial_map.places.get("start", (0.0, 0.0))
        ok, why = await navigate_to(ctx, goal)
        ok = ok or _dist((ctx.pose().x_mm, ctx.pose().y_mm), goal) < 250.0
        msg = "I'm back at my start point." if ok else f"I couldn't get back to my start point: {why}."
        return SkillResult(ok, msg, {"distance_to_goal_m": round(_dist((ctx.pose().x_mm, ctx.pose().y_mm),
                                                                         goal) / 1000, 2)})
    obj = await _resolve(ctx, target)
    asked = _asked_relation(ctx.request)
    repaired = ""
    if asked and relation in ("near", "in_front_of"):
        # The call dropped the relation the human named. Measured: told to retry, the 2B model picks yet
        # another wrong relation, so the call is repaired from the human's own words instead.
        if asked == "between" and not other:
            other = _other_named(ctx, ctx.request, obj.id) or ""
        if asked == "between" and not other:
            raise SkillError("The human asked for 'between': call go_to with relation 'between' and the "
                             "second object's id in 'other'.")
        relation, repaired = asked, f" (I went {asked.replace('_', ' ')}, as the human asked.)"
    if relation not in RELATIONS:
        raise SkillError(f"relation must be one of {RELATIONS}")
    dx, dy = obj.x_mm - pose.x_mm, obj.y_mm - pose.y_mm
    norm = math.hypot(dx, dy)
    if norm < 1.0:
        ux, uy = math.cos(math.radians(pose.heading_deg)), math.sin(math.radians(pose.heading_deg))
    else:
        ux, uy = dx / norm, dy / norm  # "behind" and "left of" are as seen from where the request was made
    gap = obj.radius_mm + (distance_cm * 10.0 if distance_cm is not None else ROBOT_CLEARANCE_MM)
    exclude: set[str] = set()
    second: TrackedObject | None = None
    if relation == "between":
        if not other:
            raise SkillError("relation 'between' needs the other object in 'other'.")
        second = await _resolve(ctx, other)
        if second.id == obj.id:
            raise SkillError("'between' needs two different objects.")
        clear = _dist((obj.x_mm, obj.y_mm), (second.x_mm, second.y_mm)) - obj.radius_mm - second.radius_mm
        if clear < 300.0:
            return SkillResult(False, f"The gap between {obj.id} and {second.id} is only {clear / 10:.0f} cm, "
                                      "too narrow for me.")
        exclude = {obj.id, second.id}  # the planner would otherwise refuse a goal touching both
    elif relation == "on":
        exclude = {obj.id}  # something flat to drive onto (a dock, a mat): not an obstacle this time

    def goal_now() -> tuple[float, float]:
        """The goal from the latest estimate of where the object(s) are."""
        o = ctx.spatial_map.objects.get(obj.id, obj)
        if second is not None:
            s = ctx.spatial_map.objects.get(second.id, second)
            return 0.5 * (o.x_mm + s.x_mm), 0.5 * (o.y_mm + s.y_mm)
        if relation == "on":
            return o.x_mm, o.y_mm
        if relation == "behind":
            return o.x_mm + ux * gap, o.y_mm + uy * gap
        if relation in ("left_of", "right_of"):
            side = 1.0 if relation == "left_of" else -1.0
            return o.x_mm - uy * gap * side, o.y_mm + ux * gap * side
        return o.x_mm - ux * gap, o.y_mm - uy * gap

    if norm > REFINE_BEYOND_MM:
        # Drive toward the object, stop at a closer viewpoint and look again before the final approach.
        here = (pose.x_mm, pose.y_mm)
        view_point = (obj.x_mm - ux * REFINE_AT_MM, obj.y_mm - uy * REFINE_AT_MM)
        if _dist(here, view_point) > 250.0:
            await navigate_to(ctx, view_point, exclude=exclude)
            await _refresh(ctx, obj)
    goal = goal_now()
    ok, why = await navigate_to(ctx, goal, exclude=exclude)
    end = ctx.pose()
    off = _dist((end.x_mm, end.y_mm), goal)
    verified = ""
    unconfirmed = False  # the target was not in view where I ended up, even after searching
    if relation != "on":
        current = ctx.spatial_map.objects.get(obj.id, obj)
        await ctx.face_point(current.x_mm, current.y_mm)
        verified = await _verify_target(ctx, obj.id)
        if "can't see" in verified and second is None:
            # Not where I thought: the map entry was wrong (a far-off guess, or a misdetection). Search
            # for it from here and go again, rather than reporting an arrival I cannot confirm.
            found = (await look_around(ctx, obj.label)).data.get("found")
            if found and found in ctx.spatial_map.objects:
                obj = ctx.spatial_map.objects[found]
                goal = goal_now()
                ok, why = await navigate_to(ctx, goal, exclude=exclude)
                end = ctx.pose()
                off = _dist((end.x_mm, end.y_mm), goal)
                await ctx.face_point(obj.x_mm, obj.y_mm)
                second_look = await _verify_target(ctx, obj.id)
                verified = f" It was not where I first thought, so I searched and found {obj.id}." + second_look
                unconfirmed = "can't see" in second_look  # two checks failed: no arrival to claim
            else:
                verified += f" I searched around but could not find {obj.label} again."
                unconfirmed = True
    obj = ctx.spatial_map.objects.get(obj.id, obj)
    end = ctx.pose()
    holds = _relation_holds(relation, (obj.x_mm, obj.y_mm), obj.radius_mm, start, (end.x_mm, end.y_mm), goal,
                            (ux, uy))
    place = relation.replace("_", " ")
    if relation == "behind":
        verified += (f" {obj.id} is now between me and where I was." if holds
                     else f" I'm beside {obj.id} rather than behind it.")
    target_txt = f"{obj.id}" + (f" and {other}" if relation == "between" else "")
    solid = (relation == "on" and not ok and "obstacle" in why.lower()
             and _dist((end.x_mm, end.y_mm), (obj.x_mm, obj.y_mm)) < obj.radius_mm + 450.0)
    if unconfirmed:
        msg = f"I went to where I thought {obj.id} was, but I can't confirm it is there."
    elif solid:
        # Asked to drive onto something the range sensor says is solid (a cube, not a mat): being right
        # next to it is the best possible outcome, and saying so stops the model retrying.
        msg = f"{obj.id} is solid, so I can't drive onto it; I'm right next to it now."
    elif ok or holds:
        msg = f"Arrived {place} {target_txt}" + ("" if ok else " (as close as the space allows)") + "."
    else:
        msg = f"I stopped {off / 1000:.1f} m short of the spot {place} {target_txt}: {why}."
    return SkillResult((ok or holds or solid) and not unconfirmed, msg + verified + repaired, {"distance_to_goal_m": round(off / 1000, 2)})


def _relation_holds(relation: str, obj: tuple[float, float], radius: float, start: tuple[float, float],
                    end: tuple[float, float], goal: tuple[float, float], view: tuple[float, float]) -> bool:
    """Did I end up in the requested spatial relation, even if not exactly on the planned spot?"""
    rel = (end[0] - obj[0], end[1] - obj[1])
    dist = math.hypot(*rel)
    if relation == "between":
        return _dist(end, goal) < 250.0
    if relation == "on":
        return dist < max(120.0, 0.7 * radius)
    if dist > radius + 650.0 or dist < 1.0:
        return False
    along = (rel[0] * view[0] + rel[1] * view[1]) / dist  # +1 = far side of the object as first seen
    side = (-rel[0] * view[1] + rel[1] * view[0]) / dist  # +1 = left of it
    if relation == "behind":
        return along > 0.7
    if relation in ("near", "in_front_of"):
        return along < -0.3 or (relation == "near" and dist < radius + 450.0)
    if relation == "left_of":
        return side > 0.6
    if relation == "right_of":
        return side < -0.6
    return False


async def face(ctx: SkillContext, target: str) -> SkillResult:
    key = target.strip().lower()
    if key in START_WORDS:
        sx, sy = ctx.spatial_map.places.get("start", (0.0, 0.0))
        await ctx.face_point(sx, sy)
        return SkillResult(True, "I'm facing my start point.")
    obj = await _resolve(ctx, target)
    result = await ctx.face_point(obj.x_mm, obj.y_mm)
    if result is not None and not result.ok:
        return SkillResult(False, result.message)
    return SkillResult(True, f"Now facing {ctx.spatial_map.describe(obj, ctx.pose())}.")


async def turn(ctx: SkillContext, direction: str, degrees: float | None = None) -> SkillResult:
    if direction == "around":
        # "around" is a half turn; a small model often adds degrees=360 to it, which would end where it began.
        angle = degrees if degrees is not None and degrees > 360.0 else 180.0
    else:
        angle = abs(degrees if degrees is not None else 90.0) * (1 if direction == "left" else -1)
    # The motor accepts at most one revolution per command; longer turns are done in pieces.
    remaining = angle
    while abs(remaining) > 1.0:
        piece = max(-360.0, min(360.0, remaining))
        result = await ctx.primitive("spin", angle_deg=round(piece, 1))
        if not result.ok:
            return SkillResult(False, f"Turned only {abs(angle - remaining):.0f} degrees: {result.message}")
        remaining -= piece
    ctx.attached_image = None
    return SkillResult(True, f"Turned {direction} {abs(angle):.0f} degrees.")


ALL_WORDS = {"all", "everything", "objects", "all objects", "all of the objects", "all the objects", "things",
             "all things", "the objects", "room"}


async def circle_around(ctx: SkillContext, target: str = "all", laps: float = 1.0) -> SkillResult:
    """Drive a loop around one object, or around everything known, at a safe distance."""
    key = target.strip().lower()
    if key in ALL_WORDS:
        objs = [o for o in ctx.spatial_map.objects.values() if o.confidence >= 0.3]
        if not objs:
            await look_around(ctx)
            objs = [o for o in ctx.spatial_map.objects.values() if o.confidence >= 0.3]
        if not objs:
            return SkillResult(False, "I don't know of any objects to circle around.")
        name = "all the objects"
    else:
        objs = [await _resolve(ctx, target)]
        name = objs[0].id
    cx = sum(o.x_mm for o in objs) / len(objs)
    cy = sum(o.y_mm for o in objs) / len(objs)
    radius = max(math.hypot(o.x_mm - cx, o.y_mm - cy) + o.radius_mm for o in objs) + ROBOT_CLEARANCE_MM + 80.0
    pose = ctx.pose()
    start_angle = math.atan2(pose.y_mm - cy, pose.x_mm - cx)
    points = max(6, int(round(2 * math.pi * radius / 450.0)))
    total = int(round(points * max(0.25, min(3.0, laps))))
    reached = 0
    for k in range(1, total + 1):
        a = start_angle + 2 * math.pi * k / points
        ok, _ = await navigate_to(ctx, (cx + radius * math.cos(a), cy + radius * math.sin(a)), max_iterations=8)
        reached += ok
    await ctx.face_point(cx, cy)
    if reached == 0:
        return SkillResult(False, f"I couldn't drive around {name}: walls or obstacles block the way.")
    note = "" if reached == total else f" ({total - reached} of {total} points were blocked, so I went past them)"
    return SkillResult(True, f"Drove a loop around {name} about {radius / 1000:.1f} m out{note}.")


async def move(ctx: SkillContext, direction: str, distance_cm: float = 30.0) -> SkillResult:
    result = await ctx.drive(distance_cm * 10.0 * (1 if direction == "forward" else -1))
    if result is not None and not result.ok:
        return SkillResult(False, f"Moved only part of the way: {result.message}")
    ctx.attached_image = None
    return SkillResult(True, f"Moved {direction} {distance_cm:.0f} cm.")


async def tilt_head(ctx: SkillContext, direction: str) -> SkillResult:
    angle = {"up": 25.0, "down": -20.0, "level": 0.0}[direction]
    result = await ctx.primitive("tilt_head", angle_deg=angle)
    return SkillResult(result.ok, result.message)


async def explore(ctx: SkillContext) -> SkillResult:
    """Go towards the least-seen reachable area, then look around there."""
    pose = ctx.pose()
    best, best_score = None, 0.05
    points = _occupied_points(ctx)
    obstacles = ctx.spatial_map.obstacles()
    for k in range(8):
        heading = math.radians(pose.heading_deg + 45.0 * k)
        for reach in (700.0, 1200.0):
            p = (pose.x_mm + reach * math.cos(heading), pose.y_mm + reach * math.sin(heading))
            plan = ctx.planner.plan((pose.x_mm, pose.y_mm), p, obstacles, points=points, goal_tolerance_mm=150.0)
            if not plan.ok:
                continue
            score = ctx.spatial_map.unseen_fraction_near(*p, 1200.0) - 0.05 * reach / 1000.0
            if score > best_score:
                best, best_score = p, score
    if best is None:
        before = set(ctx.spatial_map.objects)
        result = await look_around(ctx)
        new = sorted(set(ctx.spatial_map.objects) - before)
        msg = "The area around me already looks explored; I looked around once more."
        return SkillResult(result.ok, msg + (f" New: {', '.join(new)}." if new else ""), result.data)
    before = set(ctx.spatial_map.objects)
    ok, why = await navigate_to(ctx, best)
    if not ok:
        return SkillResult(False, f"I couldn't reach the unexplored area: {why}.")
    result = await look_around(ctx)
    new = [ctx.spatial_map.describe(ctx.spatial_map.objects[i], ctx.pose())
           for i in sorted(set(ctx.spatial_map.objects) - before)]
    msg = "I explored a new area." + (f" I found {len(new)} new thing(s)." if new else " Nothing new there.")
    return SkillResult(True, msg, {"new_objects": new} if new else {})


def skills() -> list[Skill]:
    target = {"type": "string", "description": "a known object id such as box_1; or a description such as "
              "'charging dock' (I search for it if it is not known yet); or start, the spot where I was "
              "switched on"}
    return [
        Skill("go_to", "Drive to a spot relative to a known object; plans a safe path around obstacles and "
              "turns to face the object at the end. 'behind' means the far side of the object from where you "
              "are now; 'on' drives onto something flat like a dock or mat. For 'between' also give 'other'.",
              {"target": target, "relation": {"type": "string", "enum": RELATIONS, "aliases": RELATION_ALIASES},
               "other": {"type": "string", "description": "second object id, only for between"},
               "distance_cm": {"type": "number", "minimum": 5, "maximum": 150,
                               "description": "optional gap to keep from the object"}},
              go_to, ["target"]),
        Skill("turn_to", "Turn your body toward a known object without driving, e.g. for 'look at the ball'.",
              {"target": {"type": "string", "description": "object id such as box_1"}}, face, ["target"]),
        Skill("turn", "Turn in place, e.g. 'look left' is left and 'turn around' is around (a half turn). "
              "For other amounts use left or right with degrees, e.g. 360 for a full spin.",
              {"direction": {"type": "string", "enum": ["left", "right", "around"]},
               "degrees": {"type": "number", "minimum": 5, "maximum": 1080}},
              turn, ["direction"]),
        Skill("circle_around", "Drive a loop around one object or around all known objects, keeping a safe "
              "distance.", {"target": {"type": "string", "description": "object id, or all"},
                            "laps": {"type": "number", "minimum": 0.25, "maximum": 3}},
              circle_around),
        Skill("move", "Drive straight forward or backward. 'A little' is about 20 cm.",
              {"direction": {"type": "string", "enum": ["forward", "backward"]},
               "distance_cm": {"type": "number", "minimum": 5, "maximum": 200}},
              move, ["direction"]),
        Skill("tilt_head", "Tilt your head camera up, down, or level.",
              {"direction": {"type": "string", "enum": ["up", "down", "level"]}}, tilt_head, ["direction"]),
        Skill("explore", "Drive to a part of the room you have not seen yet and look around there.", {}, explore),
    ]
