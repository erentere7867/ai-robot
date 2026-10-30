"""Moving: spatial goals resolved against the map, planned around obstacles, driven closed-loop."""

from __future__ import annotations

import math

import numpy as np

from robot.skills.base import Skill, SkillContext, SkillError, SkillResult
from robot.skills.perception import look_around, perceive
from robot.world.spatial_map import TrackedObject

ROBOT_CLEARANCE_MM = 220.0  # from an object's surface to my centre: half my length plus a margin
ARRIVE_TOLERANCE_MM = 120.0
MAX_SEGMENT_MM = 600.0
START_WORDS = {"start", "start point", "starting point", "home", "where i started", "where you started", "origin"}
RELATIONS = ["near", "behind", "in_front_of", "left_of", "right_of", "between", "on"]


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _resolve(ctx: SkillContext, target: str) -> TrackedObject:
    obj = ctx.spatial_map.resolve(target, ctx.pose())
    if obj is None:
        known = ", ".join(sorted(ctx.spatial_map.objects)) or "nothing yet"
        raise SkillError(f"I don't know where '{target}' is (known: {known}). "
                         "Use observe or look_around to find it first.")
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
        points = _occupied_points(ctx)
        plan = ctx.planner.plan(here, goal, ctx.spatial_map.obstacles(exclude=exclude), points=points)
        if not plan.ok:
            if "outside" in plan.message:
                return False, "that spot is outside the area I can map"
            return False, ("every way there is blocked by walls or objects I have sensed; "
                           "there may not be enough room for me")
        waypoint = plan.waypoints[0]
        if _dist(here, plan.waypoints[-1]) <= ARRIVE_TOLERANCE_MM:
            return True, "arrived as close as obstacles allow"
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


def _occupied_points(ctx: SkillContext) -> np.ndarray:
    mask = ctx.spatial_map.occupied_mask()
    idx = np.nonzero(mask)
    if not len(idx[0]):
        return np.zeros((0, 2))
    half, cell = ctx.spatial_map.half, 50.0
    return np.stack([-half + (idx[0] + 0.5) * cell, -half + (idx[1] + 0.5) * cell], axis=1)


async def _verify_target(ctx: SkillContext, obj_id: str) -> str:
    """Look at the target after arriving so its position in memory is refreshed."""
    obj = ctx.spatial_map.objects.get(obj_id)
    if obj is None:
        return ""
    dist, _ = obj.relative(ctx.pose())
    tilted = dist < 700.0
    if tilted:
        await ctx.primitive("tilt_head", angle_deg=-20.0)
    try:
        view = await perceive(ctx, None)
    finally:
        if tilted:
            await ctx.primitive("tilt_head", angle_deg=0.0)
    obj = ctx.spatial_map.objects.get(obj_id)
    if obj_id in view["seen"] and obj is not None:
        return f" I can see {obj_id} {obj.relative(ctx.pose())[0] / 1000:.1f} m away."
    return f" I can't see {obj_id} from here."


async def go_to(ctx: SkillContext, target: str, relation: str = "near", other: str = "",
                distance_cm: float | None = None) -> SkillResult:
    pose = ctx.pose()
    start = (pose.x_mm, pose.y_mm)
    if target.strip().lower() in START_WORDS:
        goal = ctx.spatial_map.places.get("start", (0.0, 0.0))
        ok, why = await navigate_to(ctx, goal)
        msg = "I'm back at my start point." if ok else f"I couldn't get back to my start point: {why}."
        return SkillResult(ok, msg, {"distance_to_goal_m": round(_dist((ctx.pose().x_mm, ctx.pose().y_mm),
                                                                         goal) / 1000, 2)})
    obj = _resolve(ctx, target)
    ox, oy = obj.x_mm, obj.y_mm
    dx, dy = ox - pose.x_mm, oy - pose.y_mm
    norm = math.hypot(dx, dy)
    if norm < 1.0:
        ux, uy = math.cos(math.radians(pose.heading_deg)), math.sin(math.radians(pose.heading_deg))
    else:
        ux, uy = dx / norm, dy / norm
    gap = obj.radius_mm + (distance_cm * 10.0 if distance_cm is not None else ROBOT_CLEARANCE_MM)
    exclude: set[str] = set()
    face: tuple[float, float] = (ox, oy)
    if relation == "between":
        if not other:
            raise SkillError("relation 'between' needs the other object in 'other'.")
        second = _resolve(ctx, other)
        if second.id == obj.id:
            raise SkillError("'between' needs two different objects.")
        clear = _dist((ox, oy), (second.x_mm, second.y_mm)) - obj.radius_mm - second.radius_mm
        if clear < 300.0:
            return SkillResult(False, f"The gap between {obj.id} and {second.id} is only {clear / 10:.0f} cm, "
                                      "too narrow for me.")
        goal = (0.5 * (ox + second.x_mm), 0.5 * (oy + second.y_mm))
        exclude = {obj.id, second.id}  # the planner would otherwise refuse a goal touching both
    elif relation == "on":
        goal = (ox, oy)  # something flat to drive onto (a dock, a mat): not an obstacle this time
        exclude = {obj.id}
    elif relation == "behind":
        goal = (ox + ux * gap, oy + uy * gap)
    elif relation in ("near", "in_front_of"):
        goal = (ox - ux * gap, oy - uy * gap)
    elif relation in ("left_of", "right_of"):
        side = 1.0 if relation == "left_of" else -1.0
        goal = (ox - uy * gap * side, oy + ux * gap * side)
    else:
        raise SkillError(f"relation must be one of {RELATIONS}")

    ok, why = await navigate_to(ctx, goal, exclude=exclude)
    end = ctx.pose()
    off = _dist((end.x_mm, end.y_mm), goal)
    verified = ""
    if relation != "on":
        await ctx.face_point(*face)
        verified = await _verify_target(ctx, obj.id)
    obj = ctx.spatial_map.objects.get(obj.id, obj)
    end = ctx.pose()
    holds = _relation_holds(relation, (obj.x_mm, obj.y_mm), obj.radius_mm, start, (end.x_mm, end.y_mm), goal,
                            (ux, uy))
    place = relation.replace("_", " ")
    if relation == "behind":
        verified += (f" {obj.id} is now between me and where I was." if holds
                     else f" I'm beside {obj.id} rather than behind it.")
    target_txt = f"{obj.id}" + (f" and {other}" if relation == "between" else "")
    if ok or holds:
        msg = f"Arrived {place} {target_txt}" + ("" if ok else " (as close as the space allows)") + "."
    else:
        msg = f"I stopped {off / 1000:.1f} m short of the spot {place} {target_txt}: {why}."
    return SkillResult(ok or holds, msg + verified, {"distance_to_goal_m": round(off / 1000, 2)})


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
    obj = _resolve(ctx, target)
    result = await ctx.face_point(obj.x_mm, obj.y_mm)
    if result is not None and not result.ok:
        return SkillResult(False, result.message)
    return SkillResult(True, f"Now facing {ctx.spatial_map.describe(obj, ctx.pose())}.")


async def turn(ctx: SkillContext, direction: str, degrees: float | None = None) -> SkillResult:
    if direction == "around":
        angle = degrees if degrees is not None else 180.0
    else:
        angle = abs(degrees if degrees is not None else 90.0) * (1 if direction == "left" else -1)
    result = await ctx.spin(angle)
    if result is not None and not result.ok:
        return SkillResult(False, result.message)
    ctx.attached_image = None
    return SkillResult(True, f"Turned {direction} {abs(angle):.0f} degrees.")


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
    target = {"type": "string", "description": "object id such as box_1, or a place such as start"}
    return [
        Skill("go_to", "Drive to a spot relative to a known object; plans a safe path around obstacles and "
              "turns to face the object at the end. 'behind' means the far side of the object from where you "
              "are now; 'on' drives onto something flat like a dock or mat. For 'between' also give 'other'.",
              {"target": target, "relation": {"type": "string", "enum": RELATIONS},
               "other": {"type": "string", "description": "second object id, only for between"},
               "distance_cm": {"type": "number", "minimum": 5, "maximum": 150,
                               "description": "optional gap to keep from the object"}},
              go_to, ["target"]),
        Skill("turn_to", "Turn your body toward a known object without driving, e.g. for 'look at the ball'.",
              {"target": {"type": "string", "description": "object id such as box_1"}}, face, ["target"]),
        Skill("turn", "Turn in place, e.g. 'look left' is left and 'turn around' is around (180 degrees). "
              "Optional degrees overrides the default 90.",
              {"direction": {"type": "string", "enum": ["left", "right", "around"]},
               "degrees": {"type": "number", "minimum": 5, "maximum": 360}},
              turn, ["direction"]),
        Skill("move", "Drive straight forward or backward. 'A little' is about 20 cm.",
              {"direction": {"type": "string", "enum": ["forward", "backward"]},
               "distance_cm": {"type": "number", "minimum": 5, "maximum": 200}},
              move, ["direction"]),
        Skill("tilt_head", "Tilt your head camera up, down, or level.",
              {"direction": {"type": "string", "enum": ["up", "down", "level"]}}, tilt_head, ["direction"]),
        Skill("explore", "Drive to a part of the room you have not seen yet and look around there.", {}, explore),
    ]
