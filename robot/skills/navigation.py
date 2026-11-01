"""Moving: the tools the language model calls. Each one starts a closed-loop behaviour in the fast layer.

The model decides *what* (go behind the box, follow the ball); the behaviour does *how*, re-planning and
steering 20 times a second from live perception. `follow` keeps running after the reply.
"""

from __future__ import annotations

import math
import re

from robot.control.behaviors import (
    ROBOT_CLEARANCE_MM,
    Face,
    Follow,
    GoToObject,
    GoToPoint,
    Move,
    Orbit,
    Relation,
    Turn,
    bearing_to,
    occupied_points,
)
from robot.skills.base import Skill, SkillContext, SkillError, SkillResult
from robot.skills.perception import find, look_around
from robot.world.spatial_map import TrackedObject

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
                  "left_of": re.compile(r"\bleft (?:of|side)\b"),
                  "right_of": re.compile(r"\bright (?:of|side)\b")}
_BETWEEN = re.compile(r"\bbetween (?:the )?(.+?) and (?:the )?(.+?)(?:[.,!?]|$)")
_ON_WORDS = re.compile(r"\b(?:on|onto|on top|dock|charg\w*|mat|platform|pad)\b", re.I)


def _asked_relation(request: str) -> str | None:
    """The one relation the request names, if it names exactly one."""
    named = [rel for rel, pattern in RELATION_WORDS.items() if pattern.search(request.lower())]
    return named[0] if len(named) == 1 else None


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


async def _resolve(ctx: SkillContext, target: str) -> TrackedObject:
    """Find a target in the map, searching for it first if it is not known yet."""
    obj = ctx.spatial_map.resolve(target, ctx.pose())
    if obj is not None:
        return obj
    query = re.sub(r"_\d+$", "", target.strip().lower()).replace("_", " ")
    found = await find(ctx, query)
    obj = (ctx.spatial_map.objects.get(found or "") or ctx.spatial_map.resolve(target, ctx.pose())
           or ctx.spatial_map.resolve(query, ctx.pose()))
    if obj is None:
        known = ", ".join(sorted(ctx.spatial_map.objects)) or "nothing"
        raise SkillError(f"I looked all around and could not find '{target}'. Objects I know: {known}.")
    return obj


async def go_to(ctx: SkillContext, target: str, relation: str = "near", other: str = "",
                distance_cm: float | None = None) -> SkillResult:
    if target.strip().lower() in START_WORDS:
        goal = ctx.spatial_map.places.get("start", (0.0, 0.0))
        done = await ctx.behave(GoToPoint(goal, "my start point"))
        msg = "I'm back at my start point." if done.ok else done.message
        pose = ctx.pose()
        return SkillResult(done.ok, msg, {"distance_to_goal_m": round(math.hypot(pose.x_mm - goal[0],
                                                                                 pose.y_mm - goal[1]) / 1000, 2)})
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
    if relation == "on" and ctx.request and not _ON_WORDS.search(ctx.request):
        # "on" drives onto the object (it stops being an obstacle). Measured: the 2B model sometimes says "on" for
        # "go to the red ball", and drives into the ball.
        relation = "near"
    if relation not in RELATIONS:
        raise SkillError(f"relation must be one of {RELATIONS}")
    second: TrackedObject | None = None
    if relation == "between":
        if not other:
            raise SkillError("relation 'between' needs the other object in 'other'.")
        second = await _resolve(ctx, other)
        if second.id == obj.id:
            raise SkillError("'between' needs two different objects.")
        clear = math.hypot(obj.x_mm - second.x_mm, obj.y_mm - second.y_mm) - obj.radius_mm - second.radius_mm
        if clear < 300.0:
            return SkillResult(False, f"The gap between {obj.id} and {second.id} is only {clear / 10:.0f} cm, "
                                      "too narrow for me.")
    pose = ctx.pose()
    dx, dy = obj.x_mm - pose.x_mm, obj.y_mm - pose.y_mm
    norm = math.hypot(dx, dy)
    view = (dx / norm, dy / norm) if norm > 1.0 else (math.cos(math.radians(pose.heading_deg)),
                                                      math.sin(math.radians(pose.heading_deg)))
    rel = Relation(relation, distance_cm * 10.0 if distance_cm is not None else ROBOT_CLEARANCE_MM,
                   second.id if second else None, view)
    done = await ctx.behave(GoToObject(obj.id, rel))
    if done.ok and done.data.get("confirmed") is False and second is None:
        # Not where I thought (a far-off guess or a misdetection): search from here and go again rather than
        # reporting an arrival I cannot confirm.
        found = await find(ctx, obj.label)
        if found and found in ctx.spatial_map.objects:
            retry = await ctx.behave(GoToObject(found, rel))
            if retry.ok and retry.data.get("confirmed", True):
                return SkillResult(True, f"It was not where I first thought, so I searched and found {found}. "
                                         + retry.message + repaired)
        return SkillResult(False, f"I went to where I thought {obj.id} was, but I can't confirm it is there.")
    message = done.message
    if done.ok and relation == "behind":
        end, now = ctx.pose(), ctx.spatial_map.objects.get(obj.id, obj)
        along = ((end.x_mm - now.x_mm) * view[0] + (end.y_mm - now.y_mm) * view[1])
        message += (f" {obj.id} is now between me and where I was." if along > 0
                    else f" I'm beside {obj.id} rather than behind it.")
    ctx.attached_image = None
    return SkillResult(done.ok, message + repaired)


async def face(ctx: SkillContext, target: str) -> SkillResult:
    if target.strip().lower() in START_WORDS:
        sx, sy = ctx.spatial_map.places.get("start", (0.0, 0.0))
        done = await ctx.behave(Turn(bearing_to(ctx.pose(), sx, sy)))
        return SkillResult(done.ok, "I'm facing my start point." if done.ok else done.message)
    obj = await _resolve(ctx, target)
    done = await ctx.behave(Face(obj.id))
    if not done.ok:
        return SkillResult(False, done.message)
    return SkillResult(True, f"Now facing {ctx.spatial_map.describe(obj, ctx.pose())}.")


async def turn(ctx: SkillContext, direction: str, degrees: float | None = None) -> SkillResult:
    if direction == "around":
        # "around" is a half turn; a small model often adds degrees=360 to it, which would end where it began.
        angle = degrees if degrees is not None and degrees > 360.0 else 180.0
    else:
        angle = abs(degrees if degrees is not None else 90.0) * (1 if direction == "left" else -1)
    done = await ctx.behave(Turn(angle))
    ctx.attached_image = None
    if not done.ok:
        return SkillResult(False, done.message)
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
    done = await ctx.behave(Orbit((cx, cy), radius, laps, math.atan2(pose.y_mm - cy, pose.x_mm - cx), name))
    return SkillResult(done.ok, done.message)


async def follow(ctx: SkillContext, target: str, distance_cm: float = 60.0) -> SkillResult:
    """Start following; this keeps going after the reply until stopped or the target is lost."""
    obj = await _resolve(ctx, target)
    ctx.check()
    ctx.fast.start_behavior(Follow(obj.id, distance_cm * 10.0))
    return SkillResult(True, f"I'm following {obj.id} now, about {distance_cm:.0f} cm behind it, and I'll keep "
                             "going until you tell me to stop.", {"following": obj.id})


_STOP_ASKED = re.compile(r"\b(?:stop|halt|freeze|wait|enough|quit|cancel|stay|don'?t move|hold on)\b", re.I)


async def stop(ctx: SkillContext) -> SkillResult:
    if ctx.request and not _STOP_ASKED.search(ctx.request):
        # Measured: the 2B model sometimes answers "come back to where you started" with stop ("start"/"stop").
        raise SkillError("The human did not ask you to stop. Do what they asked instead.")
    running = ctx.fast.behavior.describe() if ctx.fast.behavior else None
    ctx.fast.cancel_behavior(reason="asked to stop")
    await ctx.primitive("stop")
    return SkillResult(True, f"Stopped {running}." if running else "Stopped; I was not moving.")


async def move(ctx: SkillContext, direction: str, distance_cm: float = 30.0) -> SkillResult:
    done = await ctx.behave(Move(distance_cm * 10.0 * (1 if direction == "forward" else -1)))
    ctx.attached_image = None
    return SkillResult(done.ok, done.message)


async def tilt_head(ctx: SkillContext, direction: str) -> SkillResult:
    angle = {"up": 25.0, "down": -20.0, "level": 0.0}[direction]
    result = await ctx.primitive("tilt_head", angle_deg=angle)
    return SkillResult(result.ok, result.message)


async def explore(ctx: SkillContext) -> SkillResult:
    """Go towards the least-seen reachable area, then look all around there."""
    pose = ctx.pose()
    best, best_score = None, 0.05
    points = occupied_points(ctx.spatial_map, set())
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
    before = set(ctx.spatial_map.objects)
    if best is None:
        result = await look_around(ctx)
        new = sorted(set(ctx.spatial_map.objects) - before)
        msg = "The area around me already looks explored; I looked around once more."
        return SkillResult(result.ok, msg + (f" New: {', '.join(new)}." if new else ""), result.data)
    done = await ctx.behave(GoToPoint(best, "the unexplored area"))
    if not done.ok:
        return SkillResult(False, done.message)
    await look_around(ctx)
    new = [ctx.spatial_map.describe(ctx.spatial_map.objects[i], ctx.pose())
           for i in sorted(set(ctx.spatial_map.objects) - before) if i in ctx.spatial_map.objects]
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
        # Listed last: a small model favours tools near the top, and these two are rarely the right first step.
        Skill("follow", "Keep following a moving person or object until told to stop, e.g. 'follow me'.",
              {"target": target, "distance_cm": {"type": "number", "minimum": 30, "maximum": 200}},
              follow, ["target"]),
        Skill("stop", "Halt all motion when the human says stop or freeze.", {}, stop),
    ]

