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
from robot.skills.perception import find, in_view, look_around
from robot.world.spatial_map import ASKED_PLACE, SPEAKER_WORDS, TrackedObject, head_noun

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
# "come back", "go back", "return": a way home, not a thing. Grounded in the request like the relations.
# A distance the human said ("30 cm", "half a metre"); otherwise a distance in the call is the model's guess.
_SAID_DISTANCE = re.compile(r"\d|\b(?:half|quarter)\b|\b(?:cm|centimet\w*|met(?:er|re)s?|inch\w*|feet|foot)\b",
                            re.I)
_RETURN = re.compile(r"\b(?:come|go|get|head|drive|walk|roll) back\b|\breturn\b", re.I)
# "Come here", "come to me", "come back here": the human, wherever their voice came from.
_TO_SPEAKER = re.compile(r"\b(?:come|get|go|move|roll|drive)(?: back| over| right)? (?:here|to me|over here)\b|"
                         r"\bcome (?:closer|over)\b|\bto me\b", re.I)
PERSON_GAP_MM = 350.0  # personal space: from the edge of the person's track to my centre
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


def _means_speaker(ctx: SkillContext, target: str) -> bool:
    """Does this call mean the human who is talking? Their own words ("me", "here"), or a request to come to
    them answered with something the request does not name (the model's guess)."""
    key = target.strip().lower().strip(" .!?")
    if key in SPEAKER_WORDS:
        return True
    if not ctx.request or not _TO_SPEAKER.search(ctx.request) or not _speaker_known(ctx):
        return False  # nobody heard: "come back here" is where I was when asked (see _return_place)
    if "start" in ctx.request.lower() or key in START_WORDS:
        return False
    obj = ctx.spatial_map.resolve(target, ctx.pose())
    return obj is None or not (obj.id in ctx.request.lower()
                               or head_noun(obj.label) in ctx.request.lower().replace("_", " "))


def _speaker_known(ctx: SkillContext) -> bool:
    speaker = ctx.spatial_map.speaker
    return ctx.spatial_map.speaker_track() is not None or (speaker is not None and speaker.point is not None)


async def _find_speaker(ctx: SkillContext) -> TrackedObject | None:
    """The human who spoke; heard but not yet seen, a few looks all around first to find them."""
    heard = ctx.spatial_map.speaker
    for _ in range(3):
        if heard is None or heard.how == "seen" or ctx.fast is None:
            break
        await ctx.fast.perceive_now()
    return ctx.spatial_map.speaker_track()


def _no_speaker(ctx: SkillContext | None = None) -> SkillError:
    if ctx is not None and ctx.spatial_map.speaker is not None:
        return SkillError("I heard the human but can't see them yet. Say so; go_to me heads towards their voice.")
    return SkillError("I don't know where the human is: I have not heard or seen them. Say so, and ask them to "
                      "say something so you can hear where they are.")


async def _resolve(ctx: SkillContext, target: str) -> TrackedObject:
    """Find a target in the map ("box_1", "the red ball", "the thing behind you", "me"), searching first if
    needed.

    Directions in a description are the robot's as it was asked: after turning round to look, "the thing
    behind you" is still the thing that was behind.
    """
    if target.strip().lower().strip(" .!?") in SPEAKER_WORDS:
        person = await _find_speaker(ctx)
        if person is None:
            raise _no_speaker(ctx)
        return person
    asked_from = ctx.pose()
    obj = ctx.spatial_map.resolve(target, asked_from)
    if obj is not None:
        return obj
    query = re.sub(r"_\d+$", "", target.strip().lower()).replace("_", " ")
    found = await find(ctx, query)
    obj = (ctx.spatial_map.objects.get(found or "") or ctx.spatial_map.resolve(target, asked_from)
           or ctx.spatial_map.resolve(query, ctx.pose()))
    if obj is None:
        known = ", ".join(sorted(ctx.spatial_map.objects)) or "nothing"
        raise SkillError(f"I looked all around and could not find '{target}'. Objects I know: {known}.")
    return obj


def _return_place(ctx: SkillContext, target: str) -> str | None:
    """The place a "come back" request means when the call names something the request does not.

    Measured: handed the part "come back" after reaching the ball, the 2B model called go_to(ball_1) again
    even with the place listed and an example in the rules.
    """
    request = (ctx.request or "").lower()
    if not _RETURN.search(request) or ctx.spatial_map.place(target) is not None:
        return None
    if target.strip().lower() in START_WORDS:
        return None
    obj = ctx.spatial_map.resolve(target, ctx.pose())
    if obj is not None and (obj.id in request or head_noun(obj.label) in request.replace("_", " ")):
        return None  # "come back to the box": the object is the goal
    if "start" in request and "start" in ctx.spatial_map.places:
        return "start"
    return ASKED_PLACE if ASKED_PLACE in ctx.spatial_map.places else None


async def _go_to_speaker(ctx: SkillContext, distance_cm: float | None) -> SkillResult:
    """Come to the human: stop at a comfortable distance, facing them; follows them if they move meanwhile."""
    gap = 10.0 * distance_cm if distance_cm is not None and _SAID_DISTANCE.search(ctx.request or "") \
        else PERSON_GAP_MM
    person = await _find_speaker(ctx)
    speaker = ctx.spatial_map.speaker
    if person is None and (speaker is None or speaker.point is None):
        raise _no_speaker()
    if person is None:
        pose = ctx.pose()
        px, py = speaker.point
        dist = max(1.0, math.hypot(px - pose.x_mm, py - pose.y_mm))
        stop = max(0.0, dist - gap - 250.0)
        done = await ctx.behave(GoToPoint((pose.x_mm + (px - pose.x_mm) * stop / dist,
                                           pose.y_mm + (py - pose.y_mm) * stop / dist), "where your voice came from"))
        return SkillResult(done.ok, "I came towards your voice, but I can't see you yet." if done.ok else done.message)
    pose = ctx.pose()
    dx, dy = person.x_mm - pose.x_mm, person.y_mm - pose.y_mm
    norm = max(1.0, math.hypot(dx, dy))
    done = await ctx.behave(GoToObject(person.id, Relation("near", gap, None, (dx / norm, dy / norm))))
    ctx.attached_image = None
    if not done.ok:
        return SkillResult(False, done.message.replace(person.id, "you"))
    return SkillResult(True, "I'm here with you.", {"person": person.id})


async def go_to(ctx: SkillContext, target: str, relation: str = "near", other: str = "",
                distance_cm: float | None = None) -> SkillResult:
    if _means_speaker(ctx, target) and relation in ("near", "in_front_of", "on"):
        return await _go_to_speaker(ctx, distance_cm)
    target = _return_place(ctx, target) or target
    place = ctx.spatial_map.place(target)
    if target.strip().lower() in START_WORDS or (place is not None and place[0] == "start"):
        goal = ctx.spatial_map.places.get("start", (0.0, 0.0))
        done = await ctx.behave(GoToPoint(goal, "my start point"))
        msg = "I'm back at my start point." if done.ok else done.message
        pose = ctx.pose()
        return SkillResult(done.ok, msg, {"distance_to_goal_m": round(math.hypot(pose.x_mm - goal[0],
                                                                                 pose.y_mm - goal[1]) / 1000, 2)})
    if place is not None:
        name, goal = place
        words = "where I was when you asked" if name == ASKED_PLACE else name.replace("_", " ")
        done = await ctx.behave(GoToPoint(goal, words))
        msg = f"I'm back {words}." if done.ok else done.message
        pose = ctx.pose()
        return SkillResult(done.ok, msg, {"distance_to_goal_m": round(math.hypot(pose.x_mm - goal[0],
                                                                                 pose.y_mm - goal[1]) / 1000, 2)})
    obj = await _resolve(ctx, target)
    if distance_cm is not None and ctx.request and not _SAID_DISTANCE.search(ctx.request):
        # Measured: for "park next to it" the 2B model passed distance_cm=100 and stopped a metre away.
        distance_cm = None
    asked = _asked_relation(ctx.request)
    repaired = ""
    if asked and relation != asked:
        # The call dropped or swapped the relation the human named ("between" -> left_of, measured). Told to
        # retry, the 2B model picks yet another wrong relation, so the call is repaired from the human's words.
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
                   second.id if second else None, view, (obj.x_mm, obj.y_mm))
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
    if _means_speaker(ctx, target):
        target = "me"
    if target.strip().lower() in START_WORDS:
        sx, sy = ctx.spatial_map.places.get("start", (0.0, 0.0))
        done = await ctx.behave(Turn(bearing_to(ctx.pose(), sx, sy)))
        return SkillResult(done.ok, "I'm facing my start point." if done.ok else done.message)
    obj = await _resolve(ctx, target)
    done = await ctx.behave(Face(obj.id))
    if not done.ok:
        return SkillResult(False, done.message)
    if ctx.spatial_map.speaker is not None and ctx.spatial_map.speaker.track == obj.id:
        return SkillResult(True, "I'm facing you.")
    return SkillResult(True, f"I'm facing {ctx.spatial_map.describe(obj, ctx.pose())}.")


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
    return SkillResult(True, f"I turned {direction} {abs(angle):.0f} degrees.", in_view(ctx))


ALL_WORDS = {"all", "everything", "objects", "all objects", "all of the objects", "all the objects", "things",
             "all things", "the objects", "room"}


async def circle_around(ctx: SkillContext, target: str = "all", laps: float = 1.0, direction: str = "left",
                        radius_cm: float | None = None) -> SkillResult:
    """Drive a loop around one object, or around everything known, at a safe distance.

    "left" keeps the object on my left (anticlockwise seen from above), "right" on my right. A radius
    smaller than the safe one is raised to it.
    """
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
    if radius_cm is not None:
        radius = max(radius, min(1500.0, 10.0 * float(radius_cm)))
    pose = ctx.pose()
    clockwise = str(direction).strip().lower() in {"right", "clockwise", "cw"}
    done = await ctx.behave(Orbit((cx, cy), radius, laps, math.atan2(pose.y_mm - cy, pose.x_mm - cx), name,
                                  clockwise))
    return SkillResult(done.ok, done.message)


_FOLLOW_ASKED = re.compile(r"\bfollow|\bafter me\b|\bcome with me\b|\bkeep up\b", re.I)


async def follow(ctx: SkillContext, target: str, distance_cm: float | None = None) -> SkillResult:
    """Start following; this keeps going after the reply until stopped or the target is lost."""
    if ctx.request and _TO_SPEAKER.search(ctx.request) and not _FOLLOW_ASKED.search(ctx.request) \
            and _means_speaker(ctx, target):
        # "Come here" answered with follow (measured with the 2B model): come once, then stay.
        return await _go_to_speaker(ctx, distance_cm)
    if distance_cm is None or (ctx.request and not _SAID_DISTANCE.search(ctx.request)):
        distance_cm = 60.0  # a distance the human did not say is the model's guess (it picked 30 cm)
    obj = await _resolve(ctx, "me" if _means_speaker(ctx, target) else target)
    ctx.check()
    ctx.fast.start_behavior(Follow(obj.id, distance_cm * 10.0))
    who = "you" if ctx.spatial_map.speaker is not None and ctx.spatial_map.speaker.track == obj.id else obj.id
    return SkillResult(True, f"I'm following {who} now, about {distance_cm:.0f} cm behind, and I'll keep "
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
    target = {"type": "string", "description": "a known object id such as box_1; or the human's description, "
              "such as 'charging dock', 'the thing behind you' or 'the closest box' (I search for it if it is not "
              "known yet); or me, the human talking to you (you hear where their voice comes from); or start, "
              "the spot where I was switched on"}
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
              {"target": target}, face, ["target"]),
        Skill("turn", "Turn in place, e.g. 'look left' is left and 'turn around' is around (a half turn). "
              "For other amounts use left or right with degrees, e.g. 360 for a full spin.",
              {"direction": {"type": "string", "enum": ["left", "right", "around"]},
               "degrees": {"type": "number", "minimum": 5, "maximum": 1080}},
              turn, ["direction"]),
        Skill("circle_around", "Drive a loop around one object or around all known objects, keeping a safe "
              "distance.", {"target": {"type": "string", "description": "object id or the human's description "
                                                                        "of it, or all"},
                            "laps": {"type": "number", "minimum": 0.25, "maximum": 3},
                            "direction": {"type": "string", "enum": ["left", "right"],
                                          "description": "left = anticlockwise (default), right = clockwise"},
                            "radius_cm": {"type": "number", "description": "optional: a wider circle"}},
              circle_around),
        Skill("move", "Drive straight forward or backward. 'A little' is about 20 cm.",
              {"direction": {"type": "string", "enum": ["forward", "backward"]},
               "distance_cm": {"type": "number", "minimum": 5, "maximum": 200}},
              move, ["direction"]),
        Skill("tilt_head", "Tilt your head camera up, down, or level.",
              {"direction": {"type": "string", "enum": ["up", "down", "level"]}}, tilt_head, ["direction"]),
        Skill("explore", "Drive to a part of the room you have not seen yet and look around there.", {}, explore),
        # Listed last: a small model favours tools near the top, and these two are rarely the right first step.
        Skill("follow", "Keep following a moving person or object until told to stop, e.g. 'follow me' is "
              "target me.",
              {"target": target, "distance_cm": {"type": "number", "minimum": 30, "maximum": 200}},
              follow, ["target"]),
        Skill("stop", "Halt all motion when the human says stop or freeze.", {}, stop),
    ]

