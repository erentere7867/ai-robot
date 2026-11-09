"""Repairing a small model's tool calls from the words of the request.

A 2B model picks the right tool most of the time and gets the edges wrong: it drops the relation the human
named, calls go_to on the last object for "come back", invents a distance. Each rule here undoes one such
measured mistake, before the call reaches a skill, so skills stay plain capabilities and the rules can be
deleted together when a better model no longer needs them. Rules only ever look at what the human actually said.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from robot.world.labels import SPEAKER_WORDS, START_WORDS, head_noun
from robot.world.spatial_map import ASKED_PLACE

RELATION_WORDS = {"between": re.compile(r"\bbetween\b"), "behind": re.compile(r"\bbehind\b|\bfar side\b"),
                  "left_of": re.compile(r"\bleft (?:of|side)\b"),
                  "right_of": re.compile(r"\bright (?:of|side)\b")}
_BETWEEN = re.compile(r"\bbetween (?:the )?(.+?) and (?:the )?(.+?)(?:[.,!?]|$)")
_SAID_DISTANCE = re.compile(r"\d|\b(?:half|quarter)\b|\b(?:cm|centimet\w*|met(?:er|re)s?|inch\w*|feet|foot)\b", re.I)
_RETURN = re.compile(r"\b(?:come|go|get|head|drive|walk|roll) back\b|\breturn\b", re.I)
_TO_SPEAKER = re.compile(r"\b(?:come|get|go|move|roll|drive)(?: back| over| right)? (?:here|to me|over here)\b|"
                         r"\bcome (?:closer|over)\b|\bto me\b", re.I)
_ON_WORDS = re.compile(r"\b(?:on|onto|on top|dock|charg\w*|mat|platform|pad)\b", re.I)
_FOLLOW_ASKED = re.compile(r"\bfollow|\bafter me\b|\bcome with me\b|\bkeep up\b", re.I)
_LOOK_AT_ME = re.compile(r"\b(?:look|turn)(?: right| over)? (?:at|to|towards?) me\b|\bface me\b", re.I)
STOP_ASKED = re.compile(r"\b(?:stop|halt|freeze|wait|enough|quit|cancel|stay|don'?t move|hold on)\b", re.I)
# A request that is a plain command to move or look ("look at me", "come here", "turn around").
COMMAND = re.compile(r"^\s*(?:please\s+|(?:can|could|would) you\s+)?(?:go|come|turn|look|move|drive|spin|back|"
                     r"park|circle|follow|face|return|head|roll|get|walk|run|explore|find|search|fetch|bring)\b",
                     re.I)


@dataclass
class Call:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    note: str = ""  # added to the skill's report, so the model knows what actually ran
    refusal: str = ""  # not run; this goes back to the model instead


def _speaker_known(ctx: Any) -> bool:
    world = ctx.spatial_map
    return world.speaker_track() is not None or (world.speaker is not None and world.speaker.point is not None)


def _means_speaker(ctx: Any, request: str, target: str) -> bool:
    """Their own words ("me", "here"), or a request to come to them answered with something it does not name."""
    key = target.strip().lower().strip(" .!?")
    if key in SPEAKER_WORDS:
        return True
    if not _TO_SPEAKER.search(request) or not _speaker_known(ctx):
        return False
    if "start" in request.lower() or key in START_WORDS:
        return False
    obj = ctx.spatial_map.resolve(target, ctx.pose())
    return obj is None or not (obj.id in request.lower() or head_noun(obj.label) in request.lower().replace("_", " "))


def _return_place(ctx: Any, request: str, target: str) -> str | None:
    """The place a "come back" request means when the call names something the request does not.

    Measured: handed the part "come back" after reaching the ball, the 2B model called go_to(ball_1) again
    even with the place listed and an example in the rules.
    """
    lowered = request.lower()
    world = ctx.spatial_map
    if not _RETURN.search(lowered) or world.place(target) is not None or target.strip().lower() in START_WORDS:
        return None
    obj = world.resolve(target, ctx.pose())
    if obj is not None and (obj.id in lowered or head_noun(obj.label) in lowered.replace("_", " ")):
        return None  # "come back to the box": the object is the goal
    if "start" in lowered and "start" in world.places:
        return "start"
    return ASKED_PLACE if ASKED_PLACE in world.places else None


def _asked_relation(request: str) -> str | None:
    """The one relation the request names, if it names exactly one."""
    named = [rel for rel, pattern in RELATION_WORDS.items() if pattern.search(request.lower())]
    return named[0] if len(named) == 1 else None


def _other_named(ctx: Any, request: str, target: str) -> str | None:
    """For "between the ball and the cube": the known object named in the request that is not the target."""
    match = _BETWEEN.search(request.lower())
    if not match:
        return None
    first = ctx.spatial_map.resolve(target, ctx.pose())
    for phrase in match.groups():
        found = ctx.spatial_map.resolve(phrase.strip(), ctx.pose())
        if found is not None and (first is None or found.id != first.id):
            return found.id
    return None


def _go_to(ctx: Any, request: str, call: Call) -> Call:
    args = call.args
    target = str(args.get("target", ""))
    relation = str(args.get("relation", "near"))
    if not _SAID_DISTANCE.search(request):
        args.pop("distance_cm", None)  # measured: for "park next to it" the model passed 100 and stopped a metre away
    if _means_speaker(ctx, request, target) and relation in ("near", "in_front_of", "on"):
        args["target"] = "me"
        return call
    place = _return_place(ctx, request, target)
    if place is not None:
        args["target"] = place
        return call
    if ctx.spatial_map.place(target) is not None or target.strip().lower() in START_WORDS:
        return call
    asked = _asked_relation(request)
    if asked and relation != asked:
        # Measured: told to retry, the 2B model picks yet another wrong relation.
        if asked == "between" and not args.get("other"):
            args["other"] = _other_named(ctx, request, target) or ""
            if not args["other"]:
                call.refusal = ("The human asked for 'between': call go_to with relation 'between' and the "
                                "second object's id in 'other'.")
                return call
        args["relation"], relation = asked, asked
        call.note = f" (I went {asked.replace('_', ' ')}, as the human asked.)"
    if relation == "on" and not _ON_WORDS.search(request):
        args["relation"] = "near"  # measured: "on" for "go to the red ball", driving into the ball
    return call


def _follow(ctx: Any, request: str, call: Call) -> Call:
    target = str(call.args.get("target", ""))
    if not _SAID_DISTANCE.search(request):
        call.args.pop("distance_cm", None)  # the model's guess (it picked 30 cm)
    means_speaker = _means_speaker(ctx, request, target)
    if _TO_SPEAKER.search(request) and not _FOLLOW_ASKED.search(request) and means_speaker:
        return Call("go_to", {**call.args, "target": "me"})  # measured: "come here" answered with follow
    if means_speaker:
        call.args["target"] = "me"
    return call


def repair_call(tool: str, args: dict[str, Any], request: str, ctx: Any) -> Call:
    """`request` is the words of the current part of what the human asked; empty on background turns."""
    call = Call(tool, dict(args))
    if not request:
        return call
    if tool == "go_to":
        return _go_to(ctx, request, call)
    if tool == "follow":
        return _follow(ctx, request, call)
    if tool == "turn_to" and _means_speaker(ctx, request, str(args.get("target", ""))):
        call.args["target"] = "me"
    elif tool == "observe" and _LOOK_AT_ME.search(request) and ctx.spatial_map.speaker is not None:
        return Call("turn_to", {"target": "me"})  # measured: "look at me" read as observe
    elif tool == "stop" and COMMAND.match(request) and not STOP_ASKED.search(request):
        # Measured: "come back to where you started" answered with stop ("start"/"stop").
        call.refusal = "The human did not ask you to stop. Do what they asked instead."
    return call
