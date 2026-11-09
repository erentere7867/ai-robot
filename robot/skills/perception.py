"""Looking: the fast detector keeps the map current from every camera; these tools read it, look fresh, or search.

The cameras see all around, so "what's behind you?" needs no turn. A search for something not in view yet
is one smooth turn while every frame is checked (the front camera sees smaller things farther away). Things
outside the detector's classes ("my keys") fall back to the VLM: slower, but it recognises anything.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from robot.control.behaviors import Scan, Search, bearing_to
from robot.core.angles import wrap_deg
from robot.perception.camera import side_by_side
from robot.perception.geometry import camera_for
from robot.skills.base import Skill, SkillContext, SkillResult
from robot.world.labels import direction_words, label_score

LOOK_AROUND_STEP_DEG = 60.0
GENERIC_QUERIES = {"room", "everything", "anything", "all", "objects", "object", "things", "thing", "scene",
                   "surroundings", "around", "stuff", "view", "what", "it", "that", "front", "area", "items",
                   "item", "something"}


def _query(text: str) -> str:
    """'' for a general look ("any object", "new things in the room"), else the thing to search for."""
    text = text.strip().lower()
    words = [w for w in text.replace("_", " ").split() if w not in {"any", "a", "an", "the", "new", "interesting",
                                                                     "in", "of", "for", "some", "my", "this"}]
    return "" if not words or all(w in GENERIC_QUERIES for w in words) else text


_SIDES = {"behind": "around", "back": "around", "backwards": "around", "left": "left", "right": "right"}
_SIDE_FILLER = {"me", "you", "your", "my", "to", "the", "on", "of", "at", "side", "what", "whats", "is", "there",
                "thing", "things", "object", "objects", "hand"}


def _side(query: str) -> str | None:
    """"behind me" or "on your left" is a place to look, not a thing to search for: the turn direction."""
    words = re.findall(r"[a-z]+", query.lower())
    sides = {_SIDES[w] for w in words if w in _SIDES}
    if len(sides) == 1 and all(w in _SIDES or w in _SIDE_FILLER for w in words):
        return sides.pop()
    return None


def _streaming(ctx: SkillContext) -> bool:
    return ctx.fast.detector is not None


def _in_vocabulary(ctx: SkillContext, query: str) -> bool:
    return any(label_score(label, query) >= 1 for label in ctx.fast.detector.vocabulary)


async def _fuse(ctx: SkillContext, frame: Any, detections: list[Any], query: str | None) -> dict[str, Any]:
    """Place a view's detections in the map from the pose at capture (the robot may have moved since)."""
    ctx.check()
    update = ctx.spatial_map.integrate_view(frame.pose, frame.model, detections, query=query)
    ctx.attached_image = frame
    return await _view_result(ctx, frame, detections, update, query)


async def _view_result(ctx: SkillContext, frame: Any, detections: list[Any], update: Any, query: str | None
                       ) -> dict[str, Any]:
    unplaced = [f"{d.label} ({direction_words(frame.model.bearing_deg(d.center_u))}, far or raised)"
                for d in update.unplaced]
    await ctx.emit("perception.view", {
        "detections": [{"label": d.label, "bbox": [round(v, 3) for v in d.bbox]} for d in detections],
        "seen": [o.id for o, _, _ in update.seen], "lost": [o.id for o in update.lost], "query": query,
    })
    return {"seen": [o.id for o, _, _ in update.seen], "new": [o.id for o, _, new in update.seen if new],
            "lost": [o.id for o in update.lost], "unplaced": unplaced, "pairs": update.seen}


async def _name_unknown(ctx: SkillContext, frame: Any, pairs: list[Any]) -> None:
    """Ask the VLM what the fast detector's unnamed "object"s are (once per object)."""
    for obj, det, _ in pairs[:3]:
        if obj.label == "object":
            name = await ctx.vlm.name_object(frame, det.bbox)
            if name:
                obj.label = name


def _summary(ctx: SkillContext, ids: list[str]) -> list[str]:
    pose = ctx.pose()
    objs = [ctx.spatial_map.objects[i] for i in dict.fromkeys(ids) if i in ctx.spatial_map.objects]
    objs.sort(key=lambda o: o.relative(pose)[0])
    return [ctx.spatial_map.describe(o, pose).replace(", seen just now", "") for o in objs]


# Where each direction is (relative to my heading) and how wide "on my left" is.
DIRECTIONS = {"front": 0.0, "left": 90.0, "right": -90.0, "behind": 180.0}
SECTOR_HALF_DEG = 65.0


def _in_sector(ctx: SkillContext, obj_id: str, direction: str) -> bool:
    if direction not in DIRECTIONS:
        return True
    obj = ctx.spatial_map.objects.get(obj_id)
    if obj is None:
        return False
    return abs(wrap_deg(obj.relative(ctx.pose())[1] - DIRECTIONS[direction])) <= SECTOR_HALF_DEG


def _image_for(views: list[Any], direction: str) -> Any:
    """The view to show the model: the camera facing that way; behind me, both rear cameras side by side."""
    frames = [frame for frame, _, _ in views if frame is not None]
    if not frames:
        return None
    if direction in ("around", "front") or len(frames) == 1:
        return frames[0]
    centre = DIRECTIONS[direction]
    facing = [f for f in frames if f.model.covers(centre, 10.0)]
    if direction == "behind" and len(facing) >= 2:
        return side_by_side(sorted(facing, key=lambda f: f.model.yaw_deg)[:2])
    return min(frames, key=lambda f: abs(wrap_deg(centre - f.model.yaw_deg)))


async def observe(ctx: SkillContext, direction: str = "around") -> SkillResult:
    """Look through every camera at once: what is around me (or only in one direction), without turning."""
    pose = ctx.pose()
    key = (round(pose.x_mm, -1), round(pose.y_mm, -1), round(pose.heading_deg), round(pose.head_tilt_deg),
           direction)
    cached = ctx.turn_cache.get("observe")
    if cached is not None and cached[0] == key:
        # Looking again from the same spot shows the same thing; a small model sometimes loops here.
        return SkillResult(True, "You already looked from here and nothing moved; use what you saw.",
                           cached[1])
    if _streaming(ctx):
        views = await ctx.fast.perceive_now()
        results = [await _view_result(ctx, frame, dets, update, None) for frame, dets, update in views]
        for (frame, _, _), view in zip(views, results, strict=True):
            await _name_unknown(ctx, frame, view["pairs"])
    else:
        centre = DIRECTIONS.get(direction, 0.0)
        frame = ctx.camera.capture(camera_for(ctx.camera.models, centre).name)
        results = [await _fuse(ctx, frame, await ctx.vlm.detect(frame, None), None)]
        views = [(frame, None, None)]
    ctx.attached_image = _image_for(views, direction)
    seen_ids = [i for view in results for i in view["seen"]]
    here = [i for i in seen_ids if _in_sector(ctx, i, direction)]
    seen = _summary(ctx, here)
    data: dict[str, Any] = {"objects": seen}
    elsewhere = _summary(ctx, [i for i in seen_ids if i not in here])
    if elsewhere:
        data["elsewhere_around_me"] = elsewhere
    unplaced = [u for view in results for u in view["unplaced"]]
    if unplaced:
        data["also_visible"] = unplaced
    lost = [i for view in results for i in view["lost"]]
    if lost:
        data["no_longer_there"] = lost
    ctx.turn_cache["observe"] = (key, data)
    # Looking again from exactly where I last looked (an earlier turn) shows the same view. A small model
    # sometimes answers "now look right" by looking again without turning; saying so lets it correct that.
    same_spot = ctx.last_look == key
    ctx.last_look = key
    note = " I have not moved since I last looked." if same_spot else ""
    where = "around me" if direction == "around" else ("in front of me" if direction == "front"
                                                        else "behind me" if direction == "behind"
                                                        else f"on my {direction}")
    if not seen and not unplaced:
        return SkillResult(True, f"I see no distinct objects {where}." + note, data)
    return SkillResult(True, f"I can see {len(seen) + len(unplaced)} thing(s) {where}." + note, data)


def in_view(ctx: SkillContext) -> dict[str, Any]:
    """What the fast detector sees right now (it watched during the motion): {} without a streaming detector."""
    if ctx.fast.last_frame is None:
        return {}
    now = ctx.clock.monotonic()
    ids = [o.id for o in ctx.spatial_map.objects.values() if now - o.last_seen < 1.5 and o.confidence >= 0.4]
    return {"in_view_now": _summary(ctx, ids) or ["nothing I can name"]}


async def find(ctx: SkillContext, query: str) -> str | None:
    """Search all around for one thing; returns its id (facing it) or None."""
    result = await look_around(ctx, query)
    return result.data.get("found")


async def look_around(ctx: SkillContext, query: str = "") -> SkillResult:
    side = _side(query)
    if side is not None:
        return await observe(ctx, {"around": "behind"}.get(side, side))  # the cameras already look there
    query = _query(query)
    if _streaming(ctx) and (not query or _in_vocabulary(ctx, query)):
        started = ctx.clock.monotonic()
        if query:
            done = await ctx.behave(Search(query))
            ctx.attached_image = None
            if done.ok:
                return SkillResult(True, done.message, done.data)
            seen = [o.id for o in ctx.spatial_map.objects.values() if o.last_seen >= started]
            return SkillResult(True, done.message, {"objects": _summary(ctx, seen)})
        done = await ctx.behave(Scan())
        ctx.attached_image = None
        seen = [o.id for o in ctx.spatial_map.objects.values() if o.last_seen >= started and o.confidence >= 0.4]
        if not done.ok:
            return SkillResult(False, f"Stopped looking around: {done.message}", {"objects": _summary(ctx, seen)})
        objects = _summary(ctx, seen)
        return SkillResult(True, f"I looked all around and know of {len(objects)} object(s).", {"objects": objects})
    return await _look_around_vlm(ctx, query)


async def _face_found(ctx: SkillContext, obj_id: str) -> SkillResult:
    target = ctx.spatial_map.objects[obj_id]
    await ctx.turn_by(bearing_to(ctx.pose(), target.x_mm, target.y_mm))
    return SkillResult(True, f"Found it and turned to face it: {ctx.spatial_map.describe(target, ctx.pose())}.",
                       {"found": target.id})


async def _look_around_vlm(ctx: SkillContext, query: str) -> SkillResult:
    """Stop-and-look scan with the VLM: slow, but it recognises things the fast detector has no class for."""
    seen_ids: list[str] = []
    fallback: str | None = None  # a synonym match (asked for a box, saw a cube) if nothing better turns up
    steps = int(round(360.0 / LOOK_AROUND_STEP_DEG))
    start_heading = ctx.pose().heading_deg
    for step in range(steps):
        ctx.check()
        frame = ctx.camera.capture()
        detecting = asyncio.ensure_future(ctx.vlm.detect(frame, query or None))
        spun = None
        try:
            if step < steps - 1:
                # Turn to the next view while the model reads this one (the frame keeps its own pose).
                spun = await ctx.turn_by(wrap_deg(start_heading + (step + 1) * LOOK_AROUND_STEP_DEG
                                                  - ctx.pose().heading_deg))
            view = await _fuse(ctx, frame, await detecting, query or None)
        finally:
            detecting.cancel()
        seen_ids.extend(view["seen"])
        if query:
            pose = ctx.pose()
            scored = sorted(((label_score(ctx.spatial_map.objects[i].label, query),
                              ctx.spatial_map.objects[i].relative(pose)[0], i)
                             for i in view["seen"] if i in ctx.spatial_map.objects), key=lambda t: (-t[0], t[1]))
            if scored and scored[0][0] == 2:
                return await _face_found(ctx, scored[0][2])
            if scored and scored[0][0] == 1 and fallback is None:
                fallback = scored[0][2]
        if spun is not None and not spun.ok:
            return SkillResult(False, f"Stopped looking around: {spun.message}",
                               {"objects": _summary(ctx, seen_ids)})
    # Finish where we started so the scan does not leave the robot facing an arbitrary way.
    await ctx.turn_by(wrap_deg(start_heading - ctx.pose().heading_deg))
    ctx.attached_image = None  # the last frame is not what the robot faces now
    objects = _summary(ctx, seen_ids)
    if query and fallback is not None and fallback in ctx.spatial_map.objects:
        return await _face_found(ctx, fallback)
    if query:
        return SkillResult(True, f"I looked all around but did not find any {query}.", {"objects": objects})
    return SkillResult(True, f"I looked all around and know of {len(objects)} object(s).", {"objects": objects})


def skills() -> list[Skill]:
    return [
        Skill("observe", "Look through your cameras, which see all around you at once, e.g. for 'what do you "
              "see?' or 'what's behind you?' (direction behind; no need to turn). Returns each object with an id, "
              "direction and distance.",
              {"direction": {"type": "string", "enum": ["around", "front", "left", "right", "behind"],
                             "aliases": {"back": "behind", "ahead": "front", "forward": "front",
                                         "backward": "behind", "backwards": "behind", "all": "around",
                                         "everywhere": "around", "all_around": "around"}}},
              observe, acts=True, idempotent=True),
        Skill("look_around", "Turn a full circle while looking, to find things that are not in front of you "
              "or to learn what is in the room. With query, stops and faces the first match.",
              {"query": {"type": "string", "description": "optional: the thing you are searching for"}},
              look_around, acts=True, idempotent=True),
    ]
