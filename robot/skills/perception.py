"""Looking: camera capture + open-vocabulary detection + map fusion."""

from __future__ import annotations

from typing import Any

from robot.skills.base import Skill, SkillContext, SkillResult
from robot.world.spatial_map import direction_words, label_score

LOOK_AROUND_STEP_DEG = 60.0
GENERIC_QUERIES = {"room", "everything", "anything", "all", "objects", "object", "things", "thing", "scene",
                   "surroundings", "around", "stuff", "view", "what", "it", "that", "front", "area"}


def _query(text: str) -> str:
    text = text.strip().lower()
    return "" if text in GENERIC_QUERIES or text.endswith(" room") else text


async def perceive(ctx: SkillContext, query: str | None = None) -> dict[str, Any]:
    """One camera view fused into the map. Returns ids seen, new ids, lost ids and unplaced labels."""
    ctx.check()
    frame = ctx.camera.capture()
    detections = await ctx.detector.detect(frame, query)
    ctx.check()
    update = ctx.spatial_map.integrate_view(frame.pose, frame.model, detections, query=query)
    ctx.attached_image = frame
    unplaced = [f"{d.label} ({direction_words(frame.model.bearing_deg(d.center_u))}, far or raised)"
                for d in update.unplaced]
    await ctx.emit("perception.view", {
        "detections": [{"label": d.label, "bbox": [round(v, 3) for v in d.bbox]} for d in detections],
        "seen": [o.id for o, _, _ in update.seen], "lost": [o.id for o in update.lost], "query": query,
    })
    return {"seen": [o.id for o, _, _ in update.seen], "new": [o.id for o, _, new in update.seen if new],
            "lost": [o.id for o in update.lost], "unplaced": unplaced}


def _summary(ctx: SkillContext, ids: list[str]) -> list[str]:
    pose = ctx.pose()
    objs = [ctx.spatial_map.objects[i] for i in dict.fromkeys(ids) if i in ctx.spatial_map.objects]
    objs.sort(key=lambda o: o.relative(pose)[0])
    return [ctx.spatial_map.describe(o, pose).replace(", seen just now", "") for o in objs]


async def observe(ctx: SkillContext) -> SkillResult:
    pose = ctx.pose()
    key = (round(pose.x_mm, -1), round(pose.y_mm, -1), round(pose.heading_deg), round(pose.head_tilt_deg))
    cached = ctx.turn_cache.get("observe")
    if cached is not None and cached[0] == key:
        # Looking again from the same spot shows the same thing; a small model sometimes loops here.
        return SkillResult(True, "You already looked from here and nothing moved. Answer the human now.",
                           cached[1])
    view = await perceive(ctx, None)
    seen = _summary(ctx, view["seen"])
    data: dict[str, Any] = {"objects": seen}
    if view["unplaced"]:
        data["also_visible"] = view["unplaced"]
    if view["lost"]:
        data["no_longer_there"] = view["lost"]
    ctx.turn_cache["observe"] = (key, data)
    if not seen and not view["unplaced"]:
        return SkillResult(True, "I see no distinct objects in front of me.", data)
    return SkillResult(True, f"I can see {len(seen) + len(view['unplaced'])} thing(s). If that answers "
                             "the request, reply now.", data)


async def _face_found(ctx: SkillContext, obj_id: str) -> SkillResult:
    target = ctx.spatial_map.objects[obj_id]
    await ctx.face_point(target.x_mm, target.y_mm)
    return SkillResult(True, f"Found it and turned to face it: {ctx.spatial_map.describe(target, ctx.pose())}.",
                       {"found": target.id})


async def look_around(ctx: SkillContext, query: str = "") -> SkillResult:
    """Scan a full circle in steps, stopping early (and facing it) when `query` is clearly found."""
    query = _query(query)
    seen_ids: list[str] = []
    fallback: str | None = None  # a synonym match (asked for a box, saw a cube) if nothing better turns up
    steps = int(round(360.0 / LOOK_AROUND_STEP_DEG))
    for step in range(steps):
        view = await perceive(ctx, None)
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
        if step < steps - 1:
            result = await ctx.spin(LOOK_AROUND_STEP_DEG)
            if result is not None and not result.ok:
                return SkillResult(False, f"Stopped looking around: {result.message}",
                                   {"objects": _summary(ctx, seen_ids)})
    # Finish where we started so the scan does not leave the robot facing an arbitrary way.
    await ctx.spin(LOOK_AROUND_STEP_DEG)
    ctx.attached_image = None  # the last frame is not what the robot faces now
    objects = _summary(ctx, seen_ids)
    if query and fallback is not None and fallback in ctx.spatial_map.objects:
        return await _face_found(ctx, fallback)
    if query:
        return SkillResult(True, f"I looked all around but did not find any {query}.", {"objects": objects})
    return SkillResult(True, f"I looked all around and know of {len(objects)} object(s).", {"objects": objects})


def skills() -> list[Skill]:
    return [
        Skill("observe", "Look through your head camera at what is in front of you right now, e.g. for "
              "'what do you see?'. Returns each object with an id, direction and distance.", {}, observe),
        Skill("look_around", "Turn a full circle while looking, to find things that are not in front of you "
              "or to learn what is in the room. With query, stops and faces the first match.",
              {"query": {"type": "string", "description": "optional: the thing you are searching for"}},
              look_around),
    ]
