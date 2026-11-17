"""The context builder: formats compact 360° scene summary for the closed-loop multimodal planner."""

from __future__ import annotations

from typing import Any

from robot.agent.api import Object

MAX_OBJECTS = 12
_LABEL_ALIASES = {
    "desk": "desk(table)",
    "sofa": "sofa(couch)",
}

_AHEAD_SIDES = frozenset({"ahead", "front", "front-left", "front-right"})
_SIDE_SIDES = frozenset({"left", "right"})
_BEHIND_SIDES = frozenset(
    {"behind", "behind-left", "behind-right", "rear", "rear-left", "rear-right", "back"}
)


def _ego_region(side: str, bearing: float) -> str:
    low = (side or "").strip().lower()
    if low in _AHEAD_SIDES:
        return "ahead"
    if low in _SIDE_SIDES:
        return "sides"
    if low in _BEHIND_SIDES:
        return "behind"
    ab = abs(float(bearing or 0.0))
    if ab <= 67.5:
        return "ahead"
    if ab >= 112.5:
        return "behind"
    return "sides"


def synthesize_perception(objects: list[Object], user: int | None = None) -> str:
    """Summarize all currently known objects into sorted ego-centric 360° regions."""
    if not objects:
        return "(nothing detected yet)"

    groups: dict[str, list[str]] = {"ahead": [], "sides": [], "behind": []}
    for o in sorted(objects, key=lambda obj: float(getattr(obj, "distance", 0.0))):
        if user is not None and o.id == user:
            name = "you (person)"
        else:
            base = _LABEL_ALIASES.get(o.label, o.label).replace("_", " ")
            color = (getattr(o, "color", "") or "").strip()
            if color and color.lower() != "unknown" and not base.lower().startswith(f"{color.lower()} "):
                name = f"{color} {base}"
            else:
                name = base
        dist = float(getattr(o, "distance", 0.0))
        region = _ego_region(getattr(o, "side", ""), float(getattr(o, "bearing", 0.0)))
        groups[region].append(f"{name} ({dist:.1f} m)")

    parts = [f"{region}: {', '.join(items)}" for region, items in groups.items() if items]
    return "; ".join(parts) if parts else "(nothing detected yet)"


def scene_table(objects: list[Object], user: int | None = None) -> str:
    rows = ["id  label          color   dist    bearing  side          moving  seen"]
    for o in sorted(objects, key=lambda obj: obj.distance)[:MAX_OBJECTS]:
        label = "person(user)" if o.id == user else _LABEL_ALIASES.get(o.label, o.label)
        seen = "now" if o.last_seen < 1.0 else f"{o.last_seen:.0f} s ago"
        moving = "yes" if getattr(o, "moving", False) or getattr(o, "speed", 0.0) > 0.05 else "no"
        color = o.color or "-"
        rows.append(
            f"{o.id:<3} {label:<14} {color:<7} {o.distance:4.1f} m  {o.bearing:+5.0f}°   "
            f"{o.side:<13} {moving:<7} {seen}"
        )
    return "\n".join(rows) if len(rows) > 1 else "(nothing seen yet)"


def describe(snapshot: dict[str, Any], memory: Any = None, scratchpad: Any = None) -> str:
    """Robot state, 360° scene table, human pose, scratchpad, habit anchors and notes."""
    room = snapshot.get("room", "unknown")
    rooms = snapshot.get("rooms", [])
    battery = snapshot.get("battery", 1.0)
    posture = snapshot.get("posture", "neutral")
    others = ", ".join(r for r in rooms if r != room)

    objects = snapshot.get("objects", [])
    user_id = snapshot.get("user")
    pose = snapshot.get("pose")

    lines = [
        f"Robot: in room {room}, battery {battery:.0%}, posture {posture}. Other rooms: {others or 'none'}.",
    ]
    if pose is not None:
        if hasattr(pose, "x"):
            lines.append(f"Pose: x={pose.x:.2f}, y={pose.y:.2f}, yaw={getattr(pose, 'yaw', 0.0):.0f}°")
        elif isinstance(pose, (tuple, list)) and len(pose) >= 2:
            yaw = float(pose[2]) if len(pose) >= 3 else 0.0
            lines.append(f"Pose: x={float(pose[0]):.2f}, y={float(pose[1]):.2f}, yaw={yaw:.0f}°")

    user_obj = next(
        (
            o
            for o in objects
            if (user_id is not None and getattr(o, "id", None) == user_id)
            or "user" in str(getattr(o, "label", "")).lower()
            or "user" in str(getattr(o, "what", "")).lower()
            or getattr(o, "name", "") == "person(user)"
        ),
        None,
    )
    if user_obj is not None:
        u_id = getattr(user_obj, "id", user_id)
        u_name = (
            getattr(user_obj, "name", None)
            or ("person(user)" if getattr(user_obj, "id", None) == user_id else getattr(user_obj, "label", "person(user)"))
        )
        u_dist = float(getattr(user_obj, "distance", 0.0))
        u_side = getattr(user_obj, "side", "front")
        u_bearing = float(getattr(user_obj, "bearing", 0.0))
        lines.append(
            f"User: id={u_id} ({u_name}) at {u_dist:.1f} m {u_side} ({u_bearing:+.0f}°); "
            f"use target: user or target: {u_id} when approaching or facing the user."
        )

    lines.append(f"Surroundings: {synthesize_perception(objects, user_id)}")
    lines.append(f"Known objects:\n{scene_table(objects, user_id)}")

    human_pose = snapshot.get("human_pose")
    if human_pose is not None:
        p_posture = getattr(human_pose, "posture", "standing")
        pointing = getattr(human_pose, "pointing_ray", None)
        gaze = getattr(human_pose, "gaze_ray", None)
        details = [f"posture: {p_posture}"]
        if pointing is not None:
            details.append(f"pointing: {pointing}")
        if gaze is not None:
            details.append(f"gaze: {gaze}")
        lines.append(f"Human state: {', '.join(details)}")

    sp_summary = ""
    if scratchpad is not None:
        if hasattr(scratchpad, "summary"):
            sp_summary = scratchpad.summary()
        elif isinstance(scratchpad, str):
            sp_summary = scratchpad
        elif hasattr(scratchpad, "goal"):
            sp_summary = (
                f"goal: {scratchpad.goal}, subgoal: {getattr(scratchpad, 'subgoal', '')}, "
                f"visited: {getattr(scratchpad, 'visited_rooms', set())}, "
                f"inspected: {getattr(scratchpad, 'inspected_ids', set())}"
            )
    elif snapshot.get("scratchpad"):
        sp_summary = str(snapshot["scratchpad"])

    if sp_summary:
        lines.append(f"Scratchpad: {sp_summary}")

    if memory is not None and hasattr(memory, "usual_anchors"):
        habits: list[str] = []
        for obj in snapshot.get("objects", []):
            if hasattr(memory, "usual_anchors"):
                anchors = memory.usual_anchors(obj.label)
                if anchors:
                    top_anchor = anchors[0][0] if isinstance(anchors[0], (list, tuple)) else anchors[0]
                    habits.append(f"{obj.label}->{top_anchor}")
        if habits:
            lines.append(f"Habit anchors: {', '.join(habits)}")

    notes = snapshot.get("notes", [])
    if memory is not None and hasattr(memory, "notes"):
        try:
            notes = memory.notes()
        except Exception:
            pass
    if notes:
        lines.append(f"Route notes / facts: {'; '.join(notes[-3:])}")

    return "\n".join(lines)
