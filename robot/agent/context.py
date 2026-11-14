"""The context builder: formats compact 360° scene summary for the closed-loop multimodal planner."""

from __future__ import annotations

from typing import Any

from robot.agent.api import Object

MAX_OBJECTS = 12


def scene_table(objects: list[Object], user: int | None = None) -> str:
    rows = ["id  label          color   dist    bearing  side          moving  seen"]
    for o in sorted(objects, key=lambda obj: obj.distance)[:MAX_OBJECTS]:
        label = "person(user)" if o.id == user else o.label
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

    lines = [
        f"Robot: in room {room}, battery {battery:.0%}, posture {posture}. Other rooms: {others or 'none'}.",
        f"Known objects:\n{scene_table(snapshot.get('objects', []), snapshot.get('user'))}",
    ]

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
