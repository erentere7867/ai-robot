"""The context builder: the robot's state and nearby known objects as text, with directions already worked out.

The planner never computes "behind" or "nearest" itself: the words come from code (architecture section 6). The table
is short (the nearest MAX_OBJECTS) and every row has an id, so a program can say `robot.obj(7)` and get the object as
it is when the line runs, instead of copying coordinates that go stale while the robot moves.
"""

from __future__ import annotations

from typing import Any

from robot.agent.api import Object

MAX_OBJECTS = 10


def scene_table(objects: list[Object], user: int | None = None) -> str:
    rows = ["id  label          color   dist    bearing  side          seen"]
    for o in sorted(objects, key=lambda o: o.distance)[:MAX_OBJECTS]:
        label = "person(user)" if o.id == user else o.label
        seen = "now" if o.last_seen < 1.0 else f"{o.last_seen:.0f} s ago"
        rows.append(f"{o.id:<3} {label:<14} {o.color or '-':<7} {o.distance:4.1f} m  {o.bearing:+5.0f}°   "
                    f"{o.side:<13} {seen}")
    return "\n".join(rows) if len(rows) > 1 else "(nothing seen yet)"


def describe(snapshot: dict[str, Any]) -> str:
    """Robot state and the scene table, as the planner reads them."""
    rooms = snapshot["rooms"]
    others = ", ".join(r for r in rooms if r != snapshot["room"])
    return (f"Robot: in room {snapshot['room']}, battery {snapshot['battery']:.0%}. Other rooms: {others or 'none'}.\n"
            f"Known objects:\n{scene_table(snapshot['objects'], snapshot['user'])}")
