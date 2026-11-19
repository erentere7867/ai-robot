"""One page per 3D task, for finding out what went wrong: a map of the house with the real objects, what the robot
believed was where (false ones in red), where its body really went and where it thought it was, where each call
began; then every model turn in order with its reasoning, its calls and their results, the pictures it was given and
the search's choices. Written next to the pictures the eval keeps, and an index of the run's tasks.

Everything is drawn in the house's frame: the robot's own map starts at its dock, whose place the judge knows.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

import numpy as np

PX = 900  # the map's longer side, in pixels
NEAR = 0.75  # metres beyond its edge for a believed thing to count as a real one of its name


def to_house(dock: tuple[float, float, float], x: float, y: float) -> tuple[float, float]:
    dx, dy, dyaw = dock
    return dx + x * math.cos(dyaw) - y * math.sin(dyaw), dy + x * math.sin(dyaw) + y * math.cos(dyaw)


def same_name(a: str, b: str) -> bool:
    """A believed label and a real one name the same kind of thing: equal, or one's words inside the other's."""
    a, b = a.lower().strip(), b.lower().strip()
    return a == b or set(a.split()) <= set(b.split()) or set(b.split()) <= set(a.split())


def real_match(thing: Any, x: float, y: float, objects: list[dict]) -> dict | None:
    """The real object of the same name nearest a believed thing, if one is close enough to be it."""
    near = [o for o in objects if same_name(thing.label, o["label"])
            and math.dist((x, y), (o["x"], o["y"])) <= o["radius"] + NEAR]
    return min(near, key=lambda o: math.dist((x, y), (o["x"], o["y"]))) if near else None


def map_picture(truth: dict, dock: tuple, path: list, believed_path: list, things: list, rejected: list,
                events: list, words: set[str]) -> np.ndarray:
    """The map as a BGR picture: floor, real objects (named when the task or the robot's beliefs mention their kind),
    the true path (orange) and the believed one (thin blue), believed things (blue, red when no real one of that name
    is there, grey when the vision model rejected them), and numbered places where tool calls began."""
    import cv2

    w, h, res = truth["width"], truth["height"], truth["res"]
    k = PX / max(w, h)
    img = np.full((round(h * k), round(w * k), 3), 40, np.uint8)

    def px(x: float, y: float) -> tuple[int, int]:
        return round((x + w / 2) * k), round((h / 2 - y) * k)

    nx, ny = truth["shape"]
    free = np.unpackbits(np.frombuffer(truth["free"], np.uint8))[: nx * ny].reshape(nx, ny).astype(bool)
    cell = max(1, round(res * k))
    for i, j in np.argwhere(free):
        u, v = px(i * res - w / 2, (j + 1) * res - h / 2)
        img[v:v + cell, u:u + cell] = (90, 90, 90)
    named = {t.label for t in things} | words
    for o in truth["objects"]:
        named_here = any(same_name(n, o["label"]) for n in named)
        cv2.circle(img, px(o["x"], o["y"]), max(2, round(o["radius"] * k)), (0, 200, 0) if named_here else
                   (70, 120, 70), 2 if named_here else 1)
        if named_here:
            cv2.putText(img, o["label"], px(o["x"] + o["radius"], o["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (120, 255, 120), 1, cv2.LINE_AA)
    pts = [px(x, y) for _, x, y, _ in path]
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 140, 255), 2, cv2.LINE_AA)
    bpts = [px(*to_house(dock, x, y)) for _, x, y, _ in believed_path]
    if len(bpts) > 1:
        cv2.polylines(img, [np.array(bpts, np.int32)], False, (255, 160, 60), 1, cv2.LINE_AA)
    if pts:
        cv2.circle(img, pts[0], 7, (0, 255, 0), -1)
        cv2.circle(img, pts[-1], 7, (0, 0, 255), -1)
    for t, gone in [*((t, False) for t in things), *((t, True) for t in rejected)]:
        x, y = to_house(dock, t.x, t.y)
        colour = (150, 150, 150) if gone else (255, 160, 60) if real_match(t, x, y, truth["objects"]) else \
            (60, 60, 255)
        u, v = px(x, y)
        cv2.drawMarker(img, (u, v), colour, cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(img, f"{t.label} {t.id}", (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1,
                    cv2.LINE_AA)
    for n, (_, _, _, pose) in enumerate(events, 1):
        u, v = px(*to_house(dock, pose[0], pose[1]))
        cv2.circle(img, (u, v), 9, (255, 255, 255), -1)
        cv2.putText(img, str(n), (u - 4 * len(str(n)), v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1,
                    cv2.LINE_AA)
    return img


def write(folder: Path, task: Any, row: dict, records: list, robot: Any, home: Any, mark: int,
          pictures: list[tuple[float, str]], verdict: str) -> Path:
    """The task's page, report.html, in `folder` (where its pictures are)."""
    import cv2

    dock = home.dock()
    tracks = robot.perception.objects() if hasattr(robot, "perception") else []
    rejected = [t for t in tracks if t.rejected]
    things = list(robot.things.values())
    events = [(e.t, e.name, e.args, e.pose) for e in robot.trace.events if e.name and e.name not in ("check",)]
    words = {w for say in task.says for w in say.lower().replace(".", " ").replace(",", " ").split()}
    cv2.imwrite(str(folder / "map.png"), map_picture(home.truth, dock, home.path[mark:], robot.trace.poses, things,
                                                       rejected, events, words))
    items: list[tuple[float, int, str]] = []  # sim time, order, html
    order = 0
    for rec in records:
        for s in rec.steps:
            order += 1
            parts = []
            if s.thought:
                parts.append(f"<div class=thought>{html.escape(s.thought)}</div>")
            for name, args, result, ok in s.calls:
                a = ", ".join(f"{k}={v!r}" for k, v in args.items())
                parts.append(f"<div class='call {'ok' if ok else 'bad'}'><b>{html.escape(name)}({html.escape(a)})"
                             f"</b><br>{html.escape(result if isinstance(result, str) else str(result))}</div>")
            if s.said and not s.calls:
                parts.append(f"<div class=reply>says: {html.escape(s.said)}</div>")
            items.append((getattr(s, "t", 0.0), order, f"<div class=turn><span class=t>{getattr(s, 't', 0.0):.1f} s"
                                                        f" · model {s.model_s:.1f} s</span>{''.join(parts)}</div>"))
    for t, file in pictures:
        order += 1
        items.append((t, order, f"<figure><img src='{html.escape(file)}'><figcaption>{t:.1f} s · "
                                f"{html.escape(file)}</figcaption></figure>"))
    for t, question, answer in robot.trace.asked:
        order += 1
        items.append((t, order, f"<div class=asked>{t:.1f} s · {html.escape(str(question))}: "
                                f"<b>{html.escape(str(answer))}</b></div>"))
    legend = "".join(f"<li>{n}. {t:.1f} s {html.escape(name)}({html.escape(', '.join(f'{v}' for v in args.values()))})"
                     f"</li>" for n, (t, name, args, _) in enumerate(events, 1))
    def belief(t: Any) -> str:
        x, y = to_house(dock, t.x, t.y)
        real = real_match(t, x, y, home.truth["objects"])
        nearest = min(home.truth["objects"], key=lambda o: math.dist((x, y), (o["x"], o["y"])) - o["radius"])
        what = ("rejected by the vision model" if t.rejected else "a real one is there" if real else
                f"no real one there; nearest real thing: {nearest['label']}, "
                f"{max(0.0, math.dist((x, y), (nearest['x'], nearest['y'])) - nearest['radius']):.1f} m from its edge")
        kind = "gone" if t.rejected else "real" if real else "false"
        return f"<li class={kind}>{html.escape(t.label)} {t.id}: {html.escape(what)} (seen {t.hits} times)</li>"

    believed = "".join(belief(t) for t in tracks if not t.person)
    # what went wrong outside the model's turns (the model server timing out, a check that failed): without it a
    # task that failed before its first turn shows only its apology
    problems = "".join(f"<li>{html.escape(p)}</li>" for p in row["problems"])
    page = f"""<!doctype html><html><head><meta charset=utf-8><title>{html.escape(task.id)}</title><style>
body{{font:14px system-ui;background:#111;color:#ddd;margin:16px}} img{{max-width:100%}}
.pass{{color:#6f6}}.fail{{color:#f66}} .turn{{border-left:3px solid #48f;padding:6px 10px;margin:10px 0}}
.thought{{color:#999;font-style:italic;white-space:pre-wrap}} .call{{margin:4px 0;padding:4px 6px;background:#1c1c24}}
.call.bad{{background:#2a1515}} .reply{{color:#fd6}} .asked{{color:#9cf;margin:6px 0}} .t{{color:#777;font-size:12px}}
figure{{display:inline-block;margin:4px;width:31%}} figcaption{{font-size:11px;color:#888}}
li.false{{color:#f77}} li.gone{{color:#888}} li.real{{color:#8bf}}
.cols{{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px}}
</style></head><body>
<h1>{html.escape(task.id)}: {html.escape(' / '.join(task.says))} <span class={'pass' if row['passed'] else 'fail'}>
{'passed' if row['passed'] else 'FAILED'}</span></h1>
<p>{html.escape(verdict)} · {row['simulated_s']} s simulated · {row['bumps']} bumps</p>
{f"<h3>Problems</h3><ul class=fail>{problems}</ul>" if problems else ""}
<div class=cols><div><img src=map.png><p class=t>green: real objects (named when the task or the robot's beliefs
mention their kind) · orange: where the body really went (green dot start, red dot end) · thin blue: where the robot
thought it was · crosses: what it believed was there (blue: a real one is there, red: none, grey: rejected) · white
numbers: where each call began</p></div>
<div><h3>Calls</h3><ol>{legend}</ol><h3>What it believed</h3><ul>{believed}</ul></div></div>
<h2>Timeline</h2>{''.join(h for _, _, h in sorted(items))}</body></html>"""
    out = folder / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


def index(run: Path, rows: list[dict]) -> Path:
    """The run's page: each task, passed or not, with a link to its report."""
    lines = "".join(f"<li class={'pass' if r['passed'] else 'fail'}><a href='{html.escape(r['scene'])}/"
                    f"{html.escape(r['id'])}/report.html'>"
                    f"{html.escape(r['scene'])} · {html.escape(r['id'])}: {html.escape(' / '.join(r['says']))}</a> "
                    f"{'passed' if r['passed'] else 'FAILED'}</li>" for r in rows)
    out = run / "index.html"
    out.write_text(f"<!doctype html><html><head><meta charset=utf-8><title>{html.escape(run.name)}</title><style>"
                   "body{font:15px system-ui;background:#111;color:#ddd;margin:16px}a{color:#9cf}.fail a{color:#f88}"
                   f"</style></head><body><h1>{html.escape(run.name)}</h1><ul>{lines}</ul></body></html>",
                   encoding="utf-8")
    return out
