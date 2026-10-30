"""Spatial working memory: tracked objects, named places, ToF occupancy and camera coverage.

Everything here comes from the robot's own sensors (camera detections placed with odometry,
range readings); nothing reads simulator ground truth. Positions live in the odometry frame.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot.body.hal import BodyPose
from robot.core.clock import Clock
from robot.navigation.planner import CircleObstacle
from robot.perception.detector import Detection
from robot.perception.geometry import CameraModel

COLORS = {
    "red", "blue", "green", "yellow", "orange", "brown", "purple", "violet", "pink", "white", "black",
    "gray", "grey", "teal", "cyan", "tan", "beige", "gold", "silver", "magenta", "lavender",
}
COLOR_GROUPS = [{"brown", "tan", "beige", "orange", "cardboard", "gold"}, {"purple", "violet", "lavender", "magenta"},
                {"gray", "grey", "silver"}, {"teal", "cyan", "green"}, {"teal", "cyan", "blue"}]
SYNONYMS = [
    {"box", "carton", "crate", "package", "cube", "block"},
    {"ball", "sphere", "orb"},
    {"mug", "cup", "cylinder", "can", "glass", "jar", "bottle"},
    {"duck", "duckling", "bird", "toy"},
    {"dock", "charger", "station", "pad", "base", "platform"},
    {"person", "human", "man", "woman", "people", "child", "kid"},
]
FILLER = {"a", "an", "the", "small", "large", "big", "little", "tiny", "toy", "object", "shaped", "colored",
          "coloured", "plastic", "wooden", "rubber", "my", "your", "that", "this", "of", "on", "with", "and"}

CELL_MM = 50.0


def _tokens(label: str) -> list[str]:
    return re.findall(r"[a-z]+", label.lower())


def _colors(tokens: list[str]) -> set[str]:
    found = {t for t in tokens if t in COLORS}
    if "cardboard" in tokens:
        found.add("brown")
    return found


def _nouns(tokens: list[str]) -> set[str]:
    return {t for t in tokens if t not in COLORS and t not in FILLER}


def _colors_compatible(a: set[str], b: set[str]) -> bool:
    if not a or not b or a & b:
        return True
    return any(a & g and b & g for g in COLOR_GROUPS)


def _nouns_compatible(a: set[str], b: set[str]) -> bool:
    if a & b:
        return True
    return any(a & g and b & g for g in SYNONYMS)


def labels_match(a: str, b: str) -> bool:
    return label_score(a, b) > 0


def label_score(a: str, b: str) -> int:
    """2 = same kind of thing (shared noun), 1 = synonym (box~cube), 0 = different; colours must agree."""
    ta, tb = _tokens(a), _tokens(b)
    if not _colors_compatible(_colors(ta), _colors(tb)):
        return 0
    na, nb = _nouns(ta), _nouns(tb)
    if na & nb:
        return 2
    return 1 if _nouns_compatible(na, nb) else 0


def head_noun(label: str) -> str:
    nouns = [t for t in _tokens(label) if t not in COLORS and t not in FILLER]
    return (nouns[-1] if nouns else "object")[:16]


def direction_words(rel_bearing_deg: float) -> str:
    a = (rel_bearing_deg + 180.0) % 360.0 - 180.0
    side = "left" if a > 0 else "right"
    b = abs(a)
    if b < 10:
        return "straight ahead"
    if b < 35:
        return f"ahead, slightly {side}"
    if b < 70:
        return f"front-{side}"
    if b < 110:
        return f"to my {side}"
    if b < 160:
        return f"behind-{side}"
    return "behind me"


def age_words(age_s: float) -> str:
    if age_s < 20:
        return "seen just now"
    if age_s < 90:
        return f"seen {age_s:.0f}s ago"
    if age_s < 5400:
        return f"seen {age_s / 60:.0f} min ago"
    return f"seen {age_s / 3600:.0f} h ago"


@dataclass
class TrackedObject:
    id: str
    label: str
    x_mm: float
    y_mm: float
    radius_mm: float
    first_seen: float
    last_seen: float
    seen_count: int = 1
    confidence: float = 0.7
    stale: bool = False  # restored from a previous session; odometry frames may not match

    def relative(self, pose: BodyPose) -> tuple[float, float]:
        dx, dy = self.x_mm - pose.x_mm, self.y_mm - pose.y_mm
        bearing = math.degrees(math.atan2(dy, dx)) - pose.heading_deg
        return math.hypot(dx, dy), (bearing + 180.0) % 360.0 - 180.0


@dataclass
class ViewUpdate:
    seen: list[tuple[TrackedObject, Detection, bool]] = field(default_factory=list)  # (track, det, is_new)
    unplaced: list[Detection] = field(default_factory=list)  # above horizon / out of range
    lost: list[TrackedObject] = field(default_factory=list)


class SpatialMap:
    def __init__(self, clock: Clock, *, half_extent_mm: float = 3000.0, max_objects: int = 48) -> None:
        self.clock = clock
        self.half = float(half_extent_mm)
        self.max_objects = max_objects
        self.n = int(math.ceil(2 * self.half / CELL_MM))
        self.objects: dict[str, TrackedObject] = {}
        self.places: dict[str, tuple[float, float]] = {}
        self.occupancy = np.zeros((self.n, self.n), dtype=np.float32)  # log-odds-like evidence
        self.seen = np.zeros((self.n, self.n), dtype=bool)  # floor cells the camera has looked at
        self._counters: dict[str, int] = {}
        centers = -self.half + (np.arange(self.n) + 0.5) * CELL_MM
        self._xs = np.broadcast_to(centers[:, None], (self.n, self.n))
        self._ys = np.broadcast_to(centers[None, :], (self.n, self.n))

    # ---- objects -------------------------------------------------------------------------------
    def _new_id(self, label: str) -> str:
        noun = head_noun(label)
        self._counters[noun] = self._counters.get(noun, 0) + 1
        return f"{noun}_{self._counters[noun]}"

    def _match_by_bearing(self, pose: BodyPose, model: CameraModel, det: Detection,
                          matched: set[str]) -> TrackedObject | None:
        """Objects cut off by the image bottom are too close to range from the box; match by direction."""
        bearing = model.bearing_deg(det.center_u)
        best, best_err = None, 12.0
        for obj in self.objects.values():
            if obj.id in matched or not labels_match(obj.label, det.label):
                continue
            dist, rel = obj.relative(pose)
            if dist < 900.0 and abs(rel - bearing) < best_err:
                best, best_err = obj, abs(rel - bearing)
        return best

    def integrate_view(self, pose: BodyPose, model: CameraModel, detections: list[Detection],
                       *, query: str | None = None) -> ViewUpdate:
        """Fuse one camera view: associate detections with tracks and apply negative evidence.

        With `query` the detector only looked for that kind of object, so missing detections
        only count against tracks matching the query.
        """
        now = self.clock.monotonic()
        update = ViewUpdate()
        matched: set[str] = set()
        for det in sorted(detections, key=lambda d: -d.area):
            if det.bbox[3] >= 0.97:
                near = self._match_by_bearing(pose, model, det, matched)
                if near is not None:
                    near.last_seen, near.seen_count = now, near.seen_count + 1
                    near.confidence = min(1.0, near.confidence + 0.2)
                    matched.add(near.id)
                    update.seen.append((near, det, False))
                    continue
            located = model.locate_box(pose, det.bbox)
            if located is None:
                update.unplaced.append(det)
                continue
            x, y, radius = located
            cam_dist = math.hypot(x - pose.x_mm, y - pose.y_mm)
            gate = max(350.0, 0.3 * cam_dist)
            best, best_d = None, math.inf
            for obj in self.objects.values():
                if obj.id in matched or not labels_match(obj.label, det.label):
                    continue
                d = math.hypot(obj.x_mm - x, obj.y_mm - y)
                if d < gate and d < best_d:
                    best, best_d = obj, d
            if best is None:
                if len(self.objects) >= self.max_objects:
                    self._evict()
                best = TrackedObject(self._new_id(det.label), det.label, x, y, radius, now, now)
                self.objects[best.id] = best
                is_new = True
            else:
                alpha = 0.6 if best.stale else max(0.3, 1.0 / (best.seen_count + 1))
                best.x_mm += alpha * (x - best.x_mm)
                best.y_mm += alpha * (y - best.y_mm)
                best.radius_mm += alpha * (radius - best.radius_mm)
                best.last_seen = now
                best.seen_count += 1
                best.confidence = min(1.0, best.confidence + 0.3)
                best.stale = False
                is_new = False
            matched.add(best.id)
            update.seen.append((best, det, is_new))

        # Negative evidence: an object that should be visible but was not detected fades away.
        seen_us = [model.project(pose, o.x_mm, o.y_mm, 60.0) for o, _, _ in update.seen]
        for obj in list(self.objects.values()):
            if obj.id in matched or not model.in_view(pose, obj.x_mm, obj.y_mm, 60.0, margin=0.08):
                continue
            if query and not labels_match(obj.label, query):
                continue
            dist, _ = obj.relative(pose)
            if dist > 2600.0:
                continue
            uv = model.project(pose, obj.x_mm, obj.y_mm, 60.0)
            hidden = any(
                s is not None and uv is not None and abs(s[0] - uv[0]) < 0.08 and o.relative(pose)[0] < dist
                for s, (o, _, _) in zip(seen_us, update.seen, strict=True)
            )
            if hidden:
                continue
            obj.confidence -= 0.6 if obj.stale else 0.4
            if obj.confidence < 0.15:
                del self.objects[obj.id]
                update.lost.append(obj)
        self.mark_coverage(pose, model)
        return update

    def _evict(self) -> None:
        victim = min(self.objects.values(), key=lambda o: (o.confidence, o.last_seen))
        del self.objects[victim.id]

    def forget(self, obj_id: str) -> bool:
        return self.objects.pop(obj_id, None) is not None

    def resolve(self, target: str, pose: BodyPose | None = None) -> TrackedObject | None:
        """Find an object by id ("box_1") or description ("the brown box"); nearest wins ties."""
        key = str(target or "").strip().lower()
        if not key:
            return None
        if key in self.objects:
            return self.objects[key]
        compact = key.replace(" ", "_")
        if compact in self.objects:
            return self.objects[compact]
        if not _nouns(_tokens(key)):
            return None
        scored = [(label_score(o.label, key), o) for o in self.objects.values()]
        best = max((score for score, _ in scored), default=0)
        candidates = [o for score, o in scored if score == best and score > 0]
        if not candidates:
            return None
        if pose is None:
            return max(candidates, key=lambda o: (o.confidence, o.last_seen))
        return min(candidates, key=lambda o: o.relative(pose)[0] - 300.0 * o.confidence)

    def describe(self, obj: TrackedObject, pose: BodyPose) -> str:
        dist, bearing = obj.relative(pose)
        age = self.clock.monotonic() - obj.last_seen
        stale = ", from an earlier session" if obj.stale else ""
        return f"{obj.id} ({obj.label}): {dist / 1000:.1f} m {direction_words(bearing)}, {age_words(age)}{stale}"

    def render_for_prompt(self, pose: BodyPose, *, limit: int = 8) -> str:
        if not self.objects:
            lines = ["Known objects: none yet."]
        else:
            ordered = sorted(self.objects.values(), key=lambda o: (o.relative(pose)[0], -o.confidence))
            lines = ["Known objects:"] + [f"- {self.describe(o, pose)}" for o in ordered[:limit]]
            if len(ordered) > limit:
                lines.append(f"- ...and {len(ordered) - limit} more farther away")
        places = []
        for name, (px, py) in self.places.items():
            dist = math.hypot(px - pose.x_mm, py - pose.y_mm)
            bearing = math.degrees(math.atan2(py - pose.y_mm, px - pose.x_mm)) - pose.heading_deg
            where = "here" if dist < 150 else f"{dist / 1000:.1f} m {direction_words(bearing)}"
            places.append(f"{name} ({where})")
        if places:
            lines.append("Places: " + ", ".join(places) + ".")
        return "\n".join(lines)

    # ---- geometry for the planner ----------------------------------------------------------------
    def obstacles(self, *, exclude: set[str] | None = None) -> list[CircleObstacle]:
        exclude = exclude or set()
        return [CircleObstacle(o.x_mm, o.y_mm, o.radius_mm, o.id) for o in self.objects.values()
                if o.id not in exclude and o.confidence >= 0.3]

    def occupied_mask(self) -> np.ndarray:
        return self.occupancy > 0.6

    def to_cell(self, x_mm: float, y_mm: float) -> tuple[int, int] | None:
        i, j = int((x_mm + self.half) // CELL_MM), int((y_mm + self.half) // CELL_MM)
        return (i, j) if 0 <= i < self.n and 0 <= j < self.n else None

    def update_range(self, pose: BodyPose, range_mm: float, *, offset_mm: float = 141.0,
                     max_range_mm: float = 2400.0) -> None:
        """Integrate one forward range reading: free space along the beam, an obstacle at its end."""
        if not math.isfinite(range_mm) or range_mm <= 0:
            return
        yaw = math.radians(pose.heading_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        ox, oy = pose.x_mm + offset_mm * c, pose.y_mm + offset_mm * s
        hit = range_mm < max_range_mm
        length = min(range_mm, max_range_mm)
        steps = int(length // (CELL_MM * 0.7))
        last = None
        for k in range(steps):
            d = k * CELL_MM * 0.7
            cell = self.to_cell(ox + d * c, oy + d * s)
            if cell is None:
                break
            if cell != last and d < length - CELL_MM:
                self.occupancy[cell] = max(-2.0, self.occupancy[cell] - 0.35)
            last = cell
        if hit:
            self.add_obstacle_point(ox + range_mm * c, oy + range_mm * s, weight=0.9)

    def add_obstacle_point(self, x_mm: float, y_mm: float, *, weight: float = 1.5) -> None:
        cell = self.to_cell(x_mm, y_mm)
        if cell is not None:
            self.occupancy[cell] = min(3.0, self.occupancy[cell] + weight)

    def mark_coverage(self, pose: BodyPose, model: CameraModel, *, range_mm: float = 2200.0) -> None:
        cx, cy, _, yaw, _ = model.extrinsics(pose)
        dx, dy = self._xs - cx, self._ys - cy
        dist = np.hypot(dx, dy)
        ang = (np.degrees(np.arctan2(dy, dx) - yaw) + 180.0) % 360.0 - 180.0
        self.seen |= (dist <= range_mm) & (dist >= 250.0) & (np.abs(ang) <= 0.5 * model.hfov_deg - 2)

    def unseen_fraction_near(self, x_mm: float, y_mm: float, radius_mm: float = 1500.0) -> float:
        mask = (self._xs - x_mm) ** 2 + (self._ys - y_mm) ** 2 <= radius_mm ** 2
        free = mask & ~(self.occupancy > 0.6)
        total = int(free.sum())
        return float((free & ~self.seen).sum()) / total if total else 0.0

    # ---- persistence -----------------------------------------------------------------------------
    def snapshot(self, pose: BodyPose) -> dict[str, Any]:
        now = self.clock.monotonic()
        return {
            "objects": [{**asdict(o), "distance_mm": round(o.relative(pose)[0]),
                         "bearing_deg": round(o.relative(pose)[1], 1), "age_s": round(now - o.last_seen, 1)}
                        for o in self.objects.values()],
            "places": {k: [round(v[0]), round(v[1])] for k, v in self.places.items()},
            "occupied_cells": [[round(float(self._xs[i, j])), round(float(self._ys[i, j]))]
                               for i, j in zip(*np.nonzero(self.occupied_mask()), strict=True)][:2000],
            "coverage": round(float(self.seen.mean()), 3),
        }

    def save(self, path: Path) -> None:
        now_mono, now_wall = self.clock.monotonic(), time.time()
        data = {"saved_at": now_wall, "places": self.places, "objects": [
            {**asdict(o), "age_s": now_mono - o.last_seen} for o in self.objects.values() if o.confidence >= 0.3]}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)

    def load(self, path: Path) -> int:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        now = self.clock.monotonic()
        elapsed = max(0.0, time.time() - float(data.get("saved_at", time.time())))
        for name, xy in (data.get("places") or {}).items():
            if name != "start":
                self.places[str(name)] = (float(xy[0]), float(xy[1]))
        for raw in data.get("objects") or []:
            try:
                age = float(raw.pop("age_s", 0.0)) + elapsed
                obj = TrackedObject(**{**raw, "last_seen": now - age, "first_seen": now - age, "stale": True,
                                       "confidence": min(0.5, float(raw.get("confidence", 0.5)))})
            except TypeError:
                continue
            self.objects[obj.id] = obj
            noun, _, num = obj.id.rpartition("_")
            if num.isdigit():
                self._counters[noun] = max(self._counters.get(noun, 0), int(num))
        return len(self.objects)
