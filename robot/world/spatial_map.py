"""Spatial working memory: tracked objects and people, who is speaking, named places, occupancy, coverage.

Everything here comes from the robot's own sensors (camera detections placed with odometry and refined
by the lidar, range readings, the direction a voice came from); nothing reads simulator ground truth.
Positions live in the map frame (odometry corrected by lidar localization).
"""

from __future__ import annotations

import functools
import json
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot.body.hal import BodyPose
from robot.core.angles import wrap_deg, wrap_deg_array
from robot.core.clock import Clock
from robot.navigation.planner import CircleObstacle
from robot.perception.detection import Detection
from robot.perception.geometry import CameraModel
from robot.perception.lidar import Scan
from robot.world.labels import (
    PERSON_WORDS,
    SPEAKER_WORDS,
    _associates,
    _colors,
    _colors_compatible,
    _nouns,
    _tokens,
    age_words,
    direction_words,
    head_noun,
    label_score,
    labels_match,
    parse_reference,
)
from robot.world.occupancy import OccupancyGrid

ASKED_PLACE = "where_asked"  # where I was when the current request came: what "come back" means
PLACE_NOTES = {"start": " (where I was switched on)", ASKED_PLACE: " (where I was when the human asked)"}
# How far beyond the visible bottom edge an object's centre is. Round things: one radius. A person's box is
# as wide as their arms, but they stand on their feet: the centre is just behind the toes.
FOOTPRINT_DEPTH_MM = {"person": 110.0}
LIDAR_DEPTH_MM = {"person": 40.0}  # from the lidar's returns to the centre: it sees the fronts of the legs


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
    var_mm2: float = 300.0 ** 2  # position uncertainty (variance, per axis)
    alt_label: str = ""  # a different label the detector keeps giving this track
    alt_votes: float = 0.0
    vx_mm_s: float = 0.0  # estimated velocity of a person walking
    vy_mm_s: float = 0.0

    def predicted(self, now: float) -> tuple[float, float, float]:
        """Where it should be by now, and how uncertain that is. People walk: constant velocity, uncertainty
        growing with the time since they were last seen. Things stay put (small differences between cameras
        must not turn into motion)."""
        if not self.is_person:
            return self.x_mm, self.y_mm, self.var_mm2
        dt = max(0.0, min(MAX_PREDICT_S, now - self.last_seen))
        return (self.x_mm + self.vx_mm_s * dt, self.y_mm + self.vy_mm_s * dt,
                self.var_mm2 + (SPEED_NOISE_MM_S * dt) ** 2)

    def relative(self, pose: BodyPose) -> tuple[float, float]:
        dx, dy = self.x_mm - pose.x_mm, self.y_mm - pose.y_mm
        bearing = math.degrees(math.atan2(dy, dx)) - pose.heading_deg
        return math.hypot(dx, dy), wrap_deg(bearing)

    @property
    def is_person(self) -> bool:
        return bool(set(_tokens(self.label)) & PERSON_WORDS)


@dataclass
class Speaker:
    """The last voice heard: where it came from, and the person it belongs to when that is known."""

    origin: tuple[float, float]  # where I was when I heard it
    bearing_deg: float  # world bearing of the voice from there
    heard_at: float
    track: str | None = None  # the person it matched (a TrackedObject id)
    point: tuple[float, float] | None = None  # where they are (track, lidar or a guess)
    how: str = "voice"  # "seen" (matched a person I see), "lidar" (something solid there), "voice" (a guess)


REFERENCE_FOCAL_PX = 400.0  # the lens the error model below was measured with


def sighting_var_mm2(cam_dist_mm: float, focal_px: float = REFERENCE_FOCAL_PX) -> float:
    """Variance of a position placed from one camera box: depth from a monocular view degrades with range,
    and a pixel of error covers more ground through a wider lens (shorter focal length)."""
    return (50.0 + 0.15 * cam_dist_mm * max(1.0, REFERENCE_FOCAL_PX / focal_px)) ** 2


LIDAR_VAR_MM2 = 45.0 ** 2  # a sighting whose range the lidar measured
CLIPPED_VAR_MM2 = 450.0 ** 2  # an object cut off by the image's side
STALE_VAR_MM2 = 400.0 ** 2  # restored from disk: the odometry frame may have shifted since
MOVE_VAR_MM2 = 20.0 ** 2  # per sighting, so a track can follow an object that was moved
SPEED_NOISE_MM_S = 400.0  # how fast a person's velocity may change, per second (starting to walk)
MAX_PREDICT_S = 1.0  # predictions further ahead than this are guesses
MAX_SPEED_MM_S = 1500.0
PRECISE_VAR_MM2 = 250.0 ** 2  # sightings good enough to estimate a velocity from
VELOCITY_GAIN = 0.5  # share of an unexplained displacement taken as a change of velocity


@dataclass
class ViewUpdate:
    seen: list[tuple[TrackedObject, Detection, bool]] = field(default_factory=list)  # (track, det, is_new)
    unplaced: list[Detection] = field(default_factory=list)  # above horizon / out of range
    lost: list[TrackedObject] = field(default_factory=list)


RELABEL_VOTES = 1.0  # about one full view's worth of consistent sightings


class ObjectTable(dict):
    """Objects by id. A renamed object stays reachable under its old id, so a behaviour or a model call that
    still holds "box_2" finds it after it became "cube_2"."""

    def __init__(self) -> None:
        super().__init__()
        self.aliases: dict[str, str] = {}

    def _key(self, key: Any) -> Any:
        for _ in range(8):
            if dict.__contains__(self, key) or key not in self.aliases:
                break
            key = self.aliases[key]
        return key

    def __getitem__(self, key: Any) -> TrackedObject:
        return dict.__getitem__(self, self._key(key))

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, self._key(key))

    def get(self, key: Any, default: Any = None) -> Any:
        return dict.get(self, self._key(key), default)

    def pop(self, key: Any, *default: Any) -> Any:
        return dict.pop(self, self._key(key), *default)

    # Readers in one thread may iterate while the control thread adds a sighting: iterate over a snapshot.
    def values(self) -> list[TrackedObject]:  # type: ignore[override]
        return list(dict.values(self))

    def items(self) -> list[tuple[str, TrackedObject]]:  # type: ignore[override]
        return list(dict.items(self))

    def keys(self) -> list[str]:  # type: ignore[override]
        return list(dict.keys(self))

    def __iter__(self):
        return iter(list(dict.keys(self)))

    def rename(self, old: str, new: str) -> None:
        obj = dict.pop(self, old)
        obj.id = new
        dict.__setitem__(self, new, obj)
        self.aliases[old] = new


def _locked(method):
    """Run a SpatialMap method under the map's lock (the control thread fuses while cognition reads)."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapper


class SpatialMap:
    def __init__(self, clock: Clock, *, half_extent_mm: float = 3000.0, max_objects: int = 48) -> None:
        self.lock = threading.RLock()
        self.clock = clock
        self.max_objects = max_objects
        self.objects: ObjectTable = ObjectTable()
        self.places: dict[str, tuple[float, float]] = {}
        self.speaker: Speaker | None = None
        self.grid = OccupancyGrid(half_extent_mm)
        self._counters: dict[str, int] = {}

    # ---- objects -------------------------------------------------------------------------------
    def _new_id(self, label: str) -> str:
        noun = head_noun(label)
        self._counters[noun] = self._counters.get(noun, 0) + 1
        return f"{noun}_{self._counters[noun]}"

    def _match_by_bearing(self, pose: BodyPose, model: CameraModel, det: Detection,
                          matched: set[str], max_dist_mm: float = 900.0) -> TrackedObject | None:
        """Objects cut off by the image bottom are too close to range from the box; match by direction.

        The direction is the angular middle of the box's edges, not its middle column: on a wide lens the two
        differ by tens of degrees near the image's sides."""
        left, right = model.bearing_deg(det.bbox[0]), model.bearing_deg(det.bbox[2])
        bearing = 0.5 * (left + right)
        best, best_err = None, max(12.0, 0.5 * abs(left - right))
        for obj in self.objects.values():
            if obj.id in matched or not _associates(obj.label, det.label):
                continue
            dist, rel = obj.relative(pose)
            err = abs(wrap_deg(rel - bearing))
            if dist < max_dist_mm and err < best_err:
                best, best_err = obj, err
        return best

    @staticmethod
    def lidar_surface(scan: Any, model: CameraModel, pose: BodyPose, bbox: tuple[float, float, float, float]
                      ) -> tuple[float, float, float] | None:
        """The nearest solid thing the lidar sees inside a box's horizontal span: the centre of that cluster of
        returns (x, y) and its distance from the camera; None when the lidar sees nothing there."""
        if scan is None:
            return None
        cx, cy, _, _, _ = model.extrinsics(pose)
        pts = scan.world_points()
        if not len(pts):
            return None
        lo, hi = sorted((model.bearing_deg(bbox[0] + 0.15 * (bbox[2] - bbox[0])),
                         model.bearing_deg(bbox[2] - 0.15 * (bbox[2] - bbox[0]))))
        rel = wrap_deg_array(np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) - pose.heading_deg)
        mid = 0.5 * (lo + hi)
        inside = np.abs(wrap_deg_array(rel - mid)) <= 0.5 * (hi - lo) + 1.0
        if not inside.any():
            return None
        near, d = pts[inside], np.hypot(pts[inside, 0] - cx, pts[inside, 1] - cy)
        cluster = near[d <= d.min() + 200.0]  # the front-most thing (legs, a chair), not the wall behind it
        x, y = float(cluster[:, 0].mean()), float(cluster[:, 1].mean())
        return x, y, math.hypot(x - cx, y - cy)

    @_locked
    def integrate_view(self, pose: BodyPose, model: CameraModel, detections: list[Detection],
                       *, query: str | None = None, weight: float = 1.0, relabel: bool = False,
                       scan: Any = None, clipped_tracks: bool = True) -> ViewUpdate:
        """Fuse one camera view: associate detections with tracks and apply negative evidence.

        With `query` the detector only looked for that kind of object, so missing detections
        only count against tracks matching the query. `weight` (0..1] is how much evidence one view
        carries: 1 for an occasional look, small for a continuous stream (frames a few hundredths of a
        second apart are nearly the same observation), so a detector that flickers for a frame or two
        neither creates nor deletes an object. With `relabel` the detector's labels are trusted over the
        track's (a trained detector with a fixed vocabulary): a track it keeps calling something else is
        renamed, e.g. a "box" remembered from an older session that the detector sees as a "blue cube".
        With a lidar `scan`, a box whose span holds lidar returns at about the camera's distance (people,
        furniture: anything tall enough for the lidar) gets the lidar's range instead of the monocular guess.
        Without `clipped_tracks` (cameras all around, overlapping), a box cut off at the image's side never
        starts a track: placed from a truncated box it lands far off along the ray.
        """
        w = max(0.05, min(1.0, float(weight)))
        now = self.clock.monotonic()
        update = ViewUpdate()
        matched: set[str] = set()
        for det in sorted(detections, key=lambda d: -d.area):
            if det.bbox[3] >= 0.97:
                near = self._match_by_bearing(pose, model, det, matched)
                if near is not None:
                    near.last_seen, near.seen_count = now, near.seen_count + 1
                    near.confidence = min(1.0, near.confidence + 0.2 * w)
                    matched.add(near.id)
                    update.seen.append((near, det, False))
                    continue
            # Cut off by the image's side: the box is truncated, so the position it gives is poor. It counts as
            # weak evidence (large uncertainty), so a later full view corrects it.
            clipped = det.bbox[0] <= 0.01 or det.bbox[2] >= 0.99
            located = model.locate_box(pose, det.bbox, FOOTPRINT_DEPTH_MM.get(head_noun(det.label)))
            if located is None:
                update.unplaced.append(det)
                continue
            x, y, radius = located
            cam_dist = math.hypot(x - pose.x_mm, y - pose.y_mm)
            meas_var = sighting_var_mm2(cam_dist, model.focal_px) / w  # correlated frames: each counts less
            if clipped:
                meas_var = max(meas_var * 9.0, CLIPPED_VAR_MM2)
            surface = self.lidar_surface(scan, model, pose, det.bbox)
            lidar_placed = False
            if surface is not None:
                cx, cy, _, _, _ = model.extrinsics(pose)
                mono = math.hypot(x - cx, y - cy)
                sx, sy, ranged = surface
                depth = LIDAR_DEPTH_MM.get(head_noun(det.label), min(radius, 150.0))
                if abs(ranged + depth - mono) < max(250.0, 0.35 * mono):  # the lidar sees this very thing
                    x, y = sx + (sx - cx) / ranged * depth, sy + (sy - cy) / ranged * depth
                    meas_var = LIDAR_VAR_MM2 / w
                    lidar_placed = True
            gate = max(350.0, 0.3 * cam_dist, 600.0 if clipped else 0.0)
            best, best_d = None, math.inf
            for obj in self.objects.values():
                if obj.id in matched or not _associates(obj.label, det.label):
                    continue
                px, py, pvar = obj.predicted(now)  # a walking person is where they are now, not where last seen
                d = math.hypot(px - x, py - y)
                # An uncertain track (first seen from far away) accepts a sighting further from its guess.
                if d < max(gate, min(900.0, 2.0 * math.sqrt(pvar + meas_var))) and d < best_d:
                    best, best_d = obj, d
            if best is None and clipped:
                # Cut off by the image's side (a camera seam): the same thing another camera sees whole is in the
                # same direction. Confirm it there rather than start a track from a poor position.
                seam = self._match_by_bearing(pose, model, det, matched, max_dist_mm=2500.0)
                if seam is not None:  # still there; not a placed sighting (seen_count counts only those)
                    seam.last_seen = now
                    seam.confidence = min(1.0, seam.confidence + 0.2 * w)
                    matched.add(seam.id)
                    update.seen.append((seam, det, False))
                    continue
                if not clipped_tracks and not lidar_placed and head_noun(det.label) in PERSON_WORDS:
                    # A cut-off box lands far off along its ray; a neighbouring camera sees the thing whole, or the
                    # lidar ranges it (someone standing on the seam between two cameras is cut off in both).
                    update.unplaced.append(det)
                    continue
            if best is None:
                if len(self.objects) >= self.max_objects:
                    self._evict()
                best = TrackedObject(self._new_id(det.label), det.label, x, y, radius, now, now,
                                     confidence=(0.25 if clipped else 0.35) + 0.35 * w,
                                     var_mm2=meas_var if clipped else meas_var * w)
                self.objects[best.id] = best
                is_new = True
            else:
                # Kalman-style fusion: a close sighting outweighs a distant one, and a well-known track
                # is not dragged away by one poor far-off box.
                px, py, prior = best.predicted(now)
                prior = max(prior, STALE_VAR_MM2) if best.stale else prior
                gain = prior / (prior + meas_var)
                dt = now - best.last_seen
                # Velocity (alpha-beta) learns from whole, well-placed sightings only: a box cut off at a camera
                # seam moves the estimate around and would read as walking.
                if 0.02 < dt <= MAX_PREDICT_S and best.is_person and not best.stale and not clipped                         and meas_var <= PRECISE_VAR_MM2:
                    k = VELOCITY_GAIN * gain / dt
                    best.vx_mm_s += k * (x - px)
                    best.vy_mm_s += k * (y - py)
                    speed = math.hypot(best.vx_mm_s, best.vy_mm_s)
                    if speed > MAX_SPEED_MM_S:
                        best.vx_mm_s, best.vy_mm_s = (best.vx_mm_s * MAX_SPEED_MM_S / speed,
                                                      best.vy_mm_s * MAX_SPEED_MM_S / speed)
                elif dt > MAX_PREDICT_S:
                    best.vx_mm_s = best.vy_mm_s = 0.0  # out of sight too long to carry a velocity over
                best.x_mm = px + gain * (x - px)
                best.y_mm = py + gain * (y - py)
                best.var_mm2 = (1.0 - gain) * prior + MOVE_VAR_MM2
                alpha = 0.6 if best.stale else max(0.3, 1.0 / (best.seen_count + 1))
                # A wide lens sees things small and mostly from the side (a box corner-on looks 40% wider):
                # its size estimates count for less, like its positions.
                alpha *= min(1.0, model.focal_px / REFERENCE_FOCAL_PX) ** 2
                if not clipped:  # a truncated box says little about size
                    best.radius_mm += alpha * (radius - best.radius_mm)
                best.last_seen = now
                best.seen_count += 1
                best.confidence = min(1.0, best.confidence + 0.3 * w)
                best.stale = False
                is_new = False
            matched.add(best.id)
            update.seen.append((best, det, is_new))
            if relabel and not is_new:
                self._vote_label(best, det.label, w)

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
            obj.confidence -= (0.6 if obj.stale else 0.4) * w
            if obj.confidence < 0.15:
                del self.objects[obj.id]
                update.lost.append(obj)
        self.grid.mark_coverage(pose, model)
        if self.speaker is not None and self.speaker.how != "seen":
            self.speaker_track()  # re-bind a heard speaker to the person this view may have shown
        return update

    def _vote_label(self, obj: TrackedObject, label: str, w: float) -> None:
        if label == obj.label or label == "object":  # "object" is the catch-all; a named track keeps its name
            obj.alt_label, obj.alt_votes = "", 0.0
            return
        if label != obj.alt_label:
            obj.alt_label, obj.alt_votes = label, 0.0
        obj.alt_votes += w
        if obj.alt_votes >= RELABEL_VOTES:
            obj.label, obj.alt_label, obj.alt_votes = label, "", 0.0
            if head_noun(label) != obj.id.rpartition("_")[0]:  # the id names the old kind: "box_2" for a cube
                self.objects.rename(obj.id, self._new_id(label))

    def _evict(self) -> None:
        victim = min(self.objects.values(), key=lambda o: (o.confidence, o.last_seen))
        del self.objects[victim.id]

    @_locked
    def forget(self, obj_id: str) -> bool:
        return self.objects.pop(obj_id, None) is not None

    # ---- people and the voice ------------------------------------------------------------------------
    @_locked
    def hear_voice(self, pose: BodyPose, bearing_deg: float, *, origin_mm: float, scan: Scan | None = None
                   ) -> Speaker:
        """Someone spoke from `bearing_deg` (relative to my heading, from the microphone array above the head
        pivot). Find who: a person I see in that direction; else note where they probably are (something solid
        the lidar sees that way, or a guess along the voice) until the cameras find them."""
        now = self.clock.monotonic()
        h = math.radians(pose.heading_deg)
        ox, oy = pose.x_mm + origin_mm * math.cos(h), pose.y_mm + origin_mm * math.sin(h)
        world = wrap_deg(pose.heading_deg + bearing_deg)

        def off(x: float, y: float) -> float:
            return abs(wrap_deg(math.degrees(math.atan2(y - oy, x - ox)) - world))

        people = [o for o in self.objects.values() if o.is_person and o.confidence >= 0.3
                  and now - o.last_seen < 60.0 and off(o.x_mm, o.y_mm) < 25.0]
        speaker = Speaker((ox, oy), world, now)
        if people:
            best = min(people, key=lambda o: (off(o.x_mm, o.y_mm) + 0.004 * math.hypot(o.x_mm - ox, o.y_mm - oy)))
            speaker.track, speaker.point, speaker.how = best.id, (best.x_mm, best.y_mm), "seen"
        else:
            hit = None
            pts = scan.world_points() if scan is not None else np.zeros((0, 2))
            if len(pts):
                d = np.hypot(pts[:, 0] - ox, pts[:, 1] - oy)
                ang = np.abs(wrap_deg_array(np.degrees(np.arctan2(pts[:, 1] - oy, pts[:, 0] - ox)) - world))
                near = (ang < 8.0) & (d > 250.0) & (d < 5000.0)
                if near.any():
                    r = float(np.min(d[near])) + 100.0  # legs: the person stands just behind the nearest return
                    hit = (ox + r * math.cos(math.radians(world)), oy + r * math.sin(math.radians(world)))
            # No track: a point the robot can head for. The speaker becomes a person the cameras place in that
            # direction (speaker_track); a made-up track here drew stray sightings to it (measured).
            speaker.point = hit or self._clip(ox + 1500.0 * math.cos(math.radians(world)),
                                              oy + 1500.0 * math.sin(math.radians(world)))
            speaker.how = "lidar" if hit else "voice"
        self.speaker = speaker
        return speaker

    def _clip(self, x: float, y: float) -> tuple[float, float]:
        lim = self.grid.half - 100.0
        return max(-lim, min(lim, x)), max(-lim, min(lim, y))

    @_locked
    def speaker_track(self) -> TrackedObject | None:
        """The person who spoke last; heard but not yet seen, the first person the cameras place in the voice's
        direction becomes them. With no voice heard: the person seen most recently."""
        now = self.clock.monotonic()
        sp = self.speaker
        if sp is not None and sp.how != "seen":
            ox, oy = sp.origin
            seen = [o for o in self.objects.values() if o.is_person and o.seen_count >= 2
                    and o.last_seen >= sp.heard_at - 1.0
                    and abs(wrap_deg(math.degrees(math.atan2(o.y_mm - oy, o.x_mm - ox)) - sp.bearing_deg)) < 25.0]
            if seen:
                best = min(seen, key=lambda o: math.hypot(o.x_mm - ox, o.y_mm - oy))
                sp.track, sp.point, sp.how = best.id, (best.x_mm, best.y_mm), "seen"
        if sp is not None:
            return self.objects.get(sp.track) if sp.track else None
        people = [o for o in self.objects.values() if o.is_person and now - o.last_seen < 60.0]
        return max(people, key=lambda o: (o.last_seen, o.confidence)) if people else None

    def place(self, target: str) -> tuple[str, tuple[float, float]] | None:
        """A named place ("start", "where_asked", "where asked") as (name, point)."""
        key = re.sub(r"[\s_]+", "_", str(target or "").strip().lower())
        return (key, self.places[key]) if key in self.places else None

    @_locked
    def resolve(self, target: str, pose: BodyPose | None = None) -> TrackedObject | None:
        """Find an object by id ("box_1") or description ("the brown box"); nearest wins ties."""
        key = str(target or "").strip().lower()
        if not key:
            return None
        if key in self.objects:
            return self.objects[key]
        if key.strip(" .!?") in SPEAKER_WORDS:
            return self.speaker_track()
        near_me = re.search(r"\b(?:next to|near|by|beside|close to|in front of) (?:me|the human|the user)\b", key)
        if near_me and (person := self.speaker_track()) is not None:
            # "the ball next to me": the thing closest to the human, not to the robot.
            kind = parse_reference(key[:near_me.start()])[0]
            things = [o for o in self.objects.values() if o.id != person.id and not o.is_person
                      and (not kind or label_score(o.label, kind) > 0)]
            if things:
                return min(things, key=lambda o: math.hypot(o.x_mm - person.x_mm, o.y_mm - person.y_mm))
        compact = key.replace(" ", "_")
        if compact in self.objects:
            return self.objects[compact]
        kind, sector, superlative = parse_reference(key)
        described = bool(_nouns(_tokens(kind)) or _colors(_tokens(kind)))
        if not described and sector is None and superlative is None:
            return None  # "it", "there": nothing to go on
        candidates = list(self.objects.values())
        if described and _nouns(_tokens(kind)):
            scored = [(label_score(o.label, kind), o) for o in candidates]
            best = max((score for score, _ in scored), default=0)
            candidates = [o for score, o in scored if score == best and score > 0]
        elif described:  # only a colour: "the red one"
            want = _colors(_tokens(kind))
            candidates = [o for o in candidates if _colors(_tokens(o.label))
                          and _colors_compatible(want, _colors(_tokens(o.label)))]
        if sector is not None and pose is not None:
            centre, half = sector
            candidates = [o for o in candidates
                          if abs(wrap_deg(o.relative(pose)[1] - centre)) <= half]
        candidates = [o for o in candidates if o.confidence >= 0.2] or candidates
        if not candidates:
            return None
        if pose is None:
            return max(candidates, key=lambda o: (o.confidence, o.last_seen))
        if superlative == "far":
            return max(candidates, key=lambda o: o.relative(pose)[0])
        if superlative in ("big", "small"):
            pick = max if superlative == "big" else min
            return pick(candidates, key=lambda o: o.radius_mm)
        if superlative == "near":
            return min(candidates, key=lambda o: o.relative(pose)[0])
        return min(candidates, key=lambda o: o.relative(pose)[0] - 300.0 * o.confidence)

    def describe(self, obj: TrackedObject, pose: BodyPose) -> str:
        dist, bearing = obj.relative(pose)
        age = self.clock.monotonic() - obj.last_seen
        stale = ", from an earlier session" if obj.stale else ""
        who = ", the human talking to me" if self.speaker is not None and self.speaker.track == obj.id else ""
        return (f"{obj.id} ({obj.label}{who}): {dist / 1000:.1f} m {direction_words(bearing)}, "
                f"{age_words(age)}{stale}")

    @_locked
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
            what = PLACE_NOTES.get(name, "")
            places.append(f"{name}{what}: {where}")
        if places:
            lines.append("Places: " + ", ".join(places) + ".")
        if self.speaker is not None and self.speaker.how == "voice" and self.speaker.point is not None:
            bearing = math.degrees(math.atan2(self.speaker.point[1] - pose.y_mm,
                                              self.speaker.point[0] - pose.x_mm)) - pose.heading_deg
            lines.append(f"The human's voice came from {direction_words(bearing)} (I have not seen them yet).")
        return "\n".join(lines)

    # ---- geometry for the planner ----------------------------------------------------------------
    @_locked
    def obstacles(self, *, exclude: set[str] | None = None) -> list[CircleObstacle]:
        exclude = exclude or set()
        # Radii come from the visible width; a square box's corners reach sqrt(2) times further.
        return [CircleObstacle(o.x_mm, o.y_mm, 1.42 * o.radius_mm + 10.0, o.id) for o in self.objects.values()
                if o.id not in exclude and o.confidence >= 0.3]

    def occupied_points(self, exclude: set[str]) -> np.ndarray:
        """Range-sensor hits as planner points, minus those on objects I am allowed to approach."""
        points = self.grid.occupied_points()
        for obj_id in exclude:
            obj = self.objects.get(obj_id)
            if obj is not None:
                points = points[np.hypot(points[:, 0] - obj.x_mm, points[:, 1] - obj.y_mm) > obj.radius_mm + 120.0]
        return points

    # ---- persistence -----------------------------------------------------------------------------
    @_locked
    def snapshot(self, pose: BodyPose) -> dict[str, Any]:
        now = self.clock.monotonic()
        return {
            "objects": [{**asdict(o), "distance_mm": round(o.relative(pose)[0]),
                         "bearing_deg": round(o.relative(pose)[1], 1), "age_s": round(now - o.last_seen, 1)}
                        for o in self.objects.values()],
            "places": {k: [round(v[0]), round(v[1])] for k, v in self.places.items()},
            "occupied_cells": [[round(float(x)), round(float(y))] for x, y in self.grid.occupied_points()][:2000],
            "coverage": round(self.grid.coverage(), 3),
        }

    @_locked
    def save(self, path: Path) -> None:
        now_mono, now_wall = self.clock.monotonic(), time.time()
        places = {k: v for k, v in self.places.items() if k != ASKED_PLACE}  # only true for one request
        data = {"saved_at": now_wall, "places": places, "objects": [
            {**asdict(o), "age_s": now_mono - o.last_seen} for o in self.objects.values() if o.confidence >= 0.3]}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=float), encoding="utf-8")  # numpy scalars from detectors
        tmp.replace(path)

    @_locked
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
