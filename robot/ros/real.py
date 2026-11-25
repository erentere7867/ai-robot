"""Sprocket v2 full ROS 2 robot stack: Nav2 motion, 3-CSI perception, Brain VLM, and CirculatingMemory.

`RealRobot` is a `RosRobot` whose objects, 3D bounds, velocities, and people come from YOLOE-26 looking for the words
tools ask for, and the tracker (robot/ros/perceive.py); whose vision checks, room names and floor spots come from
LFM3-VL-2B via `Brain`; and whose long-term spatial habits, clear-space observations, and speaker identity persist in
`CirculatingMemory` (robot/mind/memory.py). The scene truth is used solely for post-run evaluation (`judge`).
"""

from __future__ import annotations

import math
import queue
import re
import threading
import time
from collections import Counter, deque
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from robot.agent.api import Matches, NotFound, Object, PathBlocked, Point, Pose, wrap
from robot.mind.brain import _with_image
from robot.mind.memory import ANCHOR_LABELS, CANONICAL_SYNONYMS
from robot.perception.camera import data_url
from robot.perception.locate import locate_crop
from robot.perception.rig import RIG, to_map, to_robot
from robot.perception.tracker import Track, canonical, pixels
from robot.ros.explore import MAX_PLACES, PLACE_GAP, describe_place, load_places, unique_place_name
from robot.ros.perceive import (
    Perception,
    _nanoseconds,
    _seconds,
    is_there,
    locate_in_picture,
    place_picture,
    see_probability,
    taken_with,
)
from robot.ros.robot import RosRobot
from robot.sim.world import COLORS, RES, Scene, SimRobot, Thing, visual_name

SEEN_NOW = 1.5  # seconds: a track seen this recently is "in view now"
LOOKS = 3  # frames each camera looks with a word before what it names counts as not in view: a thing takes two sightings
LOOK_S = 6.0  # wall seconds at most for them: they took 1.8-2.1 s in the 3D apartment at 0.33 x real time
COLOR_WORDS = (*sorted(COLORS), "unknown")
SMALL = 0.15  # metres across from the middle: a thing this small is something to look at, not an obstacle to plan round
ROOM_ASK_S = 8.0  # seconds between the vision model's looks at what kind of room the robot is in
ROOM_ANSWERS = 5  # answers kept per room; more change nothing
SPOT_AHEAD = (0.7, 1.1, 1.6, 2.2, 3.0, 4.0)  # metres ahead of the camera, for the floor spots the model picks from
SPOT_ACROSS = 0.5  # metres between neighbouring spots across
SPOT_GAP = 48  # pixels at least between two spots as seen in a 768-wide picture: spread across the view
SEARCH_AWAY = 1.6  # metres: keep new search stops well clear of already-driven trail (1.0 m zigzagged across one room)
STEP_BACK = 1.0  # metres a search backs off from what is too near to name: YOLOE named the blue bicycle from the dock
# (2.3 m) and 1.05 m from its centre, but beside it (0.2 m from its edge) in 1 of 2 front frames and in 0 of 8
TRAIL_STEP = 0.25  # metres between lightweight mid-drive snapshots (umbrella corridor window measured ~0.30 m wide)
ELEVATED_Z = 0.45  # metres above floor: above the 22 cm camera, occluded by tabletops/counters once closer than ~1.2 m


class RealRobot(RosRobot):
    def __init__(self, scene: Scene, *, detector: Any, brain: Any = None, max_time: float = 600.0,
                 nudge: Any = None, store: Any = None, grid: Any = None,
                 places_file: Path | None = None, owns_store: bool = False) -> None:
        """`store` (CirculatingMemory) persists seen objects, habits, speakers, facts and places across restarts;
        `grid` is the occupancy of the map the robot built itself (robot/ros/explore.py), for planning;
        `places_file` is the optional semantic area map built by the robot's cameras alongside `maps/<scene>.yaml`."""
        self._ready = False
        super().__init__(scene, max_time=max_time, nudge=nudge)
        self._mapped_places: set[str] = set()
        for name, pose in load_places(places_file).items():
            self.places[name] = pose
            self._mapped_places.add(name)
        self._rooms: list = []  # room masks of the robot's own map, named again as it learns (robot/ros/rooms.py)
        self._named_at = -1e9
        if grid is not None:
            from robot.ros.rooms import segment

            self.set_occupied(grid)
            self.walls = grid  # raycast camera coverage against the real SLAM walls/obstacles so _unseen() finds new areas
            self.coverage = np.zeros_like(grid)
            self._rooms = segment(~grid, RES)
        self._room_answers: list[Counter] = [Counter() for _ in self._rooms]  # the vision model's, per room
        self.search_log: list[str] = []  # the last search: where the model thought the thing was, stop by stop
        self._trail_views: deque = deque(maxlen=24)  # (mono_t, pose, {cam: (stamp, rgb, depth)}) every TRAIL_STEP m (6 m)
        self._searching = False
        self._search_snap_after = 0.0
        self._search_coverage: dict[str, np.ndarray] = {}
        self.perception = Perception(self.node, detector, self._pose_at, brain,
                                     scan_bridge=getattr(self, "_scan_bridge", None))
        self.brain = brain
        self.things, self.memory = {}, {}
        self._cleared_tracks: set[int] = set()
        self._habit_recorded: dict[int, str] = {}
        self.human_pose: Any = None
        self.speaker_profile: Any = None
        self.answers: queue.Queue[str] = queue.Queue()  # what the user types or says while listening
        self.waiting = False
        self._user: int | None = None
        self.store, self._owns_store, self._closed = store, owns_store, False
        if store is not None:
            known = store.restore(self.perception.tracker)
            self.notes = [*self.notes, *store.notes()]
            self.places.update({name: Pose(x, y, yaw) for name, (x, y, yaw) in store.places().items()})
            print(f"memory: {known} objects, {len(self.notes)} facts and {len(self.places) - 1} places remembered",
                  flush=True)
            threading.Thread(target=self._keep, daemon=True).start()
        if brain is not None and self._rooms:
            threading.Thread(target=self._ask_rooms, daemon=True).start()
        self._ready = True

    def _keep(self) -> None:
        """Write memory every few seconds, so a crash loses little."""
        while not self._closed:
            time.sleep(5.0)
            if self._closed:
                break
            self._save()

    def _save(self) -> None:
        with self.perception.lock:
            if self.store is not None:
                self.store.save(self.perception.tracker, self.notes, self.places)

    def remember(self, fact: str) -> None:
        super().remember(fact)
        if self.store is not None and hasattr(self.store, "remember_fact"):
            self.store.remember_fact(fact, room=self._room_at(self.x, self.y))

    def shutdown(self) -> None:
        self._closed = True
        with self.perception.lock:
            if self.store is not None:
                self.store.save(self.perception.tracker, self.notes, self.places)
                if self._owns_store and hasattr(self.store, "close"):
                    self.store.close()
                self.store = None
        super().shutdown()

    # ---- perception from the cameras ------------------------------------------------------------------------------
    def _pose_at(self, stamp: Any) -> tuple[float, float, float] | None:
        from rclpy.time import Time

        for when in (Time.from_msg(stamp), Time()):
            try:
                tr = self.tf.lookup_transform("map", "base_link", when).transform
            except Exception:  # noqa: BLE001 - not at that time: try the latest
                self.stale_poses = getattr(self, "stale_poses", 0) + 1
                continue
            q = tr.rotation
            return tr.translation.x, tr.translation.y, math.atan2(2 * (q.w * q.z + q.x * q.y),
                                                                   1 - 2 * (q.y * q.y + q.z * q.z))
        return None

    def _perceive_objects(self) -> None:
        if not getattr(self, "_ready", False):
            return
        now = time.monotonic()
        things, memory, detected = {}, {}, set()
        people = []
        tracks = self.perception.objects()
        anchors: list[Any] = []
        movables: list[Any] = []
        human_pose = None
        for tr in tracks:
            if not tr.present:
                if self.store is not None and tr.id not in self._cleared_tracks and hasattr(
                    self.store, "observe_clear_space"
                ):
                    self.store.observe_clear_space(tr.x, tr.y, radius=max(0.6, tr.radius + 0.3))
                    self._cleared_tracks.add(tr.id)
                continue  # looked at clearly and not there: marked clear in CirculatingMemory
            if hasattr(self, "scene") and hasattr(self.scene, "width") and hasattr(self.scene, "height"):
                has_enclosed_walls = (
                    getattr(self, "walls", None) is not None
                    and getattr(self.walls, "ndim", 0) == 2
                    and self.walls.shape[0] > 1 and self.walls.shape[1] > 1
                    and bool(self.walls[0, :].all() and self.walls[-1, :].all() and self.walls[:, 0].all() and self.walls[:, -1].all())
                )
                is_indoor = not getattr(self.scene, "outdoor", False) and (
                    bool(getattr(self.scene, "rooms", None))
                    or bool(getattr(self.scene, "walls", None))
                    or has_enclosed_walls
                )
                if is_indoor and (abs(tr.x) > self.scene.width / 2 - 0.12 or abs(tr.y) > self.scene.height / 2 - 0.12):
                    continue
            if tr.rejected:  # the vision model looked at it and said it is not what YOLOE called it
                continue
            self._cleared_tracks.discard(tr.id)
            thing = Thing(tr.label, round(tr.x, 3), round(tr.y, 3), round(tr.radius, 3), tr.color,
                          solid=not tr.person and tr.radius >= SMALL, id=tr.id)
            things[tr.id] = thing
            memory[tr.id] = ((thing.x, thing.y), self.t - (now - tr.last_seen))
            if now - tr.last_seen <= SEEN_NOW:
                detected.add(tr.id)
            if tr.person:
                people.append((tr, thing))
                if getattr(tr, "pose", None) is not None and human_pose is None:
                    human_pose = tr.pose
            elif tr.confirmed and not tr.rejected:
                c_label = CANONICAL_SYNONYMS.get(tr.label.lower(), tr.label.lower())
                if c_label in ANCHOR_LABELS or tr.label.lower() in ANCHOR_LABELS:
                    anchors.append(tr)
                else:
                    movables.append(tr)
        if self.store is not None and anchors and movables and hasattr(self.store, "record_habit"):
            for mv in movables:
                best_anc = min(anchors, key=lambda a: math.hypot(mv.x - a.x, mv.y - a.y))
                if math.hypot(mv.x - best_anc.x, mv.y - best_anc.y) <= (best_anc.radius + mv.radius + 1.2):
                    if self._habit_recorded.get(mv.id) != best_anc.label:
                        self.store.record_habit(mv.label, best_anc.label)
                        self._habit_recorded[mv.id] = best_anc.label
        if people:
            speaker_name = getattr(self.speaker_profile, "name", None)
            matched_speaker = next((t.id for tr, t in people if speaker_name and tr.voice_id == speaker_name), None)
            if matched_speaker is not None:
                self._user = matched_speaker
            elif self._user not in things:
                self._user = min(t.id for _, t in people)
            things[self._user].user = True
        self.things, self.memory, self._detected = things, memory, detected
        self.human_pose = human_pose
        self._record_trail(now)
        if getattr(self, "_rooms", None) and now - getattr(self, "_named_at", 0.0) > 5.0:  # from the tour and the model
            from robot.ros.rooms import name

            self._named_at = now
            mapped = getattr(self, "_mapped_places", ())
            tour_places = {k: v for k, v in self.places.items() if k not in mapped and k != "where you were before"}
            self.scene.rooms = name(self._rooms, RES, (self.scene.width, self.scene.height), self._room_answers,
                                    tour_places, before=list(self.scene.rooms or {}))

    def _note_place(self) -> None:
        """Record a compact VLM area caption when stopping in an unmapped part of the home (>= PLACE_GAP from existing
        mapped places), so even a brand-new home builds a semantic place map on the fly (~60 ms per new area)."""
        if not getattr(self, "_ready", False) or self.brain is None or not hasattr(self.brain, "describe"):
            return
        mapped = getattr(self, "_mapped_places", None)
        if mapped is None or len(mapped) >= MAX_PLACES:
            return
        here = (self.x, self.y)
        if any(math.dist(here, (self.places[n].x, self.places[n].y)) < PLACE_GAP for n in mapped if n in self.places):
            return
        frames = {c: f[1] for c, f in getattr(self.perception, "frames", {}).items() if f is not None}
        if not frames:
            return
        try:
            label = describe_place(self.brain, frames)
        except Exception:  # noqa: BLE001 - optional semantic caption must not interrupt motion
            return
        if label:
            key = unique_place_name(label, self.places)
            self.places[key] = Pose(self.x, self.y, math.degrees(self.yaw))
            mapped.add(key)

    def _record_trail(self, now: float) -> None:
        """Keep a lightweight snapshot of the three cameras every TRAIL_STEP m of travel, for mid-drive search
        detections and elevated-target approach confirmation (zero model cost while driving)."""
        trail = getattr(self, "_trail_views", None)
        if trail is None or not getattr(self.perception, "frames", None):
            return
        if trail and math.dist(trail[-1][1][:2], (self.x, self.y)) < TRAIL_STEP:
            return
        bridge = getattr(self.perception, "scan_bridge", None)
        cams: dict[str, tuple[Any, np.ndarray, np.ndarray | None, tuple[float, float, float]]] = {}
        for cam_name in RIG:
            frame = self.perception.frames.get(cam_name)
            pose = self._pose_at(frame[0]) if frame is not None else None
            if pose is None:
                continue
            depth = None
            if bridge is not None and cam_name in bridge.cam_depths:
                depth, _ = taken_with(bridge.cam_depths[cam_name], _nanoseconds(frame[0]), 500_000_000)
            cams[cam_name] = (frame[0], frame[1], depth, pose)
        if cams:
            trail.append((now, (self.x, self.y, self.yaw), cams))

    def _ask_rooms(self) -> None:
        """Now and then, ask the vision model what kind of room the front camera shows, for the room the robot is in:
        rooms are named by what the model sees rather than by a table of furniture words."""
        from robot.ros.rooms import room_of

        while not self._closed:
            time.sleep(ROOM_ASK_S)
            k = room_of(self._rooms, RES, (self.scene.width, self.scene.height), self.x, self.y)
            frame = self.perception.frames.get("front_camera")
            if k is None or frame is None or sum(self._room_answers[k].values()) >= ROOM_ANSWERS:
                continue
            try:
                kind = self.brain.room_kind(data_url(frame[1]))
            except Exception as exc:  # noqa: BLE001 - the model busy or down: ask again later
                print(f"rooms: {exc}", flush=True)
                continue
            if kind:
                self._room_answers[k][kind] += 1

    def observe_clear_space(self, x: float | None = None, y: float | None = None, radius: float = 0.6) -> None:
        """Record empty floor in CirculatingMemory, clearing stale movable items and route blockers."""
        if self.store is not None and hasattr(self.store, "observe_clear_space"):
            px = self.x if x is None else float(x)
            py = self.y if y is None else float(y)
            self.store.observe_clear_space(px, py, radius=radius)

    def record_habit(self, item_label: str, anchor_name: str, alpha: float = 0.3) -> None:
        if self.store is not None and hasattr(self.store, "record_habit"):
            self.store.record_habit(item_label, anchor_name, alpha=alpha)

    def usual_anchors(self, item_label: str, limit: int = 5) -> list[tuple[str, float]]:
        if self.store is not None and hasattr(self.store, "usual_anchors"):
            return self.store.usual_anchors(item_label, limit=limit)
        return []

    def identify_speaker(self, voice_vec: Any, look_vec: Any = None, threshold: float = 0.65) -> Any:
        """Identify the active speaker via CirculatingMemory and bind them as the active user track."""
        if self.store is None or not hasattr(self.store, "identify_speaker"):
            return None
        profile = self.store.identify_speaker(voice_vec, look_vec=look_vec, threshold=threshold)
        if profile is None:
            return None
        self.speaker_profile = profile
        with self.perception.lock:
            person_tracks = [t for t in self.perception.tracker.tracks.values() if t.person and t.present]
            for tr in person_tracks:
                if self._user == tr.id or len(person_tracks) == 1:
                    tr.voice_id = profile.name
        return profile

    def snapshot(self) -> dict[str, Any]:
        snap = super().snapshot()
        snap["human_pose"] = self.human_pose
        snap["posture"] = self.posture
        if self.speaker_profile is not None:
            snap["speaker"] = getattr(self.speaker_profile, "name", "")
        if self.store is not None and hasattr(self.store, "notes"):
            try:
                stored_notes = list(self.store.notes())
                merged = list(dict.fromkeys([*stored_notes, *snap.get("notes", [])]))
                snap["notes"] = merged
            except Exception:
                pass
        return snap

    def camera_view(self, camera: str = "front_camera", width: int = 512, height: int = 288) -> np.ndarray | None:
        """The latest real frame from one camera with the tracker's ids marked, as the vision model sees it."""
        return self.perception.render_annotated_view(camera, width, height, self._now() - SEEN_NOW)

    def marked_cameras(self, ids: set[int]) -> list[str]:
        """Fewest current views covering the visible choices. At a seam one camera showed only the farther chair."""
        p, now = self.perception, self._now()
        with p.lock:
            ids = {i for i in ids if (t := p.tracker.tracks.get(i)) and t.present and not t.rejected}
            views = sorted(((stamp, cam, ids.intersection(i for _, i in p.boxes[cam]))
                            for cam, (stamp, _) in p.detected.items() if 0 <= now - stamp <= SEEN_NOW), reverse=True)
        visible = set().union(*(marks for _, _, marks in views))
        for n in range(1, len(views) + 1):
            for group in combinations(views, n):
                if visible and visible == set().union(*(marks for _, _, marks in group)):
                    return [cam for _, cam, _ in group]
        return []

    def target_crop(self, object_id: int) -> np.ndarray | None:
        p = self.perception
        with p.lock:
            track = p.tracker.tracks.get(p.tracker.current(object_id))
            return p.crops.get(track.id) if track and track.present and not track.rejected else None

    def live_view(self, camera: str = "front_camera") -> np.ndarray | None:
        """The same exposure/ID pairing as the model, resized for the live page."""
        return self.perception.render_annotated_view(camera, 640, 360, self._now() - SEEN_NOW)

    def obj(self, id: Any) -> Object:  # noqa: A002 - the name programs use
        """An object by any id it has had: two sightings found to be one thing share the older id from then on. An
        id given out earlier that is no longer an object says why: the model was told of it, and "no object 35"
        alone sends it searching for a thing the robot already knows about."""
        if isinstance(id, list) and id:
            id = id[0]  # noqa: A001
        given = int(id.id if isinstance(id, Object) else id)
        with self.perception.lock:
            now = self.perception.tracker.current(given)
            track = self.perception.tracker.tracks.get(now)
        if track is not None:
            if track.rejected:
                raise NotFound(f"object {given} was not a {track.label} after all: a closer look found none there")
            if not track.present:
                raise NotFound(f"object {given} ({track.label}) was not there when the cameras last looked at its "
                               "place; it may have been moved")
            if track.confirmed and now not in self.things:
                # Perception worker merged `given -> now` since the last _perceive_objects() sync; refresh before lookup.
                self._perceive_objects()
        return super().obj(now)

    def sees_now(self, target: Object) -> bool:
        """Seen by the detector a moment ago, or else by the vision model in the part of the front picture where the
        thing is mapped. Parked facing a real beanbag 1 m away, YOLOE had not named it for 6-160 s, and all seven
        arrivals were refused as "not confirmed" until the task ran out of turns 0.86 m from it. A visual-only target
        always verifies against the post-arrival front picture (never fast-passing on last_seen alone), confirms
        against the 768px front picture or — when elevated above the 22 cm camera (z > 0.45 m, e.g. a monitor on a
        desk has 0 visible pixels once closer than 1.15 m, 76-89% at 1.5-1.7 m) — against a closer approach snapshot
        taken during the drive, and drops itself otherwise."""
        tracker = getattr(self.perception, "tracker", None)
        track = None
        visual = elevated = on_floor = False
        vcolor = ""
        first_seen, spotted_from = 0.0, (0.0, 0.0)
        if tracker is not None:
            with self.perception.lock:
                track = tracker.tracks.get(tracker.current(target.id))
                if track is not None:
                    visual = bool(track.visual)
                    elevated = bool(track.z > ELEVATED_Z)
                    on_floor = bool(track.radius == 0.0)
                    vcolor = track.visual_color
                    first_seen = track.first_seen
                    spotted_from = track.spotted_from
        if not visual and target.last_seen <= SEEN_NOW:
            return True
        # Right after face() finishes spinning, frames["front_camera"] can still hold a mid-spin picture; wait up to
        # 1 s for one frame taken after the robot settled before deciding the target isn't there.
        if getattr(self, "node", None) is not None:
            settled, deadline = self._now(), time.monotonic() + 1.0
            while time.monotonic() < deadline:
                f = self.perception.frames.get("front_camera")
                if f is not None and _seconds(f[0]) >= settled - 0.05:
                    break
                self._step(0.0, 0.0)
        frame = self.perception.frames.get("front_camera")
        pose = self._pose_at(frame[0]) if frame is not None else None
        if pose is None or self.brain is None:
            return False
        # Include the VLM-verified requested colour (never raw noisy pixel median colours): passing only target.label
        # ("bike") let a distant black wheel under a desk confirm arrival for "red bike" (0.78 vs 0.36; 768px 0.001).
        label = f"{vcolor} {target.label}".strip() if vcolor else target.label
        bx, by = to_robot(pose, *target.position)
        where = place_picture(frame[1], RIG["front_camera"], bx, by, target.radius)
        if where is not None and is_there(self.brain, where, label):
            return True
        if not visual:
            if track is not None and where is not None and not elevated and math.hypot(bx, by) <= target.radius + 1.0:
                wrong_color_only = bool(vcolor) and is_there(self.brain, where, target.label)
                with self.perception.lock:
                    if vcolor:
                        track.visual_color, track.color_from = "", 0
                    if not wrong_color_only:
                        track.present = False
                self._perceive_objects()
            return False
        if see_probability(self.brain, frame[1], label, max_dim=768) >= 0.5:
            return True
        if on_floor and is_there(self.brain, frame[1][frame[1].shape[0] // 2:], label):
            # Standing on a floor surface beside furniture (e.g. cabinet on a rug), the furniture fills the top half
            # (whole-image 47.3%) while the floor surface fills the bottom half (is_there True).
            return True
        if elevated:
            d_orig = math.dist(spotted_from, target.position)
            cands = []
            for mono_t, _, cams in getattr(self, "_trail_views", ()):
                front = cams.get("front_camera")
                if mono_t <= first_seen or front is None:
                    continue
                _, snap_rgb, _, snap_pose = front
                d_snap = math.dist(snap_pose[:2], target.position)
                if 1.40 <= d_snap <= 3.5 and d_orig - d_snap >= 0.20 and RIG["front_camera"].sees(
                    *to_robot(snap_pose, *target.position), margin=0.2
                ):
                    cands.append((d_snap, snap_rgb))
            if cands:
                _, snap_rgb = min(cands, key=lambda c: abs(c[0] - 1.6))
                if see_probability(self.brain, snap_rgb, label, max_dim=768) >= 0.5:
                    return True
        with self.perception.lock:
            if track is not None:
                track.present = False
        self._perceive_objects()
        return False

    def _visible(self) -> list[Thing]:
        return [self.things[i] for i in self._detected if i in self.things]

    def _detectable(self, t: Thing) -> bool:
        return True

    def _watch(self, description: str) -> None:
        phrase = visual_name(description)
        if phrase:
            self.perception.detector.watch(phrase)

    def _in_view(self) -> list[Thing]:
        return self._visible()

    def _ask_color(self, crop: np.ndarray, label: str) -> str:
        """Ask the vision model what colour the `label` in the middle of `crop` is (ignoring the background)."""
        question = (f"What colour would a person call the {label} in the middle of this picture? "
                    "Ignore the background. Answer with one colour or unknown.")
        scores = self.brain.decide(_with_image(data_url(crop), question), choices=COLOR_WORDS)
        scores["grey"] += scores.pop("gray")
        word = max(scores, key=scores.get)
        return word if scores[word] > 0.5 else "unknown"

    def _matching(self, description: str, things: list[Thing]) -> list[Thing]:
        wanted = set(description.lower().split()) & COLORS
        p = getattr(self, "perception", None)
        if self.brain is None or p is None:
            return super()._matching(description, things)
        if "grey" in wanted or "gray" in wanted:
            wanted.update(("grey", "gray"))
        label_query = visual_name(description)
        candidates = super()._matching(label_query if wanted else description, things)
        if not candidates:
            return []
        matches = []
        rejected = False
        can_vet = hasattr(self.brain, "yes_probability")
        q_canon = canonical(label_query) if label_query else ""
        for thing in candidates:
            with p.lock:
                track = p.tracker.tracks.get(thing.id)
                if track is None or track.rejected or not track.present:
                    continue
                crop = p.crops.get(thing.id)
                same_label = not q_canon or canonical(track.label).rstrip("s") == q_canon.rstrip("s")
                need_vet = (
                    can_vet and not track.person and not track.visual and crop is not None
                    and (not track.sure or (track.named_from > 0 and pixels(crop) >= 2 * track.named_from))
                )
                need_query = can_vet and not track.person and not track.visual and crop is not None and not same_label
                color = track.visual_color
                need_color = bool(wanted) and crop is not None and (not color or pixels(crop) >= 2 * track.color_from)
            if need_vet:
                # Vet unconfirmed or newly-closer YOLOE tracks synchronously before tools drive to them:
                # YOLOE fired on cloths as "shoes", chairs as "stool" and tables as "tv"; rejecting here
                # lets detect() fall straight through to _see_target() instead of chasing a false box.
                yes = is_there(self.brain, crop, track.label)
                with p.lock:
                    if p.tracker.tracks.get(thing.id) is not track or track.rejected or not track.present:
                        continue
                    p.tracker.verify(track, track.label if yes else None, pixels(crop), sure=yes)
                if not yes:
                    rejected = True
                    continue
            if need_query and not is_there(self.brain, crop, label_query):
                continue
            if wanted:
                if need_color:
                    # Most of a bicycle box was wall or tyres. Ask about its colour, not the median pixel;
                    # only when a tool needs the attribute, and never while holding the camera worker's lock.
                    color = self._ask_color(crop, thing.label)
                    with p.lock:
                        if p.tracker.tracks.get(thing.id) is not track or track.rejected or not track.present:
                            continue
                        track.visual_color, track.color_from = color, pixels(crop)
                if color not in wanted:
                    continue
                thing.color = color
            matches.append(thing)
        if rejected:
            self._perceive_objects()
        return matches

    def detect(self, description: str) -> list[Object]:
        """What matches the description in view now, looked for with its own words: each camera looks LOOKS times
        with them before a thing counts as not in view. Waiting for two frames instead, whatever words they had, let
        the first run with the old words; beside the blue bicycle only one frame looked for it, one sighting is not a
        thing (it takes two), and go_to said no bicycle was in view with the body 0.19 m from it."""
        self._call()
        word = visual_name(description)
        self._watch(description)
        looks = self.perception.looks
        before = {n: looks.get(n, Counter())[word] for n in RIG}
        end = time.monotonic() + LOOK_S
        while word and time.monotonic() < end and any(looks.get(n, Counter())[word] < before[n] + LOOKS for n in RIG):
            self._step(0.0, 0.0)
        if word and all(looks.get(n, Counter())[word] == before[n] for n in RIG):
            # the vision server died twice in one session; the robot then answered "not in view" and searched blind
            raise RuntimeError(f"no camera picture was looked at in {LOOK_S:g} s: is the vision server running, and "
                               "is the robot localised?")
        self._perceive_objects()
        found = self._matching(description, self._in_view())
        if not found and self.brain is not None and description.strip():
            seen = self._see_target(description)
            if seen is not None:
                found = [seen]
        return Matches(sorted((self._object(t, *self.memory[t.id]) for t in found), key=lambda o: o.distance))

    def _spotted_standpoint(self, target: Object, distance: float) -> Point | None:
        """For a standing/elevated visual track spotted close up (ds <= reach + 0.5), the free map point that puts
        front_camera on the unobstructed spotting camera spot `spotted_from`."""
        tracker = getattr(self.perception, "tracker", None)
        if tracker is None:
            return None
        with self.perception.lock:
            track = tracker.tracks.get(tracker.current(target.id))
            if track is None or not track.visual or not track.present or track.radius <= 0.0:
                return None
            spotted = track.spotted_from
        goal, reach = self._reach(target, distance)
        ds = math.dist(spotted, goal)
        if ds < 0.2 or ds > reach + 0.5:
            return None
        ux, uy = (spotted[0] - goal[0]) / ds, (spotted[1] - goal[1]) / ds
        # Stay on the unobstructed goal->spotted sightline at normal `reach` (~0.65 m) rather than backing off past it
        # (backing off to reach+0.25=0.90 m stopped 1.13 m from E2's umbrella, 0.23 m too far).
        for r in (min(ds, reach), min(ds, reach + 0.10), ds):
            cand = (goal[0] + r * ux, goal[1] + r * uy)
            if not self._occ(*cand, self.inflated):
                return cand
        return None

    def _goal_point(self, target: Point | Pose | Object, distance: float) -> tuple[Point, float] | None:
        """For a standing/elevated visual track (e.g. an umbrella tucked behind a wall corner), approach from the
        visible `spotted_from` side rather than stopping on the occluded side of a corner."""
        where = super()._goal_point(target, distance)
        if not isinstance(target, Object):
            return where
        goal, reach = self._reach(target, distance)
        cand = self._spotted_standpoint(target, distance)
        if cand is not None:
            if math.dist((self.x, self.y), cand) <= 0.25:
                return where
            if self._plan(cand, 0.15) is not None:
                return cand, math.atan2(goal[1] - cand[1], goal[0] - cand[0])
        tracker = getattr(self.perception, "tracker", None)
        if tracker is None:
            return where
        with self.perception.lock:
            track = tracker.tracks.get(tracker.current(target.id))
            if track is None or not track.visual or not track.present or track.radius <= 0.0:
                return where
            spotted = track.spotted_from
        ds = math.dist(spotted, goal)
        if ds < 0.2 or where is None or self._occ(*spotted, self.inflated):
            return where
        end, _ = where
        # If A* stopped >60 deg off the side the robot actually saw it from, step onto the spotted_from ray.
        a_end = math.atan2(end[1] - goal[1], end[0] - goal[0])
        a_spot = math.atan2(spotted[1] - goal[1], spotted[0] - goal[0])
        if abs(math.atan2(math.sin(a_end - a_spot), math.cos(a_end - a_spot))) <= math.radians(60):
            return where
        ux, uy = (spotted[0] - goal[0]) / ds, (spotted[1] - goal[1]) / ds
        for r in (reach, reach + 0.15, reach + 0.3, ds):
            cand = (goal[0] + min(r, ds) * ux, goal[1] + min(r, ds) * uy)
            if not self._occ(*cand, self.inflated) and self._plan(cand, 0.15) is not None:
                return cand, math.atan2(goal[1] - cand[1], goal[0] - cand[0])
        return where

    def _go(self, target: Point | Pose | Object, distance: float, *, side: str = "") -> None:
        if isinstance(target, Object) and hasattr(getattr(self, "perception", None), "detector"):
            self._watch(target.label)
        super()._go(target, distance, side=side)
        self._note_place()
        if side or not isinstance(target, Object):
            return
        cand = self._spotted_standpoint(target, distance)
        if cand is None:
            return
        d = math.dist((self.x, self.y), cand)
        if 0.10 < d <= 0.40 and self._clear_line((self.x, self.y), cand, grid=self.inflated):
            # Nav2's xy_goal_tolerance (0.25 m) can stop just off a narrow corner sightline (E2 umbrella: ~18 cm off
            # hid it behind the wall corner); step straight onto the spotting standpoint via drive_on_heading.
            try:
                self._rotate(wrap(math.atan2(cand[1] - self.y, cand[0] - self.x) - self.yaw))
                self._move(d)
            except PathBlocked:
                pass

    def _see_target(self, description: str) -> Thing | None:
        """When the detector finds nothing for `description`, ask the vision model across the cameras at 768px (0.35 s
        each; 98-100% yes on monitor/kitchen/rug/umbrella/shoes vs 5-47% at 512px, 0/145 FP after crop check on 30
        random apt_0 views), locate it by yes/no crops, range the crop, and register a visual track. During a search,
        if the stop itself shows nothing, also scan mid-leg trail snapshots so targets in narrow corridor windows
        (umbrella visible only over ~0.30 m along the hallway) are not jumped past."""
        phrase = re.sub(r"^(?:the|a|an|my)\s+", "", " ".join(description.lower().split())).strip()
        if not phrase:
            return None
        p = self.perception
        bridge = getattr(p, "scan_bridge", None)
        wanted = set(phrase.split()) & COLORS
        if "grey" in wanted or "gray" in wanted:
            wanted.update(("grey", "gray"))
        wanted_color = "grey" if ("grey" in wanted or "gray" in wanted) else next(iter(wanted), "")
        base = visual_name(phrase)
        label = canonical(base) if base else phrase

        def try_views(cam_entries: list[tuple[str, Any, np.ndarray, np.ndarray | None, tuple[float, float, float]]],
                      tag: str = "") -> Thing | None:
            # Mid-leg trail scans test up to 48 snapshots per stop: require >=0.80 to avoid borderline 54-69% distant
            # false stops (real mid-leg passes on umbrella/shoes scored 97-100%).
            min_score = 0.80 if tag else 0.50
            views = []
            for cam_name, stamp, rgb, depth, pose in cam_entries:
                score = see_probability(self.brain, rgb, phrase, max_dim=768)
                if score >= min_score:
                    views.append((score, cam_name, stamp, rgb, depth, pose))
            views.sort(key=lambda v: -v[0])
            placed: list[tuple[float, str, Any, np.ndarray, tuple[float, float, float], Any, float, float,
                               tuple[int, int, int, int]]] = []
            for score, cam_name, stamp, rgb, depth, pose in views:
                hit = locate_in_picture(self.brain, rgb, phrase)
                if hit is None:
                    continue
                (x1, y1, x2, y2), on_floor = hit
                # When a colour was requested ("red bike"), verify the crop's colour before placing a track: a
                # distant black wheel peeking under a desk scored black=98.8%, red=0.7% on _ask_color.
                if wanted and hasattr(self.brain, "decide") and self._ask_color(rgb[y1:y2, x1:x2], base or label) not in wanted:
                    continue
                scan, _ = taken_with(p.scans, _seconds(stamp), 0.5)
                if scan is None:
                    scan = p.scan
                place = locate_crop(RIG[cam_name], (x1, y1, x2, y2), depth, scan, on_floor=on_floor)
                if place is None:
                    continue
                mx, my = to_map(pose, place.x, place.y)
                placed.append((score, cam_name, stamp, rgb, pose, place, mx, my, (x1, y1, x2, y2)))
                if place.radius > 0.0:  # standing/elevated object: best camera is enough
                    break
            if not placed:
                return None
            score, cam_name, stamp, rgb, pose, place, mx, my, (x1, y1, x2, y2) = placed[0]
            # A wide floor-flat surface (e.g. rug) beside the robot spans neighbouring cameras; each camera sees one
            # edge, while the score-weighted centroid of nearby floor sightings lands on its middle (1.11-1.54 m -> 0.65 m).
            if place.radius == 0.0 and len(placed) > 1:
                near = [q for q in placed if q[5].radius == 0.0 and math.dist((q[6], q[7]), (mx, my)) <= 3.0]
                if len(near) > 1:
                    wsum = sum(q[0] for q in near)
                    mx = sum(q[0] * q[6] for q in near) / wsum
                    my = sum(q[0] * q[7] for q in near) / wsum
                    tag = f"{tag} [{len(near)} cams]"
            crop = rgb[y1:y2, x1:x2].copy()
            now = time.monotonic()
            with p.lock:
                existing = next(
                    (t for t in p.tracker.objects()
                     if t.present and t.label == label and (not wanted_color or t.color == wanted_color)
                     and math.dist((t.x, t.y), (mx, my)) <= max(0.8, t.radius + place.radius)),
                    None,
                )
                cam_xy = to_map(pose, RIG[cam_name].x, RIG[cam_name].y)
                if existing is not None:
                    existing.x, existing.y, existing.radius = mx, my, place.radius
                    existing.last_seen, existing.misses, existing.present = now, 0, True
                    existing.camera, existing.z, existing.spotted_from = cam_name, place.z, cam_xy
                    tid = existing.id
                else:
                    tid = p.tracker._next
                    p.tracker._next += 1
                    p.tracker.tracks[tid] = Track(
                        tid, label, mx, my, place.radius, now, now, hits=2, verified=True, sure=True,
                        var=place.sigma ** 2, noise=place.sigma, camera=cam_name,
                        visual_color=wanted_color, color_from=pixels(crop), visual=True,
                        z=place.z, spotted_from=cam_xy,
                    )
                p.crops[tid] = crop
            self._perceive_objects()
            thing = self.things.get(tid)
            if thing is not None:
                info = (f"{cam_name.replace('_camera', '').replace('_', '-')} ({score:.0%}), "
                        f"{math.hypot(mx - pose[0], my - pose[1]):.1f} m ({place.how}){tag}")
                self.trace.asked.append((round(self.t, 2), f"see {phrase}", info))
                print(f"see {phrase!r}: {info}", flush=True)
                return thing
            return None

        current = []
        for cam_name in RIG:
            frame = p.frames.get(cam_name)
            pose = self._pose_at(frame[0]) if frame is not None else None
            if pose is None:
                continue
            depth = None
            if bridge is not None and cam_name in bridge.cam_depths:
                depth, _ = taken_with(bridge.cam_depths[cam_name], _nanoseconds(frame[0]), 500_000_000)
            current.append((cam_name, frame[0], frame[1], depth, pose))
        found = try_views(current)
        if found is not None or not getattr(self, "_searching", False):
            return found
        after = getattr(self, "_search_snap_after", 0.0)
        snaps = [s for s in getattr(self, "_trail_views", ()) if s[0] > after][-16:]  # up to 4 m of a frontier leg
        self._search_snap_after = time.monotonic()
        for _, _, cams in reversed(snaps):
            entries = [(c, st, rgb, dep, pose) for c, (st, rgb, dep, pose) in cams.items()]
            found = try_views(entries, tag=" [passed]")
            if found is not None:
                return found
        return None

    # ---- spots on the floor, picked by the vision model -----------------------------------------------------------
    def floor_spots(self, camera: str = "front_camera", width: int = 768, away: tuple = (),
                    region: str | None = None) -> tuple[np.ndarray | None, list[Point]]:
        """The camera's newest picture, `width` wide, and the floor spots in its view that the robot can reach (free
        on its map, with nothing on the map between it and them), on the map; none within SEARCH_AWAY of a point in
        `away`, nor outside `region` (a room) when one is given."""
        import cv2

        frame = self.perception.frames.get(camera)
        pose = self._pose_at(frame[0]) if frame is not None else None
        if pose is None:
            return None, []
        cam, rgb = RIG[camera], frame[1]
        k = width / cam.width
        img = cv2.resize(rgb, (width, round(rgb.shape[0] * k)), interpolation=cv2.INTER_AREA)
        spots: list[Point] = []
        marks: list[tuple[float, float]] = []
        for ahead in SPOT_AHEAD:
            for i in range(-6, 7):
                across = i * SPOT_ACROSS
                bx = cam.x + ahead * math.cos(cam.yaw) - across * math.sin(cam.yaw)
                by = cam.y + ahead * math.sin(cam.yaw) + across * math.cos(cam.yaw)
                uv = cam.project(bx, by, 0.0)
                if uv is None:
                    continue
                u, v = uv[0] * k, uv[1] * k
                if any(math.dist((u, v), m) < SPOT_GAP * k * cam.width / 768 for m in marks):
                    continue
                mx, my = to_map(pose, bx, by)
                if self._occ(mx, my, self.inflated) or not self._clear_line(pose[:2], (mx, my)):
                    continue
                if any(math.dist((mx, my), a) < SEARCH_AWAY for a in away) or \
                        (region and self._room_at(mx, my) != region):
                    continue
                spots.append((mx, my))
                marks.append((u, v))
        return img, spots

    def begin(self) -> None:
        super().begin()
        self._search_coverage.clear()
        det = getattr(getattr(self, "perception", None), "detector", None)
        if hasattr(det, "clear"):
            det.clear()

    def search(self, description: str, area: str = "here", *, toward: str = "") -> Object | None:
        self.search_log = []
        self._search_from = len(self.trace.poses)  # where this search's own trail begins
        self._stepped_back = False  # once a search: a second step back backed away from "kitchen" again and again
        self._searching = True
        self._search_snap_after = time.monotonic()
        # Keep per-target coverage within one request (cleared in begin()) so a follow-up search(near=...) checks
        # the new area and still-uncovered floor instead of re-sweeping the whole home from scratch.
        key = canonical(visual_name(description) or description)
        cov = self._search_coverage.get(key)
        self.coverage = cov if cov is not None else np.zeros_like(self.occupied)
        self._search_coverage[key] = self.coverage
        self._cover()
        try:
            return super().search(description, area, toward=toward)
        finally:
            self._searching = False

    def _search_trail(self, every: float = 0.5) -> list[Point]:
        """Where the robot has been during this search, a point every half metre: the next stop keeps away from all
        of it. Keeping away only from the stops let it pick the floor it had just driven over, and it went back and
        forth along one path."""
        trail: list[Point] = []
        for _, x, y, _ in self.trace.poses[getattr(self, "_search_from", 0):]:
            if not trail or math.dist(trail[-1], (x, y)) >= every:
                trail.append((x, y))
        return trail

    def _frontier_step(self, frontiers: list[Pose], max_leg: float = 3.2) -> Point | None:
        """Nearest frontier by A* path length (not Euclidean through walls), capped to `max_leg` m so cameras and
        mid-leg trail snapshots (4 m window) inspect every corridor along the way instead of driving 12 m blind."""
        best: tuple[float, list[Point]] | None = None
        for v in frontiers[:6]:
            path = self._plan((v.x, v.y), 0.25)
            if not path:
                continue
            pts = [(self.x, self.y), *path]
            total = sum(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False))
            if best is None or total < best[0]:
                best = (total, pts)
        if best is None:
            return None
        dist, pts = 0.0, best[1]
        for a, b in zip(pts, pts[1:], strict=False):
            seg = math.dist(a, b)
            if dist + seg >= max_leg and seg > 1e-6:
                f = (max_leg - dist) / seg
                cand = (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
                if not self._occ(*cand, self.inflated):
                    return cand
            dist += seg
        return pts[-1]

    def _next_stop(self, description: str, region: str | None, tried: list[Point]) -> Point | None:
        """Where a search looks next, judged by the vision model from what each camera shows: for each direction with
        floor it can reach away from where it already looked, how likely going that way leads to the thing (a yes/no
        answer's probability); it drives to the farthest such floor in the likeliest direction. Asked to pick one of
        20-odd numbered dots across the three pictures instead, the 2B model took the last number offered nearly
        every time ("dot 23 of 23", "dot 4 of 4") with reasons like "most likely at Dot July".

        Every camera is asked, also one with no floor to drive to that way: there the thing may be right in front,
        too near for the detector to name. Beside the blue bicycle YOLOE found no bike in 24 frames, the vision model
        said yes to that front picture with 95% (7% for the way the search then went), and only the cameras with floor
        had been asked. Then the robot steps back, still facing that way, to see the thing whole."""
        options = []
        for cam in RIG:
            img, spots = self.floor_spots(cam, width=512, away=(*tried, *self._search_trail(), (self.x, self.y)),
                                          region=region)
            if img is None:
                continue
            p = self.brain.yes_probability(_with_image(data_url(img), (
                f"This picture is from the camera of a small robot looking for {description} in a home. Would going "
                f"the way this picture shows likely lead to {description}? Answer yes or no.")))
            options.append((p, cam, spots))
        if not options:
            return None
        named = ", ".join(f"{c.replace('_camera', '').replace('_', '-')} {q:.0%}" for q, c, _ in options)
        p, cam, spots = max(options, key=lambda o: o[0])
        facing = self.yaw + RIG[cam].yaw
        back = self._floor_toward(facing + math.pi, STEP_BACK)
        if not spots and p >= 0.5 and not self._stepped_back and all(
                math.dist(back, a) >= STEP_BACK / 2 for a in (*tried, (self.x, self.y))):
            stop, went, self._stepped_back = Pose(*back, math.degrees(facing)), "stepped back to see {} whole", True
        else:
            p, cam, spots = max((o for o in options if o[2]), key=lambda o: o[0], default=(0.0, cam, []))
            cov = self.coverage if getattr(self, "_searching", False) and self.walls.any() else None
            fresh = [s for s in spots if cov is None or not self._occ(*s, cov)]
            if p >= 0.5 and spots:
                # Strong visual lead: prefer uncovered floor in that camera's view before already-covered floor.
                stop = max(fresh or spots, key=lambda s: math.dist(s, (self.x, self.y)))
                went = "went {}"
            elif cov is not None:
                # No promising visual lead in any camera (<50%): drive to the nearest uncovered map frontier instead
                # of zigzagging across already-seen rooms (cut apt_0 full sweep from 14+ zigzags to 5 stops).
                away = (*tried, *self._search_trail(), (self.x, self.y))
                frontiers = [v for v in self._unseen(region or "all")
                             if all(math.dist((v.x, v.y), a) >= 1.2 for a in away)]
                step = self._frontier_step(frontiers) if frontiers else None
                if step is not None:
                    stop, went = step, "explored unseen area"
                else:
                    stop, went = max(fresh, key=lambda s: math.dist(s, (self.x, self.y)), default=None), "went {}"
            else:
                stop, went = max(spots, key=lambda s: math.dist(s, (self.x, self.y)), default=None), "went {}"
        if stop is None:
            return None
        self.search_log.append(f"{named}: {went.format(cam.replace('_camera', '').replace('_', '-'))}")
        self.trace.asked.append((round(self.t, 2), f"search {description}: which way", self.search_log[-1]))
        print(f"search {description!r}: {self.search_log[-1]}", flush=True)
        return stop

    def _clear_line(self, a: Point, b: Point, step: float = 0.1, *, grid: np.ndarray | None = None) -> bool:
        n = max(1, int(math.dist(a, b) / step))
        return not any(self._occ(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, grid)
                       for i in range(1, n + 1))

    def past_views(self, seconds: float, count: int = 4) -> list[tuple[float, np.ndarray]]:
        """Up to `count` front pictures from the last `seconds`, oldest first, each with how long ago it was taken."""
        past = list(self.perception.past)
        if not past:
            return []
        now = past[-1][0]
        recent = [(t, rgb) for t, rgb in past if now - t <= seconds]
        if len(recent) > count:
            recent = [recent[round(i * (len(recent) - 1) / max(1, count - 1))] for i in range(count)]
        return [(now - t, rgb) for t, rgb in recent]

    # ---- the vision model on what the cameras saw ----------------------------------------------------------------
    def _image_of(self, target: Any) -> str | None:
        target = self._target(target) if target is not None else None
        if isinstance(target, Object):
            crop = self.target_crop(target.id)
        else:
            frame = self.perception.frames.get("front_camera")
            crop = frame[1] if frame else None
        return data_url(crop) if crop is not None and crop.size else None

    def check(self, question: str, target: Any = None) -> bool:
        self._call("check", question=question, target=target)
        url = self._image_of(target)
        answer = self.brain.check(url, question)[0] if self.brain and url else False
        self.trace.asked.append((round(self.t, 2), question, "yes" if answer else "no"))
        self._sync()
        return answer

    def describe(self, target: Any = None, question: str = "What is this?") -> str:
        self._call("describe", question=question, target=target)
        url = self._image_of(target)
        answer = self.brain.describe(url, question).text.strip() if self.brain and url else "I can't tell"
        self.trace.asked.append((round(self.t, 2), question, answer))
        self._sync()
        return answer

    # ---- the user's answers --------------------------------------------------------------------------------------
    def _say(self, text: str) -> None:
        # Speech is played by the client. The 2D word timer kept a 161-word answer busy for ~147 wall seconds in 3D.
        self.trace.said.append((round(self.t, 2), text))

    def _listen(self, seconds: float) -> str | None:
        self.trace.listened.append(round(self.t, 2))
        self.waiting = True
        try:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                try:
                    return self.answers.get(timeout=0.1)
                except queue.Empty:
                    self._step(0.0, 0.0)
            return None
        finally:
            self.waiting = False

def judge(robot: SimRobot, truth: Scene) -> SimRobot:
    """The run as the scene's truth sees it: the suite's checks read things and rooms from the scene the simulator was
    built from, and the robot's trace from the run."""
    referee = SimRobot(truth)
    referee.trace, referee.t = robot.trace, robot.t
    referee.x, referee.y, referee.yaw = robot.x, robot.y, robot.yaw
    referee.notes = list(robot.notes)
    return referee
