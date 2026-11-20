"""Sprocket v2 full ROS 2 robot stack: Nav2 motion, 3-CSI perception, Brain VLM, and CirculatingMemory.

`RealRobot` is a `RosRobot` whose objects, 3D bounds, velocities, and people come from YOLOE-26 looking for the words
tools ask for, and the tracker (robot/ros/perceive.py); whose vision checks, room names and floor spots come from
LFM3-VL-2B via `Brain`; and whose long-term spatial habits, clear-space observations, and speaker identity persist in
`CirculatingMemory` (robot/mind/memory.py). The scene truth is used solely for post-run evaluation (`judge`).
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np

from robot.agent.api import Matches, NotFound, Object, Point, Pose
from robot.mind.brain import _with_image
from robot.mind.memory import ANCHOR_LABELS, CANONICAL_SYNONYMS
from robot.perception.camera import data_url
from robot.perception.rig import RIG, to_map
from robot.perception.tracker import pixels
from robot.ros.perceive import Perception
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
SEARCH_AWAY = 1.0  # metres: a search offers no dot this near a place it already looked from
STEP_BACK = 1.0  # metres a search backs off from what is too near to name: YOLOE named the blue bicycle from the dock
# (2.3 m) and 1.05 m from its centre, but beside it (0.2 m from its edge) in 1 of 2 front frames and in 0 of 8


class RealRobot(RosRobot):
    def __init__(self, scene: Scene, *, detector: Any, brain: Any = None, max_time: float = 600.0,
                 nudge: Any = None, store: Any = None, grid: Any = None) -> None:
        """`store` (CirculatingMemory) persists seen objects, habits, speakers, facts and places across restarts;
        `grid` is the occupancy of the map the robot built itself (robot/ros/explore.py), for planning."""
        self._ready = False
        super().__init__(scene, max_time=max_time, nudge=nudge)
        self._rooms: list = []  # room masks of the robot's own map, named again as it learns (robot/ros/rooms.py)
        self._named_at = -1e9
        if grid is not None:
            from robot.ros.rooms import segment

            self.set_occupied(grid)
            self._rooms = segment(~grid, RES)
        self._room_answers: list[Counter] = [Counter() for _ in self._rooms]  # the vision model's, per room
        self.search_log: list[str] = []  # the last search: where the model thought the thing was, stop by stop
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
        self.store, self._closed = store, False
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
            self._save()

    def _save(self) -> None:
        if self.store is not None:
            with self.perception.lock:
                self.store.save(self.perception.tracker, self.notes, self.places)

    def shutdown(self) -> None:
        self._save()
        self._closed = True
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
        if getattr(self, "_rooms", None) and now - getattr(self, "_named_at", 0.0) > 5.0:  # from the tour and the model
            from robot.ros.rooms import name

            self._named_at = now
            self.scene.rooms = name(self._rooms, RES, (self.scene.width, self.scene.height), self._room_answers,
                                    self.places, before=list(self.scene.rooms or {}))

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
        return super().obj(now)

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

    def _matching(self, description: str, things: list[Thing]) -> list[Thing]:
        wanted = set(description.lower().split()) & COLORS
        if not wanted or self.brain is None:
            return super()._matching(description, things)
        if "grey" in wanted or "gray" in wanted:
            wanted.update(("grey", "gray"))
        matches = []
        p = self.perception
        for thing in super()._matching(visual_name(description), things):
            with p.lock:
                track = p.tracker.tracks.get(thing.id)
                if track is None or track.rejected or not track.present:
                    continue
                crop, color = p.crops.get(thing.id), track.visual_color
                recheck = crop is not None and (not color or pixels(crop) >= 2 * track.color_from)
            if recheck:
                # Most of a bicycle box was wall or tyres. Ask about its colour, not the median pixel;
                # only when a tool needs the attribute, and never while holding the camera worker's lock.
                question = (f"What colour would a person call the {thing.label} in the middle of this picture? "
                            "Ignore the background. Answer with one colour or unknown.")
                scores = self.brain.decide(_with_image(data_url(crop), question), choices=COLOR_WORDS)
                scores["grey"] += scores.pop("gray")
                word = max(scores, key=scores.get)
                color = word if scores[word] > 0.5 else "unknown"
                with p.lock:
                    if p.tracker.tracks.get(thing.id) is not track or track.rejected or not track.present:
                        continue
                    track.visual_color, track.color_from = color, pixels(crop)
            if color in wanted:
                thing.color = color
                matches.append(thing)
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
        end = time.monotonic() + 3.0  # and the vision model's check of what it found: a false "tv" is no target
        while self.brain is not None and time.monotonic() < end and any(
            not t.verified and self._matching(word, [Thing(t.label, 0, 0, 0)])
            for t in self.perception.objects() if t.present and not t.person
        ):
            self._step(0.0, 0.0)
        self._perceive_objects()
        found = self._matching(description, self._in_view())
        return Matches(sorted((self._object(t, *self.memory[t.id]) for t in found), key=lambda o: o.distance))

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

    def search(self, description: str, area: str = "here") -> Object | None:
        self.search_log = []
        self._search_from = len(self.trace.poses)  # where this search's own trail begins
        return super().search(description, area)

    def _search_trail(self, every: float = 0.5) -> list[Point]:
        """Where the robot has been during this search, a point every half metre: the next stop keeps away from all
        of it. Keeping away only from the stops let it pick the floor it had just driven over, and it went back and
        forth along one path."""
        trail: list[Point] = []
        for _, x, y, _ in self.trace.poses[getattr(self, "_search_from", 0):]:
            if not trail or math.dist(trail[-1], (x, y)) >= every:
                trail.append((x, y))
        return trail

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
        if not spots and all(math.dist(back, a) >= STEP_BACK / 2 for a in (*tried, (self.x, self.y))):
            stop, went = Pose(*back, math.degrees(facing)), "stepped back to see {} whole"
        else:  # with no floor behind either (a stop that did not move came up 20 times), the likeliest way with floor
            p, cam, spots = max((o for o in options if o[2]), key=lambda o: o[0], default=(0.0, cam, []))
            stop, went = max(spots, key=lambda s: math.dist(s, (self.x, self.y)), default=None), "went {}"
        if stop is None:
            return None
        self.search_log.append(f"{named}: {went.format(cam.replace('_camera', '').replace('_', '-'))}")
        self.trace.asked.append((round(self.t, 2), f"search {description}: which way", self.search_log[-1]))
        print(f"search {description!r}: {self.search_log[-1]}", flush=True)
        return stop  # floor: the farthest in that view, the most new floor to see

    def _clear_line(self, a: Point, b: Point, step: float = 0.1) -> bool:
        n = max(1, int(math.dist(a, b) / step))
        return not any(self._occ(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(1, n + 1))

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
