"""Sprocket v2 full ROS 2 robot stack: Nav2 motion, 3-CSI perception, Brain VLM, and CirculatingMemory.

`RealRobot` is a `RosRobot` whose objects, 3D bounds, velocities, and people come from YOLOE-26 and the tracker
(robot/ros/perceive.py), whose vision checks/descriptions go to LFM3-VL-2B via `Brain` with Set-of-Mark / object
crops, and whose long-term spatial habits, clear-space observations, and speaker identity persist in
`CirculatingMemory` (robot/mind/memory.py). The scene truth is used solely for post-run evaluation (`judge`).
"""

from __future__ import annotations

import math
import queue
import threading
import time
from typing import Any

import numpy as np

from robot.agent.api import Object, Pose
from robot.mind.memory import ANCHOR_LABELS, CANONICAL_SYNONYMS
from robot.perception.camera import data_url
from robot.perception.rig import to_robot
from robot.ros.perceive import Perception
from robot.ros.robot import RosRobot, _resize_rgb
from robot.sim.world import COLORS, RES, Scene, SimRobot, Thing

# Other names for what perception calls a thing. "desk" answers to "table" only when no table is known (a word match
# scores below a whole name): from low down a coffee table is often seen as a desk.
NAMES = {
    "desk": ("table", "work table", "coffee table", "furniture"),
    "table": ("desk", "coffee table", "furniture"),
    "sofa": ("couch", "seat", "something to sit on"),
    "chair": ("seat", "something to sit on"),
    "person": ("human",),
    "cup": ("mug",),
    "backpack": ("bag",),
    "ball": ("toy",),
    "bin": ("trash can",),
}
SMALL = {"cup", "bottle", "ball", "shoe", "phone", "keys", "remote", "book", "toy"}  # small objects, not obstacles
SEEN_NOW = 1.5  # seconds: a track seen this recently is "in view now"


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
        self.perception = Perception(self.node, detector, self._pose_at, brain,
                                     scan_bridge=getattr(self, "_scan_bridge", None))
        self.brain = brain
        self.things, self.memory = {}, {}
        self._people_velocities: dict[int, tuple[float, float]] = {}
        self._cleared_tracks: set[int] = set()
        self._habit_recorded: dict[int, str] = {}
        self.human_pose: Any = None
        self.speaker_profile: Any = None
        self._candidate_paths: list[tuple[str, list[tuple[float, float]]]] = []
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
        people, people_vel = [], {}
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
            self._cleared_tracks.discard(tr.id)
            thing = Thing(tr.label, round(tr.x, 3), round(tr.y, 3), round(tr.radius, 3), tr.color,
                          names=NAMES.get(tr.label, ()), solid=not tr.person and tr.label not in SMALL,
                          id=tr.id)
            things[tr.id] = thing
            memory[tr.id] = ((thing.x, thing.y), self.t - (now - tr.last_seen))
            if now - tr.last_seen <= SEEN_NOW:
                detected.add(tr.id)
            if tr.person:
                people.append((tr, thing))
                people_vel[tr.id] = (float(getattr(tr, "vx", 0.0)), float(getattr(tr, "vy", 0.0)))
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
        self._people_velocities = people_vel
        self.human_pose = human_pose
        if getattr(self, "_rooms", None) and now - getattr(self, "_named_at", 0.0) > 5.0:  # rooms take names from the tour and the furniture seen
            from robot.ros.rooms import name

            self._named_at = now
            self.scene.rooms = name(self._rooms, RES, (self.scene.width, self.scene.height), list(things.values()),
                                    self.places)

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

    def _pivot_paths(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if self._candidate_paths:
            return list(self._candidate_paths)
        pose = (self.x, self.y, self.yaw)
        u = self.user()
        goal = u.position if u is not None else (self.x + 2.0 * math.cos(self.yaw), self.y + 2.0 * math.sin(self.yaw))
        try:
            rollouts = self.body.rollout_trajectories(
                pose, goal, lambda x, y: self._occ(x, y, self.inflated), people=self._people_with_vel(), horizon_s=1.2
            )
        except Exception:
            return []
        return [(c.label, [to_robot(pose, wx, wy) for wx, wy in c.points]) for c in rollouts[:3]]

    def camera_view(self, camera: str = "front_camera", width: int = 512, height: int = 288) -> np.ndarray | None:
        """The latest real frame from one camera with the tracker's ids marked, as the vision model sees it."""
        return self.perception.render_annotated_view(camera=camera, width=width, height=height)

    def render_view(self, width: int = 320, height: int = 240, candidate_paths: Any = ()) -> np.ndarray:
        """Latest front CSI view with Set-of-Mark box IDs and *PIVOT* floor trajectory ribbons."""
        paths = list(candidate_paths) if candidate_paths else self._pivot_paths()
        annotated = self.perception.render_annotated_view(candidate_paths=paths)
        if annotated is None or annotated.size == 0:
            return super().render_view(width=width, height=height)
        return _resize_rgb(annotated, width, height)

    def _visible(self) -> list[Thing]:
        return [self.things[i] for i in self._detected if i in self.things]

    def _detectable(self, t: Thing) -> bool:
        return True

    def _watch(self, description: str) -> None:
        phrase = " ".join(w for w in description.lower().split() if w not in COLORS)
        if phrase:
            self.perception.detector.watch(phrase)

    def _in_view(self) -> list[Thing]:
        return self._visible()

    # ---- the vision model on what the cameras saw ----------------------------------------------------------------
    def _image_of(self, target: Any) -> str | None:
        target = self._target(target) if target is not None else None
        crop = self.perception.crops.get(target.id) if isinstance(target, Object) else None
        if crop is None:
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
