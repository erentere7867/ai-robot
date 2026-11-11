"""The robot API on the full stack: motion through Nav2, and perception from the cameras instead of the scene.

`RealRobot` is a `RosRobot` whose objects, memory and user come from YOLOE-26 and the tracker (robot/ros/perceive.py),
whose vision questions go to LFM3-VL-2B with the object's camera crop, and whose answers from the user arrive from the
page or the microphone. Planning uses the saved map, the one the robot localises in. The gate's dry run plays out in
the world as perceived, never in the scene the simulator was built from: that scene is used only to judge a run
afterwards (`judge`).
"""

from __future__ import annotations

import math
import queue
import time
from dataclasses import replace
from typing import Any

from robot.agent.api import Object, Pose
from robot.perception.camera import data_url
from robot.ros.perceive import Perception
from robot.ros.robot import RosRobot
from robot.sim.world import COLORS, Scene, SimRobot, Thing

NAMES = {"sofa": ("couch", "seat", "something to sit on"), "chair": ("seat", "something to sit on"),
         "cup": ("mug",), "backpack": ("bag",), "ball": ("toy",), "bin": ("trash can",)}
SMALL = {"cup", "bottle", "ball", "shoe", "phone", "keys", "remote", "book", "toy"}  # under the lidar, not obstacles
SEEN_NOW = 1.5  # seconds: a track seen this recently is "in view now"


class RealRobot(RosRobot):
    def __init__(self, scene: Scene, *, detector: Any, brain: Any = None, max_time: float = 600.0,
                 nudge: Any = None) -> None:
        self._ready = False
        super().__init__(scene, max_time=max_time, nudge=nudge)
        self.perception = Perception(self.node, detector, self._pose_at, brain)
        self.brain = brain
        self.things, self.memory = {}, {}
        self.answers: queue.Queue[str] = queue.Queue()  # what the user types or says while a program waits
        self.waiting = False
        self._user: int | None = None
        self._ready = True

    # ---- perception from the cameras ------------------------------------------------------------------------------
    def _pose_at(self, stamp: Any) -> tuple[float, float, float] | None:
        from rclpy.time import Time

        for when in (Time.from_msg(stamp), Time()):
            try:
                tr = self.tf.lookup_transform("map", "base_link", when).transform
            except Exception:  # noqa: BLE001 - not at that time: try the latest
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
        for tr in self.perception.objects():
            if not tr.present:
                continue  # looked at clearly and not there: forgotten as a place to go
            thing = Thing(tr.label, round(tr.x, 3), round(tr.y, 3), round(tr.radius, 3), tr.color,
                          names=NAMES.get(tr.label, ()), solid=not tr.person and tr.label not in SMALL,
                          id=tr.id)
            things[tr.id] = thing
            memory[tr.id] = ((thing.x, thing.y), self.t - (now - tr.last_seen))
            if now - tr.last_seen <= SEEN_NOW:
                detected.add(tr.id)
            if tr.person:
                people.append(thing)
        if people:  # the user is the person the robot has known longest (on the robot: the voice's direction too)
            self._user = self._user if self._user in things else min(p.id for p in people)
            things[self._user].user = True
        self.things, self.memory, self._detected = things, memory, detected

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

    # ---- the dry run happens in the world as perceived -----------------------------------------------------------
    def clone(self, **limits: Any) -> SimRobot:
        things = [Thing(t.label, t.x, t.y, t.radius, t.color, names=t.names, solid=t.solid, user=t.user, id=t.id)
                  for t in self.things.values()]
        scene = replace(self.scene, things=things, start=Pose(self.x, self.y, math.degrees(self.yaw)), seen=[],
                        replies=[], noise=0.0)
        twin = SimRobot(scene, **limits)
        twin.set_occupied(self.occupied | twin.occupied)  # the saved map, plus what perception found
        twin.t, twin.memory, twin.notes = self.t, dict(self.memory), list(self.notes)
        twin.places, twin.coverage = dict(self.places), self.coverage.copy()
        return twin


def judge(robot: SimRobot, truth: Scene) -> SimRobot:
    """The run as the scene's truth sees it: the suite's checks read things and rooms from the scene the simulator was
    built from, and the robot's trace from the run."""
    referee = SimRobot(truth)
    referee.trace, referee.t = robot.trace, robot.t
    referee.x, referee.y, referee.yaw = robot.x, robot.y, robot.yaw
    referee.notes = list(robot.notes)
    return referee
