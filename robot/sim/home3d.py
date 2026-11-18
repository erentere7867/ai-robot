"""A 3D home for the robot: a real-looking apartment rendered by Habitat-Sim on the GPU, behind the 2D world's API.

The robot drives as in the 2D world (the same kinematics, planner and skills) on the home's real drivable floor, and
everything it sees comes from 3D rendering: the three cameras' pictures, and which objects are in view (an object
counts as seen when enough of its pixels are visible in a camera, as a detector would find it, so furniture hides
what is behind or under it). Habitat runs in its own Python; robot/sim/habitat_server.py is started on demand.

    robot = Home3D()                       # ReplicaCAD apartment apt_0
    robot.camera_view("front_camera")      # a real 3D picture with the tracker's ids marked
"""

from __future__ import annotations

import atexit
import math
import os
import socket
import subprocess
import threading
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np

from robot.agent.api import PathBlocked, Pose
from robot.perception import colour
from robot.perception.camera import resize_rgb
from robot.perception.overlay import draw_set_of_mark
from robot.ros.rooms import KIND, segment
from robot.sim.world import PLAN, RES, Scene, SimRobot, Thing

HABITAT_PYTHON = os.environ.get("SPROCKET_HABITAT_PYTHON", str(Path("~/micromamba/envs/habitat/bin/python").expanduser()))
SERVER = Path(__file__).with_name("habitat_server.py")
CAMERAS = ("front_camera", "rear_left_camera", "rear_right_camera")
MIN_PIXELS = 60  # an object this many pixels in view (of 512 x 288) is detected
COLOUR_PIXELS = 60  # and named a colour as soon as it is detected (a far bowl is small but still has a colour)


class _Server:
    """One Habitat process per scene, kept for the session: loading a home takes a few seconds."""

    running: dict[str, _Server] = {}

    def __init__(self, scene: str) -> None:
        self.lock = threading.Lock()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env = {**os.environ, "GALLIUM_DRIVER": os.environ.get("GALLIUM_DRIVER", "d3d12")}
        self.proc = subprocess.Popen([HABITAT_PYTHON, str(SERVER), "--port", str(port), "--scene", scene],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True,
                                     cwd="/tmp")
        for line in self.proc.stdout or ():  # Habitat prints its renderer details first
            if line.strip() == "ready":
                break
        else:
            raise RuntimeError(f"the Habitat server did not start ({HABITAT_PYTHON}); is the habitat env installed?")
        self.proc.stdout.close()  # nothing more is read; an unread pipe could fill and block the server
        for _ in range(50):
            try:
                self.conn = Client(("127.0.0.1", port), authkey=b"sprocket")
                break
            except ConnectionRefusedError:
                time.sleep(0.1)
        atexit.register(self.close)

    def call(self, **request: Any) -> Any:
        with self.lock:  # the agent and the live page's camera feeds share one connection
            self.conn.send(request)
            return self.conn.recv()

    def close(self) -> None:
        try:
            self.conn.send({"op": "quit"})
        except (OSError, AttributeError):
            pass
        self.proc.terminate()

    @classmethod
    def for_scene(cls, scene: str) -> _Server:
        if scene not in cls.running or cls.running[scene].proc.poll() is not None:
            cls.running[scene] = cls(scene)
        return cls.running[scene]


class Home3D(SimRobot):
    """The robot API in a 3D home. Things are the home's objects; nobody else is home (there is no user)."""

    def __init__(self, scene: str = "apt_0", *, realtime: bool = False, max_time: float = 600.0) -> None:
        self.server = _Server.for_scene(scene)
        self.info = self.server.call(op="info")
        nx, ny = self.info["shape"]
        bits = np.unpackbits(np.frombuffer(self.info["occupied"], np.uint8))[: nx * ny]
        self._floor = bits.reshape(nx, ny).astype(bool)
        self._seen: dict[int, int] = {}  # id -> pixels in view at the last look
        self._boxes: dict[str, dict[int, tuple]] = {}
        w, h = self.info["width"], self.info["height"]
        things = [Thing(o["label"], o["x"], o["y"], o["radius"], solid=False, id=o["id"],
                        facts={"height": o["height"]}) for o in self.info["objects"]]
        x, y, yaw = self.info["start"]
        self._room_masks = self._rooms(things, w, h)
        rooms = {}
        for room, mask in self._room_masks.items():  # the rest of the robot keeps a rectangle per room
            xs, ys = np.nonzero(mask)
            rooms[room] = (float(-w / 2 + xs.min() * RES), float(-h / 2 + ys.min() * RES),
                           float(-w / 2 + (xs.max() + 1) * RES), float(-h / 2 + (ys.max() + 1) * RES))
        scene_ = Scene(w, h, things, rooms, start=Pose(x, y, math.degrees(yaw)))
        super().__init__(scene_, realtime=realtime, max_time=max_time)

    def _rooms(self, things: list[Thing], w: float, h: float) -> dict[str, np.ndarray]:
        """Rooms as the robot would find them on its own map: the floor split at doorways (robot/ros/rooms.py), each
        named by its furniture. An open-plan room with, say, a fridge and a sofa is split again into areas, each floor
        cell going to the kind of furniture nearest it: people still say "the kitchen" for the end with the fridge."""
        free = ~self._floor
        gx, gy = np.meshgrid(np.arange(free.shape[0]) * RES + RES / 2 - w / 2,
                             np.arange(free.shape[1]) * RES + RES / 2 - h / 2, indexing="ij")
        masks: dict[str, np.ndarray] = {}
        unnamed = 0
        for mask in segment(free, RES):
            inside = [t for t in things if t.label in KIND and self._cell_in(mask, t.x, t.y, w, h)]
            kinds = sorted({KIND[t.label] for t in inside})
            if len(kinds) <= 1:
                if kinds and kinds[0] not in masks:
                    room = kinds[0]
                else:
                    unnamed += 1
                    room = f"room {unnamed}"
                masks[room] = mask
                continue
            near = np.argmin([(gx - t.x) ** 2 + (gy - t.y) ** 2 for t in inside], axis=0)
            for kind in kinds:
                mine = [k for k, t in enumerate(inside) if KIND[t.label] == kind]
                masks[kind] = masks.get(kind, np.zeros_like(mask)) | (mask & np.isin(near, mine))
        return masks or {"home": free}

    @staticmethod
    def _cell_in(mask: np.ndarray, x: float, y: float, w: float, h: float) -> bool:
        i, j = int((x + w / 2) / RES), int((y + h / 2) / RES)
        return 0 <= i < mask.shape[0] and 0 <= j < mask.shape[1] and bool(mask[i, j])

    def _room_at(self, x: float, y: float) -> str:
        w, h = self.scene.width, self.scene.height
        return next((room for room, mask in self._room_masks.items() if self._cell_in(mask, x, y, w, h)), "home")

    def _named(self, name: str) -> Any:
        """A room by name is the reachable spot nearest the middle of its floor (its rectangle's middle can lie in
        the next area of an open-plan home); anything else as in the 2D world."""
        key = str(name).lower().strip().removeprefix("the ").strip()
        mask = self._room_masks.get(key) if key not in self.places else None
        if mask is None:
            return super()._named(name)
        w, h = self.scene.width, self.scene.height
        xs, ys = np.nonzero(mask)
        middle = (-w / 2 + (xs.mean() + 0.5) * RES, -h / 2 + (ys.mean() + 0.5) * RES)
        reach, res = self._reachable(), RES * PLAN
        cells = [(i, j) for i, j in zip(*np.nonzero(reach), strict=True)
                 if self._cell_in(mask, *self._centre(i, j, res), w, h)]
        if not cells:
            raise PathBlocked(f"no reachable spot in the {key}")
        return self._centre(*min(cells, key=lambda c: math.dist(self._centre(*c, res), middle)), res)

    # ---- the home's floor instead of drawn walls -------------------------------------------------------------
    def _build_grid(self) -> None:
        self.walls = self._floor.copy()  # anything not drivable blocks the view at camera height, as furniture does
        self.set_occupied(self._floor.copy())
        self.coverage = np.zeros_like(self.walls)

    # ---- seeing by rendering ---------------------------------------------------------------------------------
    def _look(self, cameras: tuple[str, ...] = CAMERAS, rgb: bool = False, full: bool = False) -> dict:
        reply = self.server.call(op="render", pose=(self.x, self.y, self.yaw), cameras=list(cameras), rgb=rgb,
                                 full=full)
        self._boxes.update(reply["boxes"])
        return reply

    def _perceive_objects(self) -> None:
        reply = self._look()
        pixels: dict[int, int] = {}
        for boxes in reply["boxes"].values():
            for k, box in boxes.items():
                pixels[k] = pixels.get(k, 0) + box[4]
        self._seen = pixels
        self._detected = set()
        for k, n in pixels.items():
            t = self.things.get(k)
            if t is None or n < MIN_PIXELS:  # size in the picture, not a distance cap: a sofa 6 m off is plain
                continue                     # to see (and to the model looking at the same picture)
            self._detected.add(k)
            self.memory[k] = ((t.x, t.y), self.t)
        # a colour for each newly detected object, from its pixels, as the real tracker does (perception/colour.py)
        new = [k for k in self._detected if not self.things[k].color and pixels[k] >= COLOUR_PIXELS]
        if new:
            sampled = self.server.call(op="pixels", pose=(self.x, self.y, self.yaw), ids=new)
            for k, raw in sampled.items():
                self.things[k].color = colour.name(np.frombuffer(raw, np.uint8).reshape(-1, 3))

    def _visible(self) -> list[Thing]:
        return [self.things[k] for k in self._detected if k in self.things]

    def _detectable(self, t: Thing) -> bool:
        return True  # an open-vocabulary detector names anything it sees

    def camera_view(self, camera: str = "front_camera", width: int = 1024, height: int = 576) -> np.ndarray:
        """The rendered picture from one camera at `width` x `height` (drawn large and scaled down smoothly), with the
        tracker's ids on the objects it has detected, drawn at the final size so they stay readable."""
        h, w = self.info["image"]
        full = width > w
        reply = self._look((camera,), rgb=True, full=full)
        s = self.info.get("ss", 1) if full else 1
        img = np.frombuffer(reply["rgb"][camera], np.uint8).reshape(h * s, w * s, 3)
        if (width, height) != (w * s, h * s):
            img = resize_rgb(img, width, height)
        fx, fy = width / w, height / h
        marks = [((round(b[0] * fx), round(b[1] * fy), round((b[2] + 1) * fx) - 1, round((b[3] + 1) * fy) - 1), k)
                 for k, b in reply["boxes"][camera].items() if k in self.memory and b[4] >= MIN_PIXELS]
        return draw_set_of_mark(img, marks) if marks else img.copy()

    def render_view(self, width: int = 512, height: int = 288) -> np.ndarray:
        return self.camera_view("front_camera", width, height)
