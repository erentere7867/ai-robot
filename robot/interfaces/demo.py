"""Try Sprocket: a simulated home behind the live page, with a box to talk to it.

    python -m robot.interfaces.demo                     # the 3D apartment on the robot's real software (WSL, ROS,
                                                        # Habitat; llama-server on 8081): slower than real time
    python -m robot.interfaces.demo --scene "two rooms" # a fast 2D room instead
    python -m robot.interfaces.demo --case O3           # watch one 2D suite case
    python -m robot.interfaces.demo --cameras all       # send all three cameras with each request

Then open http://127.0.0.1:8765. Type an instruction and press Enter. While a task runs, whatever you type joins
that task's conversation (the model can change course); "stop" and the Stop button halt the robot at once; the room
picker swaps the room ("random room" builds a new one each time).
"""

from __future__ import annotations

import argparse
import base64
import queue
import random
import struct
import threading
import time
import zlib
from typing import Any

import numpy as np

from robot.agent.agent import Agent
from robot.agent.api import Preempted, SafetyStop
from robot.agent.instant import instant
from robot.interfaces.live import Live
from robot.mind.brain import Brain
from robot.perception.camera import encode_png
from robot.ros.robot import RosRobot
from robot.sim.world import SimRobot

SCENES = ("3D apartment", "living room", "two rooms", "office", "random room")


HOMES: dict[str, Any] = {}  # the running 3D home: its simulator, drivers and robot software stay up


def make_robot(name: str, brain: Any = None) -> Any:
    """The 3D apartment, run as the real robot (robot/sim/stack.py: its own map, Nav2, YOLOE, camera depth), or one
    of the 2D rooms in real time."""
    if name == "3D apartment":
        from robot.perception.detector import Detector
        from robot.sim.stack import Home

        if "home" not in HOMES:
            HOMES["home"], HOMES["detector"] = Home("apt_0"), Detector()
        HOMES["home"].reset()
        return HOMES["home"].robot(HOMES["detector"], brain)
    return SimRobot(make_scene(name), realtime=True)


def make_scene(name: str) -> Any:
    from robot.eval.suite import living_room, office, two_rooms

    if name == "random room":
        from robot.train.scenes import random_scene

        rng = random.Random()
        scene = None
        while scene is None:
            scene = random_scene(rng)
        return scene
    return {"living room": living_room, "two rooms": two_rooms, "office": office}[name]()


class Holder:
    """The robot now on screen; swapped by the room picker."""

    def __init__(self, sim: Any, home: Any = None) -> None:
        self.sim = sim
        self.home = home


def _slam_overlay(sim: Any) -> dict[str, Any] | None:
    """Encode the robot's 2D occupancy grid as a transparent RGBA PNG for the 3D floor overlay."""
    grid = getattr(sim, "walls", None)
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or not grid.size:
        return None
    occ = np.ascontiguousarray(grid.T[::-1, :])
    h, w = occ.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[occ] = (47, 111, 222, 145)
    raw = np.hstack([np.zeros((h, 1), dtype=np.uint8), rgba.reshape(h, w * 4)]).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, level=1)) + chunk(b"IEND", b"")
    return {"cx": 0.0, "cy": 0.0, "w": float(sim.scene.width), "h": float(sim.scene.height),
            "png": base64.b64encode(png).decode("ascii")}


def attach_3d(live: Live, holder: Holder) -> None:
    """Expose the 3D scene manifest (ReplicaCAD meshes when in a 3D home, plus SLAM floor overlay) to `Live`."""

    def world3d() -> dict[str, Any]:
        home = holder.home
        slam = _slam_overlay(holder.sim) if holder.sim is not None else None
        if home is None or not hasattr(home, "world"):
            return {"mode": "2d", "slam": slam}
        cached = getattr(home, "_scene_3d", None)
        if cached is None:
            cached = home.world.call(op="scene_3d")
            home._scene_3d = cached
        return {"mode": "3d", **cached, "slam": slam}

    live.world3d = world3d


CAMERAS = ("front_camera", "rear_left_camera", "rear_right_camera")


def show_cameras(live: Live, holder: Holder) -> None:
    """Live feeds of the three cameras on the page (the model itself only gets snapshots)."""

    def frame(name: str) -> bytes | None:
        if holder.sim is None or name not in CAMERAS:
            return None
        view = getattr(holder.sim, "live_view", holder.sim.camera_view)  # the newest frame, where the robot has one
        rgb = view(name)
        return encode_png(rgb) if rgb is not None else None

    live.camera = frame
    live.update(cameras=list(CAMERAS))


def refresh(live: Live, holder: Holder, stop_event: threading.Event | None = None) -> None:
    """Keep the map moving while the robot drives: one step can last many seconds."""
    while stop_event is None or not stop_event.is_set():
        if holder.sim is not None:
            try:
                live.show_world(holder.sim, home=holder.home)
            except (RuntimeError, AttributeError):  # world swapped mid-copy; take the next frame
                pass
        time.sleep(0.2)


def drive(live: Live, agent: Agent, holder: Holder, new_scene: Any = None, *,
          stop_event: threading.Event | None = None) -> None:
    """Act on what arrives in Live.inbox until `stop_event` is set."""
    worker: threading.Thread | None = None

    def busy() -> bool:
        return worker is not None and worker.is_alive()

    def halt() -> None:
        if busy():
            agent.stop()
            worker.join(5.0)
        if holder.sim is not None:
            try:
                holder.sim.stop()
            except Preempted:
                pass

    def do(text: str) -> None:
        print(f"instruct  {text}", flush=True)
        try:
            record = agent.do(text, holder.sim)
        except Exception as exc:  # noqa: BLE001 - the worker must never die silently while showing "running"
            print(f"error     {text}: {exc}", flush=True)
            live.update(status="failed", problems=[str(exc)])
            return
        live.show_world(holder.sim)
        entry = {"instruction": text, "outcome": record.outcome, "steps": record.lines()}
        live.update(history=(live.state.history + [entry])[-12:])
        print(f"{record.outcome:<9} {text}", flush=True)

    while stop_event is None or not stop_event.is_set():
        try:
            kind, text = live.inbox.get(timeout=0.1)
        except queue.Empty:
            continue
        if kind == "instruct" and text and getattr(holder.sim, "waiting", False) and instant(text) is None:
            holder.sim.answers.put(text)  # the answer to the robot's own question
        elif kind == "scene" and new_scene is not None and text in SCENES:
            halt()
            try:
                if busy():
                    raise RuntimeError("The previous task is still stopping; try again once it exits.")
                new_scene(text)
            except Exception as exc:  # noqa: BLE001 - a room that cannot load must not take the page down
                print(f"error     room {text}: {exc}", flush=True)
                live.update(status="failed", problems=[f"Could not load {text}: {exc}"])
            finally:
                live.update(scene_loading="")
        elif kind == "stop" or (kind == "instruct" and instant(text) == "stop"):
            halt()
            live.update(status="stopped", posture="neutral", chime="uh_oh")
        elif kind == "instruct" and text and holder.sim is None:
            live.update(status="failed", problems=["No robot is loaded; select a room."])
        elif kind == "instruct" and text and busy():
            agent.interject(text)  # joins the running task's conversation
        elif kind == "instruct" and text:
            worker = threading.Thread(target=do, args=(text,), daemon=True)
            worker.start()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="run this suite case and keep the page open")
    parser.add_argument("--scene", choices=SCENES, default="3D apartment")
    parser.add_argument("--host", default="127.0.0.1", help="dashboard bind address; use 0.0.0.0 on a trusted LAN")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--brain", default="http://127.0.0.1:8081", help="llama-server URL")
    parser.add_argument("--cameras", choices=("front", "all"), default="all")
    parser.add_argument("--think", choices=("off", "low", "medium", "high"), default="low", help="how long the model may reason before each turn")
    args = parser.parse_args()

    live = Live()
    live.serve(args.port, host=args.host)
    from robot.interfaces.voice import attach

    attach(live)
    print(f"Dashboard listening on {args.host}:{args.port}", flush=True)
    brain = Brain(args.brain)
    if not brain.healthy():
        print(f"llama-server is not answering on {args.brain}; start it first (tools/linux/brain.sh)", flush=True)
    agent = Agent(brain, live=live, cameras=args.cameras, think=args.think)

    if args.case:
        from robot.eval.suite import CASES

        case = next(c for c in CASES if c.id == args.case)
        holder = Holder(SimRobot(case.scene(), realtime=True))
        attach_3d(live, holder)
        live.update(scene=case.id, scenes=[case.id])
        show_cameras(live, holder)
        threading.Thread(target=refresh, args=(live, holder), daemon=True).start()
        record = agent.do(case.say, holder.sim)
        print(f"{record.outcome}; passes its check: {bool(case.check(holder.sim))}. Ctrl+C to quit.", flush=True)
        while True:
            time.sleep(1)

    holder = Holder(make_robot(args.scene, brain), HOMES.get("home") if args.scene == "3D apartment" else None)
    attach_3d(live, holder)
    live.update(scene=args.scene, scenes=list(SCENES))
    show_cameras(live, holder)
    threading.Thread(target=refresh, args=(live, holder), daemon=True).start()

    shutdown_failed = False

    def new_scene(name: str) -> None:
        nonlocal shutdown_failed
        if shutdown_failed:
            raise RuntimeError("Restart the demo and Nav2 before changing rooms.")
        previous = holder.sim
        reset_apartment = isinstance(previous, RosRobot) and name == "3D apartment"
        replacement = None if reset_apartment else make_robot(name, brain)
        if isinstance(previous, RosRobot):
            holder.sim = holder.home = None
            shutdown_failed = True
            previous.perception.stop()
            try:
                previous.shutdown()
            except SafetyStop as exc:
                raise RuntimeError("Restart the demo and Nav2 before changing rooms.") from exc
            shutdown_failed = False
        holder.sim = make_robot(name, brain) if reset_apartment else replacement
        holder.home = HOMES.get("home") if name == "3D apartment" else None
        agent.forget()  # a new room: earlier requests no longer apply
        live.update(scene=name, status="idle", instruction="", steps=[], todo="", problems=[], said=[], chime="",
                    true_pose=None, true_trail=[], dock=None, true_objects=[])

    drive(live, agent, holder, new_scene)


if __name__ == "__main__":
    main()
