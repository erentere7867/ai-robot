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
import queue
import random
import threading
import time
from typing import Any

from robot.agent.agent import Agent
from robot.agent.api import Preempted
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

    def __init__(self, sim: Any) -> None:
        self.sim = sim


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
                live.show_world(holder.sim)
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
                new_scene(text)
            except Exception as exc:  # noqa: BLE001 - a room that cannot load must not take the page down
                print(f"error     room {text}: {exc}", flush=True)
                live.update(status="failed", problems=[f"Could not load {text}: {exc}"])
            finally:
                live.update(scene_loading="")
        elif kind == "stop" or (kind == "instruct" and instant(text) == "stop"):
            halt()
            live.update(status="stopped", posture="neutral", chime="uh_oh")
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
    parser.add_argument("--think", choices=("off", "low", "medium", "high"), default="medium", help="how long the model may reason before each turn")
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
        live.update(scene=case.id, scenes=[case.id])
        show_cameras(live, holder)
        threading.Thread(target=refresh, args=(live, holder), daemon=True).start()
        record = agent.do(case.say, holder.sim)
        print(f"{record.outcome}; passes its check: {bool(case.check(holder.sim))}. Ctrl+C to quit.", flush=True)
        while True:
            time.sleep(1)

    holder = Holder(make_robot(args.scene, brain))
    live.update(scene=args.scene, scenes=list(SCENES))
    show_cameras(live, holder)
    threading.Thread(target=refresh, args=(live, holder), daemon=True).start()

    def new_scene(name: str) -> None:
        previous = holder.sim
        holder.sim = make_robot(name, brain)
        if isinstance(previous, RosRobot):
            previous.perception.stop()
            previous.shutdown()
        agent.forget()  # a new room: earlier requests no longer apply
        live.update(scene=name, status="idle", instruction="", steps=[], todo="", problems=[], said=[], chime="")

    drive(live, agent, holder, new_scene)


if __name__ == "__main__":
    main()
