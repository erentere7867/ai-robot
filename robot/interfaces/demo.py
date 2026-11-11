"""Try the robot: a simulated room in real time behind the live page, with a box to type instructions into.

    python -m robot.interfaces.demo                  # needs llama-server on port 8080
    python -m robot.interfaces.demo --case O3        # a hand-written suite program, no model needed

Then open http://localhost:8765. Type an instruction and press Enter; Stop halts the robot at once; the room picker
swaps the room ("random room" builds a new one each time). A new instruction interrupts the one running, and "stop",
"wait" and "come back" are acted on at once without the model, as on the robot.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from typing import Any

from robot.interfaces.live import Live
from robot.sim.world import SimRobot

SCENES = ("living room", "two rooms", "office", "random room")


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
    """The room now on screen; swapped by the room picker."""

    def __init__(self, sim: SimRobot) -> None:
        self.sim = sim


def _refresh(live: Live, holder: Holder) -> None:
    """Keep the map moving between robot calls: one call (a go_to) can last many seconds."""
    while True:
        try:
            live.show_world(holder.sim)
        except RuntimeError:  # the world changed while being copied; take the next frame
            pass
        time.sleep(0.2)


def run_case(live: Live, case_id: str) -> None:
    from robot.agent.gate import check
    from robot.agent.readback import readback
    from robot.agent.runner import Runner
    from robot.eval.programs import PROGRAMS
    from robot.eval.suite import CASES

    case = next(c for c in CASES if c.id == case_id)
    sim = SimRobot(case.scene(), realtime=True)
    threading.Thread(target=_refresh, args=(live, Holder(sim)), daemon=True).start()
    program = PROGRAMS[case_id]
    gate = check(program, sim)
    live.update(status="running", instruction=case.say, program=program, plan="hand-written program",
                readback=readback(gate.trace, gate.ended_by) if gate.ok else str(gate))
    result = Runner(sim, on_call=lambda name, line: live.update(line=line)).run(program)
    live.update(status="done" if result.ok else "failed", line=0, problems=[result.error] if result.error else [])
    print(f"{result.status}; passes its check: {bool(case.check(sim))}. Ctrl+C to quit.")
    while True:
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="run this suite case's hand-written program instead of the planner")
    parser.add_argument("--scene", choices=SCENES, default="living room")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    live = Live()
    live.serve(args.port)
    print(f"Open http://localhost:{args.port}", flush=True)
    if args.case:
        run_case(live, args.case)
        return

    from robot.agent.agent import Agent
    from robot.agent.planner import LocalPlanner

    holder = Holder(SimRobot(make_scene(args.scene), realtime=True))
    live.update(scene=args.scene, scenes=list(SCENES))
    threading.Thread(target=_refresh, args=(live, holder), daemon=True).start()
    agent = Agent(LocalPlanner(), live=live)
    if not agent.planner.brain.healthy():
        print("llama-server is not answering on http://127.0.0.1:8080; start it first", flush=True)

    def new_scene(name: str) -> None:
        holder.sim = SimRobot(make_scene(name), realtime=True)
        live.update(scene=name, status="idle", instruction="", program="", plan="", readback="", problems=[],
                    say_first="", line=0)

    drive(live, agent, holder, new_scene)


def drive(live: Live, agent: Any, holder: Holder, new_scene: Any = None) -> None:
    """Act on what the page sends, for ever: an instruction runs in its own thread and pre-empts the one running;
    "stop", "wait" and "come back" act at once without the model; while a program waits for the user's answer
    (`robot.waiting`), what is typed is that answer."""
    from robot.agent.instant import COME_BACK, instant
    from robot.agent.runner import Runner

    worker: threading.Thread | None = None

    def busy() -> bool:
        return worker is not None and worker.is_alive()

    def interrupt() -> None:
        if busy():
            holder.sim.cancel()
            worker.join(5.0)

    def do(text: str) -> None:
        record = agent.do(text, holder.sim)
        history = live.state.history + [{"instruction": text, "outcome": record.outcome}]
        live.update(history=history[-12:])
        print(f"{record.outcome:<9} {text}", flush=True)

    def come_back() -> None:
        holder.sim.begin()
        live.update(status="running", instruction="come back", program=COME_BACK, plan="instant command",
                    readback="", say_first="")
        result = Runner(holder.sim, on_call=lambda name, line: live.update(line=line)).run(COME_BACK)
        live.update(status="done" if result.ok else "stopped", line=0)

    while True:
        kind, text = live.inbox.get()
        if kind == "instruct" and text and getattr(holder.sim, "waiting", False) and instant(text) is None:
            holder.sim.answers.put(text)
        elif kind == "scene" and new_scene is not None and text in SCENES:
            interrupt()
            new_scene(text)
        elif kind == "stop" or (kind == "instruct" and instant(text) in ("stop", "wait")):
            interrupt()
            live.update(status="stopped", line=0)
        elif kind == "instruct" and text:
            interrupt()  # a new instruction pre-empts the running one
            target = come_back if instant(text) == "come back" else (lambda t=text: do(t))
            worker = threading.Thread(target=target, daemon=True)
            worker.start()


if __name__ == "__main__":
    main()
