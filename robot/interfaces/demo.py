"""Try Sprocket v2: a simulated room in real time behind the live page, with a box to command it.

    python -m robot.interfaces.demo                              # needs llama-server on port 8080
    python -m robot.interfaces.demo --case O3                  # run a suite case with closed-loop Agent
    python -m robot.interfaces.demo --brain http://127.0.0.1:8081

Then open http://localhost:8765. Type an instruction and press Enter; Stop halts the robot at once; the room picker
swaps the room ("random room" builds a new one each time). A new instruction pre-empts the one running, and "stop",
"wait" and "come back" act at once via the instant reflex layer without waiting for the planner model.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from typing import Any

from robot.agent.agent import Agent
from robot.agent.api import Preempted
from robot.agent.instant import instant
from robot.agent.planner import LocalPlanner
from robot.interfaces.live import Live
from robot.mind.brain import Brain
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

    def __init__(self, sim: Any) -> None:
        self.sim = sim


def _refresh(live: Live, holder: Holder, stop_event: threading.Event | None = None) -> None:
    """Keep the map moving between robot calls: one navigation step can last many seconds."""
    tick = 0
    while stop_event is None or not stop_event.is_set():
        sim = holder.sim
        if sim is not None:
            try:
                live.show_world(sim, include_view=(tick % 5 == 0))
            except (RuntimeError, AttributeError):  # world swapped mid-copy; take the next frame
                pass
        tick += 1
        time.sleep(0.2)


class _ObservingPlanner:
    """Forwards step planning to an underlying planner while streaming each Intent to Live."""

    def __init__(self, inner: Any, live: Live) -> None:
        self._inner = inner
        self._live = live
        self._intents: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._intents.clear()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def plan_step(
        self,
        context: str,
        instruction: str,
        image_url: str | None = None,
        scratchpad: str = "",
        earlier: tuple[str, ...] | list[str] = (),
    ) -> Any:
        self._live.update(status="planning", scratchpad=str(scratchpad))
        intent = self._inner.plan_step(
            context,
            instruction,
            image_url=image_url,
            scratchpad=scratchpad,
            earlier=earlier,
        )
        self._intents.append({
            "say": getattr(intent, "say", ""),
            "chime": getattr(intent, "chime", ""),
            "posture": getattr(intent, "posture", "neutral"),
            "action": getattr(intent, "action", "done"),
            "target": getattr(intent, "target", None),
            "params": dict(getattr(intent, "params", {}) or {}),
            "subgoal": getattr(intent, "subgoal", ""),
        })
        self._live.update(
            status="running",
            subgoal=getattr(intent, "subgoal", "") or getattr(intent, "action", ""),
            action=getattr(intent, "action", ""),
            posture=getattr(intent, "posture", "neutral"),
            chime=getattr(intent, "chime", ""),
            intents=list(self._intents),
        )
        return intent


def _wrap_planner(planner: Any, live: Live) -> Any:
    if isinstance(planner, _ObservingPlanner) or not hasattr(planner, "plan_step"):
        return planner
    return _ObservingPlanner(planner, live)


def _format_scratchpad(agent: Any, sim: Any = None) -> str:
    sp = getattr(getattr(agent, "short_term", None), "scratchpad", None)
    if sp is not None:
        parts = []
        if getattr(sp, "goal", ""):
            parts.append(f"goal: {sp.goal}")
        if getattr(sp, "visited_rooms", None):
            parts.append(f"visited: {', '.join(sorted(sp.visited_rooms))}")
        if getattr(sp, "inspected_ids", None):
            parts.append(f"inspected: {sorted(sp.inspected_ids)}")
        if getattr(sp, "blockers", None):
            parts.append(f"blockers: {'; '.join(sp.blockers[-3:])}")
        if parts:
            return " | ".join(parts)
    return str(getattr(sim, "scratchpad_summary", "") or "")


def _apply_record_to_live(live: Live, record: Any, agent: Any = None, sim: Any = None) -> None:
    intents = list(getattr(record, "intents", []) or [])
    last = intents[-1] if intents else None
    outcome = getattr(record, "outcome", "") or "done"
    status = "stopped" if outcome == "preempted" else outcome
    changes: dict[str, Any] = {
        "status": status,
        "intents": intents,
        "problems": list(getattr(record, "problems", []) or []),
        "scratchpad": _format_scratchpad(agent, sim),
    }
    if last is not None:
        changes["subgoal"] = getattr(last, "subgoal", "") or getattr(last, "action", "")
        changes["action"] = getattr(last, "action", "")
        changes["posture"] = getattr(last, "posture", "neutral")
        changes["chime"] = getattr(last, "chime", "")
    live.update(**changes)
    if sim is not None:
        live.show_world(sim)


def run_case(
    live: Live,
    case_id: str,
    brain_url: str = "http://127.0.0.1:8080",
    *,
    planner: Any = None,
    realtime: bool = True,
    keep_alive: bool = False,
) -> tuple[Any, SimRobot]:
    """Run one suite case closed-loop via Agent on SimRobot and stream Sprocket v2 state to Live."""
    from robot.eval.suite import CASES

    case = next((c for c in CASES if c.id == case_id), None)
    if case is None:
        raise ValueError(f"Unknown suite case: {case_id!r}")

    sim = SimRobot(case.scene(), realtime=realtime)
    holder = Holder(sim)
    live.update(
        scene=case.id,
        scenes=[case.id],
        status="idle",
        instruction=case.say,
        subgoal="",
        action="",
        posture="neutral",
        chime="",
        intents=[],
        scratchpad="",
        problems=[],
    )
    live.show_world(sim)

    if planner is None:
        brain = Brain(brain_url)
        if not brain.healthy():
            msg = f"Brain server is not answering on {brain_url}; start llama-server first."
            print(msg, flush=True)
            live.update(
                status="failed",
                instruction=case.say,
                subgoal="brain unavailable",
                action="none",
                posture="uh_oh",
                chime="uh_oh",
                intents=[],
                scratchpad="",
                problems=[msg],
            )
            return None, sim
        planner = LocalPlanner(brain)

    stop_refresh = threading.Event()
    threading.Thread(
        target=_refresh,
        args=(live, holder, None if keep_alive else stop_refresh),
        daemon=True,
    ).start()
    try:
        agent = Agent(_wrap_planner(planner, live), live=live)
        live.update(status="running", instruction=case.say)
        record = agent.do(case.say, sim)
        _apply_record_to_live(live, record, agent=agent, sim=sim)
        passed = bool(case.check(sim))
        print(f"{record.outcome}; passes its check: {passed}.", flush=True)
        return record, sim
    finally:
        if not keep_alive:
            stop_refresh.set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="run this suite case closed-loop with the planner")
    parser.add_argument("--scene", choices=SCENES, default="living room")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--brain", default="http://127.0.0.1:8080", help="llama-server URL")
    args = parser.parse_args()

    live = Live()
    live.serve(args.port)
    from robot.interfaces.voice import attach

    attach(live)
    print(f"Open http://localhost:{args.port}", flush=True)
    if args.case:
        run_case(live, args.case, brain_url=args.brain, keep_alive=True)
        print("Ctrl+C to quit.", flush=True)
        while True:
            time.sleep(1)

    holder = Holder(SimRobot(make_scene(args.scene), realtime=True))
    live.update(scene=args.scene, scenes=list(SCENES))
    threading.Thread(target=_refresh, args=(live, holder), daemon=True).start()
    planner = LocalPlanner(Brain(args.brain))
    agent = Agent(_wrap_planner(planner, live), live=live)
    if not planner.brain.healthy():
        print(f"llama-server is not answering on {args.brain}; start it first", flush=True)

    def new_scene(name: str) -> None:
        holder.sim = SimRobot(make_scene(name), realtime=True)
        live.update(
            scene=name,
            status="idle",
            instruction="",
            subgoal="",
            action="",
            posture="neutral",
            chime="",
            intents=[],
            scratchpad="",
            problems=[],
            view="",
        )

    drive(live, agent, holder, new_scene)


def drive(
    live: Live,
    agent: Any,
    holder: Holder,
    new_scene: Any = None,
    *,
    stop_event: threading.Event | None = None,
) -> None:
    """Act on commands from Live.inbox: a new instruction pre-empts the running one;
    "stop", "wait", and "come back" act at once via Agent.do / instant reflex without waiting for the model."""
    import queue

    worker: threading.Thread | None = None

    def busy() -> bool:
        return worker is not None and worker.is_alive()

    def interrupt() -> None:
        if busy():
            if holder.sim is not None and hasattr(holder.sim, "cancel"):
                holder.sim.cancel()
            worker.join(5.0)

    def do(text: str) -> None:
        cmd = instant(text)
        is_come_back = cmd == "come back"
        if holder.sim is not None and hasattr(holder.sim, "begin"):
            holder.sim.begin()
        reset_fn = getattr(agent.planner, "reset", None)
        if callable(reset_fn):
            reset_fn()
        live.update(
            status="running",
            instruction=text,
            subgoal="returning to user" if is_come_back else ("waiting" if cmd == "wait" else ""),
            action="pursue" if is_come_back else ("stop" if cmd == "wait" else ""),
            posture="listening" if cmd in ("come back", "wait") else "neutral",
            chime="listen" if cmd in ("come back", "wait") else "",
            intents=[],
            scratchpad="",
            problems=[],
        )
        try:
            record = agent.do(text, holder.sim)
        except Preempted:
            live.update(status="stopped")
            return
        _apply_record_to_live(live, record, agent=agent, sim=holder.sim)
        history = live.state.history + [{"instruction": text, "outcome": record.outcome}]
        live.update(history=history[-12:])
        print(f"{record.outcome:<9} {text}", flush=True)

    while stop_event is None or not stop_event.is_set():
        try:
            kind, text = live.inbox.get(timeout=0.1) if stop_event is not None else live.inbox.get()
        except queue.Empty:
            continue
        if kind == "instruct" and text and getattr(holder.sim, "waiting", False) and instant(text) is None:
            holder.sim.answers.put(text)
        elif kind == "scene" and new_scene is not None and text in SCENES:
            interrupt()
            new_scene(text)
        elif kind == "stop" or (kind == "instruct" and instant(text) == "stop"):
            interrupt()
            if holder.sim is not None and hasattr(holder.sim, "stop"):
                try:
                    holder.sim.stop()
                except Preempted:
                    pass
            live.update(status="stopped", subgoal="stopped", action="stop", posture="neutral", chime="uh_oh")
        elif kind == "instruct" and text:
            interrupt()  # a new instruction pre-empts the running one
            worker = threading.Thread(target=lambda t=text: do(t), daemon=True)
            worker.start()


if __name__ == "__main__":
    main()
