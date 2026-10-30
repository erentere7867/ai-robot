"""Plain terminal console interface showing replies, expressions, actions, and memory ops."""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import TYPE_CHECKING, Any

from robot.core.bus import EventBus
from robot.core.events import Event, EventPriority

if TYPE_CHECKING:
    from robot.main import RobotHarness


async def _read_line(prompt: str) -> str:
    """Read stdin on a daemon thread; an executor thread blocked in input() stalls interpreter exit."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def deliver(line: str | None, error: Exception | None) -> None:
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(line or "")

    def reader() -> None:
        try:
            line, error = input(prompt), None
        except Exception as exc:
            line, error = None, exc
        try:
            loop.call_soon_threadsafe(deliver, line, error)
        except RuntimeError:
            pass  # The loop closed while we were blocked on the terminal.

    threading.Thread(target=reader, name="console-input", daemon=True).start()
    return await future


class ConsoleInterface:
    """Interactive and one-shot text console for Milestone H1."""

    def __init__(self, harness: RobotHarness, *, verbose_events: bool = True) -> None:
        self.harness = harness
        self.bus: EventBus = harness.bus
        self.verbose_events = verbose_events
        self._sub_ids: list[str] = []
        self._response_ready = asyncio.Event()
        self.last_response: dict[str, Any] | None = None

    def bind(self) -> None:
        self._sub_ids.append(
            self.bus.subscribe("robot.response", self._on_robot_response, name="console:resp")
        )
        if self.verbose_events:
            self._sub_ids.append(
                self.bus.subscribe("expression.updated", self._on_expression, name="console:expr")
            )
            self._sub_ids.append(self.bus.subscribe("action.*", self._on_action_event, name="console:act"))
            self._sub_ids.append(
                self.bus.subscribe("memory.updated", self._on_memory_updated, name="console:mem")
            )
            self._sub_ids.append(
                self.bus.subscribe("reflex.triggered", self._on_reflex, name="console:reflex")
            )

    def unbind(self) -> None:
        for sid in self._sub_ids:
            self.bus.unsubscribe(sid)
        self._sub_ids.clear()

    async def send_and_wait(self, text: str, *, timeout_s: float = 30.0) -> dict[str, Any] | None:
        """Publish a user message event and wait for `robot.response`."""
        try:
            self.last_response = await self.harness.submit_message(
                text, source="console", timeout_s=timeout_s
            )
        except TimeoutError:
            print("[console] Timed out waiting for robot response.")
            self.last_response = None
        except (ValueError, RuntimeError) as exc:
            print(f"[console] {exc}")
            self.last_response = None
        await self.bus.drain()
        return self.last_response

    async def run_interactive(self) -> None:
        """Run interactive REPL loop until `/quit` or EOF/Ctrl+C."""
        print("=" * 72)
        print(f" {self.harness.config.name} Harness (Milestone H3 — Voice, Pet SFX, Mouth Sync & Barge-In)")
        if self.harness.dashboard is not None:
            print(f" Web Dashboard: {self.harness.dashboard.url}")
        print(" Commands: /status | /memories | /consolidate | /world | /emotion <name>")
        print("           /theme <color> | /sound <name> | /say <text> | /mute on|off | /bargein")
        print("           /observe <label> [dist_mm] [bearing_deg] | /estop on|off | /quit")
        print("=" * 72)

        while True:
            try:
                line = await _read_line("\nYou > ")
            except (EOFError, KeyboardInterrupt, OSError):
                print("\n[console] Exiting interactive console...")
                break

            cmd = line.strip()
            if not cmd:
                continue
            if cmd.lower() in ("/quit", "/exit", "exit", "quit"):
                break
            try:
                if cmd.startswith("/"):
                    await self._handle_slash_command(cmd)
                else:
                    await self.send_and_wait(cmd)
            except Exception as exc:
                # A typo such as `/emotion happy high` must not shut the whole robot down.
                print(f"[console] Command failed: {exc}")

    async def _handle_slash_command(self, cmd: str) -> None:
        parts = cmd.split()
        head = parts[0].lower()

        if head == "/status":
            snap = self.harness.world.snapshot()
            stats = self.bus.stats()
            print(f"[status] FSM={self.harness.fsm.state} | Pose={snap['pose']} | Sensors={snap['sensors']}")
            print(f"[status] Bus published={stats['published_count']} dropped={stats['dropped_count']}")
            print(f"[status] Autonomy={self.harness.autonomy.snapshot()} | Goal={self.harness.arbiter.active_goal}")
            return

        if head == "/memories":
            mems = self.harness.memory.list_active()
            if not mems:
                print("[memories] (no active memories)")
            for m in mems:
                print(f"  - [{m.id}] ({m.subject}) {m.content} [prov={m.provenance}]")
            return

        if head == "/consolidate":
            rep = await self.harness.consolidator.consolidate()
            await self.bus.drain()
            print(
                f"[consolidate] reviewed={rep.turns_reviewed} added={rep.added} "
                f"updated={rep.updated} noop={rep.noop} blocked={rep.blocked_by_tombstone}"
            )
            return

        if head == "/theme" and len(parts) >= 2:
            await self.bus.emit(
                "face.theme",
                source="console",
                payload={"theme": parts[1].lower()},
                priority=EventPriority.HIGH,
            )
            await self.bus.drain()
            print(f"[face] Eye color theme set to '{self.harness.face_bridge.current_frame.theme}'")
            return

        if head == "/world":
            print(self.harness.world.render_for_prompt())
            return

        if head == "/emotion" and len(parts) >= 2:
            emo = parts[1].lower()
            intensity = float(parts[2]) if len(parts) >= 3 else 0.85
            await self.bus.emit(
                "expression.manual",
                source="console",
                payload={"emotion": emo, "intensity": intensity},
                priority=EventPriority.HIGH,
            )
            await self.bus.drain()
            return

        if head == "/observe" and len(parts) >= 2:
            label = parts[1]
            dist = float(parts[2]) if len(parts) >= 3 else 800.0
            bearing = float(parts[3]) if len(parts) >= 4 else 0.0
            await self.bus.emit(
                "perception.observation",
                source="console_sim",
                payload={"label": label, "distance_mm": dist, "relative_bearing_deg": bearing},
                priority=EventPriority.NORMAL,
            )
            await self.bus.drain()
            print(f"[world] Recorded observation '{label}' at {dist:.0f}mm, bearing {bearing:.0f}°")
            return

        if head == "/estop" and len(parts) >= 2:
            active = parts[1].lower() in ("on", "true", "1")
            self.harness.body.set_estop(active)
            if active:
                await self.bus.emit(
                    "action.cancel", "console", {"reason": "estop"}, priority=EventPriority.CRITICAL
                )
            await self.harness.body._emit_pose()
            print(f"[safety] Emergency stop set to {active}")
            return

        if head == "/sound" and len(parts) >= 2:
            snd = parts[1].lower()
            samples = await self.harness.audio.play_sfx(snd)
            await self.bus.drain()
            print(
                f"[audio] Played SFX '{snd}' ({len(samples)} samples @ {self.harness.audio.sfx.sample_rate} Hz)"
            )
            return

        if head == "/say" and len(parts) >= 2:
            text_to_say = cmd[len(parts[0]) :].strip()
            speech = await self.harness.audio.speak(text_to_say)
            await self.bus.drain()
            print(f"[audio] Speaking ({speech.duration_s:.2f}s via {speech.backend_name}): {speech.text}")
            return

        if head == "/mute" and len(parts) >= 2:
            muted = parts[1].lower() in ("on", "true", "1")
            self.harness.audio.muted = muted
            print(f"[audio] Speaker mute set to {muted}")
            return

        if head == "/bargein":
            await self.bus.emit(
                "audio.barge_in",
                source="console",
                payload={"rms": 0.25},
                priority=EventPriority.CRITICAL,
            )
            await self.bus.drain()
            print("[audio] Triggered barge-in interruption")
            return

        print(f"[console] Unknown command: {cmd}")

    def _on_robot_response(self, event: Event) -> None:
        p = event.payload
        if not p.get("say"):
            return
        self.last_response = dict(p)
        emo = p.get("emotion", {})
        snd = p.get("sound", "none")
        backend = p.get("backend", "unknown")
        lat = p.get("latency_ms", 0.0)
        print(
            f"\n{self.harness.config.name} [{emo.get('name', 'neutral')}:{emo.get('intensity', 0.5)} "
            f"| sound={snd} | {backend} {lat:.0f}ms] > {p.get('say', '')}"
        )
        self._response_ready.set()

    def _on_expression(self, event: Event) -> None:
        p = event.payload
        snd = f" sound={p['sound']}" if p.get("sound") else ""
        reason = f" ({p['reason']})" if p.get("reason") else ""
        print(
            f"  [face] activity={p.get('activity')} emotion={p.get('emotion')}({p.get('intensity')}) "
            f"gaze={p.get('gaze_target')}{snd}{reason}"
        )

    def _on_action_event(self, event: Event) -> None:
        p = event.payload
        if event.type in ("action.completed", "action.failed", "action.rejected", "action.cancelled"):
            msg = p.get("message") or p.get("error") or event.type
            print(f"  [{event.type}] tool={p.get('tool')} -> {msg}")

    def _on_memory_updated(self, event: Event) -> None:
        p = event.payload
        if p.get("op") == "remember":
            rec = p.get("record") or {}
            print(f"  [memory.remember] ({rec.get('subject')}) {rec.get('content')}")
        elif p.get("op") == "forget":
            print(
                f"  [memory.forget] topic='{p.get('topic')}' removed={p.get('forgotten_count')} tombstoned=True"
            )

    def _on_reflex(self, event: Event) -> None:
        p = event.payload
        print(f"  [REFLEX] {p.get('reflex')}: {p.get('reason')}", file=sys.stderr)
