"""Schedule model-owned idle decisions without prescribing tasks or physical behavior."""

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any

from robot.core.events import Event, EventPriority
from robot.core.service import ServiceHealth

_RESUME = re.compile(r"^(?:please )?(?:resume(?: autonomy)?|autonomy on)[.!?]*$", re.I)
_PAUSE = re.compile(
    r"^(?:(?:please|can you)\s+)?(?:pause(?: autonomy| exploring)?|autonomy off|stay (?:still|here|put)|"
    r"don't move|do not move|stop wandering)[.!?]*$", re.I,
)
_QUIET = re.compile(r"^(?:please )?(?:be quiet|quiet mode|stop talking)[.!?]*$", re.I)
_TALK = re.compile(r"^(?:please )?(?:you can talk|quiet mode off|speak freely)[.!?]*$", re.I)


class AutonomyController:
    """Starts bounded background tasks only when no human interaction owns the robot.

    Each tick is cheap; motor control and inference live in one owned cancellable
    task. The scheduler supplies observations; the model supplies every decision.
    A user stop suspends initiative until explicitly resumed.
    """

    name = "autonomy"

    def __init__(self, *, bus, arbiter, body, config, attention, audio):
        self.bus = bus
        self.arbiter = arbiter
        self.body = body
        self.config = config
        self.attention = attention
        self.audio = audio
        self.enabled = config.enabled
        self.paused = False
        self.quiet = False
        self.running = False
        self.reason = "waiting for idle time"
        self.last_user_at = bus.clock.monotonic()
        self.next_decision_at = 0.0
        self._wake_at: float | None = None
        self._observation: tuple | None = None
        self._last_sample_at = -math.inf
        self._task: asyncio.Task | None = None
        self._subs: list[str] = []

    def bind(self) -> None:
        self._subs.append(self.bus.subscribe("input.user_message", self._on_input, immediate=True))
        self._subs.append(self.bus.subscribe("action.cancel", self._on_cancel, immediate=True))
        self._subs.append(self.bus.subscribe("robot.response", self._on_response, immediate=True))

    async def start(self) -> None:
        self.running = True
        # Boot is a baseline, not a synthetic request to describe the room.
        self.last_user_at = self.bus.clock.monotonic()
        self._observation = self._observe_key()

    def _observe_key(self) -> tuple:
        """Cheap, quantized local perception; no rendered images or inference."""
        sandbox = self.arbiter.world.sandbox
        pose = self.body.pose()
        visible = sandbox.detect_objects(pose) if sandbox is not None else []
        objects = tuple(sorted(
            (d.id, d.label, d.color, round(d.u_norm * 20), round(d.v_norm * 20), round(d.distance_mm / 100))
            for d in visible
        ))
        sensors = self.body.sensors()
        distance = round(sensors.front_tof_mm / 100) if math.isfinite(sensors.front_tof_mm) else "invalid"
        return (objects, distance, sensors.cliff_triggered, sensors.estop_active, sensors.picked_up)

    async def stop(self) -> None:
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        for sid in self._subs:
            self.bus.unsubscribe(sid)
        self._subs.clear()

    async def health(self) -> ServiceHealth:
        return ServiceHealth(self.name, True, self.reason, self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "paused": self.paused, "quiet": self.quiet,
                "active": bool(self._task and not self._task.done()), "reason": self.reason,
                "next_decision_in_s": (max(0, round(self._wake_at - self.bus.clock.monotonic(), 1))
                                       if self._wake_at is not None else None)}

    def allow_speech(self) -> bool:
        return not self.quiet

    def _on_response(self, event: Event) -> None:
        if (event.payload.get("final", True) and event.payload.get("origin", "user") == "user"
                and event.payload.get("backend") != "autonomy_control"):
            self.next_decision_at = self.bus.clock.monotonic() + self.config.think_interval_s
            self._observation = self._observe_key()
            delay = event.payload.get("next_wake_s")
            self._wake_at = self.bus.clock.monotonic() + delay if delay is not None else None

    async def set_mode(self, *, enabled=None, paused=None, quiet=None, correlation_id=None) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if paused is not None:
            self.paused = bool(paused)
        if quiet is not None:
            self.quiet = bool(quiet)
        if self.paused or not self.enabled:
            await self.bus.emit("action.cancel", self.name, {"reason": "autonomy_pause"},
                                priority=EventPriority.CRITICAL, correlation_id=correlation_id)
        elif enabled is True or paused is False:
            self.last_user_at = self.bus.clock.monotonic() - self.config.idle_after_s
            self._wake_at = self.bus.clock.monotonic()  # Explicit user request to resume.
        self.reason = "paused by user" if self.paused else "waiting for a model decision" if self.enabled else "disabled"
        await self.bus.emit("autonomy.updated", self.name, self.snapshot(), correlation_id=correlation_id)

    async def _on_input(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        # Unaddressed microphone chatter must not take control of autonomy.
        attend, cleaned = self.attention.should_attend(text, direct_address=bool(event.payload.get("direct", False)))
        if not attend:
            return
        self.last_user_at = self.bus.clock.monotonic()
        self._wake_at = None
        mode = None
        message = ""
        if _RESUME.fullmatch(cleaned):
            mode, message = {"enabled": True, "paused": False}, "Autonomy is on. I will decide what to do from what I observe."
        elif _PAUSE.fullmatch(cleaned):
            mode, message = {"paused": True}, "I'll stay here until you ask me to move or resume exploring."
        elif _QUIET.fullmatch(cleaned):
            mode, message = {"quiet": True}, "I'll keep my autonomous observations quiet."
        elif _TALK.fullmatch(cleaned):
            mode, message = {"quiet": False}, "I'll share interesting observations as I explore."
        if mode is None:
            return
        event.payload["handled"] = True
        await self.set_mode(**mode, correlation_id=event.correlation_id or event.id)
        await self.bus.emit("robot.response", self.name, {
            "say": message, "sound": "none", "emotion": {"name": "neutral", "intensity": 0.4},
            "actions": [], "memory": [], "backend": "autonomy_control", "status": "completed", "final": True,
        }, priority=EventPriority.HIGH, correlation_id=event.correlation_id or event.id)

    def _on_cancel(self, event: Event) -> None:
        reason = event.payload.get("reason")
        if reason in ("user_stop_command", "autonomy_pause"):
            self.paused = True
            self._wake_at = None
            self.reason = "paused by user"
        elif reason in ("user_barge_in", "manual_control"):
            self.last_user_at = self.bus.clock.monotonic()

    async def step(self) -> None:
        if not self.running or not self.enabled or self.paused:
            return
        now = self.bus.clock.monotonic()
        scheduled = self._wake_at is not None and now >= self._wake_at
        if not scheduled and now - self._last_sample_at < self.config.observation_interval_s:
            return
        self._last_sample_at = now
        active = self.arbiter._active_turn
        if active and not active.done():
            return
        if self.arbiter._input_task and not self.arbiter._input_task.done():
            return
        if self._task and not self._task.done():
            return
        if now - self.last_user_at < self.config.idle_after_s or self.arbiter.fsm.in_follow_up_window():
            self.reason = "giving the user time to interact"
            return
        if self.audio.is_speaking or (self.audio._utterance_task and not self.audio._utterance_task.done()):
            return
        sensors = self.body.sensors()
        if sensors.estop_active or sensors.cliff_triggered or sensors.picked_up:
            self.reason = "waiting for safety sensors to clear"
            return
        observation = self._observe_key()
        if self._observation is None:
            self._observation = observation
            return
        changed = observation != self._observation
        if not scheduled and (not changed or now < self.next_decision_at):
            self.reason = "watching for changes without inference"
            return
        trigger = "observation_changed" if changed else "scheduled_wake"
        self._observation = observation
        self._wake_at = None
        self.next_decision_at = now + self.config.think_interval_s
        self.reason = "model is deciding from the current observation"
        self._task = asyncio.create_task(self._run(trigger), name="autonomy:decision")
        self.bus.track_task(self._task)

    async def _run(self, trigger: str) -> None:
        delay = None
        cancelled = False
        try:
            result = await self.arbiter.handle_user_turn(
                json.dumps({"trigger": trigger, "user_request": None}),
                autonomous=True,
            )
            delay = result.next_wake_s
            cancelled = result.backend_name == "cancelled"
            self.reason = "watching for changes without inference"
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            self.reason = "model decision failed; waiting before retry"
        finally:
            if not cancelled:
                self._observation = self._observe_key()
                now = self.bus.clock.monotonic()
                self.next_decision_at = now + self.config.think_interval_s
                self._wake_at = now + delay if delay is not None else None
