"""Background decisions when no human owns the robot: drives, scene changes and requested wakes.

Each tick is cheap (numbers and a 32x24 thumbnail). A drive picks a behaviour itself (curious: explore;
playful: a gesture; low battery: go charge), like a pet's habits, and the model is only woken when there may
be something worth saying: a scene change, a wake it asked for, or loneliness when speaking is allowed.
"""

from __future__ import annotations

import asyncio
import math
import random
import re
from typing import Any

from robot.core.events import CancelReason, Event, EventPriority
from robot.core.service import ServiceHealth
from robot.interaction.drives import DriveSystem
from robot.interaction.pacing import SpeechPacer
from robot.perception.change import ChangeDetector

_RESUME = re.compile(r"^(?:please )?(?:resume(?: autonomy)?|autonomy on)[.!?]*$", re.I)
_PAUSE = re.compile(
    r"^(?:(?:please|can you)\s+)?(?:pause(?: autonomy| exploring)?|autonomy off|stay (?:still|here|put)|"
    r"don't move|do not move|stop wandering)[.!?]*$", re.I,
)
_QUIET = re.compile(r"^(?:please )?(?:be quiet|quiet mode|stop talking)[.!?]*$", re.I)
_TALK = re.compile(r"^(?:please )?(?:you can talk|quiet mode off|speak freely)[.!?]*$", re.I)


class AutonomyController:
    name = "autonomy"

    def __init__(self, *, bus, arbiter, body, config, audio, camera, spatial_map, drives: DriveSystem,
                 pacer: SpeechPacer):
        self.bus = bus
        self.arbiter = arbiter
        self.body = body
        self.config = config
        self.audio = audio
        self.camera = camera
        self.spatial_map = spatial_map
        self.drives = drives
        self.enabled = config.enabled
        self.paused = False
        self.pacer = pacer
        self._rng = random.Random(0)
        self.running = False
        self.reason = "waiting for idle time"
        self.last_user_at = bus.clock.monotonic()
        self.next_decision_at = 0.0
        self.last_trigger: str | None = None
        self._wake_at: float | None = None
        self._last_sample_at = -math.inf
        self._last_tick = bus.clock.monotonic()
        self._change = ChangeDetector()
        self._view_key: tuple | None = None
        self._task: asyncio.Task | None = None
        self._subs: list[str] = []

    def bind(self) -> None:
        self._subs.append(self.bus.subscribe("action.cancel", self._on_cancel, immediate=True))
        self._subs.append(self.bus.subscribe("robot.response", self._on_response, immediate=True))

    async def start(self) -> None:
        self.running = True
        self.last_user_at = self.bus.clock.monotonic()
        self._last_tick = self.bus.clock.monotonic()

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
        now = self.bus.clock.monotonic()
        return {"enabled": self.enabled, "paused": self.paused, "quiet": self.pacer.quiet,
                "active": bool(self._task and not self._task.done()), "reason": self.reason,
                "last_trigger": self.last_trigger,
                "next_decision_in_s": round(self._wake_at - now, 1) if self._wake_at is not None else None,
                **self.drives.snapshot()}

    def _habit(self, drive: str) -> list[tuple[str, dict[str, Any]]] | None:
        """What the pet does about a drive by itself; None means it is worth asking the model."""
        if drive == "curiosity":
            return [("explore", {})]
        if drive == "play":
            gesture = self._rng.choice(["wiggle", "perk_up", "curious_tilt", "spin", "nod"])
            return [("express", {"emotion": self._rng.choice(["happy", "excited", "curious"]), "gesture": gesture})]
        if drive == "energy":
            return [("go_to", {"target": "charging dock", "relation": "on"})]
        if drive == "social" and not self.pacer.due():
            return [("express", {"emotion": self._rng.choice(["love", "curious", "shy"]),
                                 "gesture": "curious_tilt"})]
        return None

    def _on_response(self, event: Event) -> None:
        if not event.payload.get("final", True) or event.payload.get("backend") == "autonomy_control":
            return
        now = self.bus.clock.monotonic()
        delay = event.payload.get("next_wake_s")
        self._wake_at = now + float(delay) if delay is not None else None
        self._view_key = None  # the robot may have moved: re-baseline the change detector
        if event.payload.get("origin", "user") == "user":
            self.next_decision_at = now + self.config.think_interval_s

    async def set_mode(self, *, enabled=None, paused=None, quiet=None, correlation_id=None) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if paused is not None:
            self.paused = bool(paused)
        if quiet is not None:
            self.pacer.quiet = bool(quiet)
        if self.paused or not self.enabled:
            await self.bus.emit("action.cancel", self.name, {"reason": CancelReason.AUTONOMY_PAUSE},
                                priority=EventPriority.CRITICAL, correlation_id=correlation_id)
        elif enabled is True or paused is False:
            self.last_user_at = self.bus.clock.monotonic() - self.config.idle_after_s
            self.next_decision_at = 0.0
        self.reason = "paused by user" if self.paused else "running" if self.enabled else "disabled"
        await self.bus.emit("autonomy.updated", self.name, self.snapshot(), correlation_id=correlation_id)

    def note_human(self) -> None:
        """Someone just addressed the robot."""
        self.last_user_at = self.bus.clock.monotonic()
        self._wake_at = None
        self.pacer.human_spoke()

    async def handle_mode_command(self, cleaned: str, correlation_id: str) -> bool:
        """A spoken switch for this controller ("pause", "be quiet"); True when that is what it was."""
        mode, message = None, ""
        if _RESUME.fullmatch(cleaned):
            mode, message = {"enabled": True, "paused": False}, "Okay, I'll go back to doing my own thing."
        elif _PAUSE.fullmatch(cleaned):
            mode, message = {"paused": True}, "Okay, I'll stay right here until you say resume."
        elif _QUIET.fullmatch(cleaned):
            mode, message = {"quiet": True}, "Okay, I'll keep quiet on my own."
        elif _TALK.fullmatch(cleaned):
            mode, message = {"quiet": False}, "Yay, I'll chat when I find something interesting."
        if mode is None:
            return False
        await self.set_mode(**mode, correlation_id=correlation_id)
        await self.bus.emit("robot.response", self.name, {
            "say": message, "sound": "none", "emotion": {"name": "neutral", "intensity": 0.4},
            "actions": [], "backend": "autonomy_control", "status": "completed", "final": True,
        }, priority=EventPriority.HIGH, correlation_id=correlation_id)
        return True

    def _on_cancel(self, event: Event) -> None:
        reason = CancelReason(event.payload.get("reason", CancelReason.OTHER))
        if reason.pauses_autonomy:
            self.paused = True
            self._wake_at = None
            self.reason = "paused by user"
        elif reason.is_human_activity:
            self.last_user_at = self.bus.clock.monotonic()

    def _scene_changed(self) -> bool:
        """Compare camera thumbnails while the robot holds still (a moving camera always 'changes')."""
        pose = self.body.pose()
        key = (round(pose.x_mm), round(pose.y_mm), round(pose.heading_deg), round(pose.head_tilt_deg))
        thumb = self.camera.thumbnail()
        if key != self._view_key:
            self._view_key = key
            self._change.reset(thumb)
            return False
        if self._change.is_changed(thumb):
            self._change.reset(thumb)
            return True
        return False

    async def step(self) -> None:
        now = self.bus.clock.monotonic()
        dt, self._last_tick = now - self._last_tick, now
        sensors = self.body.sensors()
        pose = self.body.pose()
        self.drives.step(dt, battery_pct=sensors.battery_pct, charging=sensors.charging,
                         unexplored_fraction=self.spatial_map.grid.unseen_fraction_near(pose.x_mm, pose.y_mm, 1500.0))
        if not self.running or not self.enabled or self.paused:
            return
        scheduled = self._wake_at is not None and now >= self._wake_at
        if not scheduled and now - self._last_sample_at < self.config.observation_interval_s:
            return
        self._last_sample_at = now
        if self.arbiter.busy or (self._task and not self._task.done()):
            return
        if now - self.last_user_at < self.config.idle_after_s or self.arbiter.fsm.in_follow_up_window():
            self.reason = "giving the human time to talk"
            return
        if self.audio.busy:
            return
        if sensors.estop_active or sensors.cliff_triggered or sensors.picked_up:
            self.reason = "waiting for safety sensors to clear"
            return
        trigger, text, plan = None, "", None
        changed = self._scene_changed()
        urgent = self.drives.urgent()
        if scheduled:
            trigger, text = "scheduled_wake", "You asked to think again now."
        elif now >= self.next_decision_at and urgent is not None and self._wake_at is None:
            # A requested wake (the wait tool) defers drive wakes until then; outside events still count.
            trigger, text = f"drive:{urgent[0]}", urgent[1]
            plan = self._habit(urgent[0])
        elif now >= self.next_decision_at and changed:
            trigger, text = "scene_changed", "Something in front of you just changed or moved."
        if trigger is None:
            self.reason = "content; no reason to think"
            return
        self.last_trigger = trigger
        self._wake_at = None
        self.next_decision_at = now + self.config.think_interval_s
        self.reason = f"thinking ({trigger})"
        self._task = asyncio.create_task(self._run(text, plan), name="autonomy:decision")
        self.bus.track_task(self._task)

    async def _run(self, text: str, plan: list[tuple[str, dict[str, Any]]] | None = None) -> None:
        try:
            result = await self.arbiter.handle_user_turn(text, autonomous=True, plan=plan)
            self.reason = "content; no reason to think"
            if result.next_wake_s is not None:
                self._wake_at = self.bus.clock.monotonic() + result.next_wake_s
        except asyncio.CancelledError:
            raise
        except Exception:
            self.reason = "background decision failed; waiting before retry"
        finally:
            self._view_key = None
            self.next_decision_at = self.bus.clock.monotonic() + self.config.think_interval_s
