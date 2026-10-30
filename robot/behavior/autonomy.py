"""Background decisions when no human owns the robot: drives, scene changes and requested wakes.

Each tick is cheap (numbers and a 32x24 thumbnail). The model is only woken for a reason, and
it decides what to do about it; the controller never chooses a physical behaviour itself.
"""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any

from robot.behavior.drives import DriveSystem
from robot.core.events import Event, EventPriority
from robot.core.service import ServiceHealth
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

    def __init__(self, *, bus, arbiter, body, config, attention, audio, camera=None, spatial_map=None,
                 drives: DriveSystem | None = None):
        self.bus = bus
        self.arbiter = arbiter
        self.body = body
        self.config = config
        self.attention = attention
        self.audio = audio
        self.camera = camera
        self.spatial_map = spatial_map
        self.drives = drives or DriveSystem(rate_scale=getattr(config, "drive_rate_scale", 1.0))
        self.enabled = config.enabled
        self.paused = False
        self.quiet = False
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
        self._subs.append(self.bus.subscribe("input.user_message", self._on_input, immediate=True))
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
        return {"enabled": self.enabled, "paused": self.paused, "quiet": self.quiet,
                "active": bool(self._task and not self._task.done()), "reason": self.reason,
                "last_trigger": self.last_trigger,
                "next_decision_in_s": round(self._wake_at - now, 1) if self._wake_at is not None else None,
                **self.drives.snapshot()}

    def allow_speech(self) -> bool:
        return not self.quiet

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
            self.quiet = bool(quiet)
        if self.paused or not self.enabled:
            await self.bus.emit("action.cancel", self.name, {"reason": "autonomy_pause"},
                                priority=EventPriority.CRITICAL, correlation_id=correlation_id)
        elif enabled is True or paused is False:
            self.last_user_at = self.bus.clock.monotonic() - self.config.idle_after_s
            self.next_decision_at = 0.0
        self.reason = "paused by user" if self.paused else "running" if self.enabled else "disabled"
        await self.bus.emit("autonomy.updated", self.name, self.snapshot(), correlation_id=correlation_id)

    async def _on_input(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        attend, cleaned = self.attention.should_attend(text, direct_address=bool(event.payload.get("direct", False)))
        if not attend:
            return
        self.last_user_at = self.bus.clock.monotonic()
        self._wake_at = None
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
            return
        event.payload["handled"] = True
        await self.set_mode(**mode, correlation_id=event.correlation_id or event.id)
        await self.bus.emit("robot.response", self.name, {
            "say": message, "sound": "none", "emotion": {"name": "neutral", "intensity": 0.4},
            "actions": [], "backend": "autonomy_control", "status": "completed", "final": True,
        }, priority=EventPriority.HIGH, correlation_id=event.correlation_id or event.id)

    def _on_cancel(self, event: Event) -> None:
        reason = event.payload.get("reason")
        if reason in ("user_stop_command", "autonomy_pause"):
            self.paused = True
            self._wake_at = None
            self.reason = "paused by user"
        elif reason in ("user_barge_in", "manual_control"):
            self.last_user_at = self.bus.clock.monotonic()

    def _scene_changed(self) -> bool:
        """Compare camera thumbnails while the robot holds still (a moving camera always 'changes')."""
        if self.camera is None:
            return False
        pose = self.body.pose()
        key = (round(pose.x_mm), round(pose.y_mm), round(pose.heading_deg), round(pose.head_tilt_deg))
        try:
            thumb = self.camera.thumbnail()
        except Exception:
            return False
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
        unexplored = None
        if self.spatial_map is not None:
            pose = self.body.pose()
            unexplored = self.spatial_map.unseen_fraction_near(pose.x_mm, pose.y_mm, 1500.0)
        self.drives.step(dt, battery_pct=getattr(sensors, "battery_pct", None),
                         charging=getattr(sensors, "charging", False), unexplored_fraction=unexplored)
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
        if self.audio.is_speaking or (self.audio._utterance_task and not self.audio._utterance_task.done()):
            return
        if sensors.estop_active or sensors.cliff_triggered or sensors.picked_up:
            self.reason = "waiting for safety sensors to clear"
            return
        trigger, text = None, ""
        changed = self._scene_changed()
        urgent = self.drives.urgent()
        if scheduled:
            trigger, text = "scheduled_wake", "You asked to think again now."
        elif now >= self.next_decision_at and urgent is not None and self._wake_at is None:
            # A requested wake (the wait tool) defers drive wakes until then; outside events still count.
            trigger, text = f"drive:{urgent[0]}", urgent[1]
        elif now >= self.next_decision_at and changed:
            trigger, text = "scene_changed", "Something in front of you just changed or moved."
        if trigger is None:
            self.reason = "content; no reason to think"
            return
        self.last_trigger = trigger
        self._wake_at = None
        self.next_decision_at = now + self.config.think_interval_s
        self.reason = f"thinking ({trigger})"
        self._task = asyncio.create_task(self._run(text), name="autonomy:decision")
        self.bus.track_task(self._task)

    async def _run(self, text: str) -> None:
        try:
            result = await self.arbiter.handle_user_turn(text, autonomous=True)
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
