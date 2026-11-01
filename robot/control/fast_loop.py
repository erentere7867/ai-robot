"""The fast layer: one loop, about 20 times a second, that keeps the world model current and steers.

Each tick:
  1. perception — take the newest detection (it runs in a worker thread at whatever rate the hardware
     manages) and fuse it into the spatial map from the pose at which its frame was captured;
  2. control — tick the active behaviour and send its speed command to the body.

Keeping both in one loop (instead of separate tasks) makes timing deterministic under the test clock
and avoids CPU contention on a Raspberry Pi. The language model never steers: it starts behaviours and
hears about their outcome.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from robot.control.behaviors import Behavior, Done, Drive, Tick
from robot.core.events import EventPriority

logger = logging.getLogger(__name__)

STREAM_WEIGHT_WINDOW_S = 0.5  # frames closer together than this share their evidence (see integrate_view)


class FastLoop:
    name = "fast_loop"

    def __init__(self, *, bus: Any, clock: Any, body: Any, camera: Any, spatial_map: Any, planner: Any,
                 detector: Any = None, rate_hz: float = 20.0, continuous: bool = True) -> None:
        self.bus = bus
        self.clock = clock
        self.body = body
        self.camera = camera
        self.world = spatial_map
        self.planner = planner
        self.detector = detector
        self.dt = 1.0 / rate_hz
        self.continuous = continuous  # perceive even with no behaviour running (off in unit tests)
        # True when the detector keeps up with the camera (YOLO); False for a slow detector (the VLM), which
        # is then only used for single looks, never every frame.
        self.streaming = detector is not None and not hasattr(detector, "backend")
        self._sub_id: str | None = None
        self.behavior: Behavior | None = None
        self._future: asyncio.Future | None = None
        self._blocked: str | None = None
        self._pending: tuple[Any, asyncio.Future] | None = None
        self._last_tick = clock.monotonic()
        self._last_fuse = -math.inf
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.detect_hz = 0.0
        self.tick_ms = 0.0
        self.last_detections: list[Any] = []
        self.last_frame: Any = None
        self._last_emit = -math.inf

    # -- lifecycle ---------------------------------------------------------------------------------
    async def start(self) -> None:
        if self._sub_id is None:
            self._sub_id = self.bus.subscribe("action.cancel", self._on_cancel, name="fast_loop:cancel",
                                              immediate=True)
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="fast_loop")

    def _on_cancel(self, event: Any) -> None:
        # A new request, or the human starting to speak, does not end following; stop/estop/cancel does.
        reason = event.payload.get("reason")
        if self.behavior is not None and self.behavior.persistent and reason in (
                "new_turn", "autonomous_start", "user_barge_in"):
            return
        if self.behavior is not None and reason not in ("new_turn", "autonomous_start"):
            self.cancel_behavior(reason=str(reason or "cancelled").replace("_", " "))

    async def stop(self) -> None:
        if self._sub_id is not None:
            self.bus.unsubscribe(self._sub_id)
            self._sub_id = None
        self.cancel_behavior(reason="shutting down")
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._pending is not None:
            self._pending[1].cancel()
            self._pending = None

    async def health(self) -> Any:
        from robot.core.service import ServiceHealth

        return ServiceHealth(name=self.name, healthy=self._task is not None and not self._task.done(),
                             status="ok", details=self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        return {"behavior": self.behavior.describe() if self.behavior else None,
                "detect_hz": round(self.detect_hz, 1), "tick_ms": round(self.tick_ms, 2),
                "detector": type(self.detector).__name__ if self.detector else None}

    def _emit(self, event_type: str, source: str, payload: dict[str, Any], *,
              priority: EventPriority = EventPriority.NORMAL) -> None:
        """Publish without making the control tick wait on subscribers."""
        self.bus.track_task(asyncio.ensure_future(self.bus.emit(event_type, source, payload, priority=priority)))

    # -- behaviours ----------------------------------------------------------------------------------
    def start_behavior(self, behavior: Behavior) -> asyncio.Future:
        """Run a behaviour, replacing the current one. The future resolves to its Done."""
        self.cancel_behavior(reason=f"switched to {behavior.describe()}")
        self.behavior = behavior
        self._future = asyncio.get_running_loop().create_future()
        self._wake.set()
        self._emit("behavior.started", self.name, {"behavior": behavior.name,
                                                             "description": behavior.describe(),
                                                             "persistent": behavior.persistent},
                             priority=EventPriority.NORMAL)
        return self._future

    async def run(self, behavior: Behavior) -> Done:
        """Start a behaviour and wait for it; cancelling the wait stops it."""
        future = self.start_behavior(behavior)
        try:
            return await asyncio.shield(future)
        finally:
            if not future.done():
                self.cancel_behavior(behavior, reason="cancelled")

    def cancel_behavior(self, behavior: Behavior | None = None, *, reason: str = "stopped") -> bool:
        if self.behavior is None or (behavior is not None and behavior is not self.behavior):
            return False
        self._finish(Done(False, f"Stopped {self.behavior.describe()}: {reason}."))
        return True

    def _finish(self, done: Done) -> None:
        behavior, future = self.behavior, self._future
        self.behavior, self._future = None, None
        if future is not None and not future.done():
            future.set_result(done)
        if behavior is not None:
            self._emit("behavior.finished", self.name, {
                "behavior": behavior.name, "description": behavior.describe(), "ok": done.ok,
                "message": done.message, "persistent": behavior.persistent}, priority=EventPriority.HIGH)

    # -- perception ------------------------------------------------------------------------------
    def _launch(self, frame: Any) -> asyncio.Future:
        if hasattr(self.detector, "detect_frame"):
            return asyncio.ensure_future(asyncio.to_thread(self.detector.detect_frame, frame))
        return asyncio.ensure_future(self.detector.detect(frame, None))

    def _fuse(self, frame: Any, detections: list[Any], weight: float) -> Any:
        # A trained detector's fixed classes outrank labels remembered from an older session or detector.
        update = self.world.integrate_view(frame.pose, frame.model, detections, weight=weight,
                                           relabel=hasattr(self.detector, "vocabulary"))
        self.last_detections, self.last_frame = detections, frame
        return update

    async def perceive_now(self) -> Any:
        """One full-weight look right now (for "what do you see?"); waits for the detector."""
        if self.detector is None:
            return None
        frame = self.camera.capture()
        detections = await self._launch(frame)
        self._last_fuse = self.clock.monotonic()
        return self._fuse(frame, detections, 1.0)

    def _perception_step(self, now: float) -> None:
        if self.detector is None or not self.streaming:
            return
        if self._pending is None:
            frame = self.camera.capture()
            self._pending = (frame, self._launch(frame))
            return
        frame, task = self._pending
        if not task.done():
            return
        self._pending = None
        try:
            detections = task.result()
        except Exception:
            logger.warning("Detection failed", exc_info=True)
            return
        interval = now - self._last_fuse
        self._last_fuse = now
        if math.isfinite(interval) and interval > 0:
            self.detect_hz += 0.2 * (1.0 / interval - self.detect_hz)
        update = self._fuse(frame, detections, min(1.0, interval / STREAM_WEIGHT_WINDOW_S))
        if update.seen and any(new for _, _, new in update.seen) or now - self._last_emit > 0.5:
            self._last_emit = now
            self._emit("perception.view", self.name, {
                "detections": [{"label": d.label, "bbox": [round(v, 3) for v in d.bbox],
                                "score": round(getattr(d, "score", 1.0), 2)} for d in detections],
                "seen": [o.id for o, _, _ in update.seen], "new": [o.id for o, _, new in update.seen if new],
                "lost": [o.id for o in update.lost], "query": None, "stream": True},
                priority=EventPriority.TELEMETRY)

    # -- loop ------------------------------------------------------------------------------------
    def _body_busy(self) -> bool:
        return getattr(self.body, "_active_cmd_id", None) is not None  # a trick or head move owns the motors

    async def tick(self) -> None:
        now = self.clock.monotonic()
        dt, self._last_tick = max(0.0, now - self._last_tick), now
        self._perception_step(now)
        behavior = self.behavior
        if behavior is None or self._body_busy():
            if behavior is None and not self._body_busy():
                await self.body.update_motion()
            return
        pose = self.body.pose()
        tof = self.body.sensors().front_tof_mm
        try:
            out = behavior.tick(Tick(now, dt, pose, tof, self._blocked, self.world, self.planner))
        except Exception as exc:
            logger.exception("Behaviour %s failed", behavior.name)
            out = Done(False, f"Something went wrong while {behavior.describe()}: {exc}")
        if isinstance(out, Done):
            self._blocked = await self.body.set_velocity(0.0, 0.0)
            if self.behavior is behavior:
                self._finish(out)
            return
        assert isinstance(out, Drive)
        v = out.v
        if v > 0 and math.isfinite(tof) and tof < 600.0:
            v = min(v, max(60.0, 1.4 * (tof - 140.0)))  # ease off as the range sensor closes in
        self._blocked = await self.body.set_velocity(v, out.w)

    async def _run(self) -> None:
        while True:
            if not self.continuous and self.behavior is None and self._pending is None:
                self._wake.clear()
                await self._wake.wait()
                self._last_tick = self.clock.monotonic()
            started = self.clock.monotonic()
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Fast loop tick failed")
            spent = self.clock.monotonic() - started
            self.tick_ms += 0.1 * (1000.0 * spent - self.tick_ms)
            await self.clock.sleep(max(0.002, self.dt - spent))
