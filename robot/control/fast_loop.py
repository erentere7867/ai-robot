"""The fast layer: one loop, 20-50 times a second, that keeps the world model current and steers.

Each tick:
  1. perception — take the newest detection (it runs in a worker thread at whatever rate the hardware
     manages) and fuse it into the spatial map from the pose at which its frame was captured;
  2. control — tick the active behaviour and send its speed command to the body.

Keeping both in one loop (instead of separate tasks) makes timing deterministic under the test clock.
The language model never steers: it starts behaviours and hears about their outcome.

On the robot the loop runs in a thread of its own (`threaded=True`) with its own event loop, pinned to a
fast core at real-time priority and woken on absolute deadlines: a model reply being parsed, the dashboard
or speech synthesis can then no longer delay a steering command. Every command passes the collision
monitor (robot/control/safety.py) on its way to the motors. Timing is measured continuously (`timing()`):
how late each tick started, how long it took, how many deadlines were missed.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import deque
from typing import Any

from robot.control.behaviors import Behavior, Done, Drive, Tick
from robot.core.events import EventPriority

logger = logging.getLogger(__name__)

STREAM_WEIGHT_WINDOW_S = 0.5  # frames closer together than this share their evidence (see integrate_view)


class TickStats:
    """Lateness (start vs deadline) and compute time of recent ticks, in milliseconds."""

    def __init__(self, size: int = 1000) -> None:
        self.late: deque[float] = deque(maxlen=size)
        self.work: deque[float] = deque(maxlen=size)
        self.overruns = 0
        self.ticks = 0

    def add(self, late_ms: float, work_ms: float) -> None:
        self.late.append(late_ms)
        self.work.append(work_ms)
        self.ticks += 1

    @staticmethod
    def _pct(values: deque[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)

    def summary(self) -> dict[str, float]:
        return {"ticks": self.ticks, "overruns": self.overruns,
                "late_p50_ms": self._pct(self.late, 0.5), "late_p99_ms": self._pct(self.late, 0.99),
                "late_max_ms": round(max(self.late, default=0.0), 2),
                "work_p50_ms": self._pct(self.work, 0.5), "work_p99_ms": self._pct(self.work, 0.99)}


class FastLoop:
    name = "fast_loop"

    def __init__(self, *, bus: Any, clock: Any, body: Any, camera: Any, spatial_map: Any, planner: Any,
                 detector: Any = None, rate_hz: float = 20.0, continuous: bool = True, lidar: Any = None,
                 monitor: Any = None, threaded: bool = False, control_cores: list[int] | None = None,
                 perception_cores: list[int] | None = None, priority: int = 0) -> None:
        self.bus = bus
        self.clock = clock
        self.body = body
        self.camera = camera
        self.world = spatial_map
        self.planner = planner
        self.detector = detector
        self.lidar = lidar  # robot.perception.lidar source, or None
        self.monitor = monitor  # robot.control.safety.CollisionMonitor, or None
        self.limit: Any = None  # the monitor's last verdict (for the dashboard)
        self._scan_stamp = -math.inf
        self.threaded = threaded and continuous
        self.control_cores = control_cores or []
        self.perception_cores = perception_cores or []
        self._perception_thread: threading.Thread | None = None
        self.priority = priority
        self.realtime: str = "in the main event loop"
        self.stats = TickStats()
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping = threading.Event()
        self._last_start: float | None = None
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
        if self.threaded:
            if self._thread is None:
                ready = threading.Event()
                self._stopping.clear()
                self._thread = threading.Thread(target=self._thread_main, args=(ready,), name="control",
                                                daemon=True)
                self._thread.start()
                ready.wait(2.0)
                # Camera, detector and lidar fusion get a thread of their own: a frame render or a detection
                # (10-40 ms) must never sit inside a control tick.
                self._perception_thread = threading.Thread(target=self._perception_main, name="perception",
                                                           daemon=True)
                self._perception_thread.start()
        elif self._task is None:
            self._task = asyncio.create_task(self._run(), name="fast_loop")

    def _thread_main(self, ready: threading.Event) -> None:
        """The control thread: its own event loop, pinned and prioritised where the OS allows it."""
        from robot.core.realtime import pin_current_thread, raise_current_thread

        pinned = pin_current_thread(self.control_cores)
        sched = raise_current_thread(self.priority)
        self.realtime = f"own thread, {sched}" + (f", cores {self.control_cores}" if pinned else "")
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._thread_id = threading.get_ident()
        self._wake = asyncio.Event()
        ready.set()
        try:
            self._loop.run_until_complete(self._run())
        except asyncio.CancelledError:
            pass
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    def _in_control(self, fn: Any, *args: Any) -> None:
        """Run `fn` on the control loop: now when already there (or unthreaded), else at its next turn."""
        if self._loop is None or threading.get_ident() == self._thread_id:
            fn(*args)
        else:
            self._loop.call_soon_threadsafe(fn, *args)

    @staticmethod
    def _resolve(future: asyncio.Future, done: Done) -> None:
        """Complete a caller's future on the caller's own loop."""
        def settle() -> None:
            if not future.done():
                future.set_result(done)
        loop = future.get_loop()
        try:
            same = asyncio.get_running_loop() is loop
        except RuntimeError:
            same = False
        if same:
            settle()
        elif not loop.is_closed():
            loop.call_soon_threadsafe(settle)

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
        if self._thread is not None:
            self._stopping.set()
            await asyncio.to_thread(self._thread.join, 2.0)
            if self._perception_thread is not None:
                await asyncio.to_thread(self._perception_thread.join, 2.0)
            self._thread, self._loop, self._thread_id, self._perception_thread = None, None, None, None
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._pending is not None:
            self._pending[1].cancel()
            self._pending = None

    async def health(self) -> Any:
        from robot.core.service import ServiceHealth

        running = (self._thread is not None and self._thread.is_alive()) if self.threaded else             (self._task is not None and not self._task.done())
        return ServiceHealth(name=self.name, healthy=running, status="ok" if running else "stopped",
                             details=self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        return {"behavior": self.behavior.describe() if self.behavior else None,
                "detect_hz": round(self.detect_hz, 1), "tick_ms": round(self.tick_ms, 2),
                "detector": type(self.detector).__name__ if self.detector else None,
                "control_hz": round(1.0 / self.dt, 1), "realtime": self.realtime, "timing": self.stats.summary(),
                "safety": None if self.limit is None else {"reason": self.limit.reason,
                                                          "clearance_mm": round(min(self.limit.clearance_mm, 9e3))}}

    def _emit(self, event_type: str, source: str, payload: dict[str, Any], *,
              priority: EventPriority = EventPriority.NORMAL) -> None:
        """Publish without making the control tick (or the perception thread) wait on subscribers."""
        if hasattr(self.bus, "emit_nowait"):
            self.bus.emit_nowait(event_type, source, payload, priority=priority)
        else:
            self.bus.track_task(asyncio.ensure_future(self.bus.emit(event_type, source, payload, priority=priority)))

    # -- behaviours ----------------------------------------------------------------------------------
    def start_behavior(self, behavior: Behavior) -> asyncio.Future:
        """Run a behaviour, replacing the current one. The future (on the caller's loop) resolves to its Done."""
        future = asyncio.get_running_loop().create_future()
        self._in_control(self._start, behavior, future)
        return future

    def _start(self, behavior: Behavior, future: asyncio.Future) -> None:
        self._cancel(None, f"switched to {behavior.describe()}")
        self.behavior = behavior
        self._future = future
        self._wake.set()
        self._emit("behavior.started", self.name, {"behavior": behavior.name,
                                                             "description": behavior.describe(),
                                                             "persistent": behavior.persistent},
                             priority=EventPriority.NORMAL)

    async def run(self, behavior: Behavior) -> Done:
        """Start a behaviour and wait for it; cancelling the wait stops it."""
        future = self.start_behavior(behavior)
        try:
            return await asyncio.shield(future)
        finally:
            if not future.done():
                self.cancel_behavior(behavior, reason="cancelled")

    def cancel_behavior(self, behavior: Behavior | None = None, *, reason: str = "stopped") -> bool:
        running = self.behavior is not None and (behavior is None or behavior is self.behavior)
        self._in_control(self._cancel, behavior, reason)
        return running

    def _cancel(self, behavior: Behavior | None, reason: str) -> None:
        if self.behavior is None or (behavior is not None and behavior is not self.behavior):
            return
        self._finish(Done(False, f"Stopped {self.behavior.describe()}: {reason}."))

    def _finish(self, done: Done) -> None:
        behavior, future = self.behavior, self._future
        self.behavior, self._future = None, None
        if future is not None:
            self._resolve(future, done)
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
        if self.threaded and hasattr(self.detector, "detect_frame"):
            return await asyncio.to_thread(self._look_now)  # off every event loop
        return await self._look()

    def _look_now(self) -> Any:
        frame = self.camera.capture()
        detections = self.detector.detect_frame(frame)
        self._last_fuse = self.clock.monotonic()
        return self._fuse(frame, detections, 1.0)

    async def _look(self) -> Any:
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
        self._absorb(frame, detections, now)

    def _absorb(self, frame: Any, detections: list[Any], now: float) -> None:
        """Fuse one streamed detection result, weighted by how long since the previous one."""
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

    def _perception_main(self) -> None:
        """The perception thread (threaded mode): newest frame -> detector -> map, and lidar scans -> map."""
        from robot.core.realtime import pin_current_thread

        pin_current_thread(self.perception_cores)
        stream = self.detector is not None and self.streaming and hasattr(self.detector, "detect_frame")
        while not self._stopping.is_set():
            try:
                self._scan_step()
                if stream:
                    frame = self.camera.capture()
                    self._absorb(frame, self.detector.detect_frame(frame), self.clock.monotonic())
                else:
                    time.sleep(0.02)
            except Exception:
                logger.warning("Perception step failed", exc_info=True)
                time.sleep(0.05)

    def _scan_step(self) -> None:
        if self.lidar is None:
            return
        scan = self.lidar.latest()
        if scan is None or scan.stamp == self._scan_stamp:
            return
        self._scan_stamp = scan.stamp
        self.world.update_scan(scan.pose, scan.angles_deg, scan.ranges_mm)

    # -- loop ------------------------------------------------------------------------------------
    def _body_busy(self) -> bool:
        return getattr(self.body, "_active_cmd_id", None) is not None  # a trick or head move owns the motors

    async def tick(self) -> None:
        now = self.clock.monotonic()
        dt, self._last_tick = max(0.0, now - self._last_tick), now
        if not self.threaded:  # threaded: the perception thread does both
            self._perception_step(now)
            self._scan_step()
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
        v, w = out.v, out.w
        if v > 0 and math.isfinite(tof) and tof < 600.0:
            v = min(v, max(60.0, 1.4 * (tof - 140.0)))  # ease off as the range sensor closes in
        guard = None
        if self.monitor is not None:
            scan = self.lidar.latest() if self.lidar is not None else None
            self.limit = self.monitor.limit(v, w, self.monitor.points(scan, tof))
            v, w, guard = self.limit.v, self.limit.w, self.limit.reason
        reflex = await self.body.set_velocity(v, w)
        self._blocked = reflex or guard

    async def _run(self) -> None:
        if self.threaded:
            await self._run_deadlines()
            return
        while True:
            if not self.continuous and self.behavior is None and self._pending is None:
                self._wake.clear()
                await self._wake.wait()
                self._last_tick = self.clock.monotonic()
            started = self.clock.monotonic()
            late = max(0.0, started - (self._last_start + self.dt)) if self._last_start is not None else 0.0
            self._last_start = started
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Fast loop tick failed")
            spent = self.clock.monotonic() - started
            self.tick_ms += 0.1 * (1000.0 * spent - self.tick_ms)
            self.stats.add(1000.0 * late, 1000.0 * spent)
            if late > self.dt:
                self.stats.overruns += 1
            await self.clock.sleep(max(0.002, self.dt - spent))

    async def _run_deadlines(self) -> None:
        """Ticks on absolute deadlines (no drift); a tick that starts a whole period late is an overrun and
        the schedule restarts from now rather than bursting to catch up."""
        deadline = time.perf_counter()
        while not self._stopping.is_set():
            start = time.perf_counter()
            late = start - deadline
            try:
                await self.tick()
            except Exception:
                logger.exception("Fast loop tick failed")
            end = time.perf_counter()
            self.tick_ms += 0.1 * (1000.0 * (end - start) - self.tick_ms)
            self.stats.add(1000.0 * max(0.0, late), 1000.0 * (end - start))
            deadline += self.dt
            if end > deadline:
                self.stats.overruns += 1
                deadline = end
            time.sleep(max(0.0, deadline - time.perf_counter()))  # precise; the loop has nothing else to do
            await asyncio.sleep(0)  # run callbacks that arrived meanwhile (detections, start/cancel)
        if self.behavior is not None:
            self._finish(Done(False, "Stopped: shutting down."))
            await self.body.set_velocity(0.0, 0.0)
