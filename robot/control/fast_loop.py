"""The fast layer: keeps the world model current and steers, 20-50 times a second.

Perception (newest frames -> detector -> spatial map, lidar scans -> occupancy) and control (tick the active
behaviour, send its speed command through the collision monitor) are one cycle each. On the robot they run in
two threads of their own, the control thread pinned to a fast core at real-time priority and woken on absolute
deadlines, so a model reply being parsed, the dashboard or speech synthesis can never delay a steering command.
Under a test clock both run stepped in one task, which makes timing deterministic.

The language model never steers: it starts behaviours and hears about their outcome.
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
from robot.core.events import CancelReason, EventPriority
from robot.core.realtime import pin_current_thread, raise_current_thread
from robot.core.service import ServiceHealth

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

    def __init__(self, *, bus: Any, clock: Any, body: Any, camera: Any, spatial_map: Any, planner: Any, monitor: Any,
                 detector: Any = None, lidar: Any = None, localizer: Any = None, rate_hz: float = 20.0,
                 threaded: bool = False, control_cores: list[int] | None = None,
                 perception_cores: list[int] | None = None, priority: int = 0) -> None:
        self.bus = bus
        self.clock = clock
        self.body = body
        self.cameras = list(camera.cameras)
        self.world = spatial_map
        self.planner = planner
        self.detector = detector  # a FrameDetector; None: the map runs on the lidar alone
        self.lidar = lidar
        self.monitor = monitor
        self.localizer = localizer
        self.threaded = threaded
        self.dt = 1.0 / rate_hz
        self.control_cores = control_cores or []
        self.perception_cores = perception_cores or []
        self.priority = priority
        self.limit: Any = None  # the monitor's last verdict
        self.realtime = "stepped with the test clock"
        self.stats = TickStats()
        self.detect_hz = 0.0
        self.tick_ms = 0.0
        self.last_detections: list[Any] = []  # the front camera's
        self.last_frame: Any = None
        self.views: dict[str, tuple[Any, list[Any]]] = {}  # newest (frame, detections) per camera
        self.behavior: Behavior | None = None
        self._future: asyncio.Future | None = None
        self._blocked: Any = None
        self._commanding = False  # the body holds a speed from a behaviour
        self._range_mm = math.inf
        self._cycle = 0
        self._scan_stamp = -math.inf
        self._last_fuse: dict[str, float] = {}  # per camera
        self._last_tick = clock.monotonic()
        self._last_start: float | None = None
        self._last_emit = -math.inf
        self._sub_id: str | None = None
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._thread: threading.Thread | None = None
        self._perception_thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping = threading.Event()

    # -- lifecycle ---------------------------------------------------------------------------------
    async def start(self) -> None:
        if self._sub_id is None:
            self._sub_id = self.bus.subscribe("action.cancel", self._on_cancel, name="fast_loop:cancel",
                                              immediate=True)
        if not self.threaded:
            if self._task is None:
                self._task = asyncio.create_task(self._run_stepped(), name="fast_loop")
            return
        if self._thread is None:
            ready = threading.Event()
            self._stopping.clear()
            self._thread = threading.Thread(target=self._control_main, args=(ready,), name="control", daemon=True)
            self._thread.start()
            ready.wait(2.0)
            self._perception_thread = threading.Thread(target=self._perception_main, name="perception", daemon=True)
            self._perception_thread.start()

    def _control_main(self, ready: threading.Event) -> None:
        pinned = pin_current_thread(self.control_cores)
        sched = raise_current_thread(self.priority)
        self.realtime = f"own thread, {sched}" + (f", cores {self.control_cores}" if pinned else "")
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._thread_id = threading.get_ident()
        self._wake = asyncio.Event()
        ready.set()
        try:
            self._loop.run_until_complete(self._run_deadlines())
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    def _in_control(self, fn: Any, *args: Any) -> None:
        """Run `fn` on the control loop: now when already there (or stepped), else at its next turn."""
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

    async def _on_cancel(self, event: Any) -> None:
        reason = CancelReason(event.payload.get("reason", CancelReason.OTHER))
        if reason.ends_behavior(self.behavior is not None and self.behavior.persistent):
            self.cancel_behavior(reason=reason.replace("_", " "))
            await self.body.halt()

    async def stop(self) -> None:
        if self._sub_id is not None:
            self.bus.unsubscribe(self._sub_id)
            self._sub_id = None
        self.cancel_behavior(reason="shutting down")
        if self._thread is not None:
            self._stopping.set()
            await asyncio.to_thread(self._thread.join, 2.0)
            await asyncio.to_thread(self._perception_thread.join, 2.0)
            self._thread, self._loop, self._thread_id, self._perception_thread = None, None, None, None
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def health(self) -> ServiceHealth:
        running = self._thread.is_alive() if self.threaded and self._thread else (
            self._task is not None and not self._task.done())
        return ServiceHealth(name=self.name, healthy=running, status="ok" if running else "stopped",
                             details=self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        return {"behavior": self.behavior.describe() if self.behavior else None,
                "detect_hz": round(self.detect_hz, 1), "tick_ms": round(self.tick_ms, 2),
                "detector": type(self.detector).__name__ if self.detector else None,
                "control_hz": round(1.0 / self.dt, 1), "realtime": self.realtime, "timing": self.stats.summary(),
                "localization": self.localizer.snapshot() if self.localizer is not None else None,
                "safety": None if self.limit is None else {"reason": self.limit.reason,
                                                          "clearance_mm": round(min(self.limit.clearance_mm, 9e3))}}

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
        self.bus.emit_nowait("behavior.started", self.name, {"behavior": behavior.name,
                                                             "description": behavior.describe(),
                                                             "persistent": behavior.persistent})

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
        self._finish(Done(False, f"Stopped {self.behavior.describe()}: {reason}.", cancelled=True))
        self._wake.set()

    def _finish(self, done: Done) -> None:
        behavior, future = self.behavior, self._future
        self.behavior, self._future = None, None
        if future is not None:
            self._resolve(future, done)
        if behavior is not None:
            self.bus.emit_nowait("behavior.finished", self.name, {
                "behavior": behavior.name, "description": behavior.describe(), "ok": done.ok, "cancelled": done.cancelled,
                "message": done.message, "persistent": behavior.persistent}, priority=EventPriority.HIGH)

    # -- perception ------------------------------------------------------------------------------
    def _fuse(self, frame: Any, detections: list[Any], weight: float) -> Any:
        scan = self.lidar.latest() if self.lidar is not None else None
        update = self.world.integrate_view(frame.pose, frame.model, detections, weight=weight, relabel=True, scan=scan,
                                           clipped_tracks=len(self.cameras) == 1)
        self.views[frame.model.name] = (frame, detections)
        if frame.model.name == self.cameras[0].model.name:
            self.last_detections, self.last_frame = detections, frame
        return update

    def _next_cameras(self) -> list[Any]:
        """This cycle's cameras: the front one every cycle, the others one at a time in turn."""
        self._cycle += 1
        rest = self.cameras[1:]
        return [self.cameras[0], rest[self._cycle % len(rest)]] if rest else [self.cameras[0]]

    async def perceive_now(self) -> list[tuple[Any, list[Any], Any]]:
        """One full-weight look through every camera right now: [(frame, detections, update)]."""
        return [] if self.detector is None else await asyncio.to_thread(self._look_now)

    def _look_now(self) -> list[tuple[Any, list[Any], Any]]:
        views = []
        for cam in self.cameras:
            frame = cam.capture()
            detections = self.detector.detect_frame(frame)
            self._last_fuse[frame.model.name] = self.clock.monotonic()
            views.append((frame, detections, self._fuse(frame, detections, 1.0)))
        return views

    def _perceive(self) -> None:
        """One perception cycle: the newest scan and range reading, and this cycle's cameras."""
        self._scan_step()
        self.world.grid.update_range(self.body.pose(), self._range_mm, offset_mm=self.monitor.range_x)
        if self.detector is None:
            return
        for cam in self._next_cameras():
            frame = cam.capture()
            self._absorb(frame, self.detector.detect_frame(frame), self.clock.monotonic())

    def _absorb(self, frame: Any, detections: list[Any], now: float) -> None:
        """Fuse one streamed detection result, weighted by how long since that camera's previous one."""
        name = frame.model.name
        interval = now - self._last_fuse.get(name, -math.inf)
        self._last_fuse[name] = now
        if name == self.cameras[0].model.name and math.isfinite(interval) and interval > 0:
            self.detect_hz += 0.2 * (1.0 / interval - self.detect_hz)
        update = self._fuse(frame, detections, min(1.0, interval / STREAM_WEIGHT_WINDOW_S))
        if any(new for _, _, new in update.seen) or now - self._last_emit > 0.5:
            self._last_emit = now
            self.bus.emit_nowait("perception.view", self.name, {
                "camera": name,
                "detections": [{"label": d.label, "bbox": [round(v, 3) for v in d.bbox], "score": round(d.score, 2)}
                               for d in detections],
                "seen": [o.id for o, _, _ in update.seen], "new": [o.id for o, _, new in update.seen if new],
                "lost": [o.id for o in update.lost], "query": None, "stream": True},
                priority=EventPriority.TELEMETRY)

    def _perception_main(self) -> None:
        pin_current_thread(self.perception_cores)
        while not self._stopping.is_set():
            try:
                self._perceive()
                if self.detector is None:
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
        pose = scan.pose
        if self.localizer is not None:  # match against the map first, then add the scan where it fits
            pose = self.localizer.correct(pose, scan)
        self.world.grid.update_scan(pose, scan)

    # -- control ---------------------------------------------------------------------------------
    async def tick(self) -> None:
        now = self.clock.monotonic()
        dt, self._last_tick = max(0.0, now - self._last_tick), now
        tof = self._range_mm = self.body.sensors().front_tof_mm
        if not self.threaded:
            self._perceive()
        behavior = self.behavior
        if behavior is None:
            if self._commanding:  # the behaviour was cancelled: nothing else will take the last speed back
                self._commanding = False
                await self.body.halt()
            await self.body.update_motion()
            return
        try:
            out = behavior.tick(Tick(now, dt, self.body.pose(), tof, self._blocked, self.world, self.planner))
        except Exception as exc:
            logger.exception("Behaviour %s failed", behavior.name)
            out = Done(False, f"Something went wrong while {behavior.describe()}: {exc}")
        if isinstance(out, Done):
            self._blocked = await self.body.set_velocity(0.0, 0.0)
            self._commanding = False
            if self.behavior is behavior:
                self._finish(out)
            return
        assert isinstance(out, Drive)
        if out.head_deg is not None:
            self.body.set_head(out.head_deg)
        v = out.v
        if v > 0 and math.isfinite(tof) and tof < 600.0:
            v = min(v, max(60.0, 1.4 * (tof - 140.0)))  # ease off as the range sensor closes in
        scan = self.lidar.latest() if self.lidar is not None else None
        self.limit = self.monitor.limit(v, out.w, self.monitor.points(scan, tof))
        self._commanding = True
        reflex = await self.body.set_velocity(self.limit.v, self.limit.w)
        self._blocked = reflex or self.limit.reason

    async def _timed_tick(self, late_s: float) -> float:
        """Run one tick, record its timing; the compute time in seconds."""
        started = time.perf_counter()
        try:
            await self.tick()
        except Exception:
            logger.exception("Fast loop tick failed")
        spent = time.perf_counter() - started
        self.tick_ms += 0.1 * (1000.0 * spent - self.tick_ms)
        self.stats.add(1000.0 * late_s, 1000.0 * spent)
        return spent

    async def _run_stepped(self) -> None:
        while True:
            if self.behavior is None and not self._commanding:
                self._wake.clear()
                await self._wake.wait()
                self._last_tick = self.clock.monotonic()
            started = self.clock.monotonic()
            late = max(0.0, started - (self._last_start + self.dt)) if self._last_start is not None else 0.0
            self._last_start = started
            await self._timed_tick(late)
            await self.clock.sleep(max(0.002, self.dt - (self.clock.monotonic() - started)))

    async def _run_deadlines(self) -> None:
        """Ticks on absolute deadlines (no drift); a tick that starts a whole period late is an overrun and
        the schedule restarts from now rather than bursting to catch up."""
        deadline = time.perf_counter()
        while not self._stopping.is_set():
            await self._timed_tick(max(0.0, time.perf_counter() - deadline))
            deadline += self.dt
            end = time.perf_counter()
            if end > deadline:
                self.stats.overruns += 1
                deadline = end
            time.sleep(max(0.0, deadline - time.perf_counter()))  # noqa: ASYNC251 - precise; own thread, nothing else to run
            await asyncio.sleep(0)  # run callbacks that arrived meanwhile (start/cancel)
        if self.behavior is not None:
            self._finish(Done(False, "Stopped: shutting down.", cancelled=True))
        await self.body.set_velocity(0.0, 0.0)
