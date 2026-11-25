"""Single owner for stamped velocity requests and the checked wheel output."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from threading import RLock
from typing import Any

NAV = "/cmd_vel_nav"
SELECTED = "/sprocket/cmd_vel_selected"
CHECKED = "/sprocket/cmd_vel_checked"
WHEELS = "/cmd_vel"

DONE, CANCELED, ABORTED = 4, 5, 6  # action_msgs/GoalStatus
# Measured in ROS time, like the Nav2 controller and the simulated base.
CONTROL_PERIOD = 0.025
COMMAND_AGE_NS = 250_000_000


class MotionGate:
    def __init__(self, now: Callable[[], int], select: Callable[[float, float, int], None],
                 output: Callable[[float, float, int], None], max_v: float, max_w: float) -> None:
        self.lock = RLock()
        self._now, self._select, self._output = now, select, output
        self._max_v, self._max_w = max_v, max_w
        self.owner: str | None = None
        self._clock = 0
        self._clock_fault = False
        self._selected = self._checked = 0
        self._source_stamp = 0
        self._in_flight: deque[int] = deque(maxlen=32)
        self._moving = False

    def _time(self) -> int:
        now = self._now()
        if now < self._clock:
            self.stop()
            self._clock_fault = True  # stamps from before the reset cannot identify current collision replies
        self._clock = now
        return now

    def arm(self, owner: str) -> None:
        with self.lock:
            now = self._time()
            if not self._clock_fault and owner != self.owner:
                self.stop()
                self.owner = owner
                self._source_stamp = now

    def stop(self) -> None:
        with self.lock:
            self.owner = None
            self._moving = False
            self._in_flight.clear()
            self._output(0.0, 0.0, self._now())

    def submit(self, owner: str, v: float, w: float, stamp: int | None = None) -> None:
        with self.lock:
            now = self._time()
            stamp = now if stamp is None else stamp
            if self.owner != owner or stamp <= self._selected:
                return
            if (owner == NAV and stamp <= self._source_stamp) or not 0 <= now - stamp <= COMMAND_AGE_NS:
                return
            if not (math.isfinite(v) and math.isfinite(w)):
                self.stop()
                return
            self._selected = stamp
            self._in_flight.append(stamp)
            self._select(max(-0.25, min(self._max_v, v)), max(-self._max_w, min(self._max_w, w)), stamp)

    def checked(self, v: float, w: float, stamp: int) -> None:
        with self.lock:
            now = self._time()
            # Collision Monitor preserves the selected stamp. A response in flight at stop/handoff
            # belongs to the old owner even if it arrives after we arm a new one.
            if (self.owner is None or stamp not in self._in_flight or stamp <= self._checked
                    or not 0 <= now - stamp <= COMMAND_AGE_NS):
                return
            if not (math.isfinite(v) and math.isfinite(w)):
                self.stop()
                return
            while self._in_flight and self._in_flight[0] <= stamp:
                self._in_flight.popleft()
            self._checked = stamp
            self._moving = bool(v or w)
            self._output(v, w, stamp)

    def expire(self) -> None:
        with self.lock:
            now = self._time()
            if self._moving and now - self._checked > COMMAND_AGE_NS:
                self._moving = False
                self._in_flight.clear()
                self._selected = now
                self._output(0.0, 0.0, now)


class NavGoal:
    """Keep an interrupted goal quarantined until Nav2 confirms it cannot still drive."""

    def __init__(self, sent: Any, lock: Any) -> None:
        self.sent = sent
        self._lock = lock
        self._result: Any = None
        self.cancelled = False
        self.released = False

    def result_future(self) -> Any:
        with self._lock:
            if self._result is None and self.sent.done():
                handle = self.sent.result()
                if handle.accepted:
                    self._result = handle.get_result_async()
            return self._result

    def terminal(self) -> bool:
        if not self.sent.done():
            return False
        try:
            result = self.result_future()
            return result is None or (result.done() and result.result().status in (DONE, CANCELED, ABORTED))
        except Exception:  # noqa: BLE001 - a broken action transport does not confirm the wheels are stopped
            return False

    def cancel(self) -> None:
        with self._lock:
            if not self.cancelled:
                self.cancelled = True
                self.sent.add_done_callback(self._cancel_when_accepted)

    def wait_terminal(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while not self.terminal():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)
        return True

    def _cancel_when_accepted(self, future: Any) -> None:
        try:
            handle = future.result()
            if handle.accepted and not self.terminal():
                handle.cancel_goal_async()
        except Exception:  # noqa: BLE001 - keep the wheel gate closed and the goal quarantined
            return
