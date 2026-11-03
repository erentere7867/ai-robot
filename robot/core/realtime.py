"""Real-time hygiene for the Linux side: pinned cores, scheduling priority, no paging, no surprise GC pauses.

The hard real-time loops live on the microcontroller; the Linux control thread is soft real-time (50 Hz
supervisory control), and what breaks it in Python is rarely the work itself but everything around it:
a language-model reply being parsed on the same core, a garbage-collection pass over a large heap, a page
fault after the model server pushed the controller's memory out, the GIL held for 5 ms at a time. Each
helper here removes one of those and reports what it could do; none of them is required to run.

On the VENTUNO Q (2x A78C 2.35 GHz, 2x A78C 2.1 GHz, 4x A55) the default plan is: control thread on one
fast A78 core at SCHED_FIFO, the detector on the other fast cores, llama-server on the remaining A78s
(its own process, pinned by the service file), and everything else (dashboard, audio, Python glue) on the
A55s. `plan_cores()` derives this from the CPU frequencies the kernel reports, so it adapts to other boards.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CorePlan:
    control: list[int] = field(default_factory=list)  # the control thread
    perception: list[int] = field(default_factory=list)  # detector worker threads
    inference: list[int] = field(default_factory=list)  # llama-server (set in its service file)
    other: list[int] = field(default_factory=list)  # the rest of the process

    def to_dict(self) -> dict[str, list[int]]:
        return {"control": self.control, "perception": self.perception, "inference": self.inference,
                "other": self.other}


def core_speeds() -> dict[int, int]:
    """CPU id -> maximum frequency in kHz (Linux); empty where the kernel does not say."""
    speeds = {}
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/cpuinfo_max_freq"):
        try:
            speeds[int(path.parent.parent.name[3:])] = int(path.read_text().strip())
        except (OSError, ValueError):
            continue
    return speeds


def plan_cores(speeds: dict[int, int] | None = None) -> CorePlan:
    """Fastest core for control, next fastest for perception and inference, slow cores for the rest."""
    speeds = core_speeds() if speeds is None else speeds
    if not speeds:
        count = os.cpu_count() or 1
        speeds = {i: 1 for i in range(count)}
    ordered = sorted(speeds, key=lambda c: (-speeds[c], c))
    if len(ordered) < 4:  # small machine: share everything, keep control on its own core if possible
        return CorePlan(control=ordered[:1], perception=ordered, inference=ordered, other=ordered)
    # Big cores: clearly faster than the slowest (A78 vs A55); all of them on a uniform machine.
    big = [c for c in ordered if speeds[c] > 1.05 * min(speeds.values())] or ordered
    little = [c for c in ordered if c not in big]
    control = big[:1]
    rest = big[1:] or little
    perception = rest[:1]
    inference = rest[1:] or rest
    other = little or [c for c in ordered if c not in control]
    return CorePlan(control=control, perception=perception, inference=inference, other=other)


def pin_current_thread(cores: list[int]) -> bool:
    """Pin the calling thread (Linux: sched_setaffinity on pid 0 applies to the thread)."""
    if not cores or not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, set(cores))
        return True
    except OSError as exc:
        logger.info("Could not pin thread to %s: %s", cores, exc)
        return False


def raise_current_thread(priority: int) -> str:
    """SCHED_FIFO at `priority` if allowed (CAP_SYS_NICE or an rtprio limit), else the best nice value."""
    if priority <= 0:
        return "normal"
    if hasattr(os, "sched_setscheduler"):
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
            return f"SCHED_FIFO {priority}"
        except (OSError, PermissionError):
            pass
    if hasattr(os, "setpriority"):
        for nice in (-15, -10, -5):
            try:
                os.setpriority(os.PRIO_PROCESS, 0, nice)
                return f"nice {nice}"
            except (OSError, PermissionError):
                continue
    return "normal (no permission to raise it)"


def lock_memory() -> bool:
    """mlockall(MCL_CURRENT | MCL_FUTURE): no page faults in the control path (needs a memlock limit)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        return libc.mlockall(1 | 2) == 0
    except OSError:
        return False


def quiet_garbage_collector() -> None:
    """Freeze everything allocated at start-up out of the collector's view and make full passes rare.

    The harness allocates its long-lived state once (maps, models, schemas); scanning it on every
    generation-2 pass is what produced the multi-millisecond pauses. Young objects are still collected."""
    gc.collect()
    gc.freeze()
    gc.set_threshold(50_000, 50, 1000)


def fine_timer() -> bool:
    """Windows only: 1 ms system timer (default 15.6 ms). Thread wake-ups, including a thread waiting for
    the GIL, otherwise land on 15.6 ms ticks; Linux timers are already fine-grained."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        return ctypes.windll.winmm.timeBeginPeriod(1) == 0
    except (OSError, AttributeError):
        return False


def tune_process(*, lock: bool = True) -> dict[str, object]:
    """Process-wide settings, once, after start-up. Returns what was applied."""
    sys.setswitchinterval(0.001)  # hand the GIL over every 1 ms (default 5 ms): the control thread waits less
    quiet_garbage_collector()
    locked = lock_memory() if lock else False
    return {"switch_interval_ms": 1.0, "gc": "frozen", "memory_locked": locked, "fine_timer": fine_timer()}
