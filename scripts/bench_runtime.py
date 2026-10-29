"""Measure the Python harness alone: no model, speaker, microphone or face window."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.cognition.backend import NullBackend
from robot.core.clock import ReplayClock
from robot.core.config import RobotConfig
from robot.main import RobotHarness


async def benchmark(turns: int = 300, *, config: str = "config/robot.yaml", dashboard: bool = False) -> dict:
    import psutil

    process = psutil.Process()
    with tempfile.TemporaryDirectory(prefix="robot-benchmark-") as temporary:
        cfg = RobotConfig.from_yaml(config)
        cfg.database_path = Path(temporary) / "memory.db"
        cfg.record_events = False
        harness = RobotHarness(
            cfg,
            clock=ReplayClock(),
            backend=NullBackend(),
            enable_dashboard=dashboard,
            dashboard_port=0,
            enable_speaker=False,
            tts_mode="warble",
            realtime_audio_sync=False,
        )
        harness.face_bridge._send_udp = lambda payload: None
        async with harness:
            for _ in range(20):
                await harness.submit_message("Hello!")
                await harness.bus.drain()
            tracemalloc.start()
            before = tracemalloc.get_traced_memory()[0]
            rss_before = process.memory_info().rss
            peak_rss = rss_before
            started, cpu_started = time.perf_counter(), time.process_time()
            for i in range(turns):
                await harness.submit_message("Hello!")
                await harness.bus.drain()
                if dashboard:
                    harness.dashboard.build_state_payload()
                # Exercise a long stream of distinct perception IDs without unbounded retention.
                harness.world.record_ego_observation(str(i), "test observation", ttl_s=5)
                if i % 25 == 0:
                    peak_rss = max(peak_rss, process.memory_info().rss)
            elapsed = time.perf_counter() - started
            cpu = time.process_time() - cpu_started
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rss_after = process.memory_info().rss
            peak_rss = max(peak_rss, rss_after)
            result = {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "scope": "Python harness only; deterministic offline backend; simulated body",
                "turns": turns,
                "dashboard": dashboard,
                "elapsed_s": round(elapsed, 3),
                "cpu_s": round(cpu, 3),
                "mean_turn_ms": round(elapsed * 1000 / turns, 3),
                "rss_before_mib": round(rss_before / 2**20, 2),
                "rss_after_mib": round(rss_after / 2**20, 2),
                "sampled_peak_rss_mib": round(peak_rss / 2**20, 2),
                "python_retained_growth_kib": round((current - before) / 1024, 1),
                "python_peak_kib": round(peak / 1024, 1),
                "retained_observations": len(harness.world._observations),
                "retained_history_messages": len(harness.context_builder._recent_turns),
                "pending_requests": len(harness._requests),
                "active_commands": len(harness.executor._active_tasks),
                "retained_episodes": harness.memory._conn.execute("SELECT count(*) FROM episodes").fetchone()[
                    0
                ],
                "bus": harness.bus.stats(),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=300)
    parser.add_argument("--config", default="config/robot.yaml")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.turns <= 100000:
        parser.error("--turns must be in 1..100000")
    result = asyncio.run(benchmark(args.turns, config=args.config, dashboard=args.dashboard))
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
