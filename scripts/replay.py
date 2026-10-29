"""Deterministically replay a recorded JSONL session through the RobotHarness."""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.cognition.backend import NullBackend
from robot.core.bus import EventBus
from robot.core.clock import ReplayClock
from robot.core.config import RobotConfig
from robot.core.recorder import EventRecorder, EventReplayer
from robot.main import RobotHarness


async def replay_session(
    recording_path: Path | str, *, db_path: Path | str | None = None, mode: str = "recorded"
) -> list[dict[str, object]]:
    """Recorded mode is passive and reproduces actual outputs without rerunning AI.

    Simulate mode re-runs user inputs with the offline backend in an isolated DB.
    Only the latest 1000 responses are retained; events are streamed from disk.
    """
    responses = deque(maxlen=1000)
    clock = ReplayClock()
    if mode == "recorded":
        bus = EventBus(clock)
        bus.subscribe("robot.response", lambda event: responses.append(dict(event.payload)))
        await bus.start()
        try:
            await EventReplayer(bus, clock).replay(EventRecorder.iter_events(recording_path), collect=False)
        finally:
            await bus.stop()
    elif mode == "simulate":
        with tempfile.TemporaryDirectory(prefix="robot-replay-") as temporary:
            cfg = RobotConfig.from_yaml()
            normal_db = cfg.database_path.resolve()
            cfg.database_path = (
                Path(db_path).expanduser().resolve() if db_path else Path(temporary) / "memory.db"
            )
            if cfg.database_path.resolve() == normal_db:
                raise ValueError("Replay refuses to write to the robot's normal memory database")
            cfg.record_events = False
            async with RobotHarness(cfg, clock=clock, backend=NullBackend()) as harness:
                harness.bus.subscribe("robot.response", lambda event: responses.append(dict(event.payload)))
                await EventReplayer(harness.bus, clock).replay(
                    EventRecorder.iter_events(recording_path),
                    include_types={"input.user_message", "perception.observation"},
                    collect=False,
                )
    else:
        raise ValueError("Replay mode must be recorded or simulate")
    return list(responses)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a recorded JSONL session")
    parser.add_argument("recording", help="Path to recorded .jsonl session file")
    parser.add_argument("--db", default=None, help="Optional temporary SQLite DB path")
    parser.add_argument("--mode", choices=["recorded", "simulate"], default="recorded")
    args = parser.parse_args()

    responses = asyncio.run(replay_session(args.recording, db_path=args.db, mode=args.mode))
    print(f"Replayed {len(responses)} turn(s):")
    for idx, r in enumerate(responses, 1):
        print(f"  {idx}. [{r.get('emotion')}] {r.get('say')} | actions={r.get('actions')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
