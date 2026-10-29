"""Exercise the real local model against SimBody and record actual tool execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from robot.core.clock import ReplayClock
from robot.core.config import RobotConfig
from robot.main import RobotHarness


async def check(prompts: list[str], output: Path, max_steps: int) -> None:
    with tempfile.TemporaryDirectory(prefix="robot-control-check-") as temporary:
        config = RobotConfig.from_yaml()
        config.database_path = Path(temporary) / "memory.db"
        config.record_events = False
        config.llm.backend = "llamacpp"
        config.autonomy.enabled = False
        config.agent.max_steps = max_steps
        harness = RobotHarness(config, clock=ReplayClock(), tts_mode="warble", realtime_audio_sync=False)
        harness.face_bridge._send_udp = lambda _: None
        checks, decisions, results = [], [], []
        generate = harness.backend.generate

        async def record(*args, **kwargs):
            result = await generate(*args, **kwargs)
            decisions.append({"response": json.loads(result.raw_text),
                              "native": result.metadata.get("assistant_message"),
                              "prompt_tokens": result.prompt_tokens})
            print(result.raw_text, flush=True)
            return result

        harness.backend.generate = record
        async with harness:
            harness.bus.subscribe("action.completed", lambda e: results.append(dict(e.payload)), immediate=True)
            for prompt in prompts:
                decision_start, result_start = len(decisions), len(results)
                before = harness.body.pose().to_dict()
                await harness.submit_message(prompt)
                await harness.arbiter.wait_until_done()
                await harness.bus.drain()
                checks.append({"prompt": prompt, "before": before, "after": harness.body.pose().to_dict(),
                               "decisions": decisions[decision_start:], "executed": results[result_start:],
                               "final": dict(harness.arbiter.active_goal)})
                print(f"{prompt}: {harness.arbiter.active_goal['status']}; pose={harness.body.pose()}", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"architecture": "native_function_calls", "clock": "replay",
                                      "checks": checks}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", action="append", dest="prompts")
    parser.add_argument("--output", type=Path, default=Path("artifacts/agentic-verification/native-control-check.json"))
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()
    asyncio.run(check(args.prompts or ["turn around", "check around the room"], args.output, args.max_steps))
