"""Benchmark LFM3-Omni-2B-Q4_K_M through the RobotHarness on laptop GPU."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.core.config import RobotConfig
from robot.main import RobotHarness


def query_gpu_vram_mb() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3.0,
        ).strip()
        return float(out.splitlines()[0].strip())
    except Exception:
        return None


async def run_benchmark() -> dict[str, object]:
    cfg = RobotConfig.from_yaml()
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.database_path = Path(tmpdir) / "bench_memory.db"
        cfg.record_events = False
        cfg.llm.backend = "llamacpp"

        harness = RobotHarness(cfg)
        await harness.start()
        try:
            prompts = [
                "Hi Sprocket! Do a happy dance for me!",
                "Remember my favorite color is green.",
                "What is my favorite color?",
                "Tilt your head up 200 degrees!",
            ]
            results: list[dict[str, object]] = []
            for p in prompts:
                t0 = time.perf_counter()
                resp = await harness.arbiter.handle_user_turn(p)
                wall_ms = (time.perf_counter() - t0) * 1000.0
                results.append(
                    {
                        "prompt": p,
                        "say": resp.say,
                        "emotion": resp.emotion.name,
                        "sound": resp.sound,
                        "actions": [{"tool": a.tool, "args": a.args} for a in resp.actions],
                        "memory": [{"op": m.op, "text": m.text} for m in resp.memory],
                        "llm_latency_ms": round(resp.latency_ms, 1),
                        "total_turn_ms": round(wall_ms, 1),
                        "backend": resp.backend_name,
                        "fallback_used": resp.fallback_used,
                        "warnings": resp.warnings,
                    }
                )
            vram_mb = query_gpu_vram_mb()
            summary = {
                "model": cfg.llm.model_name,
                "gpu_vram_used_mb": vram_mb,
                "turns": results,
            }
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return summary
        finally:
            await harness.stop()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
