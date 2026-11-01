"""How fast is the robot on this machine? Run it on the Raspberry Pi (or the laptop, to compare).

    python scripts/bench_pi.py                 # detector + fast-loop tick (+ decisions if llama-server runs)
    python scripts/bench_pi.py --threads 4     # NCNN threads (a Pi 5 has 4 cores)

Reports per-frame detector time for each runtime that is installed (NCNN without PyTorch, PyTorch CPU,
PyTorch CUDA), the cost of one fast-loop tick, and the language model's time per decision.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from robot.body.hal import BodyPose  # noqa: E402

MODELS = Path.home() / "robot-runtime" / "models"


def frames(n: int) -> list[np.ndarray]:
    """Simulator frames from varied poses (realistic content for the detector)."""
    try:
        from robot.world.sandbox import SimulationSandbox

        sb = SimulationSandbox()
        return [sb.render_head_rgb(BodyPose(heading_deg=30.0 * k)) for k in range(n)]
    except Exception:
        rng = np.random.default_rng(0)
        return [rng.integers(0, 255, (384, 512, 3), dtype=np.uint8) for _ in range(n)]


def time_detector(name: str, detector, images: list[np.ndarray]) -> None:
    for img in images[:3]:
        detector.detect_rgb(img)  # warm up
    times = []
    for img in images * 3:
        t0 = time.perf_counter()
        detector.detect_rgb(img)
        times.append(1000.0 * (time.perf_counter() - t0))
    med = statistics.median(times)
    print(f"  {name:<28} {med:6.1f} ms/frame  ({1000.0 / med:5.1f} fps)")


async def time_decisions(n: int) -> None:
    from robot.cognition.llamacpp import LlamaCppBackend
    from robot.cognition.prompts import PromptBuilder
    from robot.core.config import RobotConfig
    from robot.face.animator import EYE_COLOR_THEMES
    from robot.skills import build_default_registry

    cfg = RobotConfig.from_yaml()
    backend = LlamaCppBackend(cfg.llm)
    if not await backend.is_available():
        print("  (llama-server is not running; skipped)")
        return
    prompts = PromptBuilder(cfg.soul_path, cfg.context_budget, cfg.name)
    tools = build_default_registry(sorted(EYE_COLOR_THEMES)).tool_schemas()
    times = []
    for k, text in enumerate(["hi!", "turn around", "go to the red ball", "what do you see?"] * n):
        situation = prompts.situation(world="Known objects: ball_1 (red ball): 0.8 m ahead.", core_facts=[],
                                      memories=[], recent=[])
        result = await backend.chat(prompts.build(text, situation, background=False), tools=tools)
        if k:  # the first call fills the prompt cache
            times.append(result.latency_ms)
    await backend.close()
    print(f"  decision (warm cache)        {statistics.median(times):6.0f} ms median over {len(times)}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--decisions", type=int, default=2)
    args = parser.parse_args()
    images = frames(6)
    print("Detector (YOLO11n, 320 px):")
    ncnn_dir = MODELS / "fast_detector_ncnn_model"
    if ncnn_dir.exists():
        from robot.perception.fast_detector import NcnnDetector

        time_detector(f"NCNN, {args.threads} threads", NcnnDetector(ncnn_dir, threads=args.threads), images)
    pt = MODELS / "fast_detector.pt"
    if pt.exists():
        try:
            from robot.perception.fast_detector import FastDetector

            time_detector("PyTorch CPU", FastDetector(pt, device="cpu"), images)
            import torch

            if torch.cuda.is_available():
                time_detector("PyTorch CUDA", FastDetector(pt, device="cuda:0"), images)
        except ImportError:
            pass
    print("Fast loop:")
    from robot.core.clock import RealClock
    from robot.perception.detector import Detection
    from robot.perception.geometry import CameraModel
    from robot.world.spatial_map import SpatialMap

    world, model = SpatialMap(RealClock()), CameraModel()
    dets = [Detection("red ball", (0.4, 0.6, 0.5, 0.75)), Detection("blue cube", (0.6, 0.55, 0.75, 0.8))]
    t0 = time.perf_counter()
    for _ in range(200):
        world.integrate_view(BodyPose(), model, dets, weight=0.1)
    print(f"  map fusion per frame         {1000.0 * (time.perf_counter() - t0) / 200:6.2f} ms")
    print("Language model:")
    await time_decisions(args.decisions)


if __name__ == "__main__":
    asyncio.run(main())
