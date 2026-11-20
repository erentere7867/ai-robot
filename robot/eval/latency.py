"""Measure local robot wait paths with real inference and deterministic inputs."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any

import httpx

from robot.agent.agent import Agent
from robot.eval.suite import living_room, office
from robot.interfaces.demo import Holder, show_cameras
from robot.interfaces.live import Live
from robot.mind.brain import Brain, Reply
from robot.mind.memory import CirculatingMemory
from robot.perception.depth import MonoDepth
from robot.perception.detector import Detector
from robot.perception.gpu import VisionServer
from robot.perception.rig import RIG
from robot.sim.world import SimRobot


def measure(samples: dict[str, list[float]], name: str, fn: Callable, *args: Any) -> Any:
    start = time.perf_counter()
    result = fn(*args)
    samples.setdefault(name, []).append(1000 * (time.perf_counter() - start))
    return result


class MeasuredBrain(Brain):
    samples: dict[str, list[float]]
    replies: list[dict[str, Any]]

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_tokens: int = 400,
             temperature: float = 0.0, think: int = 0) -> Reply:
        reply = measure(self.samples, "Model turn (HTTP + inference)", super().chat,
                        messages, tools, max_tokens, temperature, think)
        if not {"prompt_ms", "predicted_ms"} <= (reply.raw.get("timings") or {}).keys():
            raise RuntimeError("llama-server did not report prompt and generation timings")
        self.samples.setdefault("Prompt processing", []).append(reply.read_ms)
        self.samples.setdefault("Token generation", []).append(reply.write_ms)
        self.replies.append({"said": reply.text, "prompt_tokens": reply.read_tokens, "tokens": reply.write_tokens,
                             "calls": [(c.name, c.args) for c in reply.tool_calls]})
        return reply


def agent_samples(args: argparse.Namespace, samples: dict[str, list[float]]) -> list[dict[str, Any]]:
    tasks = [(living_room, "What color is the ball?"), (office, "Use your rear-left camera and tell me what you see.")]
    brain = MeasuredBrain(args.brain_url)
    results = []
    with brain.http:
        if not brain.healthy():
            raise RuntimeError(f"Start llama-server at {args.brain_url} before running this benchmark")
        for i in range(args.warmup + args.agent_samples):
            scene, text = tasks[i % len(tasks)]
            sim = SimRobot(scene())
            # Change the pose a little per request so identical images are not cached across all tasks.
            sim.yaw = i * 0.031
            brain.samples = {} if i < args.warmup else samples
            brain.replies = []
            record = measure(brain.samples, "Agent round trip", Agent(brain).do, text, sim)
            if record.outcome != "done":
                raise RuntimeError(f"Agent task did not finish: {record.problems}")
            if i >= args.warmup:
                results.append({"task": text, "problems": record.problems, "replies": brain.replies})
            print(f"agent {i + 1}/{args.warmup + args.agent_samples}: {record.model_s:.2f}s in the model", flush=True)
    return results


def perception_frame() -> Any:
    import cv2

    # Use one 16:9 camera view from the 3D report's contact sheet.
    photo = Path(__file__).resolve().parents[2] / "docs" / "table_22cm_vs_1m.jpg"
    sheet = cv2.imread(str(photo))
    if sheet is None:
        raise RuntimeError(f"Could not load the benchmark image: {photo}")
    return cv2.resize(sheet[-270:, 480:, ::-1], (1280, 720))


def local_samples(args: argparse.Namespace, samples: dict[str, list[float]]) -> None:
    with ExitStack() as stack:
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        memory = CirculatingMemory(tmp / "memory.sqlite")
        stack.enter_context(closing(memory.db))
        for i in range(1000):
            memory.remember_fact(f"I put the spare key to drawer {i} in the kitchen cabinet.")
        live = Live()
        sim = SimRobot(living_room())
        live.show_world(sim)
        show_cameras(live, Holder(sim))
        server = stack.enter_context(live.serve(port=0))
        stack.callback(server.shutdown)
        http = stack.enter_context(httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", trust_env=False))

        def get(path: str) -> bytes:
            return http.get(path).raise_for_status().content

        paths: dict[str, Callable] = {
            "SQLite fact recall (1,000 rows)": lambda: memory.query_facts("spare key to drawer 551"),
            "Dashboard /state": lambda: get("/state"),
            "Dashboard /camera (one SimRobot frame)": lambda: get("/camera/front_camera.png"),
        }
        if args.vision_url or args.depth_weights:
            rgb = perception_frame()
        if args.vision_url:
            vision = VisionServer(args.vision_url)
            stack.callback(vision.http.close)
            detector = Detector(vision)
            detector.watch("table", "chair")
            paths["YOLOE detection (one frame)"] = lambda: detector(rgb)
        else:
            vision = None
        if args.depth_weights or vision:
            depth = MonoDepth(weights=args.depth_weights, server=vision, threads=4)
            paths["Metric depth (one frame)"] = lambda: depth(rgb, RIG["front_camera"])
        for name, fn in paths.items():
            for _ in range(args.warmup):
                fn()
            for _ in range(args.local_samples):
                measure(samples, name, fn)


def summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {"n": len(ordered), **{f"p{p}": round(ordered[math.ceil(p * len(ordered) / 100) - 1], 3)
                                 for p in (50, 95, 99)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-url", default="http://127.0.0.1:8081")
    parser.add_argument("--vision-url", help="vision server for YOLOE and depth; omit to skip YOLOE")
    parser.add_argument("--depth-weights", help="Depth Anything V2 small ONNX for CPU depth without a vision server")
    parser.add_argument("--agent-samples", type=int, default=12)
    parser.add_argument("--local-samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--only", choices=("agent", "local"))
    parser.add_argument("--json", type=Path, help="save raw measurements and replies for A/B comparison")
    args = parser.parse_args()
    if min(args.agent_samples, args.local_samples) < 1 or args.warmup < 0:
        parser.error("sample counts must be positive, warmup nonnegative")

    samples: dict[str, list[float]] = {}
    tasks = agent_samples(args, samples) if args.only != "local" else []
    if args.only != "agent":
        local_samples(args, samples)
    stages = {name: summary(samples[name]) for name in ("Prompt processing", "Token generation") if name in samples}
    totals = sorted(((name, summary(values)) for name, values in samples.items() if name not in stages),
                    key=lambda row: -row[1]["p95"])
    print(f"{'Local latency per operation (ms)':<44} {'n':>5} {'p50':>10} {'p95':>10} {'p99':>10}")
    for name, row in [*totals, *stages.items()]:
        print(f"{name:<44} {row['n']:5} {row['p50']:10.3f} {row['p95']:10.3f} {row['p99']:10.3f}")
    if args.json:
        args.json.write_text(json.dumps({"samples_ms": samples, "tasks": tasks}, indent=2) + "\n")


if __name__ == "__main__":
    main()
