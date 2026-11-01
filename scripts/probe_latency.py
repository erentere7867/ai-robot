"""Where does inference time go? Replays decision and detection requests against a running llama-server.

    python scripts/probe_latency.py              # decisions and detections interleaved, as in a real task
    python scripts/probe_latency.py --pin        # same, with decisions and perception pinned to their own slots

Prints the server's own timings (prompt tokens processed vs. reused from cache, encode and decode time).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.body.hal import BodyPose  # noqa: E402
from robot.cognition.prompts import PromptBuilder  # noqa: E402
from robot.core.clock import RealClock  # noqa: E402
from robot.core.config import RobotConfig  # noqa: E402
from robot.face.animator import EYE_COLOR_THEMES  # noqa: E402
from robot.perception.camera import encode_png  # noqa: E402
from robot.perception.detector import DETECT_ALL_PROMPT  # noqa: E402
from robot.skills import build_default_registry  # noqa: E402
from robot.world.sandbox import SimulationSandbox  # noqa: E402
from robot.world.spatial_map import SpatialMap  # noqa: E402


async def post(client: httpx.AsyncClient, payload: dict, label: str) -> None:
    t0 = time.perf_counter()
    resp = await client.post("/v1/chat/completions", json=payload)
    resp.raise_for_status()
    t = resp.json().get("timings", {})
    print(f"{label:<22} wall {1000 * (time.perf_counter() - t0):7.0f} ms | prompt {t.get('prompt_n', '?'):>5} new "
          f"/ {t.get('cache_n', '?'):>5} cached, {t.get('prompt_ms', 0):6.0f} ms | decode {t.get('predicted_n', '?'):>3} "
          f"tok {t.get('predicted_ms', 0):5.0f} ms")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pin", action="store_true", help="decisions on slot 0, perception on slot 1")
    parser.add_argument("--size", type=int, nargs=2, default=None, metavar=("W", "H"),
                        help="resize the camera image before detection")
    args = parser.parse_args()
    cfg = RobotConfig.from_yaml(Path("config/robot.yaml"))
    prompts = PromptBuilder(cfg.soul_path, cfg.context_budget, cfg.name)
    tools = build_default_registry(sorted(EYE_COLOR_THEMES)).tool_schemas()
    sandbox = SimulationSandbox()

    def image(heading: float) -> str:
        # A new view every time, as when the robot turns: nothing about the image can come from cache.
        rgb = sandbox.render_head_rgb(BodyPose(heading_deg=heading))
        if args.size:
            h, w, _ = rgb.shape
            rgb = rgb[[int(y * h / args.size[1]) for y in range(args.size[1])]][
                :, [int(x * w / args.size[0]) for x in range(args.size[0])]]
        return "data:image/png;base64," + base64.b64encode(encode_png(rgb)).decode("ascii")
    world = SpatialMap(RealClock()).render_for_prompt(BodyPose())

    def decision(text: str) -> dict:
        situation = prompts.situation(world=world, core_facts=[], memories=[], recent=[])
        payload = {"messages": prompts.build(text, situation, background=False), "tools": tools,
                   "temperature": 0, "max_tokens": 64, "cache_prompt": True,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if args.pin:
            payload["id_slot"] = 0
        return payload

    def detect(heading: float) -> dict:
        payload = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image(heading)}},
            {"type": "text", "text": DETECT_ALL_PROMPT}]}],
            "temperature": 0, "max_tokens": 200, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}
        if args.pin:
            payload["id_slot"] = 1
        return payload
    async with httpx.AsyncClient(base_url=cfg.llm.base_url, timeout=120, trust_env=False) as client:
        for i, text in enumerate(["hi!", "what do you see?", "turn around", "go to the red ball"]):
            await post(client, decision(text), f"decision {i}: {text[:10]}")
            await post(client, detect(60.0 * i + 7), f"detection {i}")


if __name__ == "__main__":
    asyncio.run(main())
