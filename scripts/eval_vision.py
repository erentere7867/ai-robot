"""Vision benchmark: how well does the VLM see? Measures the model + vision projector pair on its own.

Renders random rooms from the simulator (the same randomizer that trains the fast detector: layout,
distractors, lighting, robot pose), so every frame comes with exact per-object boxes. Two tests per frame,
both the calls the robot makes:

* "what is here": the open-vocabulary detection prompt. A known object counts as found when a box
  overlaps it (IoU >= 0.3) with the right noun; its colour is checked too. Boxes on nothing are invented.
* "is there a <thing>?": the query prompt plus its yes/no crop check, once for an object in view and once
  for one that is not.

Run against a llama-server that is already up (see scripts/start_llm.ps1):
    python scripts/eval_vision.py --frames 30
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.body.hal import BodyPose  # noqa: E402
from robot.cognition.llamacpp import LlamaCppBackend  # noqa: E402
from robot.core.config import RobotConfig  # noqa: E402
from robot.eval.scenarios import CameraModel  # noqa: E402
from robot.perception.camera import Frame  # noqa: E402
from robot.perception.datagen import OBJECT_CLASS, SceneRandomizer  # noqa: E402
from robot.perception.detector import VLMDetector, _iou  # noqa: E402
from robot.world.sandbox import SimulationSandbox  # noqa: E402
from robot.world.spatial_map import head_noun  # noqa: E402

NAMES = {"red_ball": "red ball", "blue_cube": "blue cube", "yellow_duck": "yellow duck", "green_mug": "green mug",
         "cardboard_box": "cardboard box", "charging_dock": "charging dock"}
# Nouns a person would accept for each object, and the colour words that are right for it.
NOUNS = {"red_ball": {"ball", "sphere"}, "blue_cube": {"cube", "block", "box"}, "yellow_duck": {"duck", "duckie"},
         "green_mug": {"mug", "cup"}, "cardboard_box": {"box", "carton", "cardboard"},
         "charging_dock": {"dock", "charger", "pad", "station", "base", "platform", "mat"}}
COLOURS = {"red_ball": {"red"}, "blue_cube": {"blue"}, "yellow_duck": {"yellow"}, "green_mug": {"green"},
           "cardboard_box": {"brown", "tan", "beige", "cardboard", "orange"}, "charging_dock": set()}
MIN_PIXELS = 300  # clearly visible; smaller slivers are not counted as missed


def _box(lab: dict, w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = lab["bbox_px"]
    return x1 / w, y1 / h, x2 / w, y2 / h


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", default="config/robot.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every frame")
    args = parser.parse_args()

    cfg = RobotConfig.from_yaml(args.config)
    backend = LlamaCppBackend(cfg.llm)
    if not await backend.is_available():
        print(f"llama-server is not answering at {cfg.llm.base_url}")
        return 2
    sandbox = SimulationSandbox()
    randomizer = SceneRandomizer(sandbox, args.seed)
    camera = CameraModel.from_sandbox(sandbox)
    detector = VLMDetector(backend)
    rng = random.Random(args.seed)

    known = found = colour_ok = boxes = invented = 0
    ious: list[float] = []
    present_ok = present_n = absent_ok = absent_n = 0
    detect_s: list[float] = []
    query_s: list[float] = []
    try:
        for i in range(args.frames):
            pose: BodyPose = randomizer.randomize()
            rgb, labels = sandbox.render_head_labels(pose)
            h, w, _ = rgb.shape
            frame = Frame(None, pose, camera, 0.0, None, rgb)
            truth = {oid: _box(lab, w, h) for oid, lab in labels.items() if lab["pixels"] >= MIN_PIXELS}
            every = {oid: _box(lab, w, h) for oid, lab in labels.items() if lab["pixels"] > 0}

            t0 = time.perf_counter()
            dets = await detector.detect(frame)
            detect_s.append(time.perf_counter() - t0)
            boxes += len(dets)
            invented += sum(1 for d in dets if not any(_iou(d.bbox, b) > 0.1 for b in every.values()))
            line = []
            for oid, box in truth.items():
                if oid not in OBJECT_CLASS:
                    continue
                known += 1
                hit = max(((d, _iou(d.bbox, box)) for d in dets if head_noun(d.label) in NOUNS[oid]
                           or set(d.label.split()) & NOUNS[oid]), key=lambda t: t[1], default=(None, 0.0))
                if hit[0] is not None and hit[1] >= 0.3:
                    found += 1
                    ious.append(hit[1])
                    colour_ok += not COLOURS[oid] or bool(set(hit[0].label.split()) & COLOURS[oid])
                    line.append(f"{oid}=ok")
                else:
                    line.append(f"{oid}=MISS")

            # Is there a <thing>? Once for something in view, once for something not in view at all.
            in_view = [oid for oid in truth if oid in OBJECT_CLASS]
            not_in_view = [oid for oid in OBJECT_CLASS if oid not in every]
            for oid, present in ((rng.choice(in_view), True) if in_view else (None, True),
                                 (rng.choice(not_in_view), False) if not_in_view else (None, False)):
                if oid is None:
                    continue
                t0 = time.perf_counter()
                hits = await detector.detect(frame, NAMES[oid])
                query_s.append(time.perf_counter() - t0)
                if present:
                    present_n += 1
                    ok = any(_iou(d.bbox, truth[oid]) >= 0.3 for d in hits)
                    present_ok += ok
                else:
                    absent_n += 1
                    ok = not hits
                    absent_ok += ok
                line.append(f"is-there {NAMES[oid]}: {'ok' if ok else 'WRONG'}")
            if args.verbose:
                print(f"frame {i:2d}  {', '.join(line)}  | saw {[d.label for d in dets]}")
    finally:
        await backend.close()

    def pct(a: int, b: int) -> str:
        return f"{100 * a / max(1, b):3.0f}% ({a}/{b})"

    print(f"what is here:   found {pct(found, known)}, colour right {pct(colour_ok, found)}, "
          f"box IoU {sum(ious) / max(1, len(ious)):.2f}, invented {pct(invented, boxes)} of boxes, "
          f"{sum(detect_s) / max(1, len(detect_s)):.2f} s")
    print(f"is there a ...: in view {pct(present_ok, present_n)}, not in view {pct(absent_ok, absent_n)}, "
          f"{sum(query_s) / max(1, len(query_s)):.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
