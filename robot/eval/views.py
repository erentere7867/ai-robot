"""Measure perception on pictures, not on one run: render views of a home from random spots with the truth of what is
in each, then ask YOLOE or the language model about words and score them against that truth. This is how the YOLOE
and model tables in docs/simulation_3d.md were made.

    python -m robot.eval.views render --scene apt_0 --count 40            # -> runs/views/apt_0.npz (the robot's front
    python -m robot.eval.views yoloe runs/views/apt_0.npz chair plant tv  #    camera, 22 cm up); YOLOE asked each word
    python -m robot.eval.views model runs/views/apt_0.npz chair plant tv  # the model's yes/no to "Is there a <word>?"
    python -m robot.eval.views crops runs/views/apt_0.npz chair sofa tv   # the robot's check of each YOLOE crop

A word is plainly in a picture when some object whose name matches it covers at least --clear pixels of the judge's
640 x 360 view, and absent when none covers any; found is the share of the first that got a yes (a box at the
detector's own threshold, or the model's yes), false the share of the second. Needs the vision server (yoloe) or the
language model's server (model); render needs Habitat. `crops` asks the robot's own question (perceive.is_there) of
every YOLOE detection's crop, and judges it by the object the semantic camera shows over most of its outline.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from robot import paths
from robot.eval.report import same_name


def render(scene: str, count: int, out: Path, seed: int = 3) -> None:
    from robot.sim.habitat import Habitat, free_mask, start

    server, port = start(scene)
    world = Habitat(port)
    try:
        info = world.call(op="info")
        label = {o["id"]: o["label"] for o in info["objects"]}
        free = np.argwhere(free_mask(info))
        rng = np.random.default_rng(seed)
        rgbs, pixels, sems = [], [], []
        for i, j in free[rng.choice(len(free), count, replace=False)]:
            pose = (float(i * info["res"] + info["res"] / 2 - info["width"] / 2),
                    float(j * info["res"] + info["res"] / 2 - info["height"] / 2), float(rng.uniform(-math.pi, math.pi)))
            rgbs.append(np.frombuffer(world.call(op="camera_at", pose=pose, name="front_camera"), np.uint8)
                        .reshape(720, 1280, 3))
            seen: dict[str, int] = {}
            for k, n in world.call(op="visible", pose=pose, cameras=["front_camera"])["front_camera"].items():
                if k in label:
                    seen[label[k]] = max(seen.get(label[k], 0), n)
            pixels.append(seen)
            sems.append(np.frombuffer(world.call(op="semantic_at", pose=pose, name="front_camera"), np.uint32)
                        .reshape(360, 640))
    finally:
        server.terminate()
    labels = sorted({lb for s in pixels for lb in s})
    px = np.array([[s.get(lb, 0) for lb in labels] for s in pixels], np.int32)  # per picture: biggest object per name
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rgb=np.stack(rgbs), labels=np.array(labels), px=px, sem=np.stack(sems),
                        ids=np.array(list(label)), names=np.array(list(label.values())))
    print(f"saved {count} views of {scene} to {out}; names in view: {', '.join(labels)}")


def score(views: Path, words: list[str], says_yes: Callable[[np.ndarray, str], bool], clear: int) -> None:
    data = np.load(views)
    labels, px = list(data["labels"]), data["px"]
    print(f"{'word':18s} {'with':>5s} {'without':>8s} {'found':>7s} {'false':>7s}")
    t = time.perf_counter()
    for word in words:
        cols = [k for k, lb in enumerate(labels) if same_name(word, str(lb))]
        biggest = px[:, cols].max(axis=1) if cols else np.zeros(len(px), int)
        has, hasnt = biggest >= clear, biggest == 0
        yes = np.array([says_yes(rgb, word) for rgb in data["rgb"]])
        found = f"{yes[has].mean():6.0%}" if has.any() else "     -"
        false = f"{yes[hasnt].mean():6.0%}" if hasnt.any() else "     -"
        print(f"{word:18s} {has.sum():5d} {hasnt.sum():8d} {found:>7s} {false:>7s}", flush=True)
    print(f"({(time.perf_counter() - t) / max(1, len(words) * len(data['rgb'])):.2f} s a question)")


def crops(views: Path, words: list[str]) -> None:
    from robot.mind.brain import Brain
    from robot.perception.detector import Detector
    from robot.ros.perceive import _only, is_there

    data, brain, detector = np.load(views), Brain(), Detector()
    name = dict(zip(data["ids"].tolist(), data["names"].tolist(), strict=True))
    print(f"{'word':10s} {'real':>5s} {'kept':>5s} {'false':>6s} {'kept':>5s}")
    for word in words:
        detector.asked = []
        detector.watch(word)
        tally = {True: [0, 0], False: [0, 0]}
        for rgb, sem in zip(data["rgb"], data["sem"], strict=True):
            seen = [s for s in detector(rgb) if s.label == word]
            boxes = [tuple(int(v) for v in s.box) for s in seen]
            for s, (x1, y1, x2, y2) in zip(seen, boxes, strict=True):
                ids = sem[y1 // 2:max(y1 // 2 + 1, y2 // 2), x1 // 2:max(x1 // 2 + 1, x2 // 2)]
                if s.mask is not None and s.mask.any():  # the outline, at the judge's half size
                    m = s.mask[::2, ::2][:ids.shape[0], :ids.shape[1]]
                    ids = ids[:m.shape[0], :m.shape[1]][m]
                ids = ids[ids > 0].ravel()
                truth = name.get(int(np.bincount(ids).argmax()), "") if ids.size else ""
                real = bool(truth) and same_name(word, truth)
                tally[real][0] += 1
                tally[real][1] += is_there(brain, _only(rgb, (x1, y1, x2, y2), boxes), word)
        print(f"{word:10s} {tally[True][0]:5d} {tally[True][1]:5d} {tally[False][0]:6d} {tally[False][1]:5d}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="what", required=True)
    r = sub.add_parser("render", help="views of a home with the truth of what is in each")
    r.add_argument("--scene", default="apt_0")
    r.add_argument("--count", type=int, default=40)
    r.add_argument("--out", type=Path)
    for name in ("yoloe", "model", "crops"):
        a = sub.add_parser(name, help=f"score {name} on rendered views")
        a.add_argument("views", type=Path)
        a.add_argument("words", nargs="+")
        if name != "crops":
            a.add_argument("--clear", type=int, default=300, help="pixels for a thing to be plainly in a picture")
    args = parser.parse_args()
    if args.what == "render":
        render(args.scene, args.count, args.out or paths.RUNS / "views" / f"{args.scene}.npz")
        return
    if args.what == "crops":
        crops(args.views, args.words)
        return
    if args.what == "yoloe":
        from robot.perception.detector import Detector

        detector = Detector()

        def says_yes(rgb: np.ndarray, word: str) -> bool:
            detector.asked = []
            detector.watch(word)
            return any(s.label == word for s in detector(rgb))
    else:
        from robot.mind.brain import Brain
        from robot.perception.camera import data_url

        brain = Brain()

        def says_yes(rgb: np.ndarray, word: str) -> bool:
            return brain.check(data_url(rgb), f"Is there a {word} in this picture?")[0]
    score(args.views, args.words, says_yes, args.clear)


if __name__ == "__main__":
    main()
