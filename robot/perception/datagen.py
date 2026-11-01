"""Training data for the fast detector, rendered from the simulator with exact labels.

Each sample is a head-camera frame of a randomized room: object layout, unlabeled-colour distractors,
lighting, robot pose and head tilt all vary, so the detector learns the objects rather than one scene.
Boxes come from a segmentation render, so occlusion and partial views are labelled exactly.

On the real robot the same YOLO-format dataset is produced from camera frames labelled by a teacher
(a large open-vocabulary detector or the VLM), then checked by hand. See docs/architecture.md.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from robot.body.hal import BodyPose
from robot.world.sandbox import _MJ_SHARED_LOCK, SimulationSandbox, _get_or_init_mujoco_gl

# Class 0..5 are the room's known objects; "object" is anything else (distractors here, unknown things on
# the robot). The fast layer tracks unknown objects too, and the VLM names them when it matters.
CLASSES = ["red ball", "blue cube", "yellow duck", "green mug", "cardboard box", "charging dock", "object"]
OBJECT_CLASS = {"red_ball": 0, "blue_cube": 1, "yellow_duck": 2, "green_mug": 3, "cardboard_box": 4,
                "charging_dock": 5}
# Distractor colours stay away from the known objects' hues, so "red ball" never means "any red thing".
DISTRACTOR_COLORS = ["#f472b6", "#fb923c", "#22d3ee", "#e5e7eb", "#6b7280", "#111827", "#c026d3", "#a3e635"]
MIN_PIXELS = 40
ROOM_LIMIT_MM = 1300.0


@dataclass
class Sample:
    rgb: np.ndarray
    boxes: list[tuple[int, float, float, float, float]]  # class, cx, cy, w, h (normalized)


def _free_spot(rng: random.Random, taken: list[tuple[float, float, float]], radius: float,
               limit: float = ROOM_LIMIT_MM) -> tuple[float, float] | None:
    for _ in range(200):
        x, y = rng.uniform(-limit, limit), rng.uniform(-limit, limit)
        if all(math.hypot(x - tx, y - ty) > radius + tr + 40.0 for tx, ty, tr in taken):
            return x, y
    return None


class SceneRandomizer:
    def __init__(self, sandbox: SimulationSandbox, seed: int = 0) -> None:
        self.sandbox = sandbox
        self.rng = random.Random(seed)
        self.base_ids = [oid for oid in OBJECT_CLASS if sandbox.get_object(oid) is not None]
        st = _get_or_init_mujoco_gl(sandbox.xml_path, sandbox.cam_width, sandbox.cam_height,
                                    sandbox.overview_width, sandbox.overview_height)
        self._model = st["model"]
        self._light0 = self._model.light_diffuse.copy()
        self._ambient0 = self._model.vis.headlight.ambient.copy()
        self._removed: dict[str, object] = {}

    def randomize(self) -> BodyPose:
        rng, sb = self.rng, self.sandbox
        for oid in [o["id"] for o in sb.list_objects() if o["id"].startswith("distractor")]:
            sb.remove_object(oid)
        taken: list[tuple[float, float, float]] = []
        for oid in self.base_ids:
            obj = sb.get_object(oid) or self._removed.get(oid)
            present = rng.random() < 0.8
            if not present:
                if sb.get_object(oid) is not None:
                    self._removed[oid] = sb.get_object(oid)
                    sb.remove_object(oid)
                continue
            if sb.get_object(oid) is None:
                sb._objects[oid] = self._removed.pop(oid)  # restore the same object
            spot = _free_spot(rng, taken, obj.radius_mm)
            if spot is None:
                continue
            sb.move_object(oid, *spot)
            taken.append((spot[0], spot[1], obj.radius_mm))
        for k in range(rng.randint(0, 4)):
            spot = _free_spot(rng, taken, 60.0)
            if spot is not None:
                sb.add_object(obj_id=f"distractor_{k}", label="object", shape="box", x_mm=spot[0], y_mm=spot[1],
                              size_mm=110.0, color=rng.choice(DISTRACTOR_COLORS))
                taken.append((spot[0], spot[1], 60.0))
        with _MJ_SHARED_LOCK:
            scale = rng.uniform(0.55, 1.35)
            self._model.light_diffuse[:] = np.clip(self._light0 * scale * np.array([rng.uniform(0.9, 1.1) for _ in range(3)]), 0, 1)
            self._model.vis.headlight.ambient[:] = np.clip(self._ambient0 * rng.uniform(0.6, 1.4), 0, 1)
        sb._invalidate_renders()
        # Robot somewhere free, usually looking roughly at something (that is what the camera mostly sees).
        spot = _free_spot(rng, taken, 230.0, limit=1250.0) or (0.0, 0.0)
        if taken and rng.random() < 0.7:
            tx, ty, _ = rng.choice(taken)
            heading = math.degrees(math.atan2(ty - spot[1], tx - spot[0])) + rng.uniform(-35.0, 35.0)
        else:
            heading = rng.uniform(-180.0, 180.0)
        return BodyPose(spot[0], spot[1], heading, rng.uniform(-22.0, 15.0))

    def sample(self) -> Sample:
        pose = self.randomize()
        rgb, labels = self.sandbox.render_head_labels(pose)
        h, w, _ = rgb.shape
        boxes = []
        for oid, lab in labels.items():
            if lab["pixels"] < MIN_PIXELS:
                continue
            cls = OBJECT_CLASS.get(oid, CLASSES.index("object"))
            x1, y1, x2, y2 = lab["bbox_px"]
            boxes.append((cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
        return Sample(rgb, boxes)


def _write_shard(args: tuple[str, str, int, int, int]) -> None:
    out_dir, split, start, count, seed = args
    randomizer = SceneRandomizer(SimulationSandbox(), seed)
    images, labels = Path(out_dir) / "images" / split, Path(out_dir) / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + count):
        sample = randomizer.sample()
        name = f"{split}_{i:06d}"
        cv2.imwrite(str(images / f"{name}.jpg"), sample.rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        (labels / f"{name}.txt").write_text(
            "".join(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for c, cx, cy, bw, bh in sample.boxes))


def write_dataset(out_dir: Path, n_train: int, n_val: int, seed: int = 0, workers: int = 1) -> Path:
    """Write a YOLO-format dataset (in parallel shards, each with its own seed) and return its data.yaml."""
    jobs = []
    for split, count, base_seed in (("train", n_train, seed), ("val", n_val, seed + 10_000)):
        per = -(-count // workers)
        jobs += [(str(out_dir), split, k * per, min(per, count - k * per), base_seed + k)
                 for k in range(workers) if count - k * per > 0]
    if workers > 1:
        from multiprocessing import Pool

        with Pool(workers) as pool:
            pool.map(_write_shard, jobs)
    else:
        for job in jobs:
            _write_shard(job)
    data = out_dir / "data.yaml"
    data.write_text(f"path: {out_dir.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
                    + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES)), encoding="utf-8")
    return data
