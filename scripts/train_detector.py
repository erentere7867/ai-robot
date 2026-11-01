"""Train the fast detector: render a labelled dataset in the simulator, train YOLO11n, export for the Pi.

    python scripts/train_detector.py --data-only --train 6000 --val 800      # just the dataset
    python scripts/train_detector.py --epochs 60                             # dataset (if missing) + training
    python scripts/train_detector.py --export-only runs/detect/fast/weights/best.pt

Outputs go to ~/robot-runtime/models/: `fast_detector.pt` (laptop, CUDA/CPU) and
`fast_detector_ncnn_model/` (Raspberry Pi 5: NCNN is the fastest CPU runtime for YOLO on ARM).
Colour is what tells the objects apart, so hue augmentation is off; brightness and saturation vary.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUNTIME = Path.home() / "robot-runtime"


def export(weights: Path, imgsz: int) -> None:
    from ultralytics import YOLO

    models = RUNTIME / "models"
    models.mkdir(parents=True, exist_ok=True)
    shutil.copy(weights, models / "fast_detector.pt")
    ncnn = Path(YOLO(str(weights)).export(format="ncnn", imgsz=imgsz, half=False))
    target = models / "fast_detector_ncnn_model"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ncnn, target)
    print(f"Exported {models / 'fast_detector.pt'} and {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=RUNTIME / "datasets" / "sim_v1")
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--val", type=int, default=800)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--model", default="yolo11n.pt")
    # Windows + Python 3.15: dataloader worker processes crash, so images load in-process from a RAM cache.
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--export-only", type=Path)
    args = parser.parse_args()

    if args.export_only:
        export(args.export_only, args.imgsz)
        return 0
    data = args.dataset / "data.yaml"
    if not data.exists():
        from robot.perception.datagen import write_dataset

        print(f"Rendering {args.train} + {args.val} labelled frames into {args.dataset} ...")
        data = write_dataset(args.dataset, args.train, args.val, seed=args.seed, workers=args.workers)
    if args.data_only:
        return 0

    from ultralytics import YOLO

    model = YOLO(str(RUNTIME / "models" / args.model) if (RUNTIME / "models" / args.model).exists() else args.model)
    result = model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                         project=str(RUNTIME / "runs"), name="fast", exist_ok=True, device=0, workers=args.loader_workers,
                         cache="ram",
                         hsv_h=0.0, hsv_s=0.35, hsv_v=0.45, degrees=0.0, fliplr=0.5, mosaic=1.0,
                         close_mosaic=10, patience=20, plots=False)
    best = Path(result.save_dir) / "weights" / "best.pt"
    export(best, args.imgsz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
