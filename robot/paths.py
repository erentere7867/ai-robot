"""Where Sprocket keeps what is not code: inside the repository folder, ignored by git, so the folder is the whole
workspace. An environment variable moves any of them (on the robot, models may sit on faster storage).

    models/  every model file (language model, YOLOE, depth, voice), and vision/ for the GPU server's ONNX files
    maps/    the robot's own saved maps, one per home
    runs/    run output: eval results, task reports, logs
    data/    downloaded test data (the photo check's COCO pictures)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _place(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable) or REPO / default).expanduser()


MODELS = _place("SPROCKET_MODELS_DIR", "models")
MAPS = _place("SPROCKET_MAPS_DIR", "maps")
RUNS = _place("SPROCKET_RUNS_DIR", "runs")
DATA = _place("SPROCKET_DATA_DIR", "data")
