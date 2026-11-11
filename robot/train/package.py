"""Merge the built data folders into one Kaggle dataset: train.jsonl, valid.jsonl and a README.

    python -m robot.train.package sft sft-watch sft-noisy

Reads ~/robot-runtime/datasets/<name>/{train,valid}.jsonl, drops duplicate instructions in the same room, keeps a
twentieth for validation, and writes ~/robot-runtime/datasets/sprocket-sft/ (upload that folder to Kaggle as a
dataset named "sprocket-sft").
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path("~/robot-runtime/datasets").expanduser()
README = """# sprocket-sft

Fine-tuning data for Sprocket's planner (LFM3-VL-2B): {n} conversations in the planner's own chat format, each a
room as the robot saw it, an instruction, and a program that passed its task's check in the 2D simulator. The 50-task
test suite and instructions close to it are held out.

{families}

Use with `kaggle_sft.ipynb` from the robot repository (robot/train/).
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    rows, seen = [], set()
    for folder in args.folders:
        for part in ("train", "valid"):
            path = ROOT / folder / f"{part}.jsonl"
            if not path.exists():
                continue
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                key = (row["messages"][1]["content"], row["messages"][2]["content"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    random.Random(args.seed).shuffle(rows)
    split = max(1, len(rows) // 20)
    out = ROOT / "sprocket-sft"
    out.mkdir(parents=True, exist_ok=True)
    for part, chunk in (("valid", rows[:split]), ("train", rows[split:])):
        with (out / f"{part}.jsonl").open("w", encoding="utf-8") as f:
            for row in chunk:
                f.write(json.dumps(row) + "\n")
    counts = Counter(r["family"] for r in rows)
    table = "\n".join(f"- {name}: {n}" for name, n in sorted(counts.items()))
    (out / "README.md").write_text(README.format(n=len(rows), families=table), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"{len(rows)} conversations ({len(rows) - split} train, {split} valid), {len(counts)} families, "
          f"{size:.0f} MB -> {out}")


if __name__ == "__main__":
    main()
