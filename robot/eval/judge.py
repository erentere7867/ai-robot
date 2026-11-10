"""Calibrate the readback self-check: does the planner's probability of "yes" separate right programs from wrong ones?

Every (instruction, readback) the self-check saw in a suite run is labelled by running the program against its task's
real check. The model's P(yes) is then read for each, and the report shows how well it ranks them (AUC) and what each
threshold would catch and wrongly reject.

    python -m robot.eval.judge --run api2-rb      # needs llama-server
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path("~/robot-runtime/runs/p5").expanduser()


def label(run: str) -> list[dict]:
    from robot.eval.suite import CASES, run_case

    by = {c.id: c for c in CASES}
    seen, rows = set(), []
    for r in json.loads((RUNS / f"{run}.json").read_text(encoding="utf-8")):
        for idx, _, said in r.get("checks", []):
            program = r["programs"][idx]
            if (r["id"], program) in seen:
                continue
            seen.add((r["id"], program))
            rows.append({"id": r["id"], "say": r["say"], "readback": said,
                         "right": bool(run_case(by[r["id"]], program)[0])})
    return rows


def main() -> None:
    from robot.agent.planner import LocalPlanner

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="api2-rb")
    args = parser.parse_args()
    rows = label(args.run)
    planner = LocalPlanner()
    for row in rows:
        row["p_yes"] = round(planner.judge(row["say"], row["readback"], threshold=0.0)[1], 4)
    (RUNS / f"{args.run}-judge.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    right = [r["p_yes"] for r in rows if r["right"]]
    wrong = [r["p_yes"] for r in rows if not r["right"]]
    pairs = [(a > b) + 0.5 * (a == b) for a in right for b in wrong]
    print(f"{len(rows)} distinct programs: {len(right)} right, {len(wrong)} wrong")
    print(f"P(yes) median: right {sorted(right)[len(right) // 2] if right else 0:.3f}, "
          f"wrong {sorted(wrong)[len(wrong) // 2] if wrong else 0:.3f}; AUC {sum(pairs) / max(1, len(pairs)):.2f} "
          "(0.5 = no better than chance)")
    for t in sorted({0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8} | set()):
        caught = sum(p < t for p in wrong)
        rejected = sum(p < t for p in right)
        print(f"  threshold {t:.2f}: catches {caught}/{len(wrong)} wrong, wrongly rejects {rejected}/{len(right)} right")


if __name__ == "__main__":
    main()
