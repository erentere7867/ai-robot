"""Build the fine-tuning set: generate tasks, keep only those whose program passes the gate's static checks, runs
cleanly in the 2D world and passes its own check, and drop any instruction too close to the 50-task test suite.

Each example is written in the planner's own chat format (the system prompt with the API card, the room as the
planner sees it before acting, the instruction, then the say/plan/program answer), so training and use match.

    python -m robot.train.build --per-family 100
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

from robot.agent.api import CATCHABLE, SAFE_BUILTINS, Preempted, api_card, geom
from robot.agent.context import describe
from robot.agent.gate import _static
from robot.agent.planner import RULES, _earlier
from robot.eval.suite import CASES
from robot.sim.world import BudgetSpent, SimRobot
from robot.train.families import FAMILIES, Task

FILLER = {"please", "hey", "sprocket", "um", "could", "you", "the", "a", "an"}
OUT = Path("~/robot-runtime/datasets/sft").expanduser()


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower())) - FILLER


SUITE = [_words(c.say) for c in CASES]


def too_close(instruction: str) -> bool:
    """Near-duplicates of a test instruction leak the test: Jaccard of content words at or above 0.8."""
    w = _words(instruction)
    return any(len(w & s) / max(1, len(w | s)) >= 0.8 for s in SUITE)


def verify(t: Task) -> tuple[bool, str, str]:
    """Run the program on its room; returns (kept, why, the room as the planner saw it before acting)."""
    sim = SimRobot(t.scene, max_time=150.0)
    sim.begin()
    context = describe(sim.snapshot())
    static = _static(t.program)
    if not static.ok:
        return False, f"gate: {static}", context
    names = {"__builtins__": dict(SAFE_BUILTINS), "robot": sim, "geom": geom, "math": math,
             **{e.__name__: e for e in CATCHABLE}}
    try:
        exec(compile(t.program, "<program>", "exec"), names)  # noqa: S102 - teacher-written templates
    except (Preempted, BudgetSpent):
        pass  # "stop" was said, or the preview budget ran out while the program was still going cleanly
    except Exception as exc:  # noqa: BLE001 - any failure drops the example
        return False, f"run: {type(exc).__name__}: {exc}", context
    if sim.trace.violations:
        return False, f"violation: {sim.trace.violations[0]}", context
    try:
        ok = bool(t.check(sim))
    except Exception as exc:  # noqa: BLE001
        return False, f"check: {exc}", context
    return ok, "passed" if ok else "check failed", context


def example(t: Task, context: str) -> dict:
    answer = f"say: {t.say}\nplan: {t.plan}\n```python\n{t.program.rstrip()}\n```"
    return {"family": t.family, "category": t.category, "instruction": t.instruction,
            "messages": [{"role": "system", "content": f"{RULES}\n\nAPI:\n{api_card()}"},
                         {"role": "user", "content": f"{context}{_earlier(t.earlier)}\nInstruction: {t.instruction}"},
                         {"role": "assistant", "content": answer}]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--noise-share", type=float, default=0.0, help="share of rooms seen through a noisy detector")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out).expanduser()
    rng = random.Random(args.seed)
    kept: list[dict] = []
    tried, why = Counter(), Counter()
    started = time.monotonic()
    for name, _, make in FAMILIES:
        if args.only and name not in args.only:
            continue
        got = 0
        for _ in range(args.per_family * 6):
            if got >= args.per_family:
                break
            t = make(random.Random(rng.random()))
            if t is None:
                continue
            if rng.random() < args.noise_share:  # some rooms through an imperfect detector, as on the robot
                t.scene.noise, t.scene.seed = rng.choice((0.3, 0.6, 1.0)), rng.randrange(10**6)
            tried[name] += 1
            if too_close(t.instruction):
                why[f"{name}: too close to the test suite"] += 1
                continue
            ok, reason, context = verify(t)
            if ok:
                kept.append(example(t, context))
                got += 1
            else:
                why[f"{name}: {reason[:70]}"] += 1
        print(f"{name:<20} kept {got:>4} of {tried[name]:>4} tried", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    rng.shuffle(kept)
    split = max(1, len(kept) // 20)
    for part, rows in (("valid", kept[:split]), ("train", kept[split:])):
        with (out / f"{part}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(f"\n{len(kept)} examples ({len(kept) - split} train, {split} valid) in {time.monotonic() - started:.0f} s "
          f"-> {out}")
    print("most common reasons for dropping:")
    for reason, n in why.most_common(15):
        print(f"  {n:4} {reason}")


if __name__ == "__main__":
    main()
