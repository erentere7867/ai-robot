"""Run capability scenarios against the simulated robot and grade them from simulator ground truth.

Examples:
  python scripts/eval_scenarios.py --list
  python scripts/eval_scenarios.py --backend llamacpp --scenarios greet,go_to_ball --seeds 3
  python scripts/eval_scenarios.py --backend null --out artifacts/eval/null-smoke.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot.eval.runner import RunOptions, format_header, format_row, format_summary, run_suite  # noqa: E402
from robot.eval.scenarios import SCENARIO_LIST, select_scenarios  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulated capability evaluation for the robot harness")
    parser.add_argument("--scenarios", default="all", help="'all' or comma-separated names (see --list)")
    parser.add_argument("--seeds", type=int, default=1, help="Number of scene seeds to run (default 1)")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed (default 0)")
    parser.add_argument("--backend", choices=["auto", "llamacpp", "null"], default=None,
                        help="Override the config's LLM backend")
    parser.add_argument("--config", default="config/robot.yaml", help="Robot config (default config/robot.yaml)")
    parser.add_argument("--out", default=None, help="JSON report path (default artifacts/eval/eval-<time>.json)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-utterance timeout in seconds")
    parser.add_argument("--no-server-start", action="store_true",
                        help="Never auto-start llama-server; use one that is already running or fail")
    parser.add_argument("--strict", action="store_true", help="Exit with status 1 if any scenario fails")
    parser.add_argument("--list", action="store_true", help="List scenarios and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show harness INFO logs")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if args.list:
        for scenario in SCENARIO_LIST:
            print(f"{scenario.name:<19} {scenario.description}")
            for text in scenario.utterances:
                print(f"{'':<21}> {text}")
        return 0
    try:
        scenarios = select_scenarios(args.scenarios)
    except ValueError as exc:
        parser.error(str(exc))
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if not 1.0 <= args.timeout <= 3600.0:
        parser.error("--timeout must be in [1, 3600] seconds")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else ROOT / "artifacts" / "eval" / f"eval-{stamp}-{args.backend or 'config'}.json"
    options = RunOptions(
        config_path=args.config,
        backend=args.backend,
        timeout_s=args.timeout,
        allow_server_start=not args.no_server_start,
    )
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    print(format_header(), flush=True)
    try:
        report = asyncio.run(
            run_suite(scenarios, seeds, options, out_path=out, on_result=lambda r: print(format_row(r), flush=True))
        )
    except KeyboardInterrupt:
        print(f"\nInterrupted; partial results are in {out}")
        return 130
    print()
    print(format_summary(report["summary"]))
    print(f"Report: {out}")
    failed = report["summary"]["total"] - report["summary"]["passed"]
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
