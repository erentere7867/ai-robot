"""Drive a real `RobotHarness` through capability scenarios and record ground truth for the graders.

Only the harness public API is used (config, start/stop, submit_message, the event bus, body pose and the
simulation sandbox), so the runner keeps working while cognition internals change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot.core.config import PROJECT_ROOT, RobotConfig
from robot.core.events import Event
from robot.eval.scenarios import CameraModel, Grade, Scenario, ScenarioRun, TurnRecord, plan_layout
from robot.main import RobotHarness

logger = logging.getLogger(__name__)

OBJECT_KEYS = ("id", "label", "shape", "x_mm", "y_mm", "z_mm", "width_mm", "depth_mm", "height_mm")
POSE_KEYS = ("x_mm", "y_mm", "heading_deg", "head_tilt_deg")
MAX_TRAJECTORY_POINTS = 4000


@dataclass
class RunOptions:
    config_path: str = "config/robot.yaml"
    backend: str | None = None
    timeout_s: float = 300.0
    allow_server_start: bool = True
    settle_s: float = 0.3


@dataclass
class ScenarioResult:
    scenario: str
    seed: int
    grade: Grade
    seconds: float
    run: ScenarioRun
    error: str | None = None
    description: str = ""

    @property
    def passed(self) -> bool:
        return self.grade.passed

    @property
    def model_calls(self) -> int:
        return sum(t.model_calls for t in self.run.turns)

    @property
    def actions_ok(self) -> int:
        return sum(1 for t in self.run.turns for a in t.actions if a.get("ok"))

    @property
    def actions_failed(self) -> int:
        return sum(1 for t in self.run.turns for a in t.actions if not a.get("ok"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "description": self.description,
            "seed": self.seed,
            "passed": self.passed,
            "reason": self.grade.reason,
            "details": self.grade.details,
            "error": self.error,
            "seconds": round(self.seconds, 2),
            "model_calls": self.model_calls,
            "actions": {"completed": self.actions_ok, "failed": self.actions_failed},
            "reflexes": sum(len(t.reflexes) for t in self.run.turns),
            "camera": self.run.camera.to_dict(),
            "objects_start": self.run.objects_start,
            "objects_end": self.run.objects_end,
            "transcript": [t.to_dict() for t in self.run.turns],
        }


def truth_pose(body: Any) -> dict[str, float]:
    """Ground-truth robot pose: `true_pose()` when the body models odometry error, else `pose()`."""
    getter = getattr(body, "true_pose", None)
    pose = getter() if callable(getter) else body.pose()
    if isinstance(pose, Mapping):
        return {k: round(float(pose.get(k, 0.0)), 1) for k in POSE_KEYS}
    return {k: round(float(getattr(pose, k, 0.0)), 1) for k in POSE_KEYS}


def snapshot_objects(sandbox: Any) -> list[dict[str, Any]]:
    return [{k: obj[k] for k in OBJECT_KEYS if k in obj} for obj in sandbox.list_objects()]


def apply_scene(sandbox: Any, scenario: Scenario, seed: int) -> None:
    """Reset the room, then apply the scenario's removals and the seeded, jittered layout."""
    reset = getattr(sandbox, "reset_default_scene", None)
    if callable(reset):
        reset()
    for obj_id in scenario.setup.remove:
        sandbox.remove_object(obj_id)
    room_half = float(getattr(sandbox, "room_half_extent_mm", 1500.0))
    for obj_id, (x, y) in plan_layout(sandbox.list_objects(), seed, scenario.setup, room_half_mm=room_half).items():
        sandbox.move_object(obj_id, x, y)


def build_config(options: RunOptions, db_path: Path) -> RobotConfig:
    cfg = RobotConfig.from_yaml(options.config_path)
    cfg.database_path = db_path
    cfg.record_events = False
    cfg.autonomy.enabled = False
    if options.backend:
        cfg.llm.backend = options.backend
    if not options.allow_server_start:
        cfg.llm.auto_start_server = False
    return cfg


class _Probe:
    """Ground-truth recorder attached to one running harness; handlers never raise into the robot."""

    def __init__(self, harness: RobotHarness) -> None:
        self.harness = harness
        self.turn: TurnRecord | None = None
        self.last_response: dict[str, Any] | None = None
        self._action_args: dict[str, Any] = {}
        self._subs: list[str] = []

    def attach(self) -> None:
        handlers: dict[str, Callable[[Event], None]] = {
            "robot.response": self._on_response,
            "cognition.model_call": self._on_model_call,
            "action.started": self._on_action_started,
            "action.completed": self._on_action_done,
            "action.failed": self._on_action_done,
            "reflex.triggered": self._on_reflex,
            "body.pose": self._on_pose,
        }
        for event_type, handler in handlers.items():
            self._subs.append(
                self.harness.bus.subscribe(event_type, self._guarded(handler), name="eval_probe", immediate=True)
            )

    def detach(self) -> None:
        for sub_id in self._subs:
            self.harness.bus.unsubscribe(sub_id)
        self._subs.clear()

    def begin(self, turn: TurnRecord) -> None:
        self.turn, self.last_response = turn, None

    def end(self) -> dict[str, Any] | None:
        self.turn = None
        return self.last_response

    @staticmethod
    def _guarded(handler: Callable[[Event], None]) -> Callable[[Event], None]:
        def run(event: Event) -> None:
            try:
                handler(event)
            except Exception:
                logger.exception("Eval probe failed on %s", event.type)

        return run

    def _on_response(self, event: Event) -> None:
        if self.turn is None or event.payload.get("origin") == "autonomous":
            return
        self.turn.responses += 1
        self.last_response = dict(event.payload)

    def _on_model_call(self, _event: Event) -> None:
        if self.turn is not None:
            self.turn.model_calls += 1

    def _on_action_started(self, event: Event) -> None:
        if self.turn is not None:
            self._action_args[str(event.payload.get("command_id"))] = event.payload.get("args", {})

    def _on_action_done(self, event: Event) -> None:
        if self.turn is None:
            return
        payload = event.payload
        self.turn.actions.append(
            {
                "tool": payload.get("tool"),
                "args": self._action_args.pop(str(payload.get("command_id")), None),
                "ok": event.type == "action.completed",
                "status": payload.get("status"),
                "message": str(payload.get("message", ""))[:200],
            }
        )

    def _on_reflex(self, event: Event) -> None:
        if self.turn is not None:
            self.turn.reflexes.append(
                {
                    "reflex": event.payload.get("reflex"),
                    "reason": event.payload.get("reason"),
                    "pose": truth_pose(self.harness.body),
                }
            )

    def _on_pose(self, _event: Event) -> None:
        turn = self.turn
        if turn is None or len(turn.trajectory) >= MAX_TRAJECTORY_POINTS:
            return
        pose = truth_pose(self.harness.body)
        last = turn.trajectory[-1] if turn.trajectory else turn.pose_before
        if (
            math.hypot(pose["x_mm"] - last["x_mm"], pose["y_mm"] - last["y_mm"]) >= 20.0
            or abs((pose["heading_deg"] - last["heading_deg"] + 180.0) % 360.0 - 180.0) >= 4.0
            or abs(pose["head_tilt_deg"] - last["head_tilt_deg"]) >= 3.0
        ):
            turn.trajectory.append(pose)


class _Session:
    """One started harness with the scenario scene applied and a probe attached."""

    def __init__(self, harness: RobotHarness) -> None:
        self.harness = harness
        self.probe = _Probe(harness)
        self.camera = CameraModel.from_sandbox(harness.sandbox)

    @classmethod
    async def open(cls, cfg: RobotConfig, scenario: Scenario, seed: int, start_timeout_s: float) -> _Session:
        harness = RobotHarness(
            cfg, enable_dashboard=False, enable_speaker=False, enable_mic=False, spawn_face_window=False
        )
        session = cls(harness)
        try:
            await asyncio.wait_for(harness.start(), timeout=start_timeout_s)
            apply_scene(harness.sandbox, scenario, seed)
            session.probe.attach()
        except BaseException:
            await session.close()
            raise
        return session

    def objects(self) -> list[dict[str, Any]]:
        return snapshot_objects(self.harness.sandbox)

    async def say(self, text: str, options: RunOptions, *, restarted: bool) -> TurnRecord:
        harness = self.harness
        turn = TurnRecord(utterance=text, pose_before=truth_pose(harness.body), restarted_before=restarted)
        self.probe.begin(turn)
        reply: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            async with asyncio.timeout(options.timeout_s):
                reply = await harness.submit_message(text, timeout_s=options.timeout_s) or {}
                await harness.arbiter.wait_until_done()
            if options.settle_s > 0:
                await asyncio.sleep(options.settle_s)
        except TimeoutError:
            turn.error = f"timeout after {options.timeout_s:.0f}s"
        except Exception as exc:
            turn.error = f"{type(exc).__name__}: {exc}"
        finally:
            turn.seconds = time.perf_counter() - started
            final = self.probe.end() or reply
            turn.say = str(final.get("say") or "")
            turn.final_status = str(final.get("status") or "")
            turn.backend = str(final.get("backend") or "")
            turn.pose_after = truth_pose(harness.body)
        return turn

    async def close(self) -> None:
        self.probe.detach()
        try:
            await asyncio.wait_for(self.harness.stop(), timeout=30.0)
        except Exception:
            logger.exception("Harness stop failed")


async def run_scenario(scenario: Scenario, seed: int, options: RunOptions) -> ScenarioResult:
    """Run one scenario on fresh harness(es); errors and timeouts become failures, never crashes."""
    started = time.perf_counter()
    turns: list[TurnRecord] = []
    objects_start: list[dict[str, Any]] = []
    objects_end: list[dict[str, Any]] = []
    camera = CameraModel()
    error: str | None = None
    session: _Session | None = None
    with tempfile.TemporaryDirectory(prefix="robot-eval-", ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "memory.db"
        try:
            for index, text in enumerate(scenario.utterances):
                restarted = session is not None and index in scenario.restart_before
                if session is None or restarted:
                    if session is not None:
                        await session.close()
                        session = None
                    session = await _Session.open(
                        build_config(options, db_path), scenario, seed, max(60.0, options.timeout_s)
                    )
                    if not objects_start:
                        objects_start, camera = session.objects(), session.camera
                turn = await session.say(text, options, restarted=restarted)
                turns.append(turn)
                if turn.error:
                    error = f"utterance {index + 1}: {turn.error}"
                    break
            if session is not None:
                objects_end = session.objects()
        except Exception as exc:
            logger.exception("Scenario %s (seed %d) crashed", scenario.name, seed)
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if session is not None:
                await session.close()

    run = ScenarioRun(scenario.name, seed, turns, objects_start, objects_end or objects_start, camera)
    if error is not None:
        grade = Grade(False, f"error: {error}")
    else:
        try:
            grade = scenario.grade(run)
        except Exception as exc:
            logger.exception("Grader %s failed", scenario.name)
            grade = Grade(False, f"grader error: {type(exc).__name__}: {exc}")
    return ScenarioResult(
        scenario.name, seed, grade, time.perf_counter() - started, run, error, scenario.description
    )


def summarize(results: Sequence[ScenarioResult]) -> dict[str, Any]:
    by_scenario: dict[str, dict[str, int]] = {}
    for r in results:
        entry = by_scenario.setdefault(r.scenario, {"passed": 0, "total": 0})
        entry["total"] += 1
        entry["passed"] += int(r.passed)
    passed = sum(int(r.passed) for r in results)
    return {
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
        "errors": sum(1 for r in results if r.error),
        "model_calls": sum(r.model_calls for r in results),
        "seconds": round(sum(r.seconds for r in results), 1),
        "by_scenario": by_scenario,
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


async def run_suite(
    scenarios: Sequence[Scenario],
    seeds: Sequence[int],
    options: RunOptions,
    *,
    out_path: Path | None = None,
    on_result: Callable[[ScenarioResult], None] | None = None,
) -> dict[str, Any]:
    """Run every scenario for every seed (seed-major), rewriting the JSON report after each result."""
    backend = options.backend or RobotConfig.from_yaml(options.config_path).llm.backend
    report: dict[str, Any] = {
        "schema": 1,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "finished_at": None,
        "git_commit": _git_commit(),
        "backend": backend,
        "options": {
            "config": options.config_path,
            "timeout_s": options.timeout_s,
            "allow_server_start": options.allow_server_start,
            "scenarios": [s.name for s in scenarios],
            "seeds": list(seeds),
        },
        "summary": {},
        "results": [],
    }
    results: list[ScenarioResult] = []
    for seed in seeds:
        for scenario in scenarios:
            result = await run_scenario(scenario, seed, options)
            results.append(result)
            report["results"].append(result.to_dict())
            report["summary"] = summarize(results)
            if on_result is not None:
                on_result(result)
            if out_path is not None:
                write_report(out_path, report)
    report["summary"] = summarize(results)
    report["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    if out_path is not None:
        write_report(out_path, report)
    return report


# --------------------------------------------------------------------------- table output

_COLUMNS = (("scenario", 19), ("seed", 4), ("pass", 4), ("secs", 7), ("calls", 5), ("actions", 7))


def format_header() -> str:
    head = " ".join(name.ljust(w) if i == 0 else name.rjust(w) for i, (name, w) in enumerate(_COLUMNS))
    return f"{head}  reason\n{'-' * (len(head) + 40)}"


def format_row(result: ScenarioResult, reason_width: int = 72) -> str:
    reason = " ".join(result.grade.reason.split())
    if len(reason) > reason_width:
        reason = reason[: reason_width - 3] + "..."
    cells = (
        result.scenario,
        str(result.seed),
        "PASS" if result.passed else "FAIL",
        f"{result.seconds:.1f}",
        str(result.model_calls),
        f"{result.actions_ok}/{result.actions_failed}",
    )
    widths = [w for _, w in _COLUMNS]
    row = " ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths, strict=True)))
    return f"{row}  {reason}"


def format_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        f"Passed {summary.get('passed', 0)}/{summary.get('total', 0)} "
        f"({100.0 * summary.get('pass_rate', 0.0):.0f}%), errors {summary.get('errors', 0)}, "
        f"model calls {summary.get('model_calls', 0)}, {summary.get('seconds', 0.0):.1f}s"
    ]
    by_scenario = summary.get("by_scenario", {})
    if any(v["total"] > 1 for v in by_scenario.values()):
        lines.extend(f"  {name:<19} {v['passed']}/{v['total']}" for name, v in by_scenario.items())
    return "\n".join(lines)

