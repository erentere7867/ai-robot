"""The gate: no program moves the robot until it passes all three checks (architecture section 7).

1. Syntax and whitelist: only robot, geom, math, the catchable exceptions, plain Python control flow, local functions
   and a short list of builtins. No imports, no private attributes, no string formatting tricks.
2. Preemptibility: every loop body calls robot, so the runner can stop it.
3. Dry run in the 2D simulator on the current map, for up to 2,000 robot calls or 120 simulated seconds: no uncaught
   error, no path through occupied cells, and no computing past a wall-clock limit. A program still running cleanly
   when that budget runs out passes. Amended from the spec, which rejects programs over 2,000 calls: that would reject
   every "follow me until I say stop" and every 20 Hz control loop the spec itself recommends.

A failure comes back as one short message the planner can act on.
"""

from __future__ import annotations

import ast
import difflib
import math
from dataclasses import dataclass
from typing import Any

from robot.agent.api import CATCHABLE, GEOM_CALLS, ROBOT_CALLS, SAFE_BUILTINS, SIGNATURES
from robot.agent.runner import Runner

MAX_CALLS = 2000
MAX_SOURCE = 4000
DRY_RUN_SIM_S = 120.0
DRY_RUN_WALL_S = 5.0
NAMES = {"robot", "geom", "math", *SAFE_BUILTINS, *(e.__name__ for e in CATCHABLE)}
CAUGHT = {e.__name__ for e in CATCHABLE}
MATH = {name for name in dir(math) if not name.startswith("_")}
BANNED_ATTRIBUTES = {"format", "format_map", "mro"}  # str.format can read attributes through its pattern
STATEMENTS = (ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.If, ast.For, ast.While, ast.Break,
              ast.Continue, ast.Pass, ast.Try, ast.ExceptHandler, ast.FunctionDef, ast.Return, ast.Delete)
EXPRESSIONS = (ast.Call, ast.keyword, ast.Name, ast.Attribute, ast.Subscript, ast.Slice, ast.Constant, ast.List,
               ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
               ast.comprehension, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.IfExp, ast.Lambda,
               ast.arguments, ast.arg, ast.JoinedStr, ast.FormattedValue, ast.Starred, ast.NamedExpr,
               ast.expr_context, ast.operator, ast.boolop, ast.unaryop, ast.cmpop)


@dataclass
class GateResult:
    ok: bool
    stage: str = ""  # "whitelist", "preempt" or "dry run"
    message: str = ""
    trace: Any = None  # the dry run's trace, when it ran
    ended_by: str = ""  # "budget" or "stop word" when the dry run was cut off while still running cleanly

    def __str__(self) -> str:
        return "passed" if self.ok else f"{self.stage}: {self.message}"


def _calls_robot(nodes: list[ast.stmt]) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
               and n.func.value.id == "robot" for stmt in nodes for n in ast.walk(stmt))


def _bound_names(tree: ast.AST) -> set[str]:
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.FunctionDef):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def _spelled(name: str) -> str:
    """How a name the API does have is written, e.g. user -> robot.user(), pi -> math.pi."""
    if name in ROBOT_CALLS:
        return f"robot.{name}()"
    if name in GEOM_CALLS:
        return f"geom.{name}"
    return f"math.{name}" if name in MATH and name not in NAMES else name


KNOWN = (*ROBOT_CALLS, *GEOM_CALLS, *MATH, *NAMES)


def _did_you_mean(name: str) -> str:
    """A repair given only "X does not exist" tends to write X again; naming the nearest real call fixes that."""
    near = [name] if name in KNOWN else difflib.get_close_matches(name, KNOWN, n=1, cutoff=0.75)
    return f"; did you mean {_spelled(near[0])}?" if near else ""


def _static(source: str) -> GateResult:
    if len(source) > MAX_SOURCE:
        return GateResult(False, "whitelist", f"the program is longer than {MAX_SOURCE} characters")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return GateResult(False, "whitelist", f"line {exc.lineno}: syntax error: {exc.msg}")
    bound = _bound_names(tree)

    def fail(node: ast.AST, why: str) -> GateResult:
        return GateResult(False, "whitelist", f"line {getattr(node, 'lineno', '?')}: {why}")

    for node in ast.walk(tree):
        if not isinstance(node, STATEMENTS + EXPRESSIONS):
            return fail(node, f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Name):
            if node.id.startswith("_") and node.id != "_":
                return fail(node, f"the name {node.id} is not allowed")
            if isinstance(node.ctx, ast.Load) and node.id not in NAMES | bound:
                hint = _did_you_mean(node.id) or "; set it before using it"
                return fail(node, f"the name {node.id} is not available{hint}")
            if isinstance(node.ctx, ast.Store) and node.id in {"robot", "geom", "math"}:
                return fail(node, f"{node.id} cannot be reassigned")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in BANNED_ATTRIBUTES:
                return fail(node, f"the attribute {node.attr} is not allowed")
            owner = node.value.id if isinstance(node.value, ast.Name) else None
            allowed = {"robot": ROBOT_CALLS, "geom": GEOM_CALLS, "math": MATH}.get(owner)
            if owner not in bound and allowed is not None and node.attr not in allowed:
                return fail(node, f"{owner}.{node.attr} does not exist{_did_you_mean(node.attr)}")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name) and node.func.value.id == "robot"
              and node.func.attr in SIGNATURES and "robot" not in bound
              and not any(isinstance(a, ast.Starred) for a in node.args) and all(k.arg for k in node.keywords)):
            try:  # the arguments must fit the signature: unit arguments by name, required words present
                SIGNATURES[node.func.attr].bind(None, *node.args, **{k.arg: None for k in node.keywords})
            except TypeError as exc:
                return fail(node, f"robot.{node.func.attr}: {exc}")
        elif isinstance(node, ast.ExceptHandler):
            caught = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            if node.type is None or not all(isinstance(c, ast.Name) and c.id in CAUGHT for c in caught):
                return fail(node, f"only {', '.join(sorted(CAUGHT))} may be caught")
        elif isinstance(node, ast.FunctionDef) and (node.decorator_list or node.returns):
            return fail(node, "functions cannot have decorators or annotations")
        elif isinstance(node, (ast.For, ast.While)) and not _calls_robot(node.body):
            return GateResult(False, "preempt",
                              f"line {node.lineno}: the loop body never calls robot, so it cannot be stopped")
        elif isinstance(node, (ast.For, ast.While)) and node.orelse:
            return fail(node, "loops cannot have an else block")
    return GateResult(True)


def check(source: str, world: Any) -> GateResult:
    """All three checks. `world` is the current state as a SimRobot; the dry run uses a copy of it."""
    result = _static(source)
    if not result.ok:
        return result
    twin = world.clone(max_calls=MAX_CALLS, max_time=world.t + DRY_RUN_SIM_S)
    run = Runner(twin, time_limit_s=DRY_RUN_WALL_S).run(source)
    if twin.trace.violations:
        return GateResult(False, "dry run", twin.trace.violations[0], twin.trace)
    if run.ok or twin.ended_by in ("budget", "stop word"):
        return GateResult(True, trace=twin.trace, ended_by=twin.ended_by)
    if run.status == "preempted":
        return GateResult(False, "dry run", f"the program computed for more than {DRY_RUN_WALL_S:.0f} s", twin.trace)
    where = f"line {run.line}: " if run.line else ""
    return GateResult(False, "dry run", where + run.error, twin.trace)
