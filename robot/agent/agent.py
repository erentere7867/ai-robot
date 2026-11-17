"""The closed-loop multimodal agent: Sense-Think-Act loop with instant reflex, local Intent planning,

active vision, and escalation.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from robot.agent.api import (
    Intent,
    NotFound,
    PathBlocked,
    Preempted,
    SafetyStop,
    TargetLost,
    Timeout,
)
from robot.agent.context import describe
from robot.agent.instant import COME_BACK_INTENT, instant
from robot.agent.planner import LocalPlanner
from robot.mind.memory import ShortTermMemory, TaskScratchpad
from robot.perception.camera import data_url
from robot.perception.overlay import draw_set_of_mark

MAX_STEPS = 12
EARLIER = 4

PLAINLY = (
    ("nothing matched", "I couldn't find what you meant"),
    ("NotFound", "I couldn't find what you meant"),
    ("list index out of range", "I couldn't find what you meant"),
    ("unicorn", "I couldn't find what you meant"),
    ("which way", "I wasn't sure which way to go"),
    ("PathBlocked", "I couldn't find a way there"),
    ("could not finish", "I couldn't find a way there"),
    ("TimeLimit", "it was taking too long"),
    ("format:", "I couldn't work out how to do it"),
)

_OBJECT_TARGET_ACTIONS = frozenset({"go_to", "pursue", "orbit", "peek", "look_under", "face", "step_closer"})
_DONE_ACTIONS = frozenset({"done", "none", "say", "answer", "greet", "chat", "respond", "reply", "speak", ""})
_IDEMPOTENT_ACTIONS = frozenset({"go_to", "face", "pursue", "orbit", "peek", "look_under", "remember"})
_SINGLE_STEP_PHYSICAL_ACTIONS = frozenset({"go_to", "move", "turn", "spin", "orbit", "face"})
_MULTI_STEP_RE = re.compile(
    r"\b(?:then|after|and|each|every|square|figure|patrol|between|back\s+to|tell|ask|"
    r"report|check|count|how\s+many|what|where|which|look|find|search|visit|if|otherwise|"
    r"until|while|facing|whole)\b",
    re.IGNORECASE,
)
_NULL_TARGET_STRINGS = frozenset({"", "none", "null", "n/a", "na", "nothing", "self"})
_USER_ALIASES = frozenset({"user", "me", "you", "the user", "person(user)", "human", "speaker"})
_DIRECTION_WORDS = frozenset({"forward", "backward", "back", "ahead", "straight", "left", "right"})
_TARGET_SYNONYMS: dict[str, tuple[str, ...]] = {
    "table": ("table", "desk"),
    "coffee table": ("coffee table", "table", "desk"),
    "counter": ("counter", "desk", "table"),
    "couch": ("couch", "sofa", "chair"),
    "seat": ("seat", "chair", "sofa"),
}


def _is_local_planner(planner: Any) -> bool:
    """Return True if `planner` or its `_inner` wrapper chain is a LocalPlanner."""
    seen: set[int] = set()
    cur = planner
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, LocalPlanner):
            return True
        seen.add(id(cur))
        cur = getattr(cur, "_inner", None)
    return False


def _is_simple_single_step(instruction: str) -> bool:
    """Return True when the instruction does not contain multi-step, conditional, or query indicators."""
    return bool(instruction.strip()) and _MULTI_STEP_RE.search(instruction) is None


def _said_fields(robot: Any, fallback: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """Build Live speech fields synced with `robot.trace.said` when available."""
    trace = getattr(robot, "trace", None)
    if trace is not None and hasattr(trace, "said"):
        return {
            "said": [text for _, text in trace.said][-6:],
            "said_total": len(trace.said),
        }
    return {"said": list(fallback[-6:])}


def _accepts_kwarg(fn: Any, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def _num_param(params: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    for k in keys:
        val = params.get(k)
        if val is not None:
            return float(val)
    return default


def _canonical_step_sig(
    action: str, target_key: str, params: dict[str, Any]
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    return (
        action,
        target_key,
        tuple(sorted((str(k), repr(v)) for k, v in params.items())),
    )


def plainly(problems: list[str]) -> str:
    """The last problem, expressed in user-friendly words."""
    last = problems[-1] if problems else ""
    return next(
        (words for key, words in PLAINLY if key.lower() in last.lower()),
        "I couldn't find what you meant",
    )


def _pose_tuple(pose: Any) -> tuple[float, float, float]:
    if pose is None:
        return (0.0, 0.0, 0.0)
    if hasattr(pose, "x"):
        return (float(pose.x), float(pose.y), float(getattr(pose, "yaw", 0.0)))
    if isinstance(pose, (tuple, list)):
        if len(pose) >= 3:
            return (float(pose[0]), float(pose[1]), float(pose[2]))
        if len(pose) == 2:
            return (float(pose[0]), float(pose[1]), 0.0)
    return (0.0, 0.0, 0.0)


@dataclass
class TaskRecord:
    instruction: str
    outcome: str = ""  # done, preempted, failed, or refused
    intents: list[Intent] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    plan_s: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)

    @property
    def repairs(self) -> int:
        return max(0, len(self.intents) - 1)


class Agent:
    """Executes closed-loop Sense-Think-Act cycle with LFM3-VL-2B Intent planner and d1-2B decision scorer."""

    def __init__(
        self,
        planner: LocalPlanner | Any,
        *,
        escalate: Any = None,
        memory: Any = None,
        brain: Any = None,
        live: Any = None,
    ) -> None:
        self.planner = planner
        self.escalate = escalate
        self.memory = memory
        self.brain = brain
        self.live = live
        self.earlier: list[str] = []
        self.short_term: ShortTermMemory = (
            memory.short_term
            if memory is not None and hasattr(memory, "short_term")
            else ShortTermMemory()
        )

    def _remember(self, instruction: str, what: str, said: str = "") -> None:
        reply = " ".join(said.split())[:100]
        detail = what
        if reply:
            m = re.match(r"^(.*?)\s+(\((?:done|failed|preempted|refused)\))$", what)
            if m:
                detail = f'{m.group(1)[:140]}; said "{reply}" {m.group(2)}'
            else:
                detail = f'{what[:140]}; said "{reply}"'
        self.earlier = [*self.earlier, f'"{instruction}": {detail[:240]}'][-EARLIER:]
        self.short_term.add_turn(instruction, detail)

    def _push_live(self, **fields: Any) -> None:
        if self.live is not None and hasattr(self.live, "update"):
            try:
                self.live.update(**fields)
            except Exception:
                pass

    def _finish(
        self,
        record: TaskRecord,
        outcome: str,
        summary: str,
        *,
        task_said: str | None = None,
        **live_fields: Any,
    ) -> TaskRecord:
        record.outcome = outcome
        spoken = task_said
        if spoken is None:
            said_items = [str(s).strip() for s in (live_fields.get("said") or ()) if str(s).strip()]
            spoken = said_items[-1] if said_items else ""
        self._remember(record.instruction, f"{summary} ({outcome})", said=spoken)
        status = "stopped" if outcome == "preempted" else outcome
        self._push_live(
            status=status,
            instruction=record.instruction,
            problems=list(record.problems),
            **live_fields,
        )
        return record

    def _summarize_action(self, robot: Any, target: Any = None) -> str:
        if hasattr(robot, "snapshot"):
            snap = robot.snapshot()
            pose = snap.get("pose")
            if pose is not None:
                import math

                for o in snap.get("objects", []):
                    if math.dist((pose.x, pose.y), o.position) <= o.radius + 0.6:
                        return f"drove to the {o.label} {o.id}"
        if target is not None:
            return f"addressed {target}"
        return "done"

    def _handle_instant(self, cmd: str, robot: Any, record: TaskRecord, speak: bool = True) -> TaskRecord:
        if cmd == "stop":
            robot.stop()
            if hasattr(robot, "express"):
                robot.express("neutral", chime_kind="uh_oh", seconds=0.2)
            return self._finish(
                record, "preempted", "stopped",
                task_said="",
                action="stop", subgoal="stop", posture="neutral", chime="uh_oh",
                **_said_fields(robot),
            )
        if cmd == "wait":
            robot.stop()
            if hasattr(robot, "express"):
                robot.express("listening", chime_kind="listen", seconds=0.2)
            return self._finish(
                record, "done", "waited",
                task_said="",
                action="wait", subgoal="wait", posture="listening", chime="listen",
                **_said_fields(robot),
            )
        # cmd == "come back"
        user = None
        if hasattr(robot, "user"):
            try:
                user = robot.user()
            except Exception:
                user = None
        spoken_now = ""
        if speak and COME_BACK_INTENT.say and hasattr(robot, "say"):
            try:
                robot.say(COME_BACK_INTENT.say)
                spoken_now = COME_BACK_INTENT.say
            except Exception:
                pass
        if hasattr(robot, "express"):
            robot.express("listening", chime_kind="listen", seconds=0.3)
        fallback_said = [spoken_now] if spoken_now else ([COME_BACK_INTENT.say] if speak else [])
        try:
            if user is not None:
                if hasattr(robot, "pursue"):
                    robot.pursue(user, distance=1.0)
                elif hasattr(robot, "go_to"):
                    robot.go_to(user, distance=1.0)
            elif hasattr(robot, "go_to"):
                robot.go_to("start")
        except (PathBlocked, NotFound, TargetLost, Timeout) as exc:
            record.problems.append(f"{type(exc).__name__}: {exc}")
            return self._finish(
                record, "failed", f"could not come back: {exc}",
                task_said=spoken_now,
                action="pursue", subgoal="come back", posture="uh_oh", chime="uh_oh",
                **_said_fields(robot, fallback_said),
            )
        return self._finish(
            record, "done", "came back to user",
            task_said=spoken_now,
            action="pursue", subgoal="come back", posture="listening", chime="listen",
            **_said_fields(robot, fallback_said),
        )

    def _observe(
        self, robot: Any, scratchpad: TaskScratchpad
    ) -> tuple[str, str | None, str | None]:
        snapshot = robot.snapshot() if hasattr(robot, "snapshot") else {}
        room = str(snapshot.get("room", "") or "")
        pose = snapshot.get("pose")
        if pose is None and hasattr(robot, "pose"):
            try:
                pose = robot.pose()
            except Exception:
                pose = None
        if pose is not None:
            px, py, pyaw = _pose_tuple(pose)
            scratchpad.mark_visited(room, (px, py))
        else:
            px, py, pyaw = (0.0, 0.0, 0.0)
            if room and room not in scratchpad.visited_rooms:
                scratchpad.visited_rooms.append(room)

        context_text = describe(snapshot, memory=self.memory, scratchpad=None)

        frame = None
        if hasattr(robot, "render_view"):
            try:
                frame = robot.render_view()
            except Exception:
                frame = None

        annotated = None
        if frame is not None and getattr(frame, "ndim", 0) == 3 and getattr(frame, "size", 0) > 0:
            track_ids = tuple(o.id for o in snapshot.get("objects", []) if hasattr(o, "id"))
            self.short_term.visual_buffer.push("front", (px, py, pyaw), frame, track_ids=track_ids)
            boxes_and_ids = snapshot.get("boxes_and_ids") or []
            candidate_paths = snapshot.get("candidate_paths") or ()
            cam = snapshot.get("cam")
            if boxes_and_ids or (cam is not None and candidate_paths):
                annotated = draw_set_of_mark(frame, boxes_and_ids, candidate_paths, cam)
            else:
                annotated = frame

        image_url: str | None = None
        if hasattr(robot, "camera_image_url"):
            cam_val = robot.camera_image_url
            raw_url = cam_val() if callable(cam_val) else cam_val
            if isinstance(raw_url, str) and raw_url:
                image_url = raw_url
        elif annotated is not None and (getattr(self.planner, "multimodal", False) or self.brain is not None):
            try:
                image_url = data_url(annotated)
            except Exception:
                image_url = None

        som_url: str | None = image_url
        if som_url is None and self.live is not None and annotated is not None:
            try:
                som_url = data_url(annotated)
            except Exception:
                som_url = None

        return context_text, image_url, som_url

    def _plan_next(
        self,
        context_text: str,
        instruction: str,
        image_url: str | None,
        scratchpad: TaskScratchpad,
    ) -> tuple[Intent, int]:
        if hasattr(self.planner, "plan_step"):
            kwargs: dict[str, Any] = {
                "image_url": image_url,
                "scratchpad": scratchpad.summary(),
                "earlier": self.earlier[-EARLIER:],
            }
            if _accepts_kwarg(self.planner.plan_step, "completed_steps"):
                kwargs["completed_steps"] = tuple(scratchpad.completed_steps)
            intent = self.planner.plan_step(context_text, instruction, **kwargs)
            if not hasattr(intent, "action"):
                raise TypeError(f"planner.plan_step returned {type(intent).__name__}, expected Intent")
            return intent, 0

        if hasattr(self.planner, "plan"):
            plan_res = self.planner.plan(
                context_text,
                instruction,
                earlier=tuple(self.earlier[-EARLIER:]),
            )
            write_tokens = int(getattr(getattr(plan_res, "reply", None), "write_tokens", 0) or 0)
            if isinstance(plan_res, Intent) or hasattr(plan_res, "action"):
                return plan_res, write_tokens
            if hasattr(plan_res, "intent"):
                return plan_res.intent, write_tokens
            raise TypeError(f"planner.plan returned {type(plan_res).__name__}, expected Intent or PlanResult")

        return Intent(action="done"), 0

    def _resolve_target(self, robot: Any, target: Any) -> tuple[Any, str, Any]:
        """Resolve an Intent target into (exec_target, canonical_key, matched_object_or_None)."""
        if target is None or isinstance(target, bool):
            return None, "none", None
        if isinstance(target, int):
            if hasattr(robot, "obj"):
                resolved = robot.obj(target)
                return resolved, f"id:{getattr(resolved, 'id', target)}", resolved
            return target, f"id:{target}", None
        if not isinstance(target, str):
            obj_id = getattr(target, "id", None)
            key = f"id:{obj_id}" if obj_id is not None else repr(target)
            return target, key, target if hasattr(target, "position") else None

        s = target.strip().strip("\"'")
        low = s.lower()
        if low not in _USER_ALIASES:
            s = re.sub(r"\s*\([^)]*\)", "", s).strip()
            low = s.lower()
        if not s or low in _NULL_TARGET_STRINGS:
            return None, "none", None
        if low in _USER_ALIASES:
            if hasattr(robot, "user"):
                u = robot.user()
                if u is None:
                    raise NotFound("no user in view")
                return u, "user", u
            return "user", "user", None

        # Parse ID prefixes/formats such as "#7", "id=4", "10  table", or "table #7"
        m_lead = re.match(r"^(?:id\s*[=:]?\s*|#\s*)?(-?\d+)(?:\s+(.*))?$", s, re.IGNORECASE)
        if m_lead:
            obj_id = int(m_lead.group(1))
            rem = (m_lead.group(2) or "").strip(" ()")
            if hasattr(robot, "obj"):
                try:
                    resolved = robot.obj(obj_id)
                    return resolved, f"id:{getattr(resolved, 'id', obj_id)}", resolved
                except NotFound:
                    if not rem:
                        raise
                    s = rem
                    low = s.lower()
            else:
                return obj_id, f"id:{obj_id}", None
        else:
            m_hash_trail = re.match(r"^(.+?)\s+#(-?\d+)$", s)
            if m_hash_trail:
                obj_id = int(m_hash_trail.group(2))
                rem = m_hash_trail.group(1).strip()
                if hasattr(robot, "obj"):
                    try:
                        resolved = robot.obj(obj_id)
                        return resolved, f"id:{getattr(resolved, 'id', obj_id)}", resolved
                    except NotFound:
                        s = rem or s
                        low = s.lower()
                else:
                    return obj_id, f"id:{obj_id}", None

        # Snapshot grounding: match object IDs for dedup/distance checks + synonym fallback
        if hasattr(robot, "snapshot"):
            try:
                snap = robot.snapshot() or {}
            except Exception:
                snap = {}
            objects = snap.get("objects", []) or []
            rooms_low = {str(r).lower() for r in snap.get("rooms", ())}
            if low not in rooms_low:
                # Trailing bare numeric suffix like "table 10" when not a room name
                m_num_trail = re.match(r"^(.+?)\s+(\d+)$", s)
                if m_num_trail and hasattr(robot, "obj"):
                    try:
                        resolved = robot.obj(int(m_num_trail.group(2)))
                        return resolved, f"id:{getattr(resolved, 'id', m_num_trail.group(2))}", resolved
                    except NotFound:
                        pass
                tokens = set(low.split())
                exact = [
                    o
                    for o in objects
                    if low in str(getattr(o, "label", "")).lower()
                    or (
                        tokens
                        and tokens
                        <= ({str(getattr(o, "label", "")).lower(), str(getattr(o, "color", "")).lower()} - {""})
                    )
                ]
                if exact:
                    nearest = min(exact, key=lambda o: float(getattr(o, "distance", 0.0)))
                    key = f"id:{nearest.id}" if hasattr(nearest, "id") else f"name:{low}"
                    return nearest, key, nearest
                syns = _TARGET_SYNONYMS.get(low)
                if syns:
                    for syn in syns:
                        matched = [o for o in objects if syn in str(getattr(o, "label", "")).lower()]
                        if matched:
                            nearest = min(matched, key=lambda o: float(getattr(o, "distance", 0.0)))
                            key = f"id:{nearest.id}" if hasattr(nearest, "id") else f"name:{syn}"
                            return nearest, key, nearest

        return s, f"name:{low}", None

    def _prepare_intent(
        self, robot: Any, intent: Intent
    ) -> tuple[str, Any, str, Any, dict[str, Any]]:
        action = intent.action.lower().strip().replace("-", "_")
        params = dict(intent.params or {})
        raw_target = intent.target

        if action == "move" and raw_target is not None:
            t_low = str(raw_target).strip().lower()
            if t_low not in _NULL_TARGET_STRINGS and t_low not in _DIRECTION_WORDS:
                action = "go_to"

        redirected = False
        if (
            action == "search_frontier"
            and raw_target is not None
            and str(raw_target).strip().lower() not in (_NULL_TARGET_STRINGS | {"frontier", "room", "here", "everywhere"})
        ):
            try:
                cand_target, cand_key, cand_obj = self._resolve_target(robot, raw_target)
                if cand_obj is not None:
                    action = "go_to"
                    exec_target, target_key, target_obj = cand_target, cand_key, cand_obj
                    params.setdefault("distance", 0.5)
                    redirected = True
            except NotFound:
                pass

        if not redirected:
            if action in _OBJECT_TARGET_ACTIONS:
                if action == "pursue" and (
                    raw_target is None or str(raw_target).strip().lower() in _NULL_TARGET_STRINGS
                ):
                    raw_target = "user"
                exec_target, target_key, target_obj = self._resolve_target(robot, raw_target)
            else:
                exec_target = raw_target
                target_key = "none" if raw_target is None else str(raw_target).strip().lower()
                target_obj = None

        if action == "go_to" and "distance" in params and params["distance"] is not None:
            try:
                d_val = float(params["distance"])
                if target_key != "user" and d_val >= 1.0:
                    params["distance"] = 0.5
                elif target_key == "user" and d_val > 1.0:
                    params["distance"] = 1.0
            except (TypeError, ValueError):
                pass

        return action, exec_target, target_key, target_obj, params

    def _format_step_summary(
        self,
        action: str,
        target_key: str,
        target_obj: Any,
        params: dict[str, Any],
        step_num: int | None = None,
    ) -> str:
        if step_num is None:
            pad = getattr(self.short_term, "scratchpad", None)
            step_num = len(pad.completed_steps) + 1 if pad is not None else 1
        t_desc = target_key
        if target_obj is not None and hasattr(target_obj, "id") and hasattr(target_obj, "label"):
            t_desc = f"{target_obj.label}#{target_obj.id}"
        elif t_desc.startswith("name:"):
            t_desc = t_desc[5:]
        params_json = json.dumps(params, sort_keys=True, default=str)
        return f"Step {step_num}: {action} target={t_desc} params={params_json} (completed)"

    def _execute_prepared(
        self, robot: Any, action: str, target: Any, params: dict[str, Any], raw_target: Any = None
    ) -> str:
        if action in ("go_to", "orbit", "peek", "look_under", "face") and target is None:
            raise NotFound(f"{action} requires a target")

        if action == "go_to":
            dist = _num_param(params, ("distance",), 0.0)
            robot.go_to(target, distance=dist)
            return f"drove to {target}"
        if action == "pursue":
            dist = _num_param(params, ("distance",), 1.0)
            sec = params.get("seconds")
            if hasattr(robot, "pursue"):
                robot.pursue(target, distance=dist, seconds=sec)
            elif hasattr(robot, "follow"):
                robot.follow(target, distance=dist, seconds=sec)
            else:
                robot.go_to(target, distance=dist)
            return f"pursued {target}"
        if action == "orbit":
            laps = _num_param(params, ("laps", "turns", "count"), 1.0)
            raw_dir = str(params.get("direction") or "left").lower().strip()
            direction = "right" if raw_dir in ("clockwise", "cw", "right") else "left"
            dist = _num_param(params, ("distance", "radius"), 0.5)
            robot.orbit(target, laps=laps, direction=direction, distance=dist)
            return f"orbited {target}"
        if action == "peek":
            if hasattr(robot, "peek"):
                robot.peek(target)
            return f"peeked at {target}"
        if action == "look_under":
            if hasattr(robot, "look_under"):
                robot.look_under(target)
            return f"looked under {target}"
        if action == "move":
            metres = _num_param(params, ("metres", "meters", "distance"), 1.0)
            default_dir = (
                "backward"
                if raw_target is not None and str(raw_target).strip().lower() in ("backward", "back")
                else "forward"
            )
            direction = str(params.get("direction") or default_dir).lower().strip() or "forward"
            robot.move(metres, direction=direction)
            return f"moved {metres}m {direction}"
        if action == "turn":
            degrees = _num_param(params, ("degrees", "angle"), 90.0)
            direction = str(params.get("direction") or "left").lower().strip() or "left"
            robot.turn(degrees, direction=direction)
            return f"turned {degrees}° {direction}"
        if action == "spin":
            turns = _num_param(params, ("turns", "laps", "count"), 1.0)
            robot.spin(turns=turns)
            return f"spun {turns} turns"
        if action == "face":
            robot.face(target)
            return f"faced {target}"
        if action == "motion_sequence":
            steps = params.get("steps", [])
            if hasattr(robot, "motion_sequence"):
                robot.motion_sequence(steps)
            return "executed motion sequence"
        if action == "ask":
            q = params.get("question", "")
            ans = robot.ask(q)
            return f"asked '{q}' -> '{ans}'"
        if action == "remember":
            fact = params.get("fact", "")
            robot.remember(fact)
            if self.memory is not None and hasattr(self.memory, "remember_fact") and fact:
                self.memory.remember_fact(fact)
            return f"remembered '{fact}'"
        if action == "search_frontier":
            if hasattr(robot, "search"):
                robot.search(str(target or "frontier"), area="here")
            return f"explored frontier for {target}"
        if action == "step_closer":
            if target is not None:
                robot.go_to(target, distance=float(params.get("distance", 0.5)))
            else:
                robot.move(0.5, "forward")
            return f"stepped closer to {target}"
        return f"performed {action}"

    def _execute_intent(self, robot: Any, intent: Intent) -> str:
        action, exec_target, _, _, params = self._prepare_intent(robot, intent)
        return self._execute_prepared(robot, action, exec_target, params, raw_target=intent.target)

    def _escalated_summary(self, robot: Any, again: TaskRecord) -> str:
        last_intent = again.intents[-1] if again.intents else None
        if again.outcome == "done":
            return self._summarize_action(robot, last_intent.target if last_intent else None)
        if again.outcome == "refused":
            reason = (last_intent.say if last_intent else "") or "cannot do it"
            return f"refused: {reason}"
        return "stopped"

    def _is_repeated_step(
        self,
        robot: Any,
        action: str,
        target_obj: Any,
        params: dict[str, Any],
        step_sig: tuple[str, str, tuple[tuple[str, str], ...]],
        succeeded_sigs: list[tuple[str, str, tuple[tuple[str, str], ...]]],
    ) -> bool:
        if action in _IDEMPOTENT_ACTIONS and step_sig in succeeded_sigs:
            return True
        if action == "go_to" and target_obj is not None and succeeded_sigs and hasattr(target_obj, "position"):
            pose = robot.pose() if hasattr(robot, "pose") else None
            if pose is None and hasattr(robot, "snapshot"):
                try:
                    pose = (robot.snapshot() or {}).get("pose")
                except Exception:
                    pose = None
            if pose is not None:
                px, py, _ = _pose_tuple(pose)
                radius = float(getattr(target_obj, "radius", 0.3))
                reach = max(0.0, float(params.get("distance", 0.0)))
                if math.dist((px, py), target_obj.position) <= radius + reach + 0.25:
                    return True
        return False

    def do(self, instruction: str, robot: Any, speak: bool = True) -> TaskRecord:
        record = TaskRecord(instruction)

        # 1. Instant reflex commands act immediately without a model
        cmd = instant(instruction)
        if cmd is not None:
            return self._handle_instant(cmd, robot, record, speak=speak)

        if hasattr(robot, "begin"):
            robot.begin()

        scratchpad = TaskScratchpad(goal=instruction)
        self.short_term.scratchpad = scratchpad
        self._push_live(status="planning", instruction=instruction, problems=[], **_said_fields(robot))

        step_failures = 0
        said_log: list[str] = []
        last_spoken = ""
        succeeded_sigs: list[tuple[str, str, tuple[tuple[str, str], ...]]] = []

        # 2. Closed-Loop Sense-Think-Act Loop
        for step_idx in range(MAX_STEPS):
            scratchpad.steps = step_idx + 1
            context_text, image_url, som_url = self._observe(robot, scratchpad)

            start_t = time.perf_counter()
            try:
                intent, write_tokens = self._plan_next(context_text, instruction, image_url, scratchpad)
                if write_tokens:
                    record.tokens.append(write_tokens)
            except Exception as exc:
                record.problems.append(f"plan: {exc}")
                step_failures += 1
                if step_failures >= 2:
                    break
                continue
            finally:
                record.plan_s.append(round(time.perf_counter() - start_t, 3))

            record.intents.append(intent)
            if intent.subgoal:
                scratchpad.subgoal = intent.subgoal

            action = intent.action.lower().strip().replace("-", "_")
            say_text = intent.say.strip() if intent.say else ""
            say_norm = " ".join(say_text.split())

            # Prepare action and guard against repetition loops BEFORE speaking redundant announcements
            prepared: tuple[str, Any, str, Any, dict[str, Any]] | None = None
            target_obj: Any = None
            step_sig: tuple[str, str, tuple[tuple[str, str], ...]] | None = None
            if action not in _DONE_ACTIONS and action != "refuse":
                try:
                    prepared = self._prepare_intent(robot, intent)
                    exec_action, exec_target, target_key, target_obj, exec_params = prepared
                    step_sig = _canonical_step_sig(exec_action, target_key, exec_params)
                    if step_idx > 0 and self._is_repeated_step(
                        robot, exec_action, target_obj, exec_params, step_sig, succeeded_sigs
                    ):
                        if record.intents:
                            record.intents.pop()
                        pop_fn = getattr(self.planner, "pop_last_intent", None)
                        if callable(pop_fn):
                            pop_fn()
                        last_kept = record.intents[-1] if record.intents else intent
                        summary = self._summarize_action(robot, exec_target)
                        return self._finish(
                            record, "done", summary,
                            task_said=said_log[-1] if said_log else "",
                            subgoal=last_kept.subgoal or scratchpad.subgoal,
                            action="done",
                            posture=last_kept.posture,
                            chime=last_kept.chime,
                            **_said_fields(robot, said_log),
                        )
                except (PathBlocked, NotFound, TargetLost, Timeout) as exc:
                    record.problems.append(f"{type(exc).__name__}: {exc}")
                    step_failures += 1
                    scratchpad.blockers.append(str(exc))
                    if step_failures >= 2:
                        break
                    continue
                except Exception as exc:
                    record.problems.append(f"{type(exc).__name__}: {exc}")
                    step_failures += 1
                    if step_failures >= 2:
                        break
                    continue

            # Speak (deduplicated) and express posture
            if say_norm and say_norm != last_spoken and speak and hasattr(robot, "say"):
                try:
                    robot.say(say_text)
                    said_log.append(say_text)
                    last_spoken = say_norm
                except Exception:
                    pass
            if hasattr(robot, "express") and (intent.posture != "neutral" or intent.chime):
                try:
                    robot.express(intent.posture, chime_kind=intent.chime, seconds=0.2)
                except Exception:
                    pass

            live_step: dict[str, Any] = {
                "status": "running",
                "instruction": instruction,
                "subgoal": intent.subgoal or scratchpad.subgoal,
                "action": intent.action,
                "posture": intent.posture,
                "chime": intent.chime,
                "plan": intent.subgoal or intent.action,
                "say_first": intent.say,
                "problems": list(record.problems),
                **_said_fields(robot, said_log),
            }
            if som_url is not None:
                live_step["set_of_mark"] = som_url
            self._push_live(**live_step)

            if action in _DONE_ACTIONS:
                summary = (
                    self._summarize_action(robot, intent.target)
                    if succeeded_sigs
                    else (f"replied: {say_text[:80]}" if say_text else "done")
                )
                return self._finish(
                    record, "done", summary,
                    task_said=said_log[-1] if said_log else "",
                    subgoal=intent.subgoal or scratchpad.subgoal,
                    action="done",
                    posture=intent.posture,
                    chime=intent.chime,
                    **_said_fields(robot, said_log),
                )

            if action == "refuse":
                return self._finish(
                    record, "refused", f"refused: {intent.say or 'cannot do it'}",
                    task_said=said_log[-1] if said_log else "",
                    subgoal=intent.subgoal or scratchpad.subgoal,
                    action="refuse",
                    posture=intent.posture,
                    chime=intent.chime,
                    **_said_fields(robot, said_log),
                )

            # Execute prepared action
            assert prepared is not None and step_sig is not None
            exec_action, exec_target, target_key, target_obj, exec_params = prepared
            try:
                self._execute_prepared(
                    robot, exec_action, exec_target, exec_params, raw_target=intent.target
                )
                succeeded_sigs.append(step_sig)
                step_num = len(scratchpad.completed_steps) + 1
                scratchpad.record_step(
                    self._format_step_summary(exec_action, target_key, target_obj, exec_params, step_num=step_num)
                )
                step_failures = 0
            except (PathBlocked, NotFound, TargetLost, Timeout) as exc:
                record.problems.append(f"{type(exc).__name__}: {exc}")
                step_failures += 1
                scratchpad.blockers.append(str(exc))
                if step_failures >= 2:
                    break
                continue
            except Preempted:
                return self._finish(
                    record, "preempted", "stopped",
                    task_said=said_log[-1] if said_log else "",
                    action="stop",
                    **_said_fields(robot, said_log),
                )
            except SafetyStop as exc:
                record.problems.append(f"SafetyStop: {exc}")
                break
            except Exception as exc:
                record.problems.append(f"{type(exc).__name__}: {exc}")
                step_failures += 1
                if step_failures >= 2:
                    break
                continue

            # Update scratchpad
            if target_obj is not None and hasattr(target_obj, "id"):
                scratchpad.inspected_ids.add(target_obj.id)
            elif intent.target is not None:
                if not isinstance(intent.target, bool) and isinstance(intent.target, int):
                    scratchpad.inspected_ids.add(intent.target)
                elif hasattr(intent.target, "id"):
                    scratchpad.inspected_ids.add(intent.target.id)

            if self.memory is not None and hasattr(self.memory, "observe_clear_space"):
                try:
                    pose = robot.pose()
                    self.memory.observe_clear_space(pose.x, pose.y, radius=1.0)
                except Exception:
                    pass

            if (
                step_idx == 0
                and _is_local_planner(self.planner)
                and exec_action in _SINGLE_STEP_PHYSICAL_ACTIONS
                and _is_simple_single_step(instruction)
            ):
                summary = self._summarize_action(robot, exec_target)
                return self._finish(
                    record, "done", summary,
                    task_said=said_log[-1] if said_log else "",
                    subgoal=intent.subgoal or scratchpad.subgoal,
                    action="done",
                    posture=intent.posture,
                    chime=intent.chime,
                    **_said_fields(robot, said_log),
                )
        else:
            if not record.problems:
                record.problems.append("could not finish within step limit")

        # 3. Escalation: if local planner failed twice, optional stronger planner gets one attempt
        if self.escalate is not None:
            again = Agent(self.escalate, live=self.live, brain=self.brain).do(instruction, robot, speak=False)
            record.intents += again.intents
            record.problems += again.problems
            record.plan_s += again.plan_s
            record.tokens += again.tokens
            if again.outcome in ("done", "refused", "preempted"):
                summary = self._escalated_summary(robot, again)
                return self._finish(
                    record, again.outcome, summary,
                    task_said=said_log[-1] if said_log else "",
                    **_said_fields(robot, said_log),
                )

        problem_text = plainly(record.problems)
        apology = f"Sorry, {problem_text}. Could you tell me another way?"
        if speak and hasattr(robot, "say"):
            try:
                robot.say(apology)
                said_log.append(apology)
            except Exception:
                pass
        return self._finish(
            record, "failed", f"could not do it: {problem_text}",
            task_said=said_log[-1] if said_log else "",
            **_said_fields(robot, said_log),
        )
