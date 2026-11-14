"""The closed-loop multimodal agent: Sense-Think-Act loop with instant reflex, local Intent planning,

active vision, and escalation.
"""

from __future__ import annotations

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
from robot.agent.instant import instant
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

    def _remember(self, instruction: str, what: str) -> None:
        self.earlier = [*self.earlier, f'"{instruction}": {what[:240]}'][-EARLIER:]
        self.short_term.add_turn(instruction, what)

    def _push_live(self, **fields: Any) -> None:
        if self.live is not None and hasattr(self.live, "update"):
            try:
                self.live.update(**fields)
            except Exception:
                pass

    def _finish(self, record: TaskRecord, outcome: str, summary: str, **live_fields: Any) -> TaskRecord:
        record.outcome = outcome
        self._remember(record.instruction, f"{summary} ({outcome})")
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

    def _handle_instant(self, cmd: str, robot: Any, record: TaskRecord) -> TaskRecord:
        if cmd == "stop":
            robot.stop()
            if hasattr(robot, "express"):
                robot.express("neutral", chime_kind="uh_oh", seconds=0.2)
            return self._finish(
                record, "preempted", "stopped",
                action="stop", subgoal="stop", posture="neutral", chime="uh_oh",
            )
        if cmd == "wait":
            robot.stop()
            if hasattr(robot, "express"):
                robot.express("listening", chime_kind="listen", seconds=0.2)
            return self._finish(
                record, "done", "waited",
                action="wait", subgoal="wait", posture="listening", chime="listen",
            )
        # cmd == "come back"
        user = robot.user() if hasattr(robot, "user") else None
        if hasattr(robot, "express"):
            robot.express("listening", chime_kind="listen", seconds=0.3)
        if user is not None:
            if hasattr(robot, "pursue"):
                robot.pursue(user, distance=1.0)
            elif hasattr(robot, "go_to"):
                robot.go_to(user, distance=1.0)
        elif hasattr(robot, "go_to"):
            robot.go_to("start")
        return self._finish(
            record, "done", "came back to user",
            action="pursue", subgoal="come back", posture="listening", chime="listen",
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

        context_text = describe(snapshot, memory=self.memory, scratchpad=scratchpad)

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
            intent = self.planner.plan_step(
                context_text,
                instruction,
                image_url=image_url,
                scratchpad=scratchpad.summary(),
                earlier=self.earlier[-EARLIER:],
            )
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

    def _execute_intent(self, robot: Any, intent: Intent) -> str:
        action = intent.action.lower().strip()
        target = intent.target
        params = intent.params or {}

        if (
            action in _OBJECT_TARGET_ACTIONS
            and not isinstance(target, bool)
            and isinstance(target, int)
            and hasattr(robot, "obj")
        ):
            target = robot.obj(target)

        if action == "go_to":
            dist = float(params.get("distance", 0.0))
            robot.go_to(target, distance=dist)
            return f"drove to {target}"
        if action == "pursue":
            dist = float(params.get("distance", 1.0))
            sec = params.get("seconds")
            if hasattr(robot, "pursue"):
                robot.pursue(target, distance=dist, seconds=sec)
            elif hasattr(robot, "follow"):
                robot.follow(target, distance=dist, seconds=sec)
            else:
                robot.go_to(target, distance=dist)
            return f"pursued {target}"
        if action == "orbit":
            laps = float(params.get("laps", 1.0))
            direction = params.get("direction", "left")
            dist = float(params.get("distance", 0.5))
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
            metres = float(params.get("metres", 1.0))
            direction = params.get("direction", "forward")
            robot.move(metres, direction=direction)
            return f"moved {metres}m {direction}"
        if action == "turn":
            degrees = float(params.get("degrees", 90.0))
            direction = params.get("direction", "left")
            robot.turn(degrees, direction=direction)
            return f"turned {degrees}° {direction}"
        if action == "spin":
            turns = float(params.get("turns", 1.0))
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
                robot.go_to(target, distance=0.5)
            else:
                robot.move(0.5, "forward")
            return f"stepped closer to {target}"
        return f"performed {action}"

    def _escalated_summary(self, robot: Any, again: TaskRecord) -> str:
        last_intent = again.intents[-1] if again.intents else None
        if again.outcome == "done":
            return self._summarize_action(robot, last_intent.target if last_intent else None)
        if again.outcome == "refused":
            reason = (last_intent.say if last_intent else "") or "cannot do it"
            return f"refused: {reason}"
        return "stopped"

    def do(self, instruction: str, robot: Any, speak: bool = True) -> TaskRecord:
        record = TaskRecord(instruction)

        # 1. Instant reflex commands act immediately without a model
        cmd = instant(instruction)
        if cmd is not None:
            return self._handle_instant(cmd, robot, record)

        if hasattr(robot, "begin"):
            robot.begin()

        scratchpad = TaskScratchpad(goal=instruction)
        self.short_term.scratchpad = scratchpad
        self._push_live(status="planning", instruction=instruction, problems=[])

        step_failures = 0
        said_log: list[str] = []

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

            # Speak and express posture
            if intent.say and speak and hasattr(robot, "say"):
                try:
                    robot.say(intent.say)
                    said_log.append(intent.say)
                except Exception:
                    pass
            if hasattr(robot, "express") and (intent.posture != "neutral" or intent.chime):
                try:
                    robot.express(intent.posture, chime_kind=intent.chime, seconds=0.2)
                except Exception:
                    pass

            action = intent.action.lower().strip()
            live_step: dict[str, Any] = {
                "status": "running",
                "instruction": instruction,
                "subgoal": intent.subgoal or scratchpad.subgoal,
                "action": intent.action,
                "posture": intent.posture,
                "chime": intent.chime,
                "said": list(said_log[-6:]),
                "plan": intent.subgoal or intent.action,
                "say_first": intent.say,
                "problems": list(record.problems),
            }
            if som_url is not None:
                live_step["set_of_mark"] = som_url
            self._push_live(**live_step)

            if action == "done":
                summary = self._summarize_action(robot, intent.target)
                return self._finish(
                    record, "done", summary,
                    subgoal=intent.subgoal or scratchpad.subgoal,
                    action="done",
                    posture=intent.posture,
                    chime=intent.chime,
                    said=list(said_log[-6:]),
                )

            if action == "refuse":
                return self._finish(
                    record, "refused", f"refused: {intent.say or 'cannot do it'}",
                    subgoal=intent.subgoal or scratchpad.subgoal,
                    action="refuse",
                    posture=intent.posture,
                    chime=intent.chime,
                    said=list(said_log[-6:]),
                )

            # Execute action
            try:
                self._execute_intent(robot, intent)
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
                    action="stop", said=list(said_log[-6:]),
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
            if intent.target is not None:
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
                    said=list(said_log[-6:]),
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
            said=list(said_log[-6:]),
        )
