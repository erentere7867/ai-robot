"""Behavior arbiter: owns turns (user or background), expression, memory hooks, and cancellation."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections import deque
from typing import Any

from robot.behavior.emotion import EmotionEngine
from robot.behavior.fsm import InteractionFSM, InteractionState
from robot.cognition.agent import AgentLoop, TurnResult
from robot.cognition.prompts import PromptBuilder
from robot.core.bus import EventBus
from robot.core.clock import RealClock
from robot.core.config import AgentConfig
from robot.core.events import Event, EventPriority
from robot.memory.consolidator import MemoryConsolidator
from robot.memory.policy import extract_explicit_memory_proposals
from robot.memory.store import MemoryStore
from robot.skills.base import SkillContext
from robot.world.spatial_map import ASKED_PLACE
from robot.world.state import WorldState

logger = logging.getLogger(__name__)

_EMPTY_REPLIES = {"", ".", "...", "(empty)", "(no reply)", "empty", "none", "(silence)", "\"\""}


# Said on its own, these halt all motion at once, before the language model has even read the message.
STOP_WORDS = re.compile(r"^\W*(?:(?:please|hey|ok|okay|no)\W+)?(?:stop|halt|freeze|stay|wait|don'?t move)"
                        r"(?:\W+(?:it|now|there|moving|right now|please))*\W*$", re.I)


class BehaviorArbiter:
    """One interruptible turn at a time; newer human input replaces older work."""

    def __init__(
        self,
        bus: EventBus,
        fsm: InteractionFSM,
        emotion: EmotionEngine,
        prompts: PromptBuilder,
        agent: AgentLoop,
        skill_context: SkillContext,
        memory: MemoryStore,
        world: WorldState,
        consolidator: MemoryConsolidator | None = None,
        agent_config: AgentConfig | None = None,
    ) -> None:
        self.bus = bus
        self.fsm = fsm
        self.emotion = emotion
        self.prompts = prompts
        self.agent = agent
        self.ctx = skill_context
        self.memory = memory
        self.world = world
        self.consolidator = consolidator
        self.agent_config = agent_config or AgentConfig()
        self.autonomy = None
        self.drives = None
        self.active_goal: dict[str, Any] | None = None
        self.recent_activity: deque[tuple[float, str]] = deque(maxlen=6)
        self._sub_ids: list[str] = []
        self._speech_correlation_id: str | None = None
        self._active_turn: asyncio.Task | None = None
        self._epoch = 0
        self._turn_lock = asyncio.Lock()
        self._input_task: asyncio.Task | None = None
        self._input_correlation_id: str | None = None

    # ---- lifecycle ---------------------------------------------------------------------------------
    def bind(self) -> None:
        subs = [
            ("attention.engaged", self._on_attention_engaged, True),
            ("expression.manual", self._on_manual_expression, False),
            ("reflex.triggered", self._on_reflex_triggered, False),
            ("audio.barge_in", self._on_barge_in, True),
            ("action.cancel", self._on_cancel, True),
            ("audio.speech_completed", self._on_speech_completed, False),
            ("behavior.finished", self._on_behavior_finished, False),
        ]
        for event_type, handler, immediate in subs:
            self._sub_ids.append(self.bus.subscribe(event_type, handler, name=f"arbiter:{event_type}",
                                                    immediate=immediate))

    def unbind(self) -> None:
        self.cancel_active()
        if self._input_task:
            self._input_task.cancel()
        for sid in self._sub_ids:
            self.bus.unsubscribe(sid)
        self._sub_ids.clear()

    def cancel_active(self) -> None:
        """Invalidate the running turn before anything acknowledges the cancellation."""
        self._epoch += 1
        self._speech_correlation_id = None
        if self.active_goal and self.active_goal.get("status") == "running":
            self.active_goal.update(status="cancelled")
        if self._active_turn and not self._active_turn.done():
            self._active_turn.cancel()
        for event_type in ("attention.engaged", "action.request", "robot.response"):
            self.bus.discard_pending(event_type)

    @property
    def busy(self) -> bool:
        return bool((self._active_turn and not self._active_turn.done())
                    or (self._input_task and not self._input_task.done()))

    async def _on_cancel(self, event: Event) -> None:
        reason = event.payload.get("reason")
        if reason in ("new_turn", "autonomous_start"):
            return
        self.cancel_active()
        if self._input_task and not self._input_task.done():
            self._input_task.cancel()
            await self.bus.emit("turn.cancelled", "arbiter", {}, correlation_id=self._input_correlation_id)
        barge_in = reason == "user_barge_in"
        self.fsm.transition(InteractionState.ATTENDING if barge_in else InteractionState.IDLE)
        snap = self.emotion.set_activity("listening" if barge_in else "idle")
        await self.bus.emit("expression.updated", "arbiter", snap.to_dict(), priority=EventPriority.CRITICAL)

    async def step_idle_timers(self) -> str | None:
        self.emotion.step_decay()
        if self._active_turn and not self._active_turn.done():
            self.fsm.touch()
            return None
        transition_event = self.fsm.check_timers()
        if self.consolidator is not None and self.consolidator.pending_count > 0 \
                and self.fsm.state == InteractionState.IDLE:
            await self.consolidator.consolidate()
        if transition_event == "became_sleepy":
            snap = self.emotion.set_emotion("sleepy", 0.7, gaze_target="down")
            self.emotion.set_activity("idle")
            await self.bus.emit("expression.updated", "arbiter", {**snap.to_dict(), "reason": "idle_timeout_sleepy"},
                                priority=EventPriority.LOW)
        return transition_event

    # ---- turns -------------------------------------------------------------------------------------
    async def handle_user_turn(self, user_text: str, *, correlation_id: str | None = None,
                               autonomous: bool = False, hint: str = "") -> TurnResult:
        if not user_text.strip() or len(user_text) > 4096:
            raise ValueError("Message must contain 1..4096 characters")
        correlation_id = correlation_id or uuid.uuid4().hex
        async with self._turn_lock:
            previous = self._active_turn
            if autonomous and self.busy:
                return TurnResult(say="", status="blocked", backend_name="busy")
            if previous and not previous.done():
                self.cancel_active()
                done, _ = await asyncio.wait({previous}, timeout=1.0)
                if not done:
                    raise RuntimeError("Previous turn is still stopping; try again")
            await self.bus.emit("action.cancel", "arbiter",
                                {"reason": "autonomous_start" if autonomous else "new_turn"},
                                priority=EventPriority.CRITICAL)
            epoch = self._epoch
            task = asyncio.create_task(
                self._run_turn(user_text, correlation_id=correlation_id, epoch=epoch, autonomous=autonomous,
                               hint=hint), name="cognition:turn")
            self._active_turn = task
            self.bus.track_task(task)
        try:
            return await task
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise
            await self.bus.emit("turn.cancelled", "arbiter", {}, correlation_id=correlation_id)
            return TurnResult(say="", status="cancelled", backend_name="cancelled")
        except Exception:
            logger.exception("Turn failed")
            await self.ctx.executor.cancel_all()
            if self.active_goal:
                self.active_goal.update(status="error")
            self.fsm.transition(InteractionState.IDLE)
            self.emotion.set_activity("idle")
            result = TurnResult(say="" if autonomous else "Something went wrong in my head, so I stopped.",
                                status="error", backend_name="error")
            await self._emit_response(result, correlation_id, autonomous)
            return result
        finally:
            if self._active_turn is task:
                self._active_turn = None

    async def wait_until_done(self) -> None:
        for task in (self._input_task, self._active_turn):
            if task is not None:
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    if asyncio.current_task().cancelling():
                        raise
                except Exception:
                    pass

    def _recent_lines(self) -> list[str]:
        now = self.bus.clock.monotonic()
        lines = []
        for ts, text in list(self.recent_activity)[-3:]:
            age = now - ts
            when = "just now" if age < 60 else f"{age / 60:.0f} min ago"
            lines.append(f"{text} ({when})")
        return lines

    async def _run_turn(self, text: str, *, correlation_id: str, epoch: int, autonomous: bool,
                        hint: str) -> TurnResult:
        self.active_goal = {"objective": text[:200], "status": "running", "step": 0,
                            "correlation_id": correlation_id, "origin": "autonomous" if autonomous else "user"}
        self.world.task_state = dict(self.active_goal)
        self.fsm.transition(InteractionState.ATTENDING)
        self.emotion.clear_manual_override()
        if self.emotion.snapshot().emotion == "sleepy":
            self.emotion.set_emotion("neutral", 0.3, gaze_target="speaker")
        snap = self.emotion.set_activity("listening", gaze_target="speaker")
        await self.bus.emit("expression.updated", "arbiter",
                            {**snap.to_dict(), "sound": "none" if autonomous else "ack_chirp"},
                            priority=EventPriority.HIGH, correlation_id=correlation_id)

        memories: list[str] = []
        if not autonomous:
            for prop in extract_explicit_memory_proposals(text):
                outcome = self.memory.apply(prop)
                if prop.op == "forget":
                    self.prompts.clear()
                    if self.consolidator:
                        self.consolidator.clear_pending()
                await self.bus.emit("memory.updated", "arbiter", outcome, correlation_id=correlation_id)
            found = self.memory.search(text, limit=4)
            memories = [m.content for m in found]
            await self.bus.emit("memory.retrieved", "arbiter",
                                {"query": text, "count": len(found), "memories": [m.to_dict() for m in found]},
                                correlation_id=correlation_id)
        core = [m.content for m in self.memory.core_facts(limit=5)] if hasattr(self.memory, "core_facts") else []
        memories = [m for m in memories if m not in core]
        pose = self.ctx.pose()
        if not autonomous:
            self.ctx.spatial_map.places[ASKED_PLACE] = (pose.x_mm, pose.y_mm)
        battery = getattr(self.ctx.body.sensors(), "battery_pct", None)
        situation = self.prompts.situation(
            world=self.ctx.spatial_map.render_for_prompt(pose), core_facts=core, memories=memories,
            recent=self._recent_lines(), battery_pct=battery, extra=hint)
        messages = self.prompts.build(text, situation, background=autonomous)

        self.fsm.transition(InteractionState.THINKING)
        snap = self.emotion.set_activity("thinking", gaze_target="speaker")
        await self.bus.emit("expression.updated", "arbiter", snap.to_dict(), correlation_id=correlation_id)

        ctx = self.ctx
        ctx.correlation_id = correlation_id
        ctx.cancelled = lambda: epoch != self._epoch
        ctx.attached_image = None
        ctx.wake_request_s = None
        ctx.expression = {}
        ctx.turn_cache = {}
        result = await self.agent.run(messages, ctx, background=autonomous)
        if epoch != self._epoch:
            raise asyncio.CancelledError
        if autonomous and (result.say.strip().lower() in _EMPTY_REPLIES
                           or (self.autonomy is not None and not self.autonomy.allow_speech())):
            result.say = ""

        self.active_goal.update(status=result.status, step=result.steps)
        self.world.task_state = dict(self.active_goal)
        notes = result.action_notes()
        if result.tool_log:
            prefix = "on my own: " if autonomous else ""
            self.recent_activity.append((self.bus.clock.monotonic(), prefix + "; ".join(notes[-3:])))
        if not autonomous:
            self.prompts.add_exchange(text, result.say, result.transcript)
            if self.consolidator is not None:
                self.consolidator.enqueue_turn(text, result.say)
        if not autonomous or result.say or result.tool_log:
            self.memory.log_episode(
                summary=(f"On my own: {'; '.join(notes[-3:])}" if autonomous
                         else f"User: {text[:80]} -> Robot: {result.say[:80]}"),
                kind="autonomous" if autonomous else "conversation",
                details={"tools": [e["tool"] for e in result.tool_log], "status": result.status},
            )
        if self.drives is not None:
            self.drives.on_turn(result, autonomous=autonomous)
        await self._emit_response(result, correlation_id, autonomous)
        return result

    async def _emit_response(self, result: TurnResult, correlation_id: str | None, autonomous: bool) -> None:
        emotion = result.expression.get("emotion") or (
            "happy" if result.status == "completed" and result.tool_log
            else "confused" if result.status in ("blocked", "error") else "neutral")
        intensity = 0.7 if result.expression.get("emotion") else 0.45
        sound = result.expression.get("sound", "none")
        gaze = "speaker" if not autonomous else "center"
        self.emotion.set_emotion(emotion, intensity, gaze_target=gaze)
        self.fsm.transition(InteractionState.SPEAKING)
        snap = self.emotion.set_activity("speaking" if result.say else "idle", gaze_target=gaze)
        await self.bus.emit("expression.updated", "arbiter", {**snap.to_dict(), "sound": sound},
                            correlation_id=correlation_id)
        self._speech_correlation_id = correlation_id
        await self.bus.emit("robot.response", "arbiter", {
            "say": result.say, "sound": sound, "emotion": {"name": emotion, "intensity": intensity},
            "actions": [{"tool": e["tool"], "args": e["args"], "ok": e["ok"]} for e in result.tool_log],
            "tools": result.tool_log, "status": result.status, "steps": result.steps,
            "backend": result.backend_name, "latency_ms": round(result.latency_ms, 1),
            "next_wake_s": result.next_wake_s, "final": True,
            "origin": "autonomous" if autonomous else "user", "goal_state": dict(self.active_goal or {}),
        }, priority=EventPriority.HIGH, correlation_id=correlation_id)
        if not isinstance(self.bus.clock, RealClock) or not result.say:
            self.fsm.transition(InteractionState.FOLLOW_UP)

    # ---- events ------------------------------------------------------------------------------------
    async def _on_attention_engaged(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        if not text:
            return
        if STOP_WORDS.match(text):
            # Reflex: halt now. The model still gets the message, to acknowledge it.
            await self.bus.emit("action.cancel", "arbiter", {"reason": "user_stop"},
                                priority=EventPriority.CRITICAL)
        if self._input_task and not self._input_task.done():
            self._input_task.cancel()
            await self.bus.emit("turn.cancelled", "arbiter", {}, correlation_id=self._input_correlation_id)
        self._input_correlation_id = event.correlation_id or event.id
        self._input_task = asyncio.create_task(self._dispatch_input(text, self._input_correlation_id),
                                               name="input:turn")
        self.bus.track_task(self._input_task)

    async def _dispatch_input(self, text: str, correlation_id: str) -> None:
        try:
            await self.handle_user_turn(text, correlation_id=correlation_id)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Input dispatch failed")
            await self.bus.emit("turn.cancelled", "arbiter", {}, correlation_id=correlation_id)

    async def _on_behavior_finished(self, event: Event) -> None:
        """A behaviour that outlives its turn (following) ended on its own: let the robot react to it."""
        p = event.payload
        if not p.get("persistent") or p.get("ok") or "Stopped" in str(p.get("message", "")) or self.busy:
            return
        task = asyncio.create_task(self.handle_user_turn(str(p.get("message", ""))[:200] + ".", autonomous=True),
                                   name="arbiter:behavior_event")
        self.bus.track_task(task)

    async def _on_manual_expression(self, event: Event) -> None:
        gaze = event.payload.get("gaze_target")
        snap = self.emotion.set_emotion(str(event.payload.get("emotion", "neutral")),
                                        float(event.payload.get("intensity", 0.8)),
                                        gaze_target=str(gaze) if gaze else None, manual_override=True)
        await self.bus.emit("expression.updated", "arbiter", {**snap.to_dict(), "reason": "manual_override"},
                            priority=EventPriority.HIGH, correlation_id=event.id)

    async def _on_reflex_triggered(self, event: Event) -> None:
        snap = self.emotion.set_emotion("surprised", 0.9, gaze_target="down")
        await self.bus.emit("expression.updated", "arbiter", {
            **snap.to_dict(), "sound": "startle_pip", "reason": f"reflex:{event.payload.get('reflex', 'unknown')}",
        }, priority=EventPriority.CRITICAL, correlation_id=event.id)

    async def _on_barge_in(self, event: Event) -> None:
        self.cancel_active()
        self.fsm.transition(InteractionState.ATTENDING)
        snap = self.emotion.set_activity("listening", gaze_target="speaker")
        await self.bus.emit("expression.updated", "arbiter", {**snap.to_dict(), "reason": "user_barge_in"},
                            priority=EventPriority.CRITICAL, correlation_id=event.id)

    async def _on_speech_completed(self, event: Event) -> None:
        if event.correlation_id != self._speech_correlation_id:
            return
        if self.fsm.state == InteractionState.SPEAKING:
            self.fsm.transition(InteractionState.FOLLOW_UP)
        if self.emotion.snapshot().activity == "speaking":
            snap = self.emotion.set_activity("idle")
            await self.bus.emit("expression.updated", "arbiter", snap.to_dict(), priority=EventPriority.LOW,
                                correlation_id=event.correlation_id)
