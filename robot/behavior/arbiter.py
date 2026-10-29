"""Behavior arbiter: coordinates attention, FSM, cognition, expressions, actions, and memory."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from robot.actions.executor import ActionExecutor
from robot.actions.registry import CapabilityRegistry
from robot.behavior.emotion import EmotionEngine
from robot.behavior.fsm import InteractionFSM, InteractionState
from robot.cognition.backend import InferenceBackend
from robot.cognition.context import ContextBuilder
from robot.cognition.schema import ValidatedResponse, build_response_json_schema
from robot.cognition.validator import ResponseValidator
from robot.core.bus import EventBus
from robot.core.clock import RealClock
from robot.core.config import AgentConfig
from robot.core.events import Event, EventPriority
from robot.memory.consolidator import MemoryConsolidator
from robot.memory.policy import extract_explicit_memory_proposals
from robot.memory.store import MemoryStore
from robot.world.state import WorldState

logger = logging.getLogger(__name__)

class BehaviorArbiter:
    """Central coordinator executing the perception -> memory -> cognition -> action loop."""

    def __init__(
        self,
        bus: EventBus,
        fsm: InteractionFSM,
        emotion: EmotionEngine,
        context_builder: ContextBuilder,
        backend: InferenceBackend,
        validator: ResponseValidator,
        registry: CapabilityRegistry,
        executor: ActionExecutor,
        memory: MemoryStore,
        world: WorldState,
        consolidator: MemoryConsolidator | None = None,
        agent_config: AgentConfig | None = None,
    ) -> None:
        self.bus = bus
        self.fsm = fsm
        self.emotion = emotion
        self.context_builder = context_builder
        self.backend = backend
        self.validator = validator
        self.registry = registry
        self.executor = executor
        self.memory = memory
        self.world = world
        self.consolidator = consolidator
        self.agent_config = agent_config or AgentConfig()
        self.active_goal: dict[str, Any] | None = None
        self._last_feedback: dict[str, Any] = {}
        self.autonomy = None
        self._autonomous = False
        self._actions_used = 0
        self._goal_started_at = 0.0
        self._prior_turns = []
        self._tool_exchanges: list[list[dict]] = []
        self._requested_wake_s: float | None = None
        self._initial_request_text: str | None = None
        self._pending_camera = None
        self._vision_requested = False
        self._schema = build_response_json_schema(
            allowed_emotions=self.registry.allowed_emotions,
            allowed_sounds=self.registry.allowed_sounds,
            allowed_tools=self.registry.tool_names,
            action_schemas=self.registry.action_schemas(),
        )
        self._sub_ids: list[str] = []
        self._speech_correlation_id: str | None = None
        self._active_turn: asyncio.Task | None = None
        self._epoch = 0
        self._turn_lock = asyncio.Lock()
        self._input_task: asyncio.Task | None = None
        self._input_correlation_id: str | None = None

    def bind(self) -> None:
        self._sub_ids.append(
            self.bus.subscribe(
                "attention.engaged", self._on_attention_engaged, name="arbiter:engaged", immediate=True
            )
        )
        self._sub_ids.append(
            self.bus.subscribe("expression.manual", self._on_manual_expression, name="arbiter:manual_expr")
        )
        self._sub_ids.append(
            self.bus.subscribe("reflex.triggered", self._on_reflex_triggered, name="arbiter:reflex")
        )
        self._sub_ids.append(
            self.bus.subscribe("audio.barge_in", self._on_barge_in, name="arbiter:barge_in", immediate=True)
        )
        self._sub_ids.append(
            self.bus.subscribe("action.cancel", self._on_cancel, name="arbiter:cancel", immediate=True)
        )
        self._sub_ids.append(
            self.bus.subscribe(
                "audio.speech_completed", self._on_speech_completed, name="arbiter:speech_done"
            )
        )

    def unbind(self) -> None:
        self.cancel_active()
        if self._input_task:
            self._input_task.cancel()
        for sid in self._sub_ids:
            self.bus.unsubscribe(sid)
        self._sub_ids.clear()

    def cancel_active(self) -> None:
        """Invalidate the entire plan before any cancellation acknowledgement."""
        self._epoch += 1
        self._speech_correlation_id = None
        if self.active_goal and self.active_goal.get("status") == "continue":
            self.active_goal.update(status="cancelled", phase="finished")
        if self._active_turn and not self._active_turn.done():
            self._active_turn.cancel()
        for event_type in ("attention.engaged", "action.request", "robot.response"):
            self.bus.discard_pending(event_type)

    async def _on_cancel(self, event: Event) -> None:
        self.cancel_active()
        if event.payload.get("reason") != "new_turn" and self._input_task and not self._input_task.done():
            self._input_task.cancel()
            await self.bus.emit("turn.cancelled", "arbiter", {}, correlation_id=self._input_correlation_id)
        barge_in = event.payload.get("reason") == "user_barge_in"
        self.fsm.transition(InteractionState.ATTENDING if barge_in else InteractionState.IDLE)
        snap = self.emotion.set_activity("listening" if barge_in else "idle")
        await self.bus.emit("expression.updated", "arbiter", snap.to_dict(), priority=EventPriority.CRITICAL)

    def _check_turn(self, epoch: int) -> None:
        if epoch != self._epoch:
            raise asyncio.CancelledError

    async def step_idle_timers(self) -> str | None:
        """Check FSM timers, emotion decay, and idle memory consolidation."""
        self.emotion.step_decay()
        if self._active_turn and not self._active_turn.done():
            self.fsm.touch()
            return None
        transition_event = self.fsm.check_timers()
        if (
            self.consolidator is not None
            and self.consolidator.pending_count > 0
            and self.fsm.state == InteractionState.IDLE
        ):
            await self.consolidator.consolidate()

        if transition_event == "became_sleepy":
            snap = self.emotion.set_emotion("sleepy", 0.7, gaze_target="down")
            self.emotion.set_activity("idle")
            await self.bus.emit(
                "expression.updated",
                source="arbiter",
                payload={**snap.to_dict(), "reason": "idle_timeout_sleepy"},
                priority=EventPriority.LOW,
            )
        return transition_event

    async def handle_user_turn(
        self,
        user_text: str,
        *,
        correlation_id: str | None = None,
        autonomous: bool = False,
    ) -> ValidatedResponse:
        """Own one interruptible turn; newer turns replace older work."""
        if not user_text.strip() or len(user_text) > 4096:
            raise ValueError("Message must contain 1..4096 characters")
        self.context_builder.validate_user_text(user_text)
        correlation_id = correlation_id or uuid.uuid4().hex
        async with self._turn_lock:
            previous = self._active_turn
            if autonomous and ((previous and not previous.done()) or (self._input_task and not self._input_task.done())):
                return ValidatedResponse(say="", status="blocked", backend_name="busy")
            if previous and not previous.done():
                self.cancel_active()
                done, _ = await asyncio.wait({previous}, timeout=0.5)
                if not done:
                    raise RuntimeError("Previous inference is still stopping; try again")
            await self.bus.emit(
                "action.cancel", "arbiter", {"reason": "autonomous_start" if autonomous else "new_turn"},
                priority=EventPriority.CRITICAL
            )
            epoch = self._epoch
            task = asyncio.create_task(
                self._run_turn(user_text, correlation_id=correlation_id, epoch=epoch,
                               autonomous=autonomous), name="cognition:turn"
            )
            self._active_turn = task
            self.bus.track_task(task)
        try:
            return await task
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise
            await self.bus.emit("turn.cancelled", "arbiter", {}, correlation_id=correlation_id)
            return ValidatedResponse(say="", backend_name="cancelled", fallback_used=True)
        except Exception:
            logger.exception("Cognitive turn failed")
            await self.executor.cancel_all()
            if self.active_goal:
                self.active_goal.update(status="blocked", message="Control loop error")
            self.fsm.transition(InteractionState.IDLE)
            self.emotion.set_activity("idle")
            response = ValidatedResponse(
                say="" if autonomous else "My control loop hit an error and stopped this goal.", backend_name="error",
                fallback_used=True, status="blocked",
            )
            await self.bus.emit(
                "robot.response", "arbiter", response.to_dict(), correlation_id=correlation_id
            )
            return response
        finally:
            if self._active_turn is task:
                self._active_turn = None

    async def wait_until_done(self) -> None:
        task = self._active_turn
        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise

    async def _run_turn(self, user_text: str, *, correlation_id: str | None, epoch: int,
                        autonomous: bool = False) -> ValidatedResponse:
        self._autonomous = autonomous
        self._actions_used = 0
        self._goal_started_at = self.bus.clock.monotonic()
        self._prior_turns = list(self.context_builder._recent_turns)
        self._tool_exchanges = []
        self._requested_wake_s = None
        self._initial_request_text = None
        self._pending_camera = None
        self._vision_requested = False
        self.active_goal = {"objective": user_text, "status": "continue", "step": 0,
                            "correlation_id": correlation_id, "origin": "autonomous" if autonomous else "user"}
        self._last_feedback = {}
        try:
            async with asyncio.timeout(self.agent_config.timeout_s):
                feedback = ""
                last_signature = None
                repeated = 0
                for step in range(self.agent_config.max_steps):
                    self._check_turn(epoch)
                    if self.bus.clock.monotonic() - self._goal_started_at >= self.agent_config.timeout_s:
                        break
                    self.active_goal.update(step=step + 1)
                    response = await self._run_step(
                        user_text, correlation_id=correlation_id, epoch=epoch, feedback=feedback, step=step,
                    )
                    if response.status != "continue" or response.fallback_used:
                        self.active_goal.update(status=response.status)
                        return response
                    if self._actions_used >= self.agent_config.max_actions:
                        break
                    signature = (response.say, json.dumps(self.world.pose, sort_keys=True),
                                 json.dumps([{k: v for k, v in r.items() if k != "tool_call_id"}
                                             for r in self._last_feedback.get("action_results", [])], sort_keys=True))
                    repeated = repeated + 1 if signature == last_signature else 0
                    last_signature = signature
                    if repeated >= 2:
                        return await self._finish_blocked("I stopped because repeated steps are not making progress.", correlation_id)
                    feedback = json.dumps(self._last_feedback, ensure_ascii=False)
                    await asyncio.sleep(0)
                return await self._finish_blocked("I reached the task step limit before verifying completion.", correlation_id)
        except TimeoutError:
            await self.executor.cancel_all()
            self._check_turn(epoch)
            return await self._finish_blocked("I reached the task time limit before verifying completion.", correlation_id)

    async def _finish_blocked(self, message: str, correlation_id: str | None) -> ValidatedResponse:
        self.active_goal.update(status="blocked", message=message)
        if self._autonomous:
            message = ""
        response = ValidatedResponse(say=message, status="blocked", backend_name="goal_runtime")
        self._speech_correlation_id = correlation_id
        self.fsm.transition(InteractionState.SPEAKING)
        await self.bus.emit("robot.response", "arbiter", {**response.to_dict(), "final": True,
                            "goal_state": self.active_goal, "origin": "autonomous" if self._autonomous else "user"},
                            correlation_id=correlation_id)
        return response

    async def _run_step(self, user_text: str, *, correlation_id: str | None, epoch: int,
                        feedback: str = "", step: int = 0) -> ValidatedResponse:
        self._check_turn(epoch)

        # 1. Immediate attending & latency-masking expression (wake up if previously sleepy)
        self.fsm.transition(InteractionState.ATTENDING)
        self.emotion.clear_manual_override()
        if self.emotion.snapshot().emotion == "sleepy":
            self.emotion.set_emotion("neutral", 0.3, gaze_target="speaker")
        attending_snap = self.emotion.set_activity("listening", gaze_target="speaker")
        await self.bus.emit(
            "expression.updated",
            source="arbiter",
            payload={**attending_snap.to_dict(), "sound": "ack_chirp" if step == 0 and not self._autonomous else "none"},
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
        )

        # 2. Retrieve relevant memories before inference
        explicit_proposals = extract_explicit_memory_proposals(user_text) if step == 0 and not self._autonomous else []
        applied_memory_ops: list[dict[str, Any]] = []
        handled_keys: set[tuple[str, str]] = set()
        for prop in explicit_proposals:
            self._check_turn(epoch)
            outcome = self.memory.apply(prop)
            if prop.op == "forget":
                self.context_builder.clear_turns()
                self._prior_turns = []
                if self.consolidator:
                    self.consolidator.clear_pending()
            applied_memory_ops.append(outcome)
            handled_keys.add((prop.op, (prop.subject or "").lower()))
            await self.bus.emit(
                "memory.updated",
                source="arbiter",
                payload=outcome,
                priority=EventPriority.NORMAL,
                correlation_id=correlation_id,
            )

        retrieved_memories = [] if self._autonomous else self.memory.search(user_text, limit=5)
        await self.bus.emit(
            "memory.retrieved",
            source="arbiter",
            payload={
                "query": user_text,
                "count": len(retrieved_memories),
                "memories": [m.to_dict() for m in retrieved_memories],
            },
            priority=EventPriority.NORMAL,
            correlation_id=correlation_id,
        )

        # 3. Transition to THINKING and build context-budgeted prompt
        self.fsm.transition(InteractionState.THINKING)
        thinking_snap = self.emotion.set_activity("thinking", gaze_target="speaker")
        await self.bus.emit(
            "expression.updated",
            source="arbiter",
            payload=thinking_snap.to_dict(),
            priority=EventPriority.NORMAL,
            correlation_id=correlation_id,
        )

        messages, token_breakdown = self.context_builder.build_messages(
            user_text,
            memories=retrieved_memories,
            world_state=self.world,
            feedback="" if self._tool_exchanges else feedback,
            autonomous=self._autonomous,
            prior_turns=self._prior_turns,
        )
        if self._tool_exchanges:
            # Relative commands must retain the pose from when the user asked.
            # Current state belongs in the tool observation, not a rewritten request.
            messages[-1]["content"] = self._initial_request_text
            messages, history_tokens, total_tokens = self.context_builder.append_tool_history(
                messages, self._tool_exchanges,
                "[Current body state after tool execution]\n" + self.world.render_for_prompt(
                    include_visual=not self.context_builder.native_tools,
                ),
            )
            token_breakdown["tool_history_tokens"] = history_tokens
            token_breakdown["total_tokens"] = total_tokens
        elif step == 0:
            self._initial_request_text = messages[-1]["content"]

        # Native vision is requested through observe; conversation has no camera input.
        sandbox = getattr(self.world, "sandbox", None)
        before_frame = None
        if sandbox is not None and (not self.context_builder.native_tools or self._pending_camera is not None):
            from robot.body.hal import BodyPose

            if self._pending_camera is not None:
                _call_id, before_frame = self._pending_camera
                self._pending_camera = None
            else:
                before_frame = sandbox.capture_camera_view(
                    BodyPose(
                        x_mm=self.world.pose.get("x_mm", 0.0),
                        y_mm=self.world.pose.get("y_mm", 0.0),
                        heading_deg=self.world.pose.get("heading_deg", 0.0),
                        head_tilt_deg=self.world.pose.get("head_tilt_deg", 0.0),
                    )
                )
            if messages:
                text_payload = str(messages[-1]["content"])
                messages = list(messages)
                messages[-1] = {
                    **messages[-1],
                    "content": [
                        {"type": "image_url", "image_url": {"url": before_frame.data_url}},
                        {"type": "text", "text": text_payload},
                    ],
                }

        # 4. Call inference backend, validate output, and couple physical embodiment
        inf_result = await self.backend.generate(
            messages, json_schema=self._schema,
        )
        self._check_turn(epoch)
        validated = self.validator.validate(
            inf_result.raw_text,
            backend_name=inf_result.backend_name,
            latency_ms=inf_result.latency_ms,
        )
        proposed_wake = validated.next_wake_s
        token_breakdown["model_prompt_tokens"] = inf_result.prompt_tokens
        if proposed_wake is not None:
            self._requested_wake_s = proposed_wake
        validated.next_wake_s = self._requested_wake_s
        gaze_target = "speaker"

        # 5. Apply any additional model-proposed memory ops not already handled explicitly
        is_question = "?" in user_text and not explicit_proposals
        for mprop in validated.memory:
            if step > 0 or self._autonomous:
                break
            self._check_turn(epoch)
            if mprop.op == "forget":
                continue
            key = (mprop.op, (mprop.subject or "").lower())
            if key in handled_keys or explicit_proposals or is_question:
                continue
            outcome = self.memory.apply(mprop)
            applied_memory_ops.append(outcome)
            handled_keys.add(key)
            await self.bus.emit(
                "memory.updated",
                source="arbiter",
                payload=outcome,
                priority=EventPriority.NORMAL,
                correlation_id=correlation_id,
            )

        # 6. Execute physical actions through ActionExecutor
        action_results: list[dict[str, Any]] = []
        assistant = inf_result.metadata.get("assistant_message")
        for call in (assistant or {}).get("tool_calls", []):
            if call["function"]["name"] != "observe":
                continue
            pose = self.executor.body.pose()
            valid = sandbox is not None and not json.loads(call["function"]["arguments"])
            frame = sandbox.capture_camera_view(pose) if valid else None
            if frame is not None:
                self._pending_camera = (call["id"], frame)
                self._vision_requested = True
            action_results.append({"tool": "observe", "tool_call_id": call["id"], "ok": valid,
                                   "status": "completed" if valid else "rejected", "pose": pose.to_dict(),
                                   "message": frame.render_grounding_summary() if frame else "Camera unavailable or invalid observe arguments."})
            self._actions_used += 1
        limitation_notes: list[str] = list(validated.clamped_notes)
        for rej in validated.rejected_actions:
            limitation_notes.append(str(rej.get("error", "Requested action rejected")))
            await self.bus.emit(
                "action.rejected",
                source="arbiter",
                payload=rej,
                priority=EventPriority.NORMAL,
                correlation_id=correlation_id,
            )

        for action_call in validated.actions:
            self._check_turn(epoch)
            if self._actions_used >= self.agent_config.max_actions:
                validated.status = "blocked"
                limitation_notes.append("Task action limit reached.")
                break
            cmd_res = await self.executor.execute_call(action_call, correlation_id=correlation_id)
            self._actions_used += 1
            self._check_turn(epoch)
            if cmd_res.pose:
                self.world.pose.update(cmd_res.pose)
            if cmd_res.tool == "set_gaze" and "target" in cmd_res.details:
                gaze_target = str(cmd_res.details["target"])
            action_results.append(
                {
                    "tool": cmd_res.tool,
                    "ok": cmd_res.ok,
                    "status": cmd_res.status,
                    "message": cmd_res.message,
                    "pose": cmd_res.pose,
                    "details": cmd_res.details,
                    "tool_call_id": action_call.call_id,
                }
            )
            if cmd_res.status in ("clamped", "reflex_stopped", "rejected", "failed"):
                if cmd_res.message not in limitation_notes:
                    limitation_notes.append(cmd_res.message)
            if not cmd_res.ok or cmd_res.tool == "stop":
                if not cmd_res.ok and cmd_res.message not in limitation_notes:
                    limitation_notes.append(cmd_res.message)
                break

        if limitation_notes:
            note_summary = " ".join(limitation_notes)
            if "limit" not in validated.say.lower() and "clamped" not in validated.say.lower():
                validated.say = f"{validated.say} (Note: {note_summary})"

        # 6b. Closed-loop next-frame camera verification after physical actions (LFM3-VL Orchestration style)
        camera_verification: dict[str, Any] | None = None
        if sandbox is not None and action_results and (not self.context_builder.native_tools or self._vision_requested):
            from robot.body.hal import BodyPose

            after_frame = sandbox.capture_camera_view(
                BodyPose(
                    x_mm=self.world.pose.get("x_mm", 0.0),
                    y_mm=self.world.pose.get("y_mm", 0.0),
                    heading_deg=self.world.pose.get("heading_deg", 0.0),
                    head_tilt_deg=self.world.pose.get("head_tilt_deg", 0.0),
                )
            )
            before_ids = [d.label for d in (before_frame.visible_objects if before_frame else [])]
            after_ids = [d.label for d in after_frame.visible_objects]
            camera_verification = {
                "before_visible": before_ids,
                "after_visible": after_ids,
                "after_summary": after_frame.render_grounding_summary(),
                "front_tof_mm": self.world.sensors.get("front_tof_mm", 1500.0),
            }
            await self.bus.emit(
                "perception.frame_verified",
                source="arbiter",
                payload=camera_verification,
                priority=EventPriority.TELEMETRY,
                correlation_id=correlation_id,
            )

        # 7. Update speaking expression & emotion
        self.fsm.transition(InteractionState.SPEAKING)
        self.emotion.set_emotion(
            validated.emotion.name,
            validated.emotion.intensity,
            gaze_target=gaze_target,
        )
        speak_snap = self.emotion.set_activity("speaking", gaze_target=gaze_target)
        await self.bus.emit(
            "expression.updated",
            source="arbiter",
            payload={**speak_snap.to_dict(), "sound": validated.sound},
            priority=EventPriority.NORMAL,
            correlation_id=correlation_id,
        )

        # 8. Record turn in working memory, episodic store, and idle consolidation queue
        self._check_turn(epoch)
        if self._autonomous and self.autonomy and not self.autonomy.allow_speech():
            validated.say = ""
        if step == 0 and not self._autonomous:
            self.context_builder.add_turn("user", user_text)
        if validated.say:
            self.context_builder.add_turn("assistant", validated.say)
        if self.consolidator is not None and not self._autonomous:
            self.consolidator.enqueue_turn(user_text, validated.say)
        if not self._autonomous or validated.say or action_results:
            self.memory.log_episode(
                summary=f"{'Background event' if self._autonomous else 'User'}: {user_text[:80]} -> Robot: {validated.say[:80]}",
                kind="autonomous" if self._autonomous else "conversation",
                details={
                    "emotion": validated.emotion.name,
                    "actions": [a.tool for a in validated.actions],
                    "backend": validated.backend_name,
                },
            )

        # 9. Emit unified robot.response and open FOLLOW_UP window
        self._speech_correlation_id = correlation_id
        self._last_feedback = {"step": step + 1, "action_results": action_results,
                               "camera_verification": camera_verification, "pose": dict(self.world.pose)}
        self._last_feedback["rejected_actions"] = validated.rejected_actions
        if assistant and assistant.get("tool_calls"):
            exchange = [assistant]
            outcomes = {r.get("tool_call_id"): r for r in action_results}
            rejected = {r.get("tool_call_id"): r for r in validated.rejected_actions}
            for call in assistant["tool_calls"]:
                call_id = call["id"]
                if call["function"]["name"] == "schedule_wake":
                    result = {"ok": proposed_wake is not None, "status": "scheduled" if proposed_wake is not None else "rejected",
                              "message": f"Wake requested in {proposed_wake} seconds." if proposed_wake is not None else "Invalid wake interval."}
                elif call_id in outcomes:
                    result = {**outcomes[call_id], **{
                        k: v for k, v in outcomes[call_id].get("details", {}).items()
                        if k == "rotation_deg" or (k == "observed_objects" and self._vision_requested)
                    }}
                elif call_id in rejected:
                    result = {"ok": False, "status": "rejected", "message": rejected[call_id]["error"]}
                else:
                    result = {"ok": False, "status": "not_executed", "message": "An earlier action stopped this batch."}
                exchange.append({"role": "tool", "tool_call_id": call_id,
                                 "content": json.dumps(result, ensure_ascii=False)})
            self._tool_exchanges.append(exchange)
            self._tool_exchanges = self._tool_exchanges[-8:]
        self.active_goal.update(status=validated.status)
        self.world.task_state = dict(self.active_goal)
        await self.bus.emit(
            "robot.response",
            source="arbiter",
            payload={
                **validated.to_dict(),
                "action_results": action_results,
                "memory_ops": applied_memory_ops,
                "token_budget": token_breakdown,
                "camera_verification": camera_verification,
                "final": validated.status != "continue",
                "goal_state": dict(self.active_goal),
                "origin": "autonomous" if self._autonomous else "user",
            },
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
        )

        if not isinstance(self.bus.clock, RealClock) or not validated.say:
            self.fsm.transition(InteractionState.FOLLOW_UP)
        return validated

    async def _on_attention_engaged(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        if not text:
            return
        if self._input_task and not self._input_task.done():
            self._input_task.cancel()
            await self.bus.emit("turn.cancelled", "arbiter", {}, correlation_id=self._input_correlation_id)
        self._input_correlation_id = event.correlation_id or event.id
        self._input_task = asyncio.create_task(
            self._dispatch_input(text, self._input_correlation_id), name="input:turn"
        )
        self.bus.track_task(self._input_task)

    async def _dispatch_input(self, text: str, correlation_id: str) -> None:
        # Account for the detached task so drain/replay includes cognitive work.
        try:
            await self.handle_user_turn(text, correlation_id=correlation_id)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Input dispatch failed")
            await self.bus.emit("turn.cancelled", "arbiter", {}, correlation_id=correlation_id)

    async def _on_manual_expression(self, event: Event) -> None:
        emo = str(event.payload.get("emotion", "neutral"))
        intensity = float(event.payload.get("intensity", 0.8))
        gaze = event.payload.get("gaze_target")
        snap = self.emotion.set_emotion(
            emo,
            intensity,
            gaze_target=str(gaze) if gaze else None,
            manual_override=True,
        )
        await self.bus.emit(
            "expression.updated",
            source="arbiter",
            payload={**snap.to_dict(), "reason": "manual_override"},
            priority=EventPriority.HIGH,
            correlation_id=event.id,
        )

    async def _on_reflex_triggered(self, event: Event) -> None:
        snap = self.emotion.set_emotion("surprised", 0.9, gaze_target="down")
        await self.bus.emit(
            "expression.updated",
            source="arbiter",
            payload={
                **snap.to_dict(),
                "sound": "startle_pip",
                "reason": f"reflex:{event.payload.get('reflex', 'unknown')}",
            },
            priority=EventPriority.CRITICAL,
            correlation_id=event.id,
        )

    async def _on_barge_in(self, event: Event) -> None:
        self.cancel_active()
        self.fsm.transition(InteractionState.ATTENDING)
        snap = self.emotion.set_activity("listening", gaze_target="speaker")
        await self.bus.emit(
            "expression.updated",
            source="arbiter",
            payload={**snap.to_dict(), "reason": "user_barge_in"},
            priority=EventPriority.CRITICAL,
            correlation_id=event.id,
        )

    async def _on_speech_completed(self, event: Event) -> None:
        if event.correlation_id != self._speech_correlation_id:
            return
        if self._active_turn and not self._active_turn.done():
            self.fsm.transition(InteractionState.THINKING)
        elif self.fsm.state == InteractionState.SPEAKING:
            self.fsm.transition(InteractionState.FOLLOW_UP)
        if self.emotion.snapshot().activity == "speaking":
            snap = self.emotion.set_activity("idle")
            await self.bus.emit(
                "expression.updated",
                "arbiter",
                snap.to_dict(),
                priority=EventPriority.LOW,
                correlation_id=event.correlation_id,
            )
