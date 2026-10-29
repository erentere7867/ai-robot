# Model-owned robot loop

The default llama.cpp path uses native function calling. The model receives the
user request, conversation, and tool results. Its calls
execute validated body actions. Matching `tool` messages and an updated camera
are returned when the model requests perception with `observe`; a normal text
reply ends the request. Ordinary conversation has no automatic image or scene
description. The model chooses whether it needs perception instead of receiving
a visual job with every greeting.
The original request is not reissued as a new command after every action.

Native calls are adapted to the existing execution path internally. The JSON
compatibility interface (`llm.native_tools: false`, or injected test backends)
uses `say`, `actions`, and `status`:

- `continue`: execute the proposed body actions, observe again, and call the model
  without requiring another user prompt.
- `completed`: emit the response and end this request immediately.
- `blocked`: emit the response and stop this request.

A completed response with `actions: []` causes no movement. Empty `say` allows a
silent decision. Sound, emotion, memory proposals, and `next_wake_s` are optional.
An old backend without a status field retains one-step completion behavior.

There are no keyword-to-task mappings, search/survey/navigation routines, automatic
object approach goals, tool masks inferred from the wording, or injected emotion
postures. A descriptive question does not start a separate search. The runtime
executes only the model's proposed, validated actions. It does not replace the
model's decisions with a task program.

Startup records a silent local baseline without calling the model. Idle checks
compare quantized visible-object positions, range, and status readings at
`autonomy.observation_interval_s`. They do not render images or run inference.
An unchanged room produces no inference, conversation history, or speech.

A changed observation wakes the model after interaction grace and cooldown. The
trigger is compact internal event data with `user_request: null`, rather than
a repeated request to describe the room. Background perception is private; the
model may act, stay idle, or address the human for a useful reason.

The generic `schedule_wake(delay_s)` tool requests one future background decision.
In JSON mode, `next_wake_s` has the same meaning. Without a change or an explicit
wake request there is no periodic model polling. `think_interval_s` is a cooldown
for change-triggered decisions. The post-action observation becomes the new
baseline, so the robot's own movement does not trigger another idle goal.

Silent idle decisions add no dialogue or episodic memory. Active native tool
exchanges are temporary and bounded to eight exchanges and 600 text tokens;
complete call/result groups are retained. Only the latest camera image is sent,
and actual server prompt-token usage is included in diagnostics.

User input preempts autonomous work. Stop, emergency stop, motor limits, and
obstacle/cliff/pickup reflexes override motor output. Each control tick samples
simulated sensors synchronously. Ordered speech allows multiple replies without
cutting off earlier sentences; interruption clears pending speech. Resource
limits for steps, actions, time, and repeated identical nonproductive decisions
remain safeguards, rather than task-completion rules.

Lightweight camera detections are sampled during motion. For a request that has
used `observe`, tool results include
objects seen along the motion and actual rotation, including during a full turn
whose final view matches the initial view. This supplies sensor feedback without
prescribing a search plan. A purely conversational request does not receive these
scene labels. Truncated tool calls never execute.

The audio response producer waits for actual realtime playback completion, not
just synthesis completion, before starting the next sentence. A new human turn
or stop can interrupt it; the robot's own next sentence does not.

`stop` pauses autonomy until `resume autonomy` or the dashboard's **Autonomy on**.
**Stay here** pauses it. Quiet mode suppresses autonomous speech. CLI
`--no-autonomy` disables idle decisions, while `--once` waits for the model loop
to end. When the real model is unavailable, auto mode reports that state and
issues no actions. `--backend null` explicitly selects a deterministic test/demo
adapter; it is not the normal offline fallback.

Run `python -m pytest -q` to verify completion, continuation, fresh camera feedback,
idle/no-op decisions, native call/result matching, silent startup, one-shot wakes,
stop, unavailable inference, and ordered speech. Real-model motor checks are in
`artifacts/agentic-verification/native-control-check.json`; conversation and
on-demand camera checks are in `artifacts/agentic-verification/conversation-control-check.json`.
`tests/test_audio_queue.py` exercises the real asynchronous playback path.
Tool use follows
[Liquid's documented LFM3 interface](https://docs.liquid.ai/lfm/key-concepts/tool-use).

The robot's behavior is limited by the connected model's planning and visual
understanding. There is no task-specific script compensating for those limits.
The current body is kinematic and shown through MuJoCo; geometric detections
provide simulated camera grounding. Physical hardware still needs motor/sensor
and perception adapters.
