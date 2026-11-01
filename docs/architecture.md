# How Sprocket thinks and moves

The robot is a small vision-language model (LFM3-VL-2B) making decisions on top of a layer of
generic, closed-loop skills. The model decides *what* to do and *why*; geometry, path planning and
motor control are done by code. Nothing is a task script: there is no "go behind the box" routine,
only general verbs (look, go somewhere relative to something, turn, remember) that the model
composes from what it perceives.

```
 human text / speech ──┐                    ┌── drives (curiosity, social, play, energy)
                       ▼                    ▼   + camera change detection   (no model calls)
                 BehaviorArbiter ◄──── AutonomyController
                       │  situation: map summary, core memories, recent activity, battery
                       ▼
                   AgentLoop  ── chat + tools ──► LFM3-VL (llama.cpp, native tool calls)
                       │  tool calls                ▲ tool results + newest camera image
                       ▼                            │
                 SkillRegistry ──────────────────────
   observe · look_around · go_to · turn_to · turn · move · tilt_head · explore
   express · set_eye_color · remember · recall · wait
                       │
     ┌─────────────────┼───────────────────────────┐
     ▼                 ▼                           ▼
 Perception        SpatialMap                 ActionExecutor ──► Body (SimBody / PhysicsBody / real HAL)
 camera + VLM      objects, places,           validation, limits,     reflexes: ToF, bumper, cliff,
 detection +       ToF occupancy, coverage    cancellation            e-stop, pick-up
 floor geometry    (robot's own sensors only)
```

## Layers

| Layer | Rate | Where | Job |
|---|---|---|---|
| Reflexes | every motor tick | `body/sim_body.py` (later: microcontroller) | stop on obstacle/bumper/cliff/e-stop |
| Skills | 1–20 Hz while running | `skills/` | closed-loop behaviours: plan, drive, replan, verify |
| Deliberation | on events | `cognition/agent.py` | pick skills and arguments, judge results, answer |
| Drives | 1 Hz, numbers only | `behavior/drives.py` | the pet's own reasons to act when idle |

## Perception (works the same on a real camera)

1. `SimCamera` renders the head camera (`sandbox.render_head_rgb`) from the body's *true* pose; the
   frame is tagged with the *estimated* pose, exactly like hardware.
2. `VLMDetector` asks the VLM for `[{"label", "bbox"}]` (LFM-VL answers on a 0..1000 grid).
3. `CameraModel.locate_box` puts each box on the floor: bottom-centre of the box → floor ray → one
   radius further along the ray. Measured error on the default scene: 11–37 mm.
4. `SpatialMap.integrate_view` associates detections with tracked objects (label + distance gate,
   colour-aware synonyms), applies negative evidence to objects that should be visible but are not,
   and marks camera coverage. Forward ToF readings build an occupancy grid, so walls are discovered,
   not given.
5. Each track keeps a position variance. A sighting's variance grows with range (depth from one
   camera box degrades with distance), and sightings are fused Kalman-style: a close look outweighs a
   distant one, and one poor far-off box cannot drag a well-known object away.
6. `look_around` turns to the next view while the model is still reading the last frame; every frame
   carries the pose it was taken at, so the map stays correct while the robot moves.
7. `go_to` on a target more than a metre away stops at a closer viewpoint, looks again and re-plans
   (look-move-look), instead of driving to where one distant sighting put it. If on arrival the target
   is not in view, the map entry was wrong: it searches for it and goes there; if it still cannot see
   it, the skill fails honestly instead of claiming an arrival.

The simulator's ground-truth detections (`sandbox.detect_objects`) are only used by the dashboard
debug view and the evaluation graders, never by the robot.

## Memory

| Kind | Where | In the prompt |
|---|---|---|
| Working | `PromptBuilder.history` (last turns incl. tool calls), `arbiter.recent_activity` | always, budgeted |
| Spatial | `SpatialMap` (saved to `spatial_map.json`; restored objects are marked stale) | "Known objects", "Places" |
| Semantic | `MemoryStore` SQLite + FTS5 facts, versioned, tombstones for "forget" | core facts always; related facts by search |
| Episodic | `MemoryStore` episodes | via `recall` |

A small model copies the shape of its history. Past turns are therefore kept as real tool-call
messages, never as text summaries of actions, or it starts *writing* actions instead of calling them.

## Making a 2B model reliable

- Few, semantic tools (`turn(direction=around)` instead of degrees), observe listed last, examples
  written in words (code-like examples make it print calls as text).
- `recover_text_tool_calls` accepts a reply that is *only* a call written as text.
- Loop guards: the same call three times, or four failures, ends the turn with an honest message;
  a second `observe` from the same pose returns the cached view.
- A successful idempotent call (`go_to`, `turn_to`, `look_around`, ...) repeated from the same spot is
  not run again: the model is told it is already done and its next answer must be a reply
  (`tool_choice: none`). A small model otherwise re-issues the call after success until the loop guard
  stops it with a misleading "I keep trying".
- The last tool result of each step carries a `next` hint built from the human's words. A request with
  several steps ("then", "after", "and go ...") is restated ("Answer the human now, unless the request
  asks for another step that is not done yet"), so "go to the ball and then to the cube" is not dropped
  after the first part; a one-step request gets "If that finishes ..., answer the human now". Merged
  into the result text instead, the hint was obeyed as an instruction and made things worse (measured).
- `go_to` checks its arguments against the request: if the human named exactly one relation
  ("between", "behind", "left of", ...) and the call says `near`, it is rejected with the relation to use.
- Only `observe` puts a camera image in the model's context. The views skills take for their own checks
  (arriving somewhere) update the map but are not shown: an extra image made the model describe or
  visit what it saw instead of answering (go to the ball: 25 s with detours -> 8 s).
- Skills return outcomes the model can reason about ("box_1 is now between me and where I was").
- `scripts/eval_decisions.py` measures the first decision for 21 situations in about a minute; use
  it after any prompt or tool change. `scripts/eval_scenarios.py` runs whole tasks in the simulator
  and grades them from ground truth.

## Running

```bash
python -m robot                                   # console + dashboard (starts llama-server if needed)
python scripts/try_robot.py "what do you see?" "go behind that box"
python scripts/try_robot.py --physics "go to the red ball"   # MuJoCo dynamics: slip, odometry drift
python scripts/try_autonomy.py --seconds 180 --speed 20      # leave it alone and watch
python scripts/eval_decisions.py                  # needs llama-server running
python scripts/eval_scenarios.py --backend llamacpp --no-server-start --seeds 2
python scripts/eval_scenarios.py --scenarios hard --no-server-start   # the harder set only
python scripts/probe_latency.py --pin             # where inference time goes (server timings)
python -m pytest -q                               # 377 tests, no model needed
```

The dashboard (http://127.0.0.1:8090) shows the robot's own spatial map (with an optional
"sim truth" overlay), every model call and skill result with the camera frame and detection boxes it
saw, and the drives that wake it when idle.

Two simulated bodies exist: `SimBody` (kinematic, perfect odometry, fast; the default) and
`PhysicsBody` (`runtime.body_backend: physics`): MuJoCo dynamics with wheel friction, pushable
objects, a contact bumper and noisy encoder/IMU odometry, so the robot's map drifts the way a real
one would. The camera always renders from the true pose; the robot only knows its estimate.

`llama-server` must be started with the vision projector, `--jinja` and two slots
(`-np 2 -c 16384`). Every request names its slot (`id_slot`): decisions use slot 0 and camera
requests slot 1, so a detection never evicts the ~2,100-token decision prefix from the cache.
`LlamaCppBackend.ensure_server_started` does this automatically.

## Measured on the laptop (LFM3-VL-2B Q4, GTX 1650)

- Decision benchmark (`eval_decisions.py`, first tool call in 21 situations): 67% at the start of the
  rework, 93-98% now, at 0.5 s per decision.
- Simulator tasks (`eval_scenarios.py`, graded from ground truth, two room layouts each): the 14 core
  tasks 28/28 in 224 s (was 28/28 in 254 s); the 12 harder tasks 22/24 (19/24 before the latest
  changes). Core: greet, describe, turn around, look left, move a little, go to the ball, go behind the
  box, go between two objects, find a hidden duck, go to the dock, come back to the start, look around
  and report, remembering a name/preference across restarts. Harder (`--scenarios hard`): a repeated
  command, turning left then right, turn 45 degrees, dance, sad face, the ball's colour, circling the
  box, find the duck and go to it, two stops in one request, back up after arriving, "the one closest
  to you" from the previous turn, the left side of the box. Runs vary: expect a miss or two.
- Where time goes (`probe_latency.py`): a decision re-reads 35-40 new tokens and reuses ~2,100 cached
  ones (0.3-0.4 s); a detection spends ~1 s encoding the image (~270 tokens) and 0.4-2 s writing the
  box list (about 30 tokens per object), so the number of objects in view sets its cost.
- Autonomy (`try_autonomy.py`): left alone, the curiosity drive woke it, it explored, found new
  objects and covered 99% of the area around it in 2.5 minutes (drives accelerated 30x).

Known limitations:

- The violet charging dock can pass a targeted search for a "blue cube" (crop check included); the
  arrival check then fails and the robot says so, but it can keep retrying until the loop guard stops it.
- The general "what is here" detection sometimes invents objects (a "laptop", a "banner") from walls
  or the dock's base. Searches for a specific thing are confirmed on a crop with a yes/no question;
  the general inventory is not, to keep it to one model call per view.
- The 2B model occasionally adds a step nobody asked for or confuses similar words; guards stop loops
  and the skills report honestly, but replies can still be slightly off.
- The same object can be listed twice when the VLM boxes it twice in one view.

## Raspberry Pi 5 (4 GB): what to expect

Laptop measurements (GTX 1650): a decision takes 0.3-0.7 s, a detection 1.4-3 s, "go behind that
box" about 25 s including driving. On the Pi's CPU the same model is several times slower (not yet
measured — run `scripts/bench_llm.py` there). Expect each decision to take several seconds and
`look_around` (six detections) to take a minute or more. Recommended path:

1. Keep LFM3-VL for decisions and descriptions.
2. Move detection to a dedicated detector: the Raspberry Pi AI HAT+ (Hailo) running a YOLO-family
   model, or an open-vocabulary detector exported to ONNX. `VLMDetector` is the only class to replace;
   it returns `Detection(label, bbox)`.
3. Run motor reflexes on a microcontroller so they never depend on Linux scheduling.

## Hands later

Add an arm to `body.yaml`/the HAL, then a `pick(target)` skill that uses the same detection →
floor-geometry pipeline to approach and a visual-servo grasp primitive. Fine manipulation is a good
fit for a learned policy (LeRobot-style, trained from demonstrations collected in MuJoCo with
`runtime.body_backend: physics`), invoked by the VLM as just another skill.
