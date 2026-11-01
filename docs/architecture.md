# How Sprocket thinks and moves

Two layers, like most robots that react in real time:

* a **fast layer** (20 Hz) that sees and steers: a small detector on every camera frame keeps a live
  map of the room, and closed-loop behaviours (go somewhere, follow, search, turn) steer the wheels
  from it, re-planning as things move;
* a **slow layer** (LFM3-VL-2B, about once a second or on events) that understands requests, picks
  goals and talks. It never steers: it starts behaviours and hears how they ended.

Nothing is a task script. "Go behind that box" is the model choosing `go_to(box_1, behind)`; the
behaviour then works out *where* behind is from the live map, plans around obstacles, and keeps
correcting while it drives.

```
 human text / speech ──► stop reflex ("stop", "freeze": halts at once, before the model reads it)
          │                                      ┌── drives (curiosity, social, play, energy)
          ▼                                      ▼
    BehaviorArbiter ◄── behaviour events ── AutonomyController
          │ situation: live map, memories, recent activity
          ▼
      AgentLoop ── tools ──► LFM3-VL-2B (llama.cpp)             SLOW LAYER  ~1 Hz / on events
          │ go_to · follow · stop · turn · move · circle_around · observe · look_around · express …
 ─────────┼──────────────────────────────────────────────────────────────────────────────────
          ▼                                                     FAST LAYER  20 Hz
      FastLoop  ── every tick ──►  1. newest detection ─► SpatialMap (objects, ToF occupancy)
          │                         2. active behaviour ─► speed command
          ▼                                   ▲
   YOLO11n (NCNN, worker thread)              │ GoToObject · Follow · Search · Scan · Turn · Move
   camera frames + pose at capture            │ Orbit · Face · GoToPoint      (control/behaviors.py)
          │                                   │
          ▼                                   ▼
   Body: set_velocity / update_motion ──► motors      REFLEXES  (microcontroller on the robot)
   SimBody · PhysicsBody · SerialBody                 range stop, bumper, cliff, e-stop, watchdog
```

## Layers

| Layer | Rate | Where | Job |
|---|---|---|---|
| Reflexes | 100 Hz | `firmware/base_controller` (sim: `body/*_body.py`) | forward stop at the range sensor, bumper, cliff, e-stop, 300 ms watchdog |
| Fast loop | 20 Hz | `control/fast_loop.py` | fuse the newest detection into the map, tick the behaviour, send speeds |
| Detector | as fast as it runs (60 fps laptop CPU) | `perception/fast_detector.py` | YOLO11n trained on this robot's objects; unknown things come out as "object" |
| Behaviours | every tick while active | `control/behaviors.py` | pure pursuit on A* paths, live goals (moving targets), honest failure reasons |
| Deliberation | on events | `cognition/agent.py` | pick behaviours and arguments, judge results, answer |
| Drives | 1 Hz, numbers only | `behavior/drives.py` | the pet's own reasons to act when idle |

What the split buys, measured in the simulator: going to the ball takes 4-9 s instead of 10-16 s,
a search is one smooth turn (6 s) instead of six stop-and-look views (11 s), the robot follows a
rolling ball at a steady distance, and "stop" halts it mid-motion.

### Behaviours

* `GoToObject(target, relation)`: the goal (near, behind, left of, between, on) is recomputed every tick
  from the live map, so a target that moves, or whose position sharpens as the robot approaches, is
  followed. A blocked exact spot falls back to spots along the same line ("behind" stays behind).
  Arrival is confirmed by seeing the target; if not, the skill searches and goes again or says so.
* `Follow(target, distance)`: persistent: it keeps running after the reply, so the robot can chat while
  following. Losing the target ends it and gives the robot a background turn to react.
* `Search(query)` / `Scan()`: one smooth turn while every frame is checked; things outside the detector's
  classes ("my keys") fall back to the VLM's stop-and-look scan, so the robot stays general.
* `Turn`, `Move`, `Face`, `Orbit`, `GoToPoint`.

The planner (`navigation/planner.py`) uses the real footprint (181 mm circumscribed radius) with a 60 mm
margin around objects placed from camera boxes, padded to cover a square's corners; walls and measured
range hits get a tighter margin. Leaving a blocked start is limited to 350 mm (it could otherwise "escape"
straight through an obstacle whose padding touches the robot's).

### What the model is protected from

The 2B model's choices are checked against the human's own words where that is unambiguous:
`go_to(..., near)` for "go *behind* the box" is repaired to `behind` (and for "between A and B" the other
object is found from the request); `on` for "go to the red ball" becomes `near` (driving *onto* a ball
rams it); `stop` without the human asking to stop is refused (the model confuses "start" and "stop").

## Perception (works the same on a real camera)

1. The camera (`SimCamera` in the simulator, `RealCamera` on the robot: Pi camera module or USB) hands
   out frames tagged with the body's pose at capture. PNG encoding is lazy: the detector reads raw pixels,
   and only the occasional VLM request pays for a PNG.
2. The **fast detector** (`perception/fast_detector.py`) is YOLO11n, trained on this robot's objects:
   `scripts/train_detector.py` renders 6,800 labelled frames from the simulator (randomised layouts,
   lighting, pose, head tilt, unlabelled-colour distractors; exact boxes from a segmentation render,
   occlusion included) and trains in ~40 minutes on the laptop GPU. It runs as an NCNN model with no
   PyTorch (`NcnnDetector`), the same code on the laptop and the Pi: 16 ms per frame on the laptop CPU.
   Validation mAP50 0.988; on freshly generated scenes 0.90 precision / 0.91 recall at IoU 0.5.
   Off-the-shelf open-vocabulary detectors (YOLO-World, YOLOE) found 50-75% of these objects.
3. The VLM stays the open-vocabulary eye: it names things the detector only knows as "object", and
   searches for things outside the detector's classes.
4. `CameraModel.locate_box` puts each box on the floor: bottom-centre of the box → floor ray → one
   radius further along the ray. Measured error on the default scene: 11–37 mm.
5. `SpatialMap.integrate_view` associates detections with tracked objects (label + distance gate,
   colour-aware synonyms), applies negative evidence to objects that should be visible but are not, and
   marks camera coverage. Forward ToF readings build an occupancy grid, so walls are discovered.
6. Fusion is Kalman-style and time-weighted. A sighting's variance grows with range (close looks
   outweigh distant ones); frames from a continuous stream each count for less (`weight`), so detector
   flicker neither creates nor deletes objects; a box cut off by the image's edge is weak evidence.

The simulator's ground truth (`sandbox.detect_objects`, `render_head_labels`) is used for training data,
the dashboard's debug view and the evaluation graders, never by the robot while it runs.

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
python scripts/train_detector.py                  # render the dataset, train YOLO11n, export NCNN
python scripts/bench_pi.py                        # detector / fast loop / decision speed on this machine
python scripts/eval_decisions.py                  # needs llama-server running
python scripts/eval_scenarios.py --no-server-start --seeds 2            # every scenario
python scripts/eval_scenarios.py --scenarios realtime --no-server-start # follow, stop, interrupt, rolling ball
python scripts/probe_latency.py --pin             # where inference time goes (server timings)
python -m pytest -q                               # 397 tests, no model needed
```

The dashboard (http://127.0.0.1:8090) shows the robot's own spatial map (with an optional
"sim truth" overlay), every model call and skill result with the camera frame and detection boxes it
saw, and the drives that wake it when idle.

Bodies: `SimBody` (kinematic, the default), `PhysicsBody` (`runtime.body_backend: physics`: MuJoCo
dynamics, wheel slip, noisy encoder/IMU odometry, a contact bumper) and `SerialBody`
(`runtime.body_backend: serial`: the real robot's motor microcontroller, see `docs/raspberry_pi.md`).
All three take speed set points (`set_velocity`, watchdog included) and discrete actions (`execute`).

`llama-server` runs with the vision projector, `--jinja` and two slots (`-np 2 -c 16384`). Decisions
use slot 0 and camera requests slot 1 (`id_slot`), so a VLM look never evicts the decision prefix.

## Measured on the laptop (LFM3-VL-2B Q4_K_M on a GTX 1650; detector on the CPU via NCNN)

- Simulator tasks (`eval_scenarios.py`, graded from ground truth, two room layouts each), two-layer
  architecture vs. the previous version: core 28/28 in 170 s (was 28/28 in 224 s); harder 22/24 in
  224 s (was 22/24 in 357 s); real-time 8/8 (new: follow a rolling ball, stop while following, stop
  part-way when interrupted, go to a ball that rolls during the approach). 58/60 overall
  (`artifacts/eval/v23-realtime-all.json`). Per task: go to the ball 9 → 4 s, go behind the box 23 → 12 s,
  find the duck and go to it 41 → 8 s, come back to the start 15 → 9 s, look around and report 4.4 → 2.7 s.
- Reaction time, request to wheels moving: median 0.84 s, 90th percentile 2.3 s (it includes the model's
  decision; "stop" bypasses the model and halts within one control tick).
- Decision benchmark (`eval_decisions.py`, first tool call in 21 situations): 95% at 0.5 s per decision
  (67% at the start of the rework). With LFM3-VL-2B-QAD-Q4_0: 83%.
- Fast detector: 16 ms per frame with NCNN on the laptop CPU (4 threads), 12 ms with PyTorch CUDA;
  validation mAP50 0.988; on fresh scenes 0.90 precision / 0.91 recall at IoU 0.5. Map fusion 0.5 ms.
- Autonomy (`try_autonomy.py`): left alone, the curiosity drive woke it, it explored, found new
  objects and covered 99% of the area around it in 2.5 minutes (drives accelerated 30x).

Known limitations:

- The detector is trained on the simulator's objects. On the real robot, add camera frames of your room
  and objects (and a `person` class) and retrain; the pipeline is the same.
- The remaining eval misses are a stop 1.5 cm past the 45 cm limit where another object blocks the
  exact spot, and the model occasionally choosing an odd tool (e.g. a half turn for "back up a little").
- The general "what is here" detection sometimes invents objects (a "laptop", a "banner") from walls
  or the dock's base. Searches for a specific thing are confirmed on a crop with a yes/no question;
  the general inventory is not, to keep it to one model call per view.
- The 2B model occasionally adds a step nobody asked for or confuses similar words; guards stop loops
  and the skills report honestly, but replies can still be slightly off.
- The same object can be listed twice when the VLM boxes it twice in one view.

## Raspberry Pi 5 (4 GB)

See `docs/raspberry_pi.md`: setup script, wiring, the motor-controller firmware and what to expect.
In short: the fast layer (NCNN detector + 20 Hz loop) is sized for the Pi's CPU, reflexes run on the
microcontroller, and the language model (Q4_K_M, CPU) is only consulted to choose what to do next.

## Hands later

Add an arm to `body.yaml`/the HAL, then a `pick(target)` skill that uses the same detection →
floor-geometry pipeline to approach and a visual-servo grasp primitive. Fine manipulation is a good
fit for a learned policy (LeRobot-style, trained from demonstrations collected in MuJoCo with
`runtime.body_backend: physics`), invoked by the VLM as just another skill.
