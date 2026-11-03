# How Sprocket works

Three layers, the way robots that react in real time are built:

* a **real-time controller** (the VENTUNO Q's STM32H5, 200 Hz): wheel loops, odometry, braking envelope,
  reflexes and watchdog. It never waits for Linux (`firmware/ventuno_mcu`, see [ventuno_q.md](ventuno_q.md));
* a **fast layer** (50 Hz control thread, a perception thread) that sees and steers: three cameras and a
  360-degree lidar keep a live map of the room and the people in it, and closed-loop behaviours (go
  somewhere, follow, search, circle, turn) steer the wheels from it, re-planning as things move;
* a **slow layer** (LFM3-VL-2B through llama.cpp, on events) that understands requests, picks skills and
  talks. It never steers: it starts behaviours and hears how they ended.

Nothing is a task script. "Do a full circle around the box behind you" is the model calling
`circle_around(target="the box behind you")`; the map resolves which box that is (behind the robot *as it was
asked*), and the behaviour drives the loop around it, skipping blocked points. "Come here" is
`go_to(target="me")`; the map knows who spoke and from where.

```
 voice ─► mic array ─► direction ─┐        ┌─ "stop" reflex: halts at once, before the model reads anything
 text / speech ───────────────────┼────────┤
                                  ▼        ▼
                         SpatialMap.hear_voice      BehaviorArbiter ◄── drives (curiosity, social, play, energy)
                         (who spoke: a person seen,       │ situation: live map, the speaker, memories
                          the lidar, or a guess)          ▼
                                                    AgentLoop ── tools ──► LFM3-VL-2B        SLOW  on events
                                                          │ go_to · follow · circle_around · observe · turn …
 ─────────────────────────────────────────────────────────┼────────────────────────────────────────────────
  perception thread                                        ▼                                   FAST  50 Hz
  front + rear cameras ─► YOLO11n (NCNN, GPU) ─┐    control thread: behaviour tick ─► collision monitor
  lidar scans ─► occupancy, localization ──────┼──► SpatialMap ◄──── GoToObject · Follow · Orbit · Search …
                                                │                          │ v, w
 ───────────────────────────────────────────────┴──────────────────────────┼─────────────────────────────
  STM32H5: wheel PI loops, odometry, braking envelope, bumper / cliff / e-stop, 300 ms watchdog   200 Hz
```

## Senses

| Sensor | Where | What it gives |
|---|---|---|
| Front camera, 90°, 640x480 | tilting head | detail ahead: the detector's range (a 10 cm toy at 2.5 m), the VLM's eye |
| Two rear cameras, 140°, 640x360 | sensor mast, facing ±115° | the rest of the circle: people, furniture, anything behind |
| 2D lidar (LD19), 360°, 10 Hz | mast top, 330 mm | walls, furniture, people's legs; localization; ranges people for the cameras |
| 4-mic array | head top | the direction a voice came from |
| Front ToF, bumpers, cliff sensors, IMU, encoders | base, on the MCU | reflexes and odometry |

The cameras are "fovea and periphery": the front camera sees far and small; the wide rear cameras cover the
remaining 270° at lower resolution. Together every bearing is covered (`test_the_rig_sees_all_around`), so
"what's behind you?" needs no turn and the map keeps objects behind the robot current while it drives.
`config/body.yaml` (`sensors.cameras`) is the single description of the rig: the perception geometry reads it,
and the simulator builds its MuJoCo cameras from it (their poses agree to 0.5 mm, checked by a test).

## Perception

1. Every camera hands out frames tagged with the body's pose at capture (`CameraRig`; `SimCamera` renders
   MuJoCo from the true pose, `RealCamera` reads a MIPI camera through the ISP and rectifies wide lenses from
   a calibration file). PNG encoding is lazy and capped at 512 px wide for the VLM.
2. The **fast detector** is YOLO11n trained on this robot's objects and people (`scripts/train_detector.py`:
   renders labelled frames from all three cameras in randomized rooms with a person in random clothes,
   random lighting, shadows and reflections; exact boxes from a segmentation render). It runs as an NCNN
   export, on the GPU through Vulkan when there is one. On freshly generated rooms: front camera recall 0.96 /
   precision 0.99, rear cameras 0.90 / 0.96, people 0.99 recall; 9 ms per frame on the laptop GPU, 25 ms on its CPU.
3. The perception thread runs the front camera every cycle and the rear cameras in turn.
4. `CameraModel.locate_box` puts each box on the floor: bottom centre → floor ray → pushed back to the
   object's centre. Width comes from the floor points under the box's bottom corners, which is exact even at
   the edge of a 140° lens (pixels × distance / focal length is off by 4x there). Error from true boxes:
   ~30 mm median on every camera.
5. **Lidar ranges what the cameras see.** A box whose span holds lidar returns at about the camera's
   distance (a person's legs, furniture) is placed at the centre of that cluster of returns, with
   centimetre variance, instead of the monocular guess.
6. `SpatialMap.integrate_view` fuses detections into tracks (label + distance gate, colour-aware synonyms,
   Kalman-style, time-weighted: a frame from a continuous stream counts for less than a deliberate look).
   A sighting's noise scales with distance over focal length, so the wide cameras' sightings and size
   estimates count for less than the front camera's. People get a motion model (constant velocity learned
   from whole, well-placed sightings, uncertainty growing while unseen), so a walking person's track keeps up
   instead of lagging until a new track splits off. A box cut off at an image side confirms a track in that
   direction but never starts one unless the lidar ranges it (someone on the seam between two cameras is cut
   off in both). Objects that should be visible but are not fade out.
   A trained detector's labels outrank remembered ones (a track it keeps calling something else is renamed).
7. The VLM stays the open-vocabulary eye: it names what the detector only knows as "object", answers
   questions about the view, and searches for things outside the detector's classes ("my keys").

## People and the voice

When a message arrives with a voice direction (`audio/doa.py`: SRP-PHAT on the array's channels, delays applied
in the frequency domain because the 64 mm array spans only three samples at 16 kHz; median error 1-4° with room
echoes), `SpatialMap.hear_voice` decides who spoke: the person track in that direction; if nobody is seen
there yet, it notes where they probably are (the nearest solid thing the lidar sees along the voice, else a point
1.5 m along it) and binds the speaker to the first person the cameras place in that direction. It never invents
a person track for the voice: a made-up track drew stray sightings to it, and following then lost its target
(measured). The speaker is "me" for the model and is marked in the prompt ("person_1 (person, the human talking
to me)"). `go_to(me)` stops at a comfortable distance facing them and follows them if they move on the way (or
heads for the voice when they are not seen yet); `follow(me)` keeps following; `turn_to(me)` faces them; "the
cup next to me" means next to the human.

In the simulator a typed message is the simulated person speaking: `SimVoice` renders their voice on the
simulated array (direct path, four wall reflections, noise), so the same estimate runs in sim and on the robot.

## Behaviours (`control/behaviors.py`)

* `GoToObject(target, relation)`: near, behind, in front of, left/right of, between, on. The goal is recomputed
  every tick from the live map. "Near" is approached from wherever the robot is; the other sides keep the view
  from where the task began, unless the object turns out to be somewhere else entirely. A blocked spot falls back
  to spots along the same side. Arrival is confirmed by seeing the target.
* `Follow(target, distance)`: persistent, keeps running after the reply; losing the target ends it.
* `Search(query)` / `Scan()`: one smooth turn while every frame of every camera is checked.
* `Orbit`, `Turn`, `Move`, `Face`, `GoToPoint`.

The planner (`navigation/planner.py`) is A* on the lidar occupancy grid plus tracked objects, with the robot's
footprint (181 mm circumscribed radius) and margins. Every speed command passes the collision monitor
(`control/safety.py`): speed capped to what still stops in time, given lidar points all around and the front ToF.

## Making a small model reliable

A 2B model chooses well among few, clear tools, and badly at the edges. The harness guards those edges in code,
each measured against the simulator before it went in:

- Few, semantic tools listed in a deliberate order; examples written in words, not code.
- The request's own words repair a call that drops what the human said: the relation ("go *behind*"), "come
  back" (where the robot was asked), "come here" (the human), "on" only when they said on/dock.
- A request with several parts is cut at sequencing words ("... and come back") and handed over one part at a
  time (`cognition/planner.py`); a part ends when its main action succeeds.
- Loop guards (the same call three times, four failures), repeated idempotent calls answered from the first
  result, calls written as text or markup recovered, claims of actions no tool performed sent back once.
- Past turns are kept as real tool-call messages (a small model copies the shape of its history), and the
  system prompt and tools are read once at start so the first request is fast.
- `scripts/eval_decisions.py` measures the first decision in ~20 situations in a minute;
  `scripts/eval_scenarios.py` runs whole tasks in the simulator and grades them from ground truth.

## Memory

| Kind | Where | In the prompt |
|---|---|---|
| Working | the last turns (tool calls included), recent activity | always, budgeted |
| Spatial | `SpatialMap`: objects, people, the speaker, places; saved on shutdown, restored as stale | "Known objects", "Places" |
| Semantic | `MemoryStore`: SQLite + FTS5 facts, versioned, tombstones for "forget" | core facts always; related facts by search |
| Episodic | `MemoryStore` episodes, compressed into daily summaries | via `recall` |

## Threads and timing

| Thread | Rate | Job |
|---|---|---|
| control | 50 Hz, absolute deadlines, SCHED_FIFO on a fast core (robot) | behaviour tick, collision monitor, speed command |
| perception | as fast as frames and the detector allow | cameras → detector → map; lidar → occupancy, localization |
| main event loop | on events | cognition, speech, memory, dashboard |
| llama-server | own process, own cores | the language and vision model |

Measured on the laptop simulator (three cameras rendered in MuJoCo, detector on the GPU): front camera fused at
~16 Hz with a rear camera each cycle, control loop at 50 Hz with 0.5 ms median and 8 ms p99 lateness. On the robot
the cameras need no rendering; `scripts/bench_control.py` measures the same numbers there.

## Hands later

Add an arm to `body.yaml` and the HAL, then a `pick(target)` skill that uses the same detection → floor
geometry to approach and a visual-servo grasp primitive (a learned policy trained from demonstrations in
MuJoCo with `runtime.body_backend: physics` fits well), called by the model as one more skill.
