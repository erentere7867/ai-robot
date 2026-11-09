# Sprocket: an embodied AI robot harness for the Arduino VENTUNO Q

A local-first harness for a small wheeled robot that sees all around it, hears where you are, remembers you,
and turns plain requests into motion: "come here", "do a full circle around the box behind you", "follow me",
"go to the ball and come back". Nothing is a task script: a language model picks general skills, and
closed-loop behaviours carry them out from live perception.

```
STM32H5 (real time, 200 Hz)        wheels, odometry, braking envelope, reflexes, watchdog
fast layer (50 Hz, own threads)    3 cameras + lidar -> detector -> spatial map; behaviours steer from it
slow layer (on events)             LFM3-VL-2B: understands, chooses skills, talks; never steers
```

## What it can do

| Ability | How |
|---|---|
| See all around | three cameras (front 90° in the tilting head, two 140° on the sensor mast), one YOLO11n on every frame |
| Know where you are | 4-mic array, direction of arrival (SRP-PHAT), matched to a person the cameras or lidar see |
| "come here", "follow me", "look at me" | `go_to` / `follow` / `turn_to` with target `me`; the request's words repair a wrong model choice |
| "the box behind you", "the closest ball", "the cup next to me" | referring expressions resolved on the live map, from where you asked |
| Go somewhere | A* on lidar occupancy + tracked objects, pure pursuit, live goals (moving targets), honest failures |
| Remember | SQLite + FTS5 facts (versioned, forgettable), episodes, a spatial map that persists |
| Stay safe | MCU reflexes and watchdog, a collision monitor on every command, "stop" bypasses the model |
| Act on its own | drives (curiosity, social, play, energy) wake the model when idle |

## Quick start (laptop simulator)

Python 3.11+, a llama.cpp `llama-server` build, and the model files in `config/robot.yaml`
(`LFM3-VL-2B-Q4_K_M.gguf` + `mmproj-LFM3-VL-2B-F16.gguf`).

```bash
python -m pip install -e ".[sim,dev,audio,face]"
python -m robot                    # console + dashboard at http://127.0.0.1:8090 (starts llama-server)
python scripts/try_robot.py "what's behind you?" "come here" "do a full circle around the box behind you"
```

In the simulator, typed messages are spoken by the simulated person: their voice reaches the simulated
microphone array from where they stand, so "come here" works exactly as it does on the robot. Drag the
person around in the dashboard's 3D view.

## On the robot

```bash
bash scripts/ventuno/setup.sh                                   # packages, llama.cpp, models, services
python scripts/ventuno/mcu_check.py --wiggle                    # MCU link, wheels off the ground
python scripts/ventuno/calibrate_camera.py front                # once per camera (front, rear_left, rear_right)
sudo systemctl start sprocket-llm sprocket                      # or: python -m robot --config config/robot.ventuno.yaml
```

See [docs/ventuno_q.md](docs/ventuno_q.md) for the hardware (parts, wiring, cameras, lidar, microphones) and bring-up.

## Testing & Measuring

Testing spans unit-level subsystem tests up to full simulator scenario grading:

### 1. Subsystem unit & integration tests (`pytest`)

No model server or hardware needed, about a minute:

```bash
python -m pytest -q                                              # full suite

# Or test individual subsystems:
pytest tests/test_control.py                                     # 50 Hz control loop, behaviors, safety monitor
pytest tests/test_people.py                                      # 4-mic DOA voice localization & person tracking
pytest tests/test_perception.py                                  # 3-camera rig, wide-lens geometry, detector, lidar
pytest tests/test_bodies.py                                      # SimBody, MuJoCo PhysicsBody, VentunoBody
pytest tests/test_skills.py                                      # go_to, follow, observe, circle_around
pytest tests/test_runtime.py                                     # harness, reflexes, watchdog, telemetry
pytest tests/test_cognition.py                                   # agent loop, prompt generation, tool calls
pytest tests/test_memory.py                                      # SQLite + FTS5 facts, episode consolidation
pytest tests/test_eval.py                                        # simulator scenario graders
```

### 2. Simulator scenario evaluations (`scripts/eval_scenarios.py`)

Runs end-to-end tasks in the MuJoCo simulator, evaluated against ground truth:

```bash
python scripts/eval_scenarios.py --backend null --scenarios greet       # smoke test harness/graders without LLM
python scripts/eval_scenarios.py --list                                 # list all scenario sets & suites
python scripts/eval_scenarios.py --scenarios people --no-server-start   # test "come here", "follow me", etc.
python scripts/eval_scenarios.py --scenarios core,tasks,realtime        # test navigation, multi-step tasks, moving targets
```

### 3. Decisions & benchmarks

```bash
python scripts/eval_decisions.py                                 # first-decision tool call accuracy (~1 min)
python scripts/probe_latency.py --pin                            # breakdown of inference latencies
python scripts/bench_control.py                                  # control-loop timing & lateness under load
python scripts/bench_localization.py                             # correlative scan-to-map matching drift
python scripts/eval_vision.py                                    # camera projection & detector mAP
python scripts/train_detector.py                                 # render synthetic dataset, train, export NCNN
```

### 4. Hardware bring-up tests (on the robot)

```bash
python scripts/ventuno/mcu_check.py --wiggle                    # MCU RPC link, telemetry & motor watchdog
python scripts/ventuno/calibrate_camera.py front                # camera calibration (front, rear_left, rear_right)
```

## Layout

```
robot/
  platform.py   the robot the harness runs on: a body and its senses, simulated or real
  body/         the body interface every robot implements (HAL), reflexes, dimensions from body.yaml
  sim/          the simulated robot: kinematic and MuJoCo bodies, room, cameras, lidar, voice, training data
  hardware/     the real robot: VENTUNO Q MCU link, MIPI/USB cameras, LD19 lidar
  control/      fast loop (control + perception threads), behaviours, collision monitor
  perception/   camera rig, geometry, the trained detector (NCNN), lidar scans
  world/        spatial map (objects, people, speaker), occupancy grid, labels, localization
  audio/        speech in/out, voice direction (doa.py), sounds
  cognition/    agent loop, prompts, call repair, to-do list for multi-part requests, llama.cpp client, VLM eye
  skills/       what the model can call: go_to, follow, observe, circle_around, express, remember, ...
  memory/       long-term memory store and consolidation
  interaction/  attention and input routing, arbitration, autonomy and drives, emotion
  eval/         simulator scenarios and ground-truth graders
config/         robot.yaml (laptop), robot.ventuno.yaml (robot), body.yaml (geometry, sensors), MuJoCo model
firmware/       ventuno_mcu: the STM32H5 sketch
scripts/        training, evaluation, benchmarks, VENTUNO Q setup
docs/           architecture, VENTUNO Q hardware, evaluation
```

Read [docs/architecture.md](docs/architecture.md) for how the layers work and why, and
[docs/evaluation.md](docs/evaluation.md) for the scenario suite and current results.
