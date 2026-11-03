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
python -m pip install -e ".[dev,audio,face]"
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

## Measuring

```bash
python -m pytest -q                                              # ~330 tests, no model needed, < 1 min
python scripts/eval_scenarios.py --scenarios people --no-server-start   # graded tasks in the simulator
python scripts/eval_decisions.py                                 # first-decision accuracy, ~1 min
python scripts/probe_latency.py --pin                            # where inference time goes
python scripts/bench_control.py                                  # control-loop timing under load
python scripts/train_detector.py                                 # render a dataset, fine-tune, export NCNN
```

## Layout

```
robot/
  body/         SimBody (kinematic), PhysicsBody (MuJoCo), VentunoBody (STM32H5 over the Arduino Router)
  control/      fast loop (control + perception threads), behaviours, collision monitor
  perception/   camera rig, geometry, YOLO (NCNN) and VLM detectors, lidar, training data
  world/        spatial map (objects, people, speaker, occupancy), localization, simulator
  audio/        speech in/out, voice direction (doa.py), sounds
  cognition/    agent loop, prompts, to-do list for multi-part requests, llama.cpp client
  skills/       what the model can call: go_to, follow, observe, circle_around, express, remember, ...
  memory/       long-term memory store and consolidation
  behavior/     attention, arbitration, autonomy and drives, emotion
  eval/         simulator scenarios and ground-truth graders
config/         robot.yaml (laptop), robot.ventuno.yaml (robot), body.yaml (geometry, sensors), MuJoCo model
firmware/       ventuno_mcu: the STM32H5 sketch
scripts/        training, evaluation, benchmarks, VENTUNO Q setup
docs/           architecture, VENTUNO Q hardware, evaluation
```

Read [docs/architecture.md](docs/architecture.md) for how the layers work and why, and
[docs/evaluation.md](docs/evaluation.md) for the scenario suite and current results.
