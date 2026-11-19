# Sprocket v2

A pet-like, helpful wheeled robot on the Arduino VENTUNO Q, with everything running on the robot.

Its mind is an **agentic tool-calling loop** around an on-board vision-language model (`LFM3-VL-2B` for now). The
model gets the three cameras' pictures and the user's words, answers what it can from the pictures, and acts by
calling tools: `go_to("sofa")`, `search("keys", area="bedroom")`, `follow("user")`, `go_toward("behind-left", 2)`. The
tools do the real-time work: YOLOE looks for the words they name, a tracker follows what it finds, depth from the
cameras makes a 360° scan, and Nav2 plans and drives. Every result goes back to the model before its next step.

It is meant to be a general-purpose robot, not a pre-programmed one: nothing in the code is written for a particular
task, room, object or phrase, and the model's judgement, not keyword rules, decides what to do. The code is kept
small and maintainable: one place for each piece of logic, nothing duplicated, nothing kept that is not used. The
rules for changing it are in [AGENTS.md](AGENTS.md).

- [docs/testing.md](docs/testing.md): how Sprocket is tested, and how to set up any Linux machine to test it the
  same way: `bash tools/linux/setup.sh` installs everything on a bare Ubuntu, and `python -m robot.eval.doctor`
  checks it and runs a 3D task.
- [docs/usage.md](docs/usage.md): set up, run the model and the vision server, run the tests, the 2D suite, the 3D
  apartment with its live page and task reports, and the demo.
- [docs/architecture.md](docs/architecture.md): the design: body, perception, memory, the agent.
- [docs/simulation_3d.md](docs/simulation_3d.md): the 3D apartment, and what has been measured in it.
- [docs/ideas/sprocket-harness.md](docs/ideas/sprocket-harness.md): how the agent harness came to be, with its numbers.
- [AGENTS.md](AGENTS.md): rules for anyone (person or coding agent) changing the code.

## Layout

```text
firmware/     STM32H5 sketch: 200 Hz wheel control, VL53L5CX step/cliff reflex, IMU + anti-snag
robot/
  agent/      the agent loop, its tools (checked, with errors that teach), to-do list, instant stop/wait
  mind/       the model's client (native tool calls, yes/no probabilities), long-term memory (SQLite)
  perception/ camera rig, YOLOE for asked words, depth scaled by the floor, synthetic /scan, tracker, pose
  body/       the body's control loops: trajectory rollout, pursuit, social yielding, aliveness
  ros/        the robot on ROS 2: Nav2, its own map and rooms, perception nodes, RealRobot
  sim/        the 2D simulator behind the same robot API, and the Habitat 3D apartment
  interfaces/ live web page, voice (Whisper, voiceprints, Piper), the demo
  eval/       the 2D suite, the 3D tasks with per-task reports, the vision model's photo check
  train/      random 2D scenes (the post-training data builder comes next)
tests/        unit and integration tests
tools/        start the model (brain.ps1, wsl/brain.sh) and the GPU vision server (win/), set up WSL and models
```

## Where the behaviour lives

| To change or understand... | Look at |
| :--- | :--- |
| What the model is told, and the agent loop | `robot/agent/agent.py` (`SYSTEM`, `Agent.do`) |
| The tools the model can call, their descriptions and checks | `robot/agent/tools.py` |
| What a tool does in the world: `search`, `go_to`, `follow`, ... | `robot/sim/world.py` (`SimRobot`), overridden on the real stack in `robot/ros/real.py` (`RealRobot`) |
| Driving through Nav2 | `robot/ros/robot.py` (`RosRobot`) |
| What YOLOE looks for, tracking, the model's check of each detection | `robot/perception/detector.py`, `robot/perception/tracker.py`, `robot/ros/perceive.py` |
| YOLOE and depth on the GPU | `tools/vision_server.py` (Windows), `robot/perception/gpu.py` (its client) |
| The model's own calls (yes/no, room names) | `robot/mind/brain.py` |
| The 3D tasks and how each is judged | `robot/eval/home3d.py` (`TASKS`); reports: `robot/eval/report.py` |

## Run it (Windows + WSL)

```powershell
powershell -ExecutionPolicy Bypass -File tools\demo.ps1    # model + vision server + the 3D apartment; talk to it at http://127.0.0.1:8765
```

```bash
# in WSL, from the repo
~/sprocket-venv/bin/python -m pytest -q && ~/sprocket-venv/bin/python -m ruff check .
. tools/linux/env.sh && python -m robot.eval.home3d --tag mine --rtf 0.5 --live 8765 --only N1 S1
```

Models, maps and run output live in this folder too (`models/`, `maps/`, `runs/`, ignored by git), so the folder is
the whole workspace; first-time setup and everything else are in [docs/usage.md](docs/usage.md).

## Debug a 3D run

1. **Watch it live**: run with `--live 8765` and open http://127.0.0.1:8765. You see the request, every tool call and
   its result, the model's reasoning, the search's choices as it makes them, the map, and the three cameras.
2. **Read the task's report**: `runs/<tag>/<scene>/<task>/report.html` (while the run is live, also
   http://127.0.0.1:8765/runs/<tag>/index.html). It has the house's map with the real objects; what the robot
   believed was where, each marked real, false (with the real thing nearest it) or rejected; where it really went and
   where it thought it was; numbered places where each call began; a timeline of every model turn with its full
   reasoning, its calls and their results, and the exact pictures the model was given; and the problems outside the
   turns (the model server timing out, a check that could not run).
3. **Name the cause** before changing code: the harness (prompt, tools, results), perception (false or missing
   detections), the body (Nav2 failing, "status 6", a goal it cannot reach), the check itself, or the model.
4. **Rerun just that task** with `--only <id>`; one run is noisy, so repeat before calling a change better.
5. **Logs**: `runs/<tag>.jsonl` (one line per task: steps, what it knew, model time),
   `runs/home_<scene>/robot.log` (ROS, Nav2, depth), `hardware.log` (the simulated body),
   `%TEMP%\sprocket-vision.log` and `%TEMP%\sprocket-llama.log` on Windows. Stop a run with `bash tools/linux/stop.sh`.
