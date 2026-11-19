# Sprocket v2

A pet-like, helpful wheeled robot on the Arduino VENTUNO Q, with everything running on the robot.

Its mind is an **agentic tool-calling loop** around an on-board vision-language model (`LFM3-VL-2B` for now). The
model gets the three cameras' pictures and the user's words, answers what it can from the pictures, and acts by
calling tools: `go_to("sofa")`, `search("keys", area="bedroom")`, `follow("user")`, `go_toward("behind-left", 2)`. The
tools do the real-time work: YOLOE looks for the words they name, a tracker follows what it finds, depth from the
cameras makes a 360° scan, and Nav2 plans and drives. Every result goes back to the model before its next step.

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

## Quick start (Windows + WSL)

```powershell
powershell -ExecutionPolicy Bypass -File tools\demo.ps1    # model + vision server + the 3D apartment; open http://127.0.0.1:8765
```

First-time setup, the tests and the evals are in [docs/usage.md](docs/usage.md).
