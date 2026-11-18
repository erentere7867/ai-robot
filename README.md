# Sprocket v2

A living, visually aware general-purpose wheeled robot on the Arduino VENTUNO Q.

Sprocket's mind is an **agentic tool-calling loop** around an on-board vision-language model (`LFM3-VL-2B` for now): it sees the cameras, calls tools in the model's native format, reads every result, keeps a to-do list, and ends a task by replying in words. Under it sit a body with its own control loops, **360° pure-vision perception (3 × MIPI-CSI cameras + 8×8 ToF)** and long-term memory.

The specification is in [docs/architecture.md](docs/architecture.md); the agent's design and measurements are in [docs/ideas/sprocket-harness.md](docs/ideas/sprocket-harness.md).

## Layout

```text
firmware/  STM32H5 sketch: 200 Hz wheel control, VL53L5CX step/cliff reflex, IMU + anti-snag
robot/
  body/       30–50 Hz Living Body Engine: trajectory rollout, social yielding, aliveness, active vision
  perception/ 3-CSI camera rig, metric floor/ToF depth calibration, synthetic /scan, pose/gaze, SoM+PIVOT overlay
  mind/       model client (native tool calls, d1-2B scoring), long-term memory (SQLite)
  agent/      the agent loop, its tools (checked, with errors that teach), to-do list, instant stop/wait
  sim/        2D simulator behind the robot API, with drawn perspective cameras
  ros/        3D Gazebo & ROS 2 stack (3-CSI rig, metric depth, synthetic /scan, rooms, exploration, RealRobot)
  interfaces/ Live web dashboard, Whisper + WeSpeaker voiceprint + DOA ring buffer, Piper + expressive chimes
  eval/       Evaluation suite and photo verification checks
  train/      random scene generator (the post-training data builder comes next)
tests/        Unit and integration tests
```

## Quick Start

The model runs on llama-server (build 11960 or newer, for LFM3) on port 8081. On Windows, from the repository
folder in PowerShell (`tools/wsl/brain.sh` does the same inside WSL or Linux with llama.cpp built there):

```powershell
wsl -- bash tools/wsl/setup_habitat.sh                             # once: GPU rendering in WSL + the 3D apartment
powershell -ExecutionPolicy Bypass -File tools\demo.ps1           # the model (if not running) + the demo; open http://127.0.0.1:8765
powershell -ExecutionPolicy Bypass -File tools\brain.ps1 -Stop    # stop the model when done
```

The demo starts in the **3D apartment**: a real-looking home (ReplicaCAD, 119 objects) rendered by Habitat-Sim on the
GPU, behind the same robot API as the fast 2D rooms (which the room picker still offers). Everything else runs in
Python with numpy (the WSL venv `~/sprocket-venv` on this machine):

```bash
python -m pip install -e ".[eval,dev]"
python -m pytest -q
python -m ruff check .
python -m robot.eval.suite --tag mine         # the 50-task suite in the fast 2D world
python -m robot.eval.home3d --tag mine        # 10 3D reasoning tasks in the 3D apartment
python -m robot.interfaces.demo               # talk to it at http://127.0.0.1:8765
```

Gazebo with ROS 2 and Nav2 (`robot/ros/`, [docs/simulation_3d.md](docs/simulation_3d.md)) is for testing the ROS
integration before the real robot, not for day-to-day work on the agent.
