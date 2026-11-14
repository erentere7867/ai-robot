# Sprocket v2

A living, visually aware general-purpose wheeled robot on the Arduino VENTUNO Q.

Sprocket pairs a **30–50 Hz Driverless-Car Living Body Engine** with **360° Pure-Vision Perception (3 × MIPI-CSI cameras + 8×8 ToF)**, a **Two-Tier Circulating Hybrid Memory**, and a **Closed-Loop Visual Mind (`LFM3-VL-2B` + `d1-2B`)** that sees annotated frames and acts continuously in dynamic environments.

The full specification is in [docs/architecture.md](docs/architecture.md).

## Layout

```text
firmware/  STM32H5 sketch: 200 Hz wheel control, VL53L5CX step/cliff reflex, IMU + anti-snag
robot/
  body/       30–50 Hz Living Body Engine: trajectory rollout, social yielding, aliveness, active vision
  perception/ 3-CSI camera rig, metric floor/ToF depth calibration, synthetic /scan, pose/gaze, SoM+PIVOT overlay
  mind/       LFM3-VL-2B + d1-2B client, Two-Tier Circulating Hybrid Memory (SQLite + vectors + RAM scratchpad)
  agent/      Closed-loop visual planner, Intent schema, continuous Sense-Think-Act controller, instant commands
  sim/        2D dynamic simulator behind the Living Body & Perception APIs
  ros/        3D Gazebo & ROS 2 stack (3-CSI rig, metric depth, synthetic /scan, rooms, exploration, RealRobot)
  interfaces/ Live web dashboard, Whisper + WeSpeaker voiceprint + DOA ring buffer, Piper + expressive chimes
  eval/       Evaluation suite and photo verification checks
  train/      Scene generator and closed-loop dataset builder
tests/        Unit and integration tests
```

## Quick Start

```bash
python -m pip install -e ".[eval,dev]"
python -m pytest -q
python -m ruff check .
python -m robot.interfaces.demo
```
