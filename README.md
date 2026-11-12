# Sprocket

A general-purpose wheeled robot on the Arduino VENTUNO Q. For every instruction a vision-language model writes a short
Python program against a small robot API, a gate checks it, and faster layers below run it in real time.

The design is in [docs/architecture.md](docs/architecture.md).

## Layout

```
firmware/  STM32H5 sketch: wheels, odometry, watchdog, reflexes
robot/
  agent/      the robot API and API card, the planner, the gate, the sandboxed runner, instant commands
  sim/        world.py: the 2D simulator behind the API (gate dry runs, test suite)
  ros/        the 3D simulation: Sprocket's body (3 cameras for 360 degrees, 2D lidar, IMU), Gazebo rooms with
              Fuel furniture, Nav2, perception on ROS, mapping by exploration (explore.py), rooms from the map
              (rooms.py), and run3d.py to run the whole stack
  perception/ YOLOE-26 detector, camera rig geometry, locating detections, colour, the map-frame tracker
  mind/       client for the model (LFM3-VL-2B through llama-server), long-term memory (SQLite)
  interfaces/ the live page (instructions typed or spoken, program and map out), speech (Whisper, Piper), 2D demo
  eval/       the 50-instruction suite, hand-written programs, red-team programs, photo check (M0)
  train/      fine-tuning data generator and the Kaggle notebook
tests/     acceptance tests
```

## Try it

```bash
python -m pip install -e ".[eval,dev]"
python -m pytest -q                 # about 75 s
python -m robot.eval.suite          # hand-written programs through gate, runner and checks
python -m robot.interfaces.demo     # the 2D room behind the live page, http://localhost:8765
```

The 3D run needs WSL (Ubuntu) with ROS 2 Lyrical, Nav2 and Gazebo, a Python environment with ultralytics, and the
YOLOE-26 weights in the working folder:

```bash
python -m robot.ros.run3d --perception --brain http://127.0.0.1:8081    # what the robot sees, against the truth
python -m robot.ros.run3d --perception --turn --watch 50 ...             # the same while turning, with steadiness
python -m robot.ros.run3d --cases O1 O3 O6 --watch 30                    # hand-written programs, judged
python -m robot.ros.run3d --cases O1 O3 --plan --brain http://127.0.0.1:8081   # the planner's programs, judged
python -m robot.ros.run3d --explore --room "two rooms" --watch 300 ...   # map from scratch, save to ~/sprocket-maps
python -m robot.ros.run3d --cases S1 S6 --own-map ...                    # localise and plan in the robot's own map
python -m robot.ros.run3d --interactive --gui --brain http://127.0.0.1:8081   # Gazebo and the page (with memory)
```

On the laptop everything heavy runs on the GPU inside WSL: llama-server built with CUDA (`-ngl 99 -fa on`, two slots,
the vision projector), YOLOE on CUDA PyTorch, and Gazebo rendering through Mesa's Direct3D 12 driver. Speech needs
`faster-whisper`, `piper-tts` and a Piper voice in `~/sprocket-models/voice`.

The photo check for the model (M0) needs a `llama-server` build (b11960 or newer) running
`LFM3-VL-2B-QAD-Q4_0.gguf` with `mmproj-LFM3-VL-2B-F16.gguf`:

```bash
python -m robot.eval.brain_check --fetch
python -m robot.eval.brain_check
```
