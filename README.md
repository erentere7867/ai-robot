# Sprocket

A general-purpose wheeled robot on the Arduino VENTUNO Q. For every instruction a vision-language model writes a short
Python program against a small robot API, a gate checks it, and faster layers below run it in real time.

The design is in [docs/architecture.md](docs/architecture.md).

## Layout

```
config/    robot geometry (body.yaml), MuJoCo model and meshes
firmware/  STM32H5 sketch: wheels, odometry, watchdog, reflexes
robot/
  agent/      the robot API and API card, the gate, the sandboxed runner
  sim/        world.py: 2D simulator behind the API (gate dry runs, test suite); MuJoCo room and physics for ROS 2
  mind/       client for the model (LFM3-VL-2B through llama-server)
  perception/ camera geometry, frames, lidar scans
  eval/       the 50-instruction suite, hand-written programs, red-team programs, photo check (M0)
tests/     phase acceptance tests
```

## Try it

```bash
python -m pip install -e ".[sim,eval,dev]"
python -m pytest -q                 # phase 4 acceptance in simulation, about 15 s
python -m robot.eval.suite          # hand-written programs through gate, runner and checks
```

The photo check for the model (M0) needs a `llama-server` build (b11960 or newer) running
`LFM3-VL-2B-QAD-Q4_0.gguf` with `mmproj-LFM3-VL-2B-F16.gguf`:

```bash
python -m robot.eval.brain_check --fetch
python -m robot.eval.brain_check
```
