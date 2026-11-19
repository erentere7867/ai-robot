# Working on Sprocket

For people and coding agents changing this repository.

## Start here

1. [README.md](README.md): what the robot is, **where each behaviour lives in the code** (a table), how to run it, and
   **how to debug a 3D run** (live page, per-task reports, logs).
2. [docs/usage.md](docs/usage.md): setup and every command.
3. [docs/architecture.md](docs/architecture.md): the design; section 6 is the agent as it is.
4. [docs/simulation_3d.md](docs/simulation_3d.md): what has been measured, including what did not work and why (read
   it before trying an idea again: picking numbered dots, the fixed word list and the fridge task are there).

When a 3D task fails, open its report (`~/robot-runtime/runs/<tag>/<task>/report.html`) first: it shows what the robot
believed against what was really there, where it went, and every model turn with the pictures it was given. Most
wrong guesses about a failure end there.

## Rules

1. **The model decides; code does not route around it.** No regular expressions, keyword lists or routers that read
   the user's words or the model's output to pick an action, and no silent rewriting of what the model chose. If the
   model chooses badly, change what it sees (pictures, tool descriptions, results, the prompt) or what the tools can
   do, and measure. Checking a tool's arguments and answering with an error that says what to do instead is fine.
2. **Delete what is not needed**, with its tests. Dead code, unused options and old experiments go; git keeps them.
3. **Fix causes, with evidence.** Find out why before changing anything (the per-task reports, the saved pictures, a
   measurement on the test pictures), say what was measured in the code's comment or the docs, and do not add
   fallbacks that hide a failure: a missing server or model fails plainly.
4. **The simulator runs the real robot.** In the 3D apartment only the world, the body and the raw sensors are
   simulated; perception, depth, Nav2 and the agent are the robot's own. Nothing from the simulator's truth reaches the
   robot; the truth is for judging only.
5. **Judge the harness in 3D, not only in 2D**, and read the task reports (`~/robot-runtime/runs/<tag>/<task>/report.html`)
   before deciding what went wrong: say whether a failure is the harness, the body (Nav2, the map), the check, or the
   model. One run per task is noisy; repeat before calling a change better or worse.
6. **Everything runs on the robot**: the Arduino VENTUNO Q (Adreno GPU, 40 TOPS NPU, 16 GB). No cloud models. Sensors
   and chassis cost under $150; indoor first. The language model (LFM3-VL-2B) is provisional, so keep the harness
   model-agnostic.

## Before you finish a change

```bash
~/sprocket-venv/bin/python -m ruff check .
~/sprocket-venv/bin/python -m pytest -q
```

Then the 3D tasks the change touches (`python -m robot.eval.home3d --only ...`, see usage.md).

## Style

- Python 3.11 (`robot/sim/habitat_server.py` runs in Habitat-Sim's Python 3.9), lines up to 120, ruff's rules in
  `pyproject.toml`.
- Comments and docstrings say why, in plain words, with the number that decided it ("found 54% of tables from 1 m and
  8% from 22 cm"), not what the next line does.
- Tool results and errors are short sentences the model reads: say what happened and what it can do next.

## Pitfalls on this machine

- Run Python in WSL (`~/sprocket-venv`); Windows' own Python has no numpy. The language model and the vision server
  are Windows programs; WSL reaches them at `127.0.0.1`.
- Source `tools/wsl/env.sh` before any ROS process: ROS 2 over UDP finds peers but delivers no data here; it uses shared
  memory instead.
- rclpy processes ignore SIGTERM: stop the stack with `tools/wsl/stop.sh`.
- Run ultralytics from `~/sprocket-models`; elsewhere it downloads its 242 MB text encoder again.
- Through `wsl -- bash -lc '...'` shell variables can arrive empty: put longer commands in a script file.
