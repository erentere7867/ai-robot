# Companion Robot Harness

Local-first embodied AI robot harness for `companion_r1` (4-wheel skid-steer base, tiltable head, 5" 800×480 face display, LFM3-VL-2B).

A small vision-language model decides what to do; generic closed-loop skills do it. The robot sees
with open-vocabulary detection (the VLM's own grounding), places what it sees on a map with camera
geometry, plans paths around obstacles it has sensed, and remembers facts, places and events.
"What do you see?" → "go behind that box" works without any task-specific code: the model calls
`observe`, then `go_to(target=box_1, relation=behind)`, and the skill plans, drives, replans and
checks the result.

When nobody is talking to it, drives (curiosity, social, play, battery) and camera changes give the
pet its own reasons to act; the model decides whether and how. Use `stop` to interrupt and pause
autonomy, and **Autonomy on** / `resume autonomy` to resume.

See [architecture](docs/architecture.md) for how it works, how to measure it, and the Raspberry Pi
plan, and [evaluation](docs/evaluation.md) for the simulator test suite.

Try it from a terminal (needs the model files below):

```bash
python scripts/try_robot.py "what do you see?" "go behind that box" "now come back to where you started"
```

## Quick Start

Python 3.11 or newer is required. Install the core harness with `python -m pip install -e .`.
Add `[audio]` for microphone/speaker support, `[face]` for the standalone face window, or `[dev]`
for tests, linting, and the harness benchmark. The core harness does not import pygame.

For the Raspberry Pi 5 (4 GB), use the harness resource profile:

```bash
python -m robot --config config/robot.pi5.yaml --headless --no-dashboard
```

This profile uses CPU inference, smaller history, and a slower autonomy cadence. Add `--mic` when
microphone input is wanted; microphone input requires a wake word by default. Console/dashboard
messages are direct input.

Use `--backend null --mute` for a wiring smoke test without the model (the robot talks but never
moves). The repository provides simulated bodies (`SimBody`, and a MuJoCo-dynamics body selected
with `runtime.body_backend: physics`); a physical motor/sensor driver is still required for real
hardware.

### 1. Run Automated Tests
```powershell
& "$env:USERPROFILE\robot-runtime\venv\Scripts\python.exe" -m pytest -v
```

### 2. Start the Local LFM3-VL-2B Server (the harness also starts it automatically)
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_llm.ps1
```

### 3. Run the Robot Harness with Web Dashboard & 60 FPS SDL2 Face Window
```powershell
# Interactive console + Web Dashboard at http://127.0.0.1:8090 + 60 FPS SDL2 Face Window
& "$env:USERPROFILE\robot-runtime\venv\Scripts\python.exe" -m robot --face-window

# Console + Web Dashboard only (face also renders live inside the browser dashboard)
& "$env:USERPROFILE\robot-runtime\venv\Scripts\python.exe" -m robot

# Standalone 60 FPS SDL2 Face Process (listens on UDP 127.0.0.1:18991)
& "$env:USERPROFILE\robot-runtime\venv\Scripts\python.exe" -m robot.face.process
```

### 4. Benchmark the Model on Your Machine
```powershell
& "$env:USERPROFILE\robot-runtime\venv\Scripts\python.exe" .\scripts\bench_llm.py
```

### 5. Replay a Recorded Session
```powershell
& "$env:USERPROFILE\robot-runtime\venv\Scripts\python.exe" .\scripts\replay.py <path-to-session.jsonl>
```

Replay defaults to passive playback of the recorded outputs. It does not run inference or write
to memory. `--mode simulate` re-runs user inputs through the offline backend using a temporary
database; its results can differ from the original session. Replay refuses the normal memory DB.

Measure the harness separately from the model:

```bash
python scripts/bench_runtime.py --config config/robot.pi5.yaml --turns 1000 --dashboard --output artifacts/runtime-benchmark.json
python -m ruff check robot scripts tests
python -m pytest -q
```

The benchmark uses an isolated temporary database, a deterministic offline backend and simulated
hardware. It reports RSS, Python allocation growth, CPU time, and retained work. `tracemalloc`
adds overhead, so its turn times are diagnostic rather than a hardware latency guarantee.

See [runtime notes](docs/runtime-hardening.md) for cancellation semantics, resource limits,
recording retention, and validation scope.
