# Companion Robot Harness

Local-first embodied AI robot harness for `companion_r1` (4-wheel skid-steer base, tiltable head, 5" 800×480 face display, LFM3-VL-2B).

The model decides what to do, which body tools to use, and when it is finished.
Native function calls execute body actions. The model requests the camera with `observe` when it needs vision; normal conversation has no automatic scene input.
It can continue without another prompt, or finish immediately with no movement.
Startup is silent. Changed observations or a model-requested wake trigger background decisions; an unchanged room makes no model calls.
Use `stop` to interrupt and pause autonomy, and **Autonomy on** / `resume autonomy`
to resume. See [model loop](docs/agentic-runtime.md) for the execution contract.

## Quick Start

Python 3.11 or newer is required. Install the core harness with `python -m pip install -e .`.
Add `[audio]` for microphone/speaker support, `[face]` for the standalone face window, or `[dev]`
for tests, linting, and the harness benchmark. The core harness does not import pygame.

For the Raspberry Pi 5 (4 GB), use the harness resource profile:

```bash
python -m robot --config config/robot.pi5.yaml --headless --no-dashboard
```

This profile changes queue/history limits, animation rate, and diagnostic recording. It leaves
model settings at their existing defaults. To retain a custom model configuration, copy its `llm`
section into the profile. Add `--mic` when microphone input is wanted; microphone input requires
a wake word by default. Console/dashboard messages are direct input.

Use `--backend null --mute` for a simulation-only smoke test. The repository currently provides
`SimBody`; a physical motor/sensor driver is still required for real hardware.

### 1. Run Automated Tests
```powershell
& "$env:USERPROFILE\robot-runtime\venv\Scripts\python.exe" -m pytest -v
```

### 2. Start the Local LFM3-Omni-2B Server (Optional — Offline Mode Works Without It)
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

### 4. Benchmark LFM3-Omni-2B on Your Laptop GPU
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
