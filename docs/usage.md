# Using the repository

Everything below is for a Windows PC with WSL 2 (Ubuntu), which is how Sprocket is developed until the board
arrives: the language model and the vision models run on the Windows GPU, the robot's software (ROS 2, Nav2, the
agent) and the 3D apartment run in WSL.

The repository folder is the whole workspace: besides the code it holds `models/`, `maps/`, `runs/` and `data/`,
ignored by git (`robot/paths.py`; an environment variable such as `SPROCKET_MODELS_DIR` moves one). Installed
outside it:

| What | Where |
| :--- | :--- |
| llama.cpp (Vulkan build) | `C:\llama-vulkan\llama-server.exe`, or `$env:LLAMA_SERVER` |
| Python for the vision server (onnxruntime-directml, numpy) | `%USERPROFILE%\sprocket-win` |
| Python for the robot (this repo installed editable, ultralytics, ROS's rclpy) | WSL `~/sprocket-venv` |
| ROS 2 lyrical with Nav2 and slam_toolbox | WSL `/opt/ros/lyrical` |
| Habitat-Sim (Python 3.9) and the apartment's data | WSL `~/micromamba/envs/habitat`, `~/habitat-data` |

## What runs where

| Piece | Where | Port | Start |
| :--- | :--- | :--- | :--- |
| Language model (LFM3-VL-2B, llama.cpp) | Windows GPU (Vulkan) | 8081 | `tools\brain.ps1` |
| Vision server (YOLOE-26l + Depth Anything, ONNX Runtime DirectML) | Windows GPU | 8090 | `tools\win\vision.ps1` |
| 3D apartment (Habitat-Sim + ReplicaCAD) | WSL, rendered on the GPU | local socket | started by the eval or the demo |
| The robot: ROS 2, Nav2, perception, the agent | WSL | live page 8765 | `python -m robot.eval.home3d` or the demo |

WSL reaches both Windows servers at `127.0.0.1`. From Windows, open pages at `http://127.0.0.1:<port>`; `localhost`
can time out.

## First-time setup

1. **Language model.** llama.cpp build 11960 or newer (LFM3 support), at `C:\llama-vulkan\llama-server.exe` or named by
   `$env:LLAMA_SERVER`. Put `LFM3-VL-2B-QAD-Q4_0.gguf` and `mmproj-LFM3-VL-2B-F16.gguf` in `llm\` under
   the repo's `models\` (or `$env:SPROCKET_MODELS_DIR`).
2. **WSL environment.** ROS 2 lyrical with Nav2 and slam_toolbox in `/opt/ros/lyrical`, and the robot's venv at
   `~/sprocket-venv` with this repo installed: `~/sprocket-venv/bin/pip install -e ".[eval,vision,dev]"` (ultralytics
   brings torch, for the words' text features).
3. **The 3D apartment.** `wsl -- bash tools/wsl/setup_habitat.sh` (GPU rendering in WSL, Habitat-Sim in its own
   Python 3.9, ReplicaCAD; about 650 MB).
4. **Models.** `bash tools/wsl/setup_models.sh` checks `models/` and fetches YOLOE-26l, the pose model,
   MobileCLIP's text encoder, the Piper voice and Whisper. `--export <dir>` and `--import <dir>` carry the whole folder
   to another machine.
5. **Vision server.** A full Python 3.10 or newer on Windows (not the embeddable one, which has no `venv`):
   ```powershell
   python -m venv $env:USERPROFILE\sprocket-win
   & $env:USERPROFILE\sprocket-win\Scripts\pip install onnxruntime-directml numpy
   ```
   then make its models in WSL: `bash tools/wsl/setup_models.sh --gpu-models` (into `models/vision/`)
   (YOLOE exported with 8 word slots by `tools/wsl/export_yoloe.py`, and the depth network).

## Every day

```powershell
powershell -ExecutionPolicy Bypass -File tools\brain.ps1          # the model, if not running
powershell -ExecutionPolicy Bypass -File tools\win\vision.ps1     # the vision server, if not running
powershell -ExecutionPolicy Bypass -File tools\demo.ps1           # or both of them and the demo: talk to it at http://127.0.0.1:8765
powershell -ExecutionPolicy Bypass -File tools\brain.ps1 -Stop    # stop the model; tools\win\vision.ps1 -Stop for the other
```

## Tests and lint (in WSL, from the repo)

```bash
~/sprocket-venv/bin/python -m pytest -q
~/sprocket-venv/bin/python -m ruff check .
```

## The 2D suite

Fifty instructions in drawn 2D rooms, fast; the model must be running.

```bash
~/sprocket-venv/bin/python -m robot.eval.suite --tag mine            # --only O3 C1 ..., --list
```

## The 3D apartment

The real robot software in a ReplicaCAD apartment: only the world, the body and the raw sensors are simulated
([simulation_3d.md](simulation_3d.md)). The first run in a scene explores it for about 30 minutes and saves the map in
`maps/`; later runs localise in it.

```bash
. tools/wsl/env.sh
python -m robot.eval.home3d --tag mine --rtf 0.5 --live 8765 --only N1 P1 S1
bash "<repo>/tools/wsl/stop.sh"                # stops everything the run started
```

- `--live 8765` serves the live page while it runs: the request, each tool call and its result, the model's
  reasoning, the search's choices, the map and the three cameras.
- Each task leaves a report at `runs/<tag>/<scene>/<task>/report.html` (also at
  `http://127.0.0.1:8765/runs/<tag>/index.html` while the run is live): the house's map with the real objects, what the
  robot believed was where (false ones in red), where it really went and where it thought it was; then every model turn
  with its reasoning, calls, results and the exact pictures it was given.
- Results: `runs/<tag>.jsonl`; logs of the robot's processes: `runs/home_<scene>/`.
- `--rtf` is simulated seconds per second; 0.5 leaves room for the GPU and the model. One task takes a few minutes.

## Other tools

- `python -m robot.eval.brain_check --fetch`, then `python -m robot.eval.brain_check`: the model's yes/no answers and
  its picking of numbered objects on real indoor photos (COCO), against human labels.
- `tools/wsl/export_yoloe.py`: export YOLOE with a different size or number of word slots.
- `tools/wsl/brain.sh`: the model on llama.cpp built in WSL or on Linux (the board).
