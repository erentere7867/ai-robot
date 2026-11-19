# Testing Sprocket

How Sprocket is tested, and how to set up a Linux machine (a coding agent's own VM, or any Ubuntu) to test it the
same way. Commands run from the repository folder.

## Three levels

| Level | Catches | Needs | Time |
| :--- | :--- | :--- | :--- |
| 1. Lint and unit tests | Broken logic, tool argument checks, the agent loop with a scripted model, perception maths, reports | Python only | seconds |
| 2. The 2D suite | The model choosing tools well in drawn rooms (50 instructions) | + the language model | minutes |
| 3. The 3D apartment | Everything together: the model, perception, depth, Nav2, the body, in a real-looking home | + ROS 2, Habitat, the vision server | minutes per task |

Level 1 runs on GitHub for every pull request and push to `main` (`.github/workflows/tests.yml`) and must pass before a
pull request can merge. Levels 2 and 3 run on a machine set up as below. A change to the robot's behaviour (prompt,
tools, perception, navigation) is judged at level 3, not only 1 and 2 ([AGENTS.md](../AGENTS.md)).

## Level 1: lint and unit tests

```bash
python3 -m venv ~/sprocket-venv
~/sprocket-venv/bin/pip install -e ".[eval,vision,dev]"
~/sprocket-venv/bin/python -m ruff check .
~/sprocket-venv/bin/python -m pytest -q
```

They need no GPU, ROS, Habitat or model: the model is scripted (`tests/test_agent.py`), the vision server's decoding is
fed made-up network output (`tests/test_vision_gpu.py`), and the 3D reports are drawn from a made-up run
(`tests/test_report.py`). A few browser tests skip themselves without Chrome and Playwright.

## Setting up a Linux machine for levels 2 and 3

Tested on Ubuntu 26.04 with its Python 3.14 and ROS 2 lyrical. Everything below except ROS and Habitat lives in the
repository folder or the venv.

1. **ROS 2 lyrical** from the ROS apt repository (see the ROS 2 installation guide), with the packages the robot's
   stack starts:
   ```bash
   sudo apt install ros-lyrical-ros-base ros-lyrical-nav2-bringup ros-lyrical-slam-toolbox \
     ros-lyrical-robot-localization ros-lyrical-robot-state-publisher
   ```
2. **The robot's Python**: the venv from level 1, made with the same Python ROS uses (the system one), so `rclpy`
   from ROS loads in it. `tools/linux/env.sh` puts ROS, the venv and the repo on the path; source it before anything
   that uses ROS.
3. **The 3D apartment**: `bash tools/linux/setup_habitat.sh` (Habitat-Sim 0.3.3 in its own Python 3.9, the ReplicaCAD
   apartment; about 650 MB). On a machine without a GPU it renders with Mesa on the CPU: slower, but it works.
4. **Models** in `models/` (ignored by git):
   - the language model: `LFM3-VL-2B-QAD-Q4_0.gguf` and `mmproj-LFM3-VL-2B-F16.gguf` from LiquidAI's LFM3-VL-2B GGUF
     release on Hugging Face, into `models/llm/`;
   - the rest: `bash tools/linux/setup_models.sh --gpu-models` fetches YOLOE-26l, the pose model, MobileCLIP's text
     encoder, the voice and Whisper, and makes the vision server's two ONNX files in `models/vision/`.
5. **The language model's server**: llama.cpp build 11960 or newer, built in `~/llama.cpp` (`cmake -B build && cmake
   --build build -j`; with `-DGGML_CUDA=ON` into `build-cuda` for an NVIDIA GPU), then `bash tools/linux/brain.sh`
   (port 8081).
6. **The vision server**: `bash tools/linux/vision.sh` (port 8090; `--stop` to stop). On the CPU by default; with
   `onnxruntime-gpu` installed, `SPROCKET_VISION_PROVIDER=CUDAExecutionProvider` puts it on an NVIDIA GPU.

How fast the pieces are, so runs can be planned:

| Piece | Windows PC, RX 6600 (DirectML, Vulkan) | 6 CPU cores |
| :--- | :--- | :--- |
| YOLOE-26l, one frame | 75 ms | 1.4 s |
| Depth, one frame | 82 ms | 0.5 s |
| The language model reading one picture | 0.15 s | several seconds |

On a CPU-only machine, run the 3D apartment slower (`--rtf 0.1`): the simulated world waits for the robot's software,
so the results stay comparable, only the wall time grows.

## Level 2: the 2D suite

```bash
~/sprocket-venv/bin/python -m robot.eval.suite --tag mine          # --only O3 C1 ..., --list
```

## Level 3: the 3D apartment

```bash
. tools/linux/env.sh
python -m robot.eval.home3d --tag mine --rtf 0.5 --live 8765 --only N1 S1
bash tools/linux/stop.sh                       # stops everything a run started
```

The first run in a layout explores it for about 30 minutes and saves the map in `maps/`. Each task's results go to
`runs/<tag>.jsonl`, and its report to `runs/<tag>/<scene>/<task>/report.html`.

## How we test a change

1. **Reproduce it in 3D first.** Run the task that shows the problem alone (`--only <id> --live 8765`) and open its
   report. The report shows the house's real objects against what the robot believed was there (false ones marked,
   with the real thing nearest each), where it really went and where it thought it was, and every model turn with its
   reasoning, its calls and their results, and the exact pictures it was given.
2. **Name the cause** before changing anything: the harness (prompt, tool descriptions, results), perception (a false or
   missing detection), the body (Nav2 failing, "status 6", a goal it cannot reach), the check, or the model. For
   perception, measure on pictures, not on one run: render views of a home with Habitat, run the detector or the model
   over them, and compare with the true objects (how `docs/simulation_3d.md`'s tables were made).
3. **Make the smallest general change** ([AGENTS.md](../AGENTS.md)): nothing written for that task, room or word.
4. **Cover the logic with a unit test** where it is logic (a parser, a decision rule, a geometric step), with a fake
   robot or a scripted model like the existing tests.
5. **Lint, unit tests, then the 3D task again, more than once.** One run per task is noisy (between two runs of the
   same 23 tasks, 6 changed outcome), so compare several runs, and other tasks the change could touch.
6. **Write down what was measured** where the next person will look: the code comment beside the decision, and
   `docs/simulation_3d.md` for findings about the robot.
