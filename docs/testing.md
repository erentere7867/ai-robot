# Testing Sprocket

How Sprocket is tested, and how to set up a Linux machine (a coding agent's own VM, or any Ubuntu) to test it the
same way. Commands run from the repository folder.

## From a bare machine

On a fresh Ubuntu 26.04 (a VM, a cloud machine, WSL), three commands give the whole stack, checked:

```bash
bash tools/linux/setup.sh                                  # everything, 15 min on a 4-core VM: what it installs below
. tools/linux/env.sh
python -m robot.eval.doctor --start --3d --rtf 0.1         # checks each piece, starts the servers, runs one 3D task
```

`setup.sh` installs the system packages, ROS 2 lyrical with Nav2 and slam_toolbox, the robot's Python environment
(`~/sprocket-venv`), Habitat-Sim and the ReplicaCAD apartment, llama.cpp (built with CUDA when `nvcc` is there), and
every model from its public source into `models/`; it is safe to run again. The doctor checks the Python packages,
ROS, Habitat, the models, both servers (starting them with `--start`), that the model reads a picture and that YOLOE
answers, and with `--3d` runs task M1 in the apartment end to end; each failure says how to fix it. The robot's own
map of the apartment is in the repo (`maps/apt_0.*`), so the first 3D run does not have to explore it.

ROS 2 lyrical is packaged for Ubuntu 26.04 only, so on another Linux (an Ubuntu 24.04 cloud VM, say) run the same
three commands in an Ubuntu 26.04 container. It needs no GPU or display: Habitat draws with Mesa on the CPU.

```bash
docker run -d --name sprocket --network host --ipc host -v "$PWD":/root/ai-robot -w /root/ai-robot \
  ubuntu:26.04 sleep infinity
docker exec -it sprocket bash                              # then the three commands above, as root
```

`--network host` puts the servers' ports and the live page on the host, and `--ipc host` gives ROS's shared-memory
transport the host's `/dev/shm`.

Verified on 2026-11-19 on a cloud VM (Ubuntu 24.04 host, 4 Xeon cores, 16 GB, no GPU) in a bare `ubuntu:26.04`
container as above, from a clean clone: `setup.sh` in 16 minutes in one go, then the doctor `ready`, with task M1
passing in 6 minutes at `--rtf 0.1`.

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
python3 -m venv --system-site-packages ~/sprocket-venv
~/sprocket-venv/bin/pip install -e ".[eval,vision,dev]"
~/sprocket-venv/bin/python -m ruff check .
~/sprocket-venv/bin/python -m pytest -q
```

They need no GPU, ROS, Habitat or model: the model is scripted (`tests/test_agent.py`), the vision server's decoding is
fed made-up network output (`tests/test_vision_gpu.py`), and the 3D reports are drawn from a made-up run
(`tests/test_report.py`). A few browser tests skip themselves without Chrome and Playwright.

To test motion wiring with ROS 2 Lyrical and Nav2 but no models or Habitat scene, use an unused ROS domain:

```bash
. tools/linux/env.sh
ROS_DOMAIN_ID=196 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RUN_ROS_MOTION_SMOKE=1 \
  python -m pytest -q tests/test_motion_ros.py
```

The test sends a real Nav2 action through Collision Monitor to the Habitat wheel callback using synthetic sensors.
It checks source stamps, cancellation, late replies and direct-drive handoff. It does not measure stopping distance.

## Setting up a Linux machine for levels 2 and 3

Tested on Ubuntu 26.04 with its Python 3.14 and ROS 2 lyrical. Everything below except ROS and Habitat lives in the
repository folder or the venv.

1. **ROS 2 lyrical** from the ROS apt repository (see the ROS 2 installation guide), with the packages the robot's
   stack starts:
   ```bash
   sudo apt install ros-lyrical-ros-base ros-lyrical-nav2-bringup ros-lyrical-slam-toolbox \
     ros-lyrical-robot-localization ros-lyrical-robot-state-publisher
   ```
2. **The robot's Python**: the venv from level 1, made with the same Python ROS uses (the system one) and with the
   system's packages, so `rclpy` from ROS and the Python packages ROS installs with apt load in it.
   `tools/linux/env.sh` puts ROS, the venv and the repo on the path; source it before anything that uses ROS.
3. **The 3D apartment**: `bash tools/linux/setup_habitat.sh` (Habitat-Sim 0.3.3 in its own Python 3.9, the ReplicaCAD
   apartment; about 650 MB). On a machine without a GPU it renders with Mesa on the CPU: slower, but it works.
4. **Models** in `models/` (ignored by git): `bash tools/linux/setup_models.sh` fetches the language model
   (LiquidAI's LFM3-VL-2B GGUF and its vision projector, into `models/llm/`), YOLOE-26l, the pose model, MobileCLIP's
   text encoder, the depth network, the voice and Whisper, and makes the vision server's two ONNX files in
   `models/vision/`.
5. **The language model's server**: llama.cpp build 11960 or newer, built in `~/llama.cpp` (`cmake -B build && cmake
   --build build -j`; with `-DGGML_CUDA=ON` into `build-cuda` for an NVIDIA GPU), then `bash tools/linux/brain.sh`
   (port 8081).
6. **The vision server**: `bash tools/linux/vision.sh` (port 8090; `--stop` to stop). On the CPU by default; with
   `onnxruntime-gpu` installed, `SPROCKET_VISION_PROVIDER=CUDAExecutionProvider` puts it on an NVIDIA GPU.

How fast the pieces are, so runs can be planned:

| Piece | Windows PC, RX 6600 (DirectML, Vulkan) | 6 CPU cores |
| :--- | :--- | :--- |
| YOLOE-26l, one frame | 75 ms | 1.4 s |
| Depth (previous Small model), one frame | 82 ms | 0.5 s |
| The language model reading one picture | 0.15 s | several seconds |

On a CPU-only machine, run the 3D apartment slower (`--rtf 0.1`): the simulated world waits for the robot's software,
so the results stay comparable, only the wall time grows. The model's turns do not wait, though: its client gives up
after 180 s (`robot/mind/brain.py`), and the report then lists `ReadTimeout` under Problems. 4 cores without a GPU is
the least that passes: task M1's first turn (3,500 tokens with the three pictures) took 172 to 176 s there, at
`--rtf 0.1` and at `0.05` alike, with the model on 2 of the cores (`brain.sh` gives it half; on all 4 it took over
180 s).

## On a Colab GPU

Levels 1 and 2 run on a Colab GPU runtime through Google's [Colab CLI](https://github.com/googlecolab/google-colab-cli)
(`uv tool install google-colab-cli`, Linux and macOS), from your machine or a coding agent's cloud session. It signs in
by printing a link: open it anywhere and paste back the code Google shows. Without a system keyring (a container), add
one so the sign-in is kept: `uv tool install --force google-colab-cli --with keyrings.alt`.

```bash
colab new -s sprocket --gpu T4                                   # the free GPU; colab usage shows the units left
echo '!git clone -q https://github.com/erentere7867/ai-robot ~/ai-robot && bash ~/ai-robot/tools/colab/setup.sh' \
  | colab exec -s sprocket --timeout 0                           # 5 minutes: venv, model, llama.cpp, model started
echo '!cd ~/ai-robot && ~/sprocket-venv/bin/python -m pytest -q' | colab exec -s sprocket --timeout 0   # level 1
echo '!cd ~/ai-robot && ~/sprocket-venv/bin/python -m robot.eval.suite --tag colab' \
  | colab exec -s sprocket --timeout 0                           # level 2
colab download -s sprocket /root/ai-robot/runs/colab.jsonl runs/colab.jsonl
colab stop -s sprocket                                           # it bills until stopped
```

A pass of the suite takes 12 to 17 minutes on the T4 (two can run at once: `brain.sh` gives the model two
slots), about 0.3 compute units. Repeat before comparing: on 26 November the same `main` scored 17/50 and then
23/50, and the 23 November code 23/50.

To test a pull request beside `main`: `git -C ~/ai-robot fetch origin pull/<n>/head:pr<n> && git -C ~/ai-robot
worktree add ~/pr<n> pr<n>`, then run the suite from `~/pr<n>` (the venv and the model are shared). `colab exec` fails
after 30 s without output unless given `--timeout 0`; for a long run, start it with `nohup` and read its log. A runtime
started this way is not listed in Colab's *Manage sessions*: `colab sessions` lists it and `colab stop` ends it.

Level 3 does not run on the runtime itself (Ubuntu 24.04, 2 cores, 12 GB, no Docker for ROS 2 lyrical's 26.04), but
it runs on any Linux machine with Docker, in the 26.04 container above, with the language model and the vision server
on the runtime through an SSH tunnel. Measured on 26 November on a 4-core cloud VM: task M1 in 3 minutes at
`--rtf 0.1`, YOLOE 174 ms and depth 203 ms a frame on the T4, 8 MB/s up the tunnel. With the vision server on the VM's
CPU instead it took 3 of the 4 cores, the robot's control loop ran late, and M1 failed.

```bash
mkdir -p ~/.ssh && ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/id_ed25519      # once; colab ssh sends this key
cat >> ~/.ssh/config <<'END'
Host colab-sprocket
  User root
  ProxyCommand colab ssh --proxy-mode -s sprocket
  ControlMaster auto
  ControlPath ~/.ssh/colab-%r@%h
  ControlPersist yes
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
END
# the runtime takes one ssh connection at a time; ControlMaster lets later commands share it, but a tunnel
# loop must not go through it: the client exits at once, so the loop retries every few seconds and is refused
ssh colab-sprocket 'mkdir -p ~/ai-robot/models/vision && cat > ~/ai-robot/models/vision/yoloe26l_8.onnx' \
  < models/vision/yoloe26l_8.onnx                                # made by setup_models.sh; too big for colab upload
ssh colab-sprocket 'cd ~/ai-robot && curl -sfL -o models/vision/depth_base.onnx \
  https://huggingface.co/onnx-community/depth-anything-v2-base/resolve/main/onnx/model.onnx \
  && uv pip uninstall -q --python ~/sprocket-venv/bin/python onnxruntime \
  && uv pip install -q --python ~/sprocket-venv/bin/python "onnxruntime-gpu[cuda,cudnn]" \
  && SPROCKET_VISION_PROVIDER=CUDAExecutionProvider bash tools/linux/vision.sh'
ssh -O exit colab-sprocket   # free the one connection for the tunnel, which reconnects: one drop failed every later task
(while :; do ssh -N -o ControlMaster=no -o ControlPath=none -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
   -L 127.0.0.1:8081:localhost:8081 -L 127.0.0.1:8090:localhost:8090 colab-sprocket; sleep 5; done) &
docker exec -it sprocket bash -c '. tools/linux/env.sh && python -m robot.eval.doctor --3d --rtf 0.1'
```

Behind an HTTPS proxy that re-signs traffic, the container needs the proxy's CA (copied into
`/usr/local/share/ca-certificates/`, then `update-ca-certificates`) and the proxy variables passed to `docker exec`, or
`setup.sh` stops at the ROS repository with "certificate signer not trusted".

## Level 2: the 2D suite

```bash
~/sprocket-venv/bin/python -m robot.eval.suite --tag mine          # --only O3 C1 ..., --list
```

Since 27 November the 2D world sends what the robot sends with each picture (the ids marked on it and the four scan
sectors) and draws an open door as open; scores from before are not comparable. Three passes of the same code on a
T4 ranged 25-30 of 50 that day, so compare the sum of three passes, and read the cases that changed.

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
   perception, measure on pictures, not on one run (how `docs/simulation_3d.md`'s tables were made):
   ```bash
   python -m robot.eval.views render --scene apt_0 --count 40            # views from random spots, with the truth
   python -m robot.eval.views yoloe runs/views/apt_0.npz chair sofa tv   # found and false, per word
   python -m robot.eval.views model runs/views/apt_0.npz chair sofa tv   # the same for the model's yes/no
   ```
3. **Make the smallest general change** ([AGENTS.md](../AGENTS.md)): nothing written for that task, room or word.
4. **Cover the logic with a unit test** where it is logic (a parser, a decision rule, a geometric step), with a fake
   robot or a scripted model like the existing tests.
5. **Lint, unit tests, then the 3D task again, more than once.** One run per task is noisy (between two runs of the
   same 23 tasks, 6 changed outcome), so compare several runs, and other tasks the change could touch.
6. **Write down what was measured** where the next person will look: the code comment beside the decision, and
   `docs/simulation_3d.md` for findings about the robot.
