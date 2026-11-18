# Sprocket v2 — 3D Simulation Guide

There are two 3D simulators. Use the first for work on the agent; the second for the ROS integration.

## 0. The 3D apartment (Habitat-Sim): the agent's 3D reasoning

`robot/sim/home3d.py` puts a real-looking apartment (ReplicaCAD `apt_0`: 119 objects, 67 m² of drivable ground
floor) behind the same robot API as the 2D world. The robot drives with the 2D world's kinematics and planner on the
home's real floor (from Habitat's navmesh, for a 0.3 m tall robot that climbs 3 cm: no stairs); everything it sees is
rendered: the three cameras at 0.22 m, and which objects are in view (at least 60 of a camera's 512×288 pixels in a
semantic camera, however far away: size in the picture, not a distance cap, decides), so furniture hides what is
behind, under or on top of it. Rooms are found on the floor map as the real robot would (split at doorways, named by
their furniture; an open-plan room with a fridge and a sofa becomes a kitchen end and a living-room end).

- **GPU in WSL:** WSL ships Direct3D 12 for Mesa, but Mesa renders on the CPU unless `GALLIUM_DRIVER=d3d12` is set.
  With it, on an AMD RX 6600: 255 robot views per second (three 512×288 cameras), against 27 on the CPU.
- **Habitat-Sim 0.3.3 is built only for Python 3.9**, so it runs in its own micromamba env
  (`~/micromamba/envs/habitat`) as `robot/sim/habitat_server.py`; the robot code talks to it over a local socket.
  Its EGL context is opened by device number (`gpu_device_id = -1`), not the CUDA lookup that fails on AMD cards.
- **Setup once:** `wsl -- bash tools/wsl/setup_habitat.sh` (the GPU setting, micromamba, the env, the free scenes).
- **Run:** `python -m robot.eval.home3d` (27 tasks with automatic checks, worked out from whichever layout they run
  in: `apt_0` and `v3_sc0_staging_00` by default, `--scenes` for any of ReplicaCAD's 91), or the demo, which starts
  in it. This is the test that counts now; the 2D suite stays only as a quick smoke test.

What it showed first: from 0.22 m the cameras never see the surface of an 86 cm kitchen counter, or a sofa's seat
from its side. From a few metres back they do see the taller things on a counter (a kettle, a bowl), but nothing flat
on it, and the closer the robot gets the less it sees. "What is on the table?" needs a taller camera mast or a camera
that tilts up.

Real homes also found a body bug the 2D rooms never did: the path follower (`robot/body/controller.py`) drove only
forward arcs, so after a search left the robot in a corner beside the TV stand, facing the wall, every route out
stalled at its first waypoint. It now turns on the spot to face a waypoint more than 45° off before driving (and
keeps turning until nearly facing it, or it wobbles between turning and arcing away). It also cut corners: it switched
to the next waypoint 28 cm early even when the straight line ran into a cabinet's corner, where the collision monitor
held it. It cuts a corner only when that line is clear. Driving to every object in `apt_0` and `v3_sc0_staging_00` in
turn, without the model, now reaches 140 of 142 (one stall, one with no path).

## 0.1 Gazebo + ROS 2 + Nav2: the ROS integration

This part documents the full 3D Gazebo + ROS 2 `lyrical` simulation stack for Sprocket v2, the on-disk AI & 3D asset inventory, verification commands, and how to export/import the self-contained model bundle to a faster PC. In WSL, Gazebo also needs `GALLIUM_DRIVER=d3d12` to render on the GPU (`tools/wsl/run3d.sh` sets it); on the CPU it takes minutes to come up.

## 1. Architecture Overview

The 3D simulator (`robot/ros/stage.py`, `robot/ros/run3d.py`) mirrors the physical robot's sensor suite, kinematics, and inference stack in Gazebo (`gz sim`, ODE physics at `max_step_size=0.003` s, Ogre2 sensor rendering):

- **Chassis & Motion (`robot/ros/model.py`)**:
  - 0.30 m diameter cylindrical differential-drive base (`BODY_R=0.15` m, `TRACK=0.24` m, `WHEEL_R=0.05` m) with front/rear casters.
  - Gazebo `DiffDrive` plugin consumes `/cmd_vel` and publishes `/odom` (50 Hz), `JointStatePublisher` publishes `joint_states`, and `OdometryPublisher` emits `/ground_truth` (for offline evaluation only; never read by the robot).
  - `robot_localization` `ekf_node` (30 Hz) fuses wheel linear velocity (`vx`, `vy` from `/odom`) with IMU yaw rate (`vyaw` from `/imu` at 100 Hz) into `odom -> base_link` so on-spot spins avoid wheel-slip drift.
- **360° Pure-Vision + Multi-Zone ToF Rig (`robot/ros/model.py`, `robot/perception/rig.py`)**:
  - **3× 140° RGB-D CSI cameras** (`hfov=2.44` rad, `1280x720` @ 10 Hz) at 120° spacing (`front_camera` at `0°` on `/rgbd_camera/*`, `rear_left_camera` at `+120°`, `rear_right_camera` at `-120°`).
  - **2× 8×8 `VL53L5CX` multi-zone ToF sensors** (`front_tof_link`, `rear_tof_link` -> `/tof/front`, `/tof/rear`) for step/rug/cliff hazard classification (`classify_tof_grid` in `robot/perception/depth.py`) and metric depth anchoring.
  - **360° Synthetic `/scan` (`SyntheticScanPublisher`, `robot/ros/perceive.py`)**: calibrates each camera's depth map with IMU roll/pitch and ToF range anchors (`calibrate_depth`) and projects 3-CSI obstacle bands (`z` in `[0.03, 0.38]` m) into a 360-beam `LaserScan` on `/scan` (`synthetic_scan`) for Nav2/AMCL localization without hardware LiDAR.
- **AI Perception & Voice (`robot/perception/detector.py`, `robot/interfaces/voice.py`)**:
  - **YOLOE-26s-seg + MobileCLIP2** (`yoloe-26s-seg.pt`, `mobileclip2_b.ts`): open-vocabulary instance segmentation across all 3 cameras, with runtime `watch()` vocabulary expansion and floor-contact mask estimation (`_feet`).
  - **YOLO26n-pose** (`yolo26n-pose.pt`): lazily executed when a `"person"` is detected to attach 17 COCO keypoints (`Seen.keypoints`) for 3D posture/pointing-ray estimation (`robot/perception/pose.py`).
  - **YOLO26n-depth** (`yolo26n-depth.pt`, enabled via `--mono-depth`): monocular depth estimation (`Detector.estimate_depth`) calibrated via floor geometry + ToF anchors.
  - **Faster-Whisper (`base.en`) & Piper TTS (`en_US-lessac-medium`)**: offline speech-to-text (`Ears`) and neural speech synthesis (`Mouth`) attached to the Live web UI (`http://localhost:8765`).
- **On-Board VLM Brain (`tools/wsl/brain.sh`, `robot/mind/brain.py`)**:
  - **`LFM3-VL-2B-QAD-Q4_0.gguf` + `mmproj-LFM3-VL-2B-F16.gguf`** served via `llama.cpp` (`llama-server`, GPU CUDA `build-cuda/bin/llama-server` if built, else CPU) on `http://127.0.0.1:8081`, powering grammar-constrained closed-loop `LocalPlanner`, Set-of-Mark grounding, and object crop verification.

## 2. Exact Model Inventory & Disk Paths

All local AI weights live under `$SPROCKET_MODELS_DIR` (default `~/sprocket-models`) and Gazebo 3D meshes under `$GZ_FUEL_CACHE` (default `~/.gz/fuel`):

| Component | Path on Disk | Approx. Size | Purpose |
| --- | --- | --- | --- |
| VLM Language Weights | `~/sprocket-models/llm/LFM3-VL-2B-QAD-Q4_0.gguf` | ~1.0 GB | `llama-server` closed-loop planner & VLM |
| VLM Vision Projector | `~/sprocket-models/llm/mmproj-LFM3-VL-2B-F16.gguf` | ~362 MB | Multimodal image encoding for `LFM3-VL-2B` |
| Open-Vocab Detector | `~/sprocket-models/yoloe-26s-seg.pt` | ~30 MB | Open-vocabulary segmentation (`Detector`) |
| Text Encoder (YOLOE) | `~/sprocket-models/mobileclip2_b.ts` | ~243 MB | TorchScript MobileCLIP2 for `set_classes()` |
| Human Pose Model | `~/sprocket-models/yolo26n-pose.pt` | ~7.6 MB | Lazy 17-keypoint person pose estimation |
| Monocular Depth Model | `~/sprocket-models/yolo26n-depth.pt` | ~13 MB | Monocular depth estimation (`--mono-depth`) |
| Piper TTS Model | `~/sprocket-models/voice/en_US-lessac-medium.onnx` (+ `.json`) | ~61 MB | Spoken responses in Live UI (`Mouth`) |
| Faster-Whisper STT | `~/sprocket-models/whisper/faster-whisper-base.en/` | ~139 MB | Local voice transcription (`Ears`) |
| Gazebo Fuel Meshes | `~/.gz/fuel/fuel.gazebosim.org/{openrobotics,makerspet,slhdn}/models/` | ~47 MB | 13 household 3D models (`robot/ros/fuel.py`) |

> [!NOTE]
> On the primary development WSL environment, `~/sprocket-models/llm/*.gguf` may be symlinks into `/mnt/c/Users/FixAccount/...`. When exporting a portable bundle with `setup_models.sh --export`, symlinks are dereferenced (`cp -rL`) so the bundle is completely self-contained.

## 3. Setup, Verification, and Running in WSL

Run all WSL commands from the repository root (`c:\Users\FixAccount\OneDrive\Masaüstü\ai robot`):

### Step 1: Verify & Prepare All Models
```bash
wsl -- bash tools/wsl/setup_models.sh
```
This ensures `~/sprocket-models/{llm,voice,whisper}` exist, links/copies `LFM3-VL-2B` + `mmproj`, verifies or downloads the Ultralytics, Piper, and Faster-Whisper assets inside `~/sprocket-venv`, checks all 13 Gazebo Fuel models in `~/.gz/fuel`, and prints a sized `[OK]` checklist.

### Step 2: Hardware / Sensor Topic Smoke Check
```bash
wsl -- bash -lc 'source /opt/ros/lyrical/setup.bash && . ~/sprocket-venv/bin/activate && python3 -m robot.ros.sensors_check'
```
Verifies publication rates for `/rgbd_camera/*`, `/rear_left_camera/*`, `/rear_right_camera/*`, `/tof/front`, `/tof/rear`, `/scan`, `/odom`, `/imu`, and confirms AMCL `map -> base_link` localization. (Pass `--gui` to open the Gazebo window.)

### Step 3: Start the VLM Brain Server (`llama-server` on port `8081`)
```bash
wsl -- bash tools/wsl/brain.sh
```
Automatically picks `~/llama.cpp/build-cuda/bin/llama-server` (CUDA GPU) if present, otherwise `build/bin/llama-server` (CPU), resolving weights from `$SPROCKET_MODELS_DIR/llm/` first.

### Step 4: Run the 3D Simulation
```bash
# Interactive mode: Gazebo 3D window + Live Web UI at http://localhost:8765
wsl -- bash tools/wsl/run3d.sh --interactive --gui

# Interactive mode with monocular depth estimation enabled
wsl -- bash tools/wsl/run3d.sh --interactive --gui --mono-depth --room "two rooms"

# Closed-loop suite evaluation cases (headless or with --gui)
wsl -- bash tools/wsl/run3d.sh --cases O1 O3 S1

# Perception accuracy & timing benchmark
wsl -- bash tools/wsl/run3d.sh --perception --tour
```
To stop Gazebo, Nav2, and (optionally with `--all`) `llama-server`:
```bash
wsl -- bash tools/wsl/stop.sh --all
```

## 4. Exporting & Importing the Portable Bundle to a Faster PC

To move all ~1.9 GB of weights and Gazebo 3D models to a faster machine without re-downloading:

1. **Export on source PC** (dereferences all symlinks into a self-contained directory):
   ```bash
   wsl -- bash tools/wsl/setup_models.sh --export /mnt/c/Users/FixAccount/sprocket-bundle
   ```
   This creates `sprocket-models/`, `gz-fuel/`, `import.sh`, and `README.txt` inside the destination directory.

2. **Copy** the `sprocket-bundle` folder to the target PC (e.g., via USB, LAN, or shared drive).

3. **Import on target PC** (inside WSL2 with ROS 2 `lyrical`, `~/sprocket-venv`, and `~/llama.cpp` installed):
   ```bash
   wsl -- bash tools/wsl/setup_models.sh --import /mnt/c/path/to/sprocket-bundle
   ```
   (Or run `bash /mnt/c/path/to/sprocket-bundle/import.sh` directly.)
   Then run `bash tools/wsl/brain.sh` and `bash tools/wsl/run3d.sh --interactive --gui`.
