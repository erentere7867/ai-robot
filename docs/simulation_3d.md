# Sprocket v2 — Portable 3D Gazebo & ROS 2 Simulation Guide

This guide documents the full 3D Gazebo + ROS 2 `lyrical` simulation stack for Sprocket v2, the on-disk AI & 3D asset inventory, verification commands, and how to export/import the self-contained model bundle to a faster PC.

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
