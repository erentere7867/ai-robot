# Sprocket on the Arduino VENTUNO Q

The VENTUNO Q has two computers on one board, and the robot uses them the way professional mobile robots split
their work: everything that must happen on time runs on the microcontroller, everything that thinks runs on
Linux, and the Linux side is itself split so that thinking can never delay steering.

![robot model](robot_ventuno.png)

## The board

| Part | What it is | What the robot uses it for |
|---|---|---|
| Qualcomm Dragonwing IQ8 (IQ-8275) | 2x Cortex-A78C 2.35 GHz, 2x A78C 2.1 GHz, 4x A55 1.95 GHz | control, perception, the language model, everything else (see the core plan) |
| Adreno 623 GPU | Vulkan, OpenCL | the detector (NCNN through Vulkan, `perception.detector_gpu`); optionally llama.cpp's OpenCL backend |
| Hexagon NPU (40 TOPS INT8) | | not used yet (a QNN export of the detector is the next step) |
| 16 GB LPDDR5-6400 | ~51 GB/s | room for a bigger model; decoding speed is bandwidth-bound |
| 3x MIPI CSI | three cameras at once | the front, rear-left and rear-right cameras |
| STM32H5F5 (Cortex-M33, 250 MHz) | the real-time controller | `firmware/ventuno_mcu`: wheels, odometry, reflexes, watchdog |
| Arduino Router | msgpack-RPC between the two | `robot/body/ventuno_body.py` over `/var/run/arduino-router.sock` |

## Parts

| Part | Suggested | Notes |
|---|---|---|
| Front camera | IMX219 or IMX708 module with a ~100-120° lens | rectified to 90° (`config/body.yaml`), mounted in the head |
| Rear cameras (2) | IMX219 modules with 160° fisheye lenses | rectified to 140°, on the mast under the lidar, facing ±115°; calibrate each (`scripts/ventuno/calibrate_camera.py`) |
| Lidar | LDROBOT LD19 or LD06 | USB-UART, 10 Hz, 0.02-12 m; `/dev/lidar` via the udev rule |
| Microphone array | 4-mic circular array with raw channels over USB (ReSpeaker 4-mic USB, XVF3800 in raw mode) | must deliver all four channels; set it as the default ALSA capture device |
| Front range sensor | VL53L1X | on the MCU's I2C |
| IMU | MPU-6050 or better | on the MCU's I2C |
| Drive | 4 geared DC motors with quadrature encoders, a dual H-bridge (TB6612 class) | encoders on MCU interrupt pins |
| Safety | 2 bumper switches, 4 IR cliff sensors, a latching e-stop | wired to the MCU, they stop the motors without Linux |
| Head | one servo for tilt | MCU PWM |

Positions, fields of view and the microphone geometry live in `config/body.yaml`; change them there if you
mount things differently, and the simulator, perception and voice direction follow.

## Layers and rates

```
 STM32H5 (firmware/ventuno_mcu)                                 hard real time, never waits for Linux
   200 Hz  encoders -> wheel speed PI (+feed-forward, anti-windup) -> PWM
   200 Hz  odometry: distance from encoders, heading from the gyro (bias re-learned at rest)
   200 Hz  safety envelope: forward speed <= sqrt(2 a d) to the range reading; bumper / cliff / lifted /
           stall / e-stop latch the motors; 300 ms command watchdog
    50 Hz  telemetry -> Linux                       ^ cmd_vel 50 Hz
 ------------------------------------------------- | -------- Arduino Router (msgpack-RPC) ----------
 Linux, one Python process                         |
   control thread   50 Hz, SCHED_FIFO 60, fast core: behaviour tick -> collision monitor -> cmd_vel
   perception thread  3 cameras -> YOLO11n (NCNN on the GPU) -> map; lidar -> occupancy, localization
   main event loop  cognition (agent, skills, memory), speech, voice direction, dashboard; the A55 cores
 llama-server (own process, sprocket-llm.service): the two 2.1 GHz A78 cores
```

Why the split matters, measured on the laptop with `scripts/bench_control.py` (simulated robot driving a
continuous loop while the main event loop is hit by 30 ms bursts of Python work, as a model reply being parsed):

| | control in the main loop | control + perception threads (now) |
|---|---|---|
| achieved control rate (target 50 Hz) | 24.7 Hz | 48.8-49.1 Hz |
| median start lateness | 11.9 ms | 0.5 ms |
| worst lateness | 75-122 ms | 29 ms |

The remaining tail under load is the interpreter lock (on Windows a waiting thread wakes on the 15.6 ms timer
tick; Linux hands the lock over at the 1 ms switch interval `tune_process` sets). Either way the microcontroller
keeps executing the last command inside its own braking envelope, so a late supervisory tick costs smoothness,
never safety.

### Core plan (`robot/core/realtime.py`)

Derived at start-up from the frequencies the kernel reports, so it adapts to other boards:

| cores | what runs there |
|---|---|
| A78 2.35 GHz #1 | control thread (SCHED_FIFO 60, pinned) |
| A78 2.35 GHz #2 | perception thread (cameras, detector, lidar fusion) |
| A78 2.1 GHz x2 | llama-server (`CPUAffinity` in `sprocket-llm.service`, 2 threads) |
| A55 x4 | the main event loop: cognition glue, speech, voice direction, dashboard |

Process tuning (`perception.tune_process`): the garbage collector's view of the start-up heap is frozen and full
collections made rare, the interpreter lock is handed over every 1 ms, and memory is locked (`mlockall`) so the
control path never page-faults after the model server pushed it out.

## Localization

Odometry drifts (skid steer slips in every turn), and everything the robot knows is placed with that pose.
`robot/world/localization.py` matches each lidar scan against the occupancy grid (correlative scan matching,
±80 mm and ±3°) on the perception thread and folds the result into a map ← odometry correction. Measured on the
MuJoCo physics body over 60 s of driving (`scripts/bench_localization.py`): odometry ended 49 mm / 2.2° off the
true pose, the corrected pose 31 mm / 0.7°.

## Safety, in depth

1. **Microcontroller** (always on, independent of Linux): braking envelope from the range sensor, reflex latches,
   stall detection, watchdog.
2. **Collision monitor** (`robot/control/safety.py`, every control tick): lidar points all around plus the range
   reading; speed capped to what still stops in time, turning slowed when something brushes the turning circle.
3. **Behaviours** plan around the map (A* on the lidar occupancy grid) and re-plan on blocks.
4. **"stop"** said or typed halts the robot before the model reads the message.

## Bring-up

1. Wire the base to the MCU header (motor driver, encoders, VL53L1X, IMU, bumpers, cliff sensors, head servo,
   e-stop, battery divider, dock contacts). Map the `PIN_*` constants at the top of
   `firmware/ventuno_mcu/ventuno_mcu.ino` to your header; the rest of the constants come from `config/body.yaml`.
2. Flash the sketch from Arduino App Lab (or `arduino-cli`) with the libraries `Arduino_RouterBridge`,
   `VL53L1X` (Pololu) and `Adafruit MPU6050`.
3. Connect the cameras to the three CSI ports (front = port 0, rear-left = 1, rear-right = 2, or change
   `device` in `body.yaml`), the lidar and the microphone array to USB.
4. On the board: `bash scripts/ventuno/setup.sh` (packages, Vulkan, llama.cpp, the model, Python, udev rule,
   CPU governor, services). `LLAMA_BACKEND=opencl` also builds the Adreno OpenCL backend for llama.cpp.
5. Check the link with the wheels off the ground: `~/robot-runtime/venv/bin/python scripts/ventuno/mcu_check.py --wiggle`
   (telemetry should arrive at ~50 Hz).
6. Calibrate each camera: `python scripts/ventuno/calibrate_camera.py front` (and `rear_left`, `rear_right`).
7. Retrain the detector on your room: add frames from the robot's cameras, labelled (a large open-vocabulary
   detector or the VLM as the teacher, checked by hand), to the simulator dataset and run
   `scripts/train_detector.py`; the pipeline and classes stay the same.
8. `sudo systemctl start sprocket-llm sprocket`; logs with `journalctl -u sprocket -f`. The control thread's
   timing and the collision monitor's verdict are in the fast loop's health details and on the dashboard.

## Not verified yet

Nothing here has run on a VENTUNO Q. The firmware compiles against the documented Router Bridge API
(`Bridge.provide_safe`, `Bridge.notify`) and the Linux side against the router's msgpack-RPC protocol, both as
published for the UNO Q; neither has met the real board. Expect to adjust pins, encoder directions, gains
(`KFF`, `KP`, `KI`), the camera pipeline (`robot/perception/camera.py` tries Qualcomm's `qtiqmmfsrc`, then
`libcamerasrc`; `runtime.camera: usb` is the safe start), the microphone channel order (match
`positions_mm` in `body.yaml` to how the array numbers its microphones) and the detector's Vulkan driver.
