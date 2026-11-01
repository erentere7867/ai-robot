# Running Sprocket on a Raspberry Pi 5

## What runs where

| Part | Runs on | Rate | Notes |
|---|---|---|---|
| Motor speed loops, range stop, bumper, cliff, e-stop, watchdog | microcontroller (`firmware/base_controller`) | 100 Hz | never waits for Linux |
| Camera + YOLO11n detector (NCNN, no PyTorch) | Pi, worker thread | 15-25 fps expected | 16 ms/frame on the laptop CPU; a Pi 5 core is ~2.5-4x slower |
| Fast loop: map fusion + behaviours + speed commands | Pi | 20 Hz | map fusion 0.5 ms per frame |
| LFM3-VL-2B (llama.cpp, CPU, 4 threads) | Pi | on requests and events | decides what to do; never steers |

Not yet measured on a Pi: run `python scripts/bench_pi.py` there and update this table.

## Hardware

* Raspberry Pi 5 (4 GB), Pi camera module (or a USB camera), active cooler.
* Microcontroller: Raspberry Pi Pico (RP2040) on USB, running `firmware/base_controller`.
* Motor driver TB6612FNG (left motors on channel A, right on B), one quadrature encoder per side,
  VL53L1X range sensor (front), MPU6050 IMU, two bumper microswitches, four IR cliff sensors, head servo,
  latching e-stop button, battery divider. Pins are at the top of `base_controller.ino`.

The link is plain text lines at 115200 baud (protocol in `robot/body/serial_body.py`):
`V <mm/s> <mdeg/s>` speed, `H <deg*10>` head, `S` stop, `E <0|1>` e-stop; the Pico answers 50 times a
second with `T <ms> <ticks_l> <ticks_r> <gyro_mdeg_s> <tof_mm> <flags> <battery_mv>`. You can drive it
from a serial terminal to test the base before starting the robot software.

## Setup

1. Raspberry Pi OS Bookworm 64-bit. Clone this repository, then:

   ```bash
   bash scripts/pi/setup_pi5.sh
   ```

   It installs packages, builds llama.cpp natively for the Cortex-A76, downloads LFM3-VL-2B (Q4_K_M,
   the QAD-Q4_0 variant and the vision projector), creates the Python environment (with access to the
   system's picamera2), sets the CPU governor to performance and installs the `sprocket` service.
2. The trained detector ships in `models/` (NCNN export, 10 MB). After retraining on the laptop
   (`python scripts/train_detector.py`), copy the new export over; one in `~/robot-runtime/models/` wins:

   ```bash
   scp -r ~/robot-runtime/models/fast_detector_ncnn_model pi@<robot>:~/robot-runtime/models/
   ```

   For the real room, add frames from the robot's camera to the dataset (label them with a large
   open-vocabulary detector or the VLM, check by hand) and retrain; the simulator frames teach shapes
   and colours, real frames teach your lighting and your objects. Add a `person` class for "follow me".
3. Flash `firmware/base_controller` to the Pico (Arduino IDE, Arduino-Pico core), check the pins.
4. `sudo systemctl start sprocket` (logs: `journalctl -u sprocket -f`). Config: `config/robot.pi5.yaml`.
   Without the motor board, set `runtime.body_backend: kinematic` to try everything else.

## Choices made for speed (and what was measured)

* **Detector instead of the VLM for seeing**: the VLM took 1.4-3 s per look on the laptop GPU; the
  detector 16 ms on the laptop CPU, with higher accuracy on these objects.
* **NCNN without PyTorch**: importing PyTorch costs ~400 MB of the Pi's 4 GB and seconds at start-up.
  `NcnnDetector` matches the ultralytics output on 148/150 test frames.
* **Lazy PNG**: frames are only PNG-encoded when the VLM needs one (~10-20 ms of a Pi core saved per frame).
* **One loop for perception and control**: no thread contention; the detector runs in one worker thread.
* **Q4_K_M, not QAD-Q4_0, by default**: QAD-Q4_0 is faster on ARM (repacked for the dot-product
  instructions) but scored 83% vs 95% on the decision benchmark. To trade accuracy for speed, swap the
  two model lines in `config/robot.pi5.yaml`.
* **Prompt caching**: decisions reuse ~2,100 cached prompt tokens and process only ~40 new ones, which is
  what makes a CPU-only 2B model usable for decisions at all.
