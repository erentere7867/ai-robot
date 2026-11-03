#!/usr/bin/env bash
# Set up the robot on an Arduino VENTUNO Q (Ubuntu or Debian, arm64). Safe to re-run.
#
#   bash scripts/ventuno/setup.sh                  # CPU build of llama.cpp (always works)
#   LLAMA_BACKEND=opencl bash scripts/ventuno/setup.sh   # also try the Adreno OpenCL backend
#
# Then flash firmware/ventuno_mcu to the STM32H5 (Arduino App Lab or arduino-cli), check the link with
#   ~/robot-runtime/venv/bin/python scripts/ventuno/mcu_check.py
# and start everything:  sudo systemctl start sprocket-llm sprocket   (logs: journalctl -u sprocket -f)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RT="$HOME/robot-runtime"
HF="https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/resolve/main"
BACKEND="${LLAMA_BACKEND:-cpu}"
mkdir -p "$RT/models/llm" "$RT/data"

echo "== system packages"
sudo apt-get update
sudo apt-get install -y git cmake build-essential python3-venv python3-dev curl libopenblas-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good python3-opencv \
    portaudio19-dev v4l-utils
if [ "$BACKEND" = "opencl" ]; then
    sudo apt-get install -y ocl-icd-opencl-dev opencl-headers clinfo || true
fi

echo "== core plan (fast cores for control/perception/inference, little cores for the rest)"
PLAN="$(python3 - "$REPO" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from robot.core.realtime import plan_cores
p = plan_cores()
print(" ".join(map(str, p.inference)), "|", " ".join(map(str, p.control + p.perception)), "|", len(p.inference))
PY
)"
INFER_CPUS="$(echo "$PLAN" | cut -d'|' -f1 | xargs)"
INFER_THREADS="$(echo "$PLAN" | cut -d'|' -f3 | xargs)"
echo "   llama-server on cores: $INFER_CPUS ($INFER_THREADS threads)"

echo "== llama.cpp ($BACKEND; Cortex-A78: dot-product kernels, Q4_0 is repacked for them at load)"
BUILD="$RT/llama.cpp/build"
if [ ! -x "$BUILD/bin/llama-server" ]; then
    [ -d "$RT/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$RT/llama.cpp"
    EXTRA=()
    if [ "$BACKEND" = "opencl" ]; then
        EXTRA=(-DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON)
    fi
    cmake -S "$RT/llama.cpp" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF "${EXTRA[@]}"
    cmake --build "$BUILD" --config Release -j"$(nproc)" --target llama-server llama-bench
fi

echo "== models (LFM3-VL-2B 4-bit and its vision projector)"
for f in LFM3-VL-2B-Q4_K_M.gguf mmproj-LFM3-VL-2B-F16.gguf; do
    [ -s "$RT/models/llm/$f" ] || curl -fL -o "$RT/models/llm/$f" "$HF/$f"
done

echo "== python"
[ -d "$RT/venv" ] || python3 -m venv --system-site-packages "$RT/venv"   # system OpenCV has GStreamer
"$RT/venv/bin/pip" install -U pip
"$RT/venv/bin/pip" install -e "$REPO[robot]" ncnn msgpack pyserial pyyaml

echo "== devices, limits, performance"
sudo usermod -aG dialout,video,audio "$USER"
sudo cp "$REPO/scripts/ventuno/99-sprocket.rules" /etc/udev/rules.d/99-sprocket.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo tee /etc/systemd/system/cpu-performance.service >/dev/null <<'UNIT'
[Unit]
Description=CPU performance governor for the robot
[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $g; done'
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl enable --now cpu-performance.service

echo "== Arduino Router (MCU link)"
if [ -S /var/run/arduino-router.sock ]; then
    echo "   router socket present"
else
    echo "   WARNING: /var/run/arduino-router.sock not found; is the arduino-router service running?"
fi

echo "== services"
for unit in sprocket sprocket-llm; do
    sed -e "s|@USER@|$USER|g" -e "s|@REPO@|$REPO|g" -e "s|@RT@|$RT|g" -e "s|@INFER_CPUS@|$INFER_CPUS|g" \
        -e "s|@INFER_THREADS@|$INFER_THREADS|g" "$REPO/scripts/ventuno/$unit.service" \
        | sudo tee "/etc/systemd/system/$unit.service" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable sprocket-llm.service sprocket.service

echo "Done. Next: flash firmware/ventuno_mcu, then run scripts/ventuno/mcu_check.py and scripts/bench_control.py."
