#!/usr/bin/env bash
# Set up the robot on a Raspberry Pi 5 (Raspberry Pi OS Bookworm, 64-bit). Safe to re-run.
#
#   bash scripts/pi/setup_pi5.sh
#
# The trained detector ships in models/; a retrained one in ~/robot-runtime/models/ takes precedence.
# Start it:  sudo systemctl start sprocket   (logs: journalctl -u sprocket -f)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RT="$HOME/robot-runtime"
HF="https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/resolve/main"
mkdir -p "$RT/models/llm" "$RT/data"

echo "== system packages"
sudo apt-get update
sudo apt-get install -y git cmake build-essential python3-venv python3-dev python3-picamera2 python3-libcamera \
    libopenblas-dev curl

echo "== llama.cpp (native build: Cortex-A76 dot-product + fp16 kernels; Q4_0 is repacked for them at load)"
if [ ! -x "$RT/llama.cpp/build/bin/llama-server" ]; then
    [ -d "$RT/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$RT/llama.cpp"
    cmake -S "$RT/llama.cpp" -B "$RT/llama.cpp/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    cmake --build "$RT/llama.cpp/build" --config Release -j4 --target llama-server llama-bench
fi

echo "== models (LFM3-VL-2B 4-bit and its vision projector; QAD-Q4_0 is the faster, less accurate option)"
for f in LFM3-VL-2B-Q4_K_M.gguf LFM3-VL-2B-QAD-Q4_0.gguf mmproj-LFM3-VL-2B-Q8_0.gguf; do
    [ -s "$RT/models/llm/$f" ] || curl -fL -o "$RT/models/llm/$f" "$HF/$f"
done

echo "== python (system site packages so the venv sees picamera2)"
[ -d "$RT/venv" ] || python3 -m venv --system-site-packages "$RT/venv"
"$RT/venv/bin/pip" install -U pip
# ncnn runs the detector without PyTorch; ultralytics is only needed to train or export (on the laptop).
"$RT/venv/bin/pip" install -e "$REPO[robot]" ncnn opencv-python-headless pyyaml

echo "== permissions and performance"
sudo usermod -aG dialout,video "$USER"
# Keep the CPU at full clock while the robot runs (the default governor ramps up too slowly for 20 Hz work).
sudo tee /etc/systemd/system/cpu-performance.service >/dev/null <<'UNIT'
[Unit]
Description=CPU performance governor for the robot
[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl enable --now cpu-performance.service

echo "== service"
sed -e "s|@USER@|$USER|g" -e "s|@REPO@|$REPO|g" -e "s|@RT@|$RT|g" "$REPO/scripts/pi/sprocket.service" \
    | sudo tee /etc/systemd/system/sprocket.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable sprocket.service

echo "Done. Check speed with: $RT/venv/bin/python $REPO/scripts/bench_pi.py"
