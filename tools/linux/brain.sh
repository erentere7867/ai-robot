#!/usr/bin/env bash
# Start the robot's model (LFM3-VL-2B) with llama.cpp built in WSL or on Linux (the board), on port 8081: on the GPU
# when llama.cpp was built with CUDA (~/llama.cpp/build-cuda), else on the CPU. On Windows use tools\brain.ps1.
#   bash tools/linux/brain.sh                        # half the available cores by default
#   SPROCKET_BRAIN_THREADS=8 bash tools/linux/brain.sh  # CPU-only server with spare cores
set -euo pipefail

MODELS_DIR="${SPROCKET_MODELS_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/models}"
MODEL="$MODELS_DIR/llm/LFM3-VL-2B-QAD-Q4_0.gguf"
MMPROJ="$MODELS_DIR/llm/mmproj-LFM3-VL-2B-F16.gguf"

if curl -s --max-time 2 http://127.0.0.1:8081/health | grep -q ok; then
  echo "the model is already running on port 8081"
  exit 0
fi
cd ~/llama.cpp 2>/dev/null || {
  echo "no llama.cpp build here (~/llama.cpp). On Windows start the model with:"
  echo "  powershell -ExecutionPolicy Bypass -File tools\\brain.ps1"
  exit 1
}
# Half the cores: the robot's other software (YOLOE, depth, Habitat, Nav2) needs the rest, and llama.cpp slows down
# badly when its threads wait for a core. On 4 cores without a GPU, task M1's model turns took 275, 124 and 201 s with
# 4 threads (the client gives up at 180 s) and 176, 56 and 62 s with 2.
threads=$((($(nproc) + 1) / 2))
if [ -x build-cuda/bin/llama-server ]; then bin=build-cuda/bin/llama-server; gpu="-ngl 99 -fa on"; where=GPU
else
  bin=build/bin/llama-server; gpu=""; where=CPU
  threads="${SPROCKET_BRAIN_THREADS:-$threads}"
  if ! [[ "$threads" =~ ^[1-9][0-9]*$ ]]; then
    echo "SPROCKET_BRAIN_THREADS must be a positive integer" >&2
    exit 2
  fi
fi
[ -f "$MODEL" ] || { echo "missing model: $MODEL (see tools/linux/setup_models.sh)"; exit 1; }
[ -f "$MMPROJ" ] || { echo "missing vision projector: $MMPROJ (see tools/linux/setup_models.sh)"; exit 1; }
nohup "$bin" -m "$MODEL" --mmproj "$MMPROJ" $gpu -c 32768 -np 2 --jinja --swa-full --reasoning-budget 128 --port 8081 \
  -t "$threads" > ~/llama-wsl.log 2>&1 &
echo "starting LFM3-VL-2B on the $where ($MODEL)..."
for _ in $(seq 1 120); do curl -s http://127.0.0.1:8081/health | grep -q ok && break; sleep 2; done
if curl -s http://127.0.0.1:8081/health | grep -q ok; then echo "the model is ready on port 8081"
else echo "the model did not start: see ~/llama-wsl.log"; exit 1; fi
