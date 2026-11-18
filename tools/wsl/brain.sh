#!/usr/bin/env bash
# Start the robot's VLM brain (LFM3-VL-2B) in WSL on port 8081, and optionally a dedicated d1-2B decision scorer
# (--with-d1) on port 8082, on the GPU when llama.cpp was built with CUDA (~/llama.cpp/build-cuda), else on the CPU.
#   wsl -- bash tools/wsl/brain.sh
#   wsl -- bash tools/wsl/brain.sh --with-d1
set -euo pipefail

WITH_D1=0
for arg in "$@"; do
  case "$arg" in
    --with-d1) WITH_D1=1 ;;
    -h|--help) echo "Usage: brain.sh [--with-d1]"; exit 0 ;;
    *) echo "error: unknown option '$arg'" >&2; exit 2 ;;
  esac
done

MODELS_DIR="${SPROCKET_MODELS_DIR:-$HOME/sprocket-models}"
MODEL="$MODELS_DIR/llm/LFM3-VL-2B-QAD-Q4_0.gguf"
[ -f "$MODEL" ] || MODEL=/mnt/c/Users/FixAccount/robot-runtime/models/llm/LFM3-VL-2B-QAD-Q4_0.gguf
[ -f "$MODEL" ] || MODEL=/mnt/c/Users/FixAccount/Downloads/LFM3-VL-2B-QAD-Q4_0.gguf
MMPROJ="$MODELS_DIR/llm/mmproj-LFM3-VL-2B-F16.gguf"
[ -f "$MMPROJ" ] || MMPROJ=/mnt/c/Users/FixAccount/Downloads/mmproj-LFM3-VL-2B-F16.gguf

cd ~/llama.cpp 2>/dev/null || {
  echo "no llama.cpp build in WSL (~/llama.cpp). On Windows start the model with:"
  echo "  powershell -ExecutionPolicy Bypass -File tools\\brain.ps1"
  exit 1
}
if [ -x build-cuda/bin/llama-server ]; then bin=build-cuda/bin/llama-server; gpu="-ngl 99 -fa on"; where=GPU
else bin=build/bin/llama-server; gpu=""; where=CPU; fi

if curl -s --max-time 2 http://127.0.0.1:8081/health | grep -q ok; then
  echo "the model is already running on port 8081"
else
  [ -f "$MODEL" ] || { echo "missing LLM model: $MODEL (run tools/wsl/setup_models.sh)"; exit 1; }
  [ -f "$MMPROJ" ] || { echo "missing mmproj: $MMPROJ (run tools/wsl/setup_models.sh)"; exit 1; }
  nohup "$bin" -m "$MODEL" --mmproj "$MMPROJ" $gpu -c 32768 -np 2 --jinja --swa-full --port 8081 -t 4 \
    > ~/llama-wsl.log 2>&1 &
  echo "starting LFM3-VL-2B on the $where ($MODEL)..."
  for _ in $(seq 1 120); do curl -s http://127.0.0.1:8081/health | grep -q ok && break; sleep 2; done
  if curl -s http://127.0.0.1:8081/health | grep -q ok; then echo "the model is ready on port 8081"
  else echo "the model did not start: see ~/llama-wsl.log"; exit 1; fi
fi

if [ "$WITH_D1" = "1" ]; then
  D1_MODEL="$MODELS_DIR/llm/d1-2B-Q4_0.gguf"
  D1_MMPROJ="$MODELS_DIR/llm/mmproj-d1-2B-Q8_0.gguf"
  if curl -s --max-time 2 http://127.0.0.1:8082/health | grep -q ok; then
    echo "d1-2B is already running on port 8082"
  else
    [ -f "$D1_MODEL" ] || { echo "missing d1-2B model: $D1_MODEL (run tools/wsl/setup_models.sh --with-d1)"; exit 1; }
    [ -f "$D1_MMPROJ" ] || { echo "missing d1-2B mmproj: $D1_MMPROJ (run tools/wsl/setup_models.sh --with-d1)"; exit 1; }
    nohup "$bin" -m "$D1_MODEL" --mmproj "$D1_MMPROJ" $gpu -c 4096 -np 1 --jinja --swa-full --port 8082 -t 4 \
      > ~/d1-wsl.log 2>&1 &
    echo "starting d1-2B on the $where ($D1_MODEL)..."
    for _ in $(seq 1 120); do curl -s http://127.0.0.1:8082/health | grep -q ok && break; sleep 2; done
    if curl -s http://127.0.0.1:8082/health | grep -q ok; then echo "d1-2B is ready on port 8082"
    else echo "d1-2B did not start: see ~/d1-wsl.log"; exit 1; fi
  fi
fi
