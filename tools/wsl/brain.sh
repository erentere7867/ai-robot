#!/usr/bin/env bash
# Start the robot's model (LFM3-VL-2B with its vision projector) in WSL on port 8081, on the GPU when llama.cpp was
# built with CUDA (~/llama.cpp/build-cuda), else on the CPU. Does nothing if it is already running.
#   wsl -- bash tools/wsl/brain.sh
MODEL=/mnt/c/Users/FixAccount/robot-runtime/models/llm/LFM3-VL-2B-QAD-Q4_0.gguf
MMPROJ=/mnt/c/Users/FixAccount/Downloads/mmproj-LFM3-VL-2B-F16.gguf
if curl -s --max-time 2 http://127.0.0.1:8081/health | grep -q ok; then echo "the model is already running"; exit 0; fi
cd ~/llama.cpp || { echo "no ~/llama.cpp"; exit 1; }
if [ -x build-cuda/bin/llama-server ]; then bin=build-cuda/bin/llama-server; gpu="-ngl 99 -fa on"; where=GPU
else bin=build/bin/llama-server; gpu=""; where=CPU; fi
nohup $bin -m "$MODEL" --mmproj "$MMPROJ" $gpu -c 16384 -np 2 --jinja --swa-full --port 8081 -t 4 \
  > ~/llama-wsl.log 2>&1 &
echo "starting the model on the $where..."
for _ in $(seq 1 120); do curl -s http://127.0.0.1:8081/health | grep -q ok && break; sleep 2; done
if curl -s http://127.0.0.1:8081/health | grep -q ok; then echo "the model is ready on port 8081"
else echo "the model did not start: see ~/llama-wsl.log"; exit 1; fi
