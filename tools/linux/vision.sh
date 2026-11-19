#!/usr/bin/env bash
# The vision server (YOLOE + depth, tools/vision_server.py) on a Linux machine, on 127.0.0.1:8090, in the background.
# On the CPU by default; with onnxruntime-gpu, SPROCKET_VISION_PROVIDER=CUDAExecutionProvider puts it on an NVIDIA GPU.
# On the Windows development PC use tools\win\vision.ps1 (DirectML) instead.
#   bash tools/linux/vision.sh          # start it if it is not running (log: runs/vision.log)
#   bash tools/linux/vision.sh --stop
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
ready() { curl -s --max-time 2 http://127.0.0.1:8090/health | grep -q '"ok"'; }

if [ "${1:-}" = "--stop" ]; then pkill -f "[v]ision_server.py" && echo "stopped" || echo "not running"; exit 0; fi
if ready; then echo "vision server already running"; exit 0; fi
mkdir -p "$repo/runs"
setsid nohup ~/sprocket-venv/bin/python -u "$repo/tools/vision_server.py" --models "$repo/models/vision" \
  --provider "${SPROCKET_VISION_PROVIDER:-CPUExecutionProvider}" > "$repo/runs/vision.log" 2>&1 &
for _ in $(seq 1 60); do
  if ready; then echo "vision server ready on 127.0.0.1:8090"; exit 0; fi
  sleep 1
done
echo "vision server did not start: see $repo/runs/vision.log"; exit 1
