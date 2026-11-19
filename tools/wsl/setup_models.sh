#!/usr/bin/env bash
# The models Sprocket uses, in the repo's models/ (or $SPROCKET_MODELS_DIR): check what is there, fetch what can
# be fetched, make the GPU vision server's ONNX files, and carry everything to another machine as one folder.
#   bash tools/wsl/setup_models.sh                      # check and fetch
#   bash tools/wsl/setup_models.sh --gpu-models         # also make the vision server's models in models/vision
#   bash tools/wsl/setup_models.sh --export <dir>       # copy all models to <dir>/sprocket-models (links followed)
#   bash tools/wsl/setup_models.sh --import <dir>       # copy them back from <dir>/sprocket-models
# The language model (LFM3-VL-2B GGUF and its vision projector) is not fetched: put both in models/llm/, where
# tools\brain.ps1 looks on Windows.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${SPROCKET_MODELS_DIR:-$(cd "$HERE/../.." && pwd)/models}"
VENV_PY="$HOME/sprocket-venv/bin/python"
MODE="setup"; DIR=""
case "${1:-}" in
  "") ;;
  --gpu-models) MODE="gpu-models"; DIR="$MODELS_DIR/vision" ;;
  --export|--import) [ $# -ge 2 ] || { echo "error: $1 needs a folder" >&2; exit 2; }
    MODE="${1#--}"; DIR="$2" ;;
  -h|--help) sed -n 2,10p "$0"; exit 0 ;;
  *) echo "error: unknown option '$1'" >&2; exit 2 ;;
esac
mkdir -p "$MODELS_DIR/llm" "$MODELS_DIR/voice" "$MODELS_DIR/whisper"

fetch() {
  [ -x "$VENV_PY" ] || { echo "no $VENV_PY: make the venv first (docs/usage.md)"; return; }
  # ultralytics fetches from its own releases; from $MODELS_DIR, so it finds its text encoder there too
  for asset in yoloe-26l-seg.pt yolo26n-pose.pt mobileclip2_b.ts; do
    [ -f "$MODELS_DIR/$asset" ] && continue
    echo "downloading $asset..."
    (cd "$MODELS_DIR" && "$VENV_PY" -c "from ultralytics.utils.downloads import attempt_download_asset as a; a('$asset')")
  done
  if [ ! -f "$MODELS_DIR/voice/en_US-lessac-medium.onnx" ]; then
    echo "downloading the Piper voice en_US-lessac-medium..."
    "$VENV_PY" -c "from pathlib import Path; from piper.download_voices import download_voice as d; \
d('en_US-lessac-medium', Path('$MODELS_DIR/voice'))"
  fi
  if [ ! -f "$MODELS_DIR/whisper/faster-whisper-base.en/model.bin" ]; then
    echo "downloading faster-whisper base.en..."
    "$VENV_PY" -c "from faster_whisper.utils import download_model as d; \
d('base.en', output_dir='$MODELS_DIR/whisper/faster-whisper-base.en')"
  fi
}

gpu_models() {  # the vision server's files: YOLOE with 8 word slots, and the depth network under the name it uses
  mkdir -p "$1"
  if [ ! -f "$1/yoloe26l_8.onnx" ]; then
    (cd "$MODELS_DIR" && "$VENV_PY" "$HERE/export_yoloe.py" yoloe-26l-seg.pt "$1/yoloe26l_8.onnx")
  fi
  cp -n "$MODELS_DIR/depth-anything-v2-small-onnx/model.onnx" "$1/depth_small.onnx"
  echo "vision server models in $1"
}

check() {
  local missing=0
  echo "=== $MODELS_DIR ==="
  for rel in llm/LFM3-VL-2B-QAD-Q4_0.gguf llm/mmproj-LFM3-VL-2B-F16.gguf yoloe-26l-seg.pt yolo26n-pose.pt \
             mobileclip2_b.ts depth-anything-v2-small-onnx/model.onnx voice/en_US-lessac-medium.onnx \
             whisper/faster-whisper-base.en/model.bin; do
    if [ -e "$MODELS_DIR/$rel" ]; then printf "  [OK]   %-44s %s\n" "$rel" "$(du -shL "$MODELS_DIR/$rel" | cut -f1)"
    else printf "  [MISS] %s\n" "$rel"; missing=$((missing + 1)); fi
  done
  return "$missing"
}

case "$MODE" in
  setup) fetch ;;
  gpu-models) fetch; gpu_models "$DIR" ;;
  export) mkdir -p "$DIR/sprocket-models"; cp -rL "$MODELS_DIR/." "$DIR/sprocket-models/"; echo "exported to $DIR" ;;
  import) cp -rL "$DIR/sprocket-models/." "$MODELS_DIR/"; echo "imported into $MODELS_DIR" ;;
esac
check
