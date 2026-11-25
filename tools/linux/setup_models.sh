#!/usr/bin/env bash
# Every model Sprocket uses, in the repo's models/ (or $SPROCKET_MODELS_DIR): fetch what is missing from its public
# source, make the vision server's ONNX files, and check what is there.
#   bash tools/linux/setup_models.sh                  # fetch what is missing, make models/vision, then check
#   bash tools/linux/setup_models.sh --check          # only check (exit status: how many are missing)
#   bash tools/linux/setup_models.sh --export <dir>   # copy all models to <dir>/sprocket-models (links followed)
#   bash tools/linux/setup_models.sh --import <dir>   # copy them back from <dir>/sprocket-models
# Sources: LiquidAI/LFM3-VL-2B-GGUF and onnx-community/depth-anything-v2-base on Hugging Face,
# ultralytics' releases (YOLOE-26l, the pose model, MobileCLIP's text encoder), Piper's voices, faster-whisper.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${SPROCKET_MODELS_DIR:-$(cd "$HERE/../.." && pwd)/models}"
VENV_PY="$HOME/sprocket-venv/bin/python"
HF=https://huggingface.co
MODE="setup"; DIR=""
case "${1:-}" in
  "") ;;
  --check) MODE="check" ;;
  --export|--import) [ $# -ge 2 ] || { echo "error: $1 needs a folder" >&2; exit 2; }
    MODE="${1#--}"; DIR="$2" ;;
  -h|--help) sed -n 2,10p "$0"; exit 0 ;;
  *) echo "error: unknown option '$1'" >&2; exit 2 ;;
esac

get() {  # get <url> <file>: download unless present (resumes a partial download)
  [ -f "$2" ] && return
  mkdir -p "$(dirname "$2")"
  echo "downloading $(basename "$2")..."
  curl -fL --retry 3 -C - -o "$2.part" "$1" && mv "$2.part" "$2"
}

fetch() {
  get "$HF/LiquidAI/LFM3-VL-2B-GGUF/resolve/main/LFM3-VL-2B-QAD-Q4_0.gguf" "$MODELS_DIR/llm/LFM3-VL-2B-QAD-Q4_0.gguf"
  get "$HF/LiquidAI/LFM3-VL-2B-GGUF/resolve/main/mmproj-LFM3-VL-2B-F16.gguf" "$MODELS_DIR/llm/mmproj-LFM3-VL-2B-F16.gguf"
  get "$HF/onnx-community/depth-anything-v2-base/resolve/main/onnx/model.onnx" "$MODELS_DIR/vision/depth_base.onnx"
  [ -x "$VENV_PY" ] || { echo "no $VENV_PY: run tools/linux/setup.sh first"; exit 1; }
  for asset in yoloe-26l-seg.pt yolo26n-pose.pt mobileclip2_b.ts; do  # from models/: ultralytics looks there
    [ -f "$MODELS_DIR/$asset" ] || (cd "$MODELS_DIR" && "$VENV_PY" -c \
      "from ultralytics.utils.downloads import attempt_download_asset as a; a('$asset')")
  done
  [ -f "$MODELS_DIR/voice/en_US-lessac-medium.onnx" ] || "$VENV_PY" -c "from pathlib import Path; \
from piper.download_voices import download_voice as d; p = Path('$MODELS_DIR/voice'); p.mkdir(exist_ok=True); \
d('en_US-lessac-medium', p)"  # piper writes into the folder but does not make it
  [ -f "$MODELS_DIR/whisper/faster-whisper-base.en/model.bin" ] || "$VENV_PY" -c "from faster_whisper.utils import \
download_model as d; d('base.en', output_dir='$MODELS_DIR/whisper/faster-whisper-base.en')"
  # YOLOE with 8 word slots (a few minutes on a CPU)
  [ -f "$MODELS_DIR/vision/yoloe26l_8.onnx" ] || (cd "$MODELS_DIR" && "$VENV_PY" "$HERE/export_yoloe.py" \
    yoloe-26l-seg.pt vision/yoloe26l_8.onnx)
}

check() {
  local missing=0
  echo "=== $MODELS_DIR ==="
  for rel in llm/LFM3-VL-2B-QAD-Q4_0.gguf llm/mmproj-LFM3-VL-2B-F16.gguf yoloe-26l-seg.pt yolo26n-pose.pt \
             mobileclip2_b.ts voice/en_US-lessac-medium.onnx \
             whisper/faster-whisper-base.en/model.bin vision/yoloe26l_8.onnx vision/depth_base.onnx; do
    if [ -e "$MODELS_DIR/$rel" ]; then printf "  [OK]   %-44s %s\n" "$rel" "$(du -shL "$MODELS_DIR/$rel" | cut -f1)"
    else printf "  [MISS] %s\n" "$rel"; missing=$((missing + 1)); fi
  done
  return "$missing"
}

case "$MODE" in
  setup) fetch ;;
  export) mkdir -p "$DIR/sprocket-models"; cp -rL "$MODELS_DIR/." "$DIR/sprocket-models/"; echo "exported to $DIR" ;;
  import) cp -rL "$DIR/sprocket-models/." "$MODELS_DIR/"; echo "imported into $MODELS_DIR" ;;
esac
check
