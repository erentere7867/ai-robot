#!/usr/bin/env bash
# Verify, download, link, export, and import all Sprocket 3D simulation & AI models.
#   wsl -- bash tools/wsl/setup_models.sh                    # verify/download into $MODELS_DIR & $FUEL_DIR
#   wsl -- bash tools/wsl/setup_models.sh --export <dest>    # create portable self-contained bundle (dereferenced)
#   wsl -- bash tools/wsl/setup_models.sh --import <src>     # restore portable bundle onto target machine
set -euo pipefail

MODELS_DIR="${SPROCKET_MODELS_DIR:-$HOME/sprocket-models}"
FUEL_DIR="${GZ_FUEL_CACHE:-$HOME/.gz/fuel}"
VENV_PY="$HOME/sprocket-venv/bin/python"

usage() {
  cat <<'EOF'
Usage:
  setup_models.sh                  Verify/link/download all models and Gazebo Fuel cache
  setup_models.sh --with-d1        Also download optional LiquidAI/d1-2B-GGUF decision model
  setup_models.sh --export <dir>   Export dereferenced sprocket-models + gz-fuel bundle to <dir>
  setup_models.sh --import <dir>   Import portable bundle from <dir> into $MODELS_DIR and $FUEL_DIR
EOF
}

MODE="setup"
BUNDLE_DIR=""
WITH_D1=0
while [ $# -gt 0 ]; do
  case "$1" in
    --with-d1)
      WITH_D1=1
      shift
      ;;
    --export)
      [ $# -ge 2 ] || { echo "error: --export requires <dest_dir>" >&2; exit 2; }
      MODE="export"
      BUNDLE_DIR="$2"
      shift 2
      ;;
    --import)
      [ $# -ge 2 ] || { echo "error: --import requires <src_dir>" >&2; exit 2; }
      MODE="import"
      BUNDLE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ensure_dirs() {
  mkdir -p "$MODELS_DIR/llm" "$MODELS_DIR/voice" "$MODELS_DIR/whisper" "$FUEL_DIR"
}

link_if_missing() {
  local dst="$1"
  shift
  if [ -L "$dst" ] && [ ! -e "$dst" ]; then
    rm -f "$dst"
  fi
  if [ -e "$dst" ]; then
    return 0
  fi
  for src in "$@"; do
    if [ -f "$src" ]; then
      ln -s "$src" "$dst"
      echo "linked: $dst -> $src"
      return 0
    fi
  done
  return 0
}

verify_and_fetch() {
  ensure_dirs

  link_if_missing "$MODELS_DIR/llm/LFM3-VL-2B-QAD-Q4_0.gguf" \
    "/mnt/c/Users/FixAccount/robot-runtime/models/llm/LFM3-VL-2B-QAD-Q4_0.gguf" \
    "/mnt/c/Users/FixAccount/Downloads/LFM3-VL-2B-QAD-Q4_0.gguf"

  link_if_missing "$MODELS_DIR/llm/mmproj-LFM3-VL-2B-F16.gguf" \
    "/mnt/c/Users/FixAccount/Downloads/mmproj-LFM3-VL-2B-F16.gguf"

  if [ "$WITH_D1" = "1" ]; then
    if [ -x "$VENV_PY" ]; then
      for d1_file in d1-2B-Q4_0.gguf mmproj-d1-2B-Q8_0.gguf; do
        if [ ! -f "$MODELS_DIR/llm/$d1_file" ]; then
          echo "downloading $d1_file from LiquidAI/d1-2B-GGUF..."
          "$VENV_PY" -c "from huggingface_hub import hf_hub_download; hf_hub_download('LiquidAI/d1-2B-GGUF', filename='$d1_file', local_dir='$MODELS_DIR/llm')"
        fi
      done
    else
      echo "warning: $VENV_PY not found; cannot download LiquidAI/d1-2B-GGUF"
    fi
  fi

  if [ -x "$VENV_PY" ]; then
    for asset in yoloe-26s-seg.pt yolo26n-pose.pt yolo26n-depth.pt mobileclip2_b.ts; do
      if [ ! -f "$MODELS_DIR/$asset" ]; then
        echo "downloading $asset into $MODELS_DIR..."
        (cd "$MODELS_DIR" && "$VENV_PY" -c "from ultralytics.utils.downloads import attempt_download_asset; attempt_download_asset('$asset')")
      fi
    done

    if [ ! -f "$MODELS_DIR/voice/en_US-lessac-medium.onnx" ] || [ ! -f "$MODELS_DIR/voice/en_US-lessac-medium.onnx.json" ]; then
      echo "downloading Piper TTS voice en_US-lessac-medium..."
      "$VENV_PY" -c "from pathlib import Path; from piper.download_voices import download_voice; download_voice('en_US-lessac-medium', Path('$MODELS_DIR/voice'))"
    fi

    if [ ! -f "$MODELS_DIR/whisper/faster-whisper-base.en/model.bin" ]; then
      snap_model=""
      for cand in "$HOME/.cache/huggingface/hub/models--Systran--faster-whisper-base.en/snapshots/"*/model.bin; do
        if [ -e "$cand" ]; then
          snap_model="$cand"
          break
        fi
      done
      if [ -n "$snap_model" ]; then
        mkdir -p "$MODELS_DIR/whisper/faster-whisper-base.en"
        cp -rL "$(dirname "$snap_model")/." "$MODELS_DIR/whisper/faster-whisper-base.en/"
        echo "copied faster-whisper-base.en from HuggingFace cache"
      else
        echo "downloading faster-whisper-base.en..."
        "$VENV_PY" -c "from faster_whisper.utils import download_model; download_model('base.en', output_dir='$MODELS_DIR/whisper/faster-whisper-base.en')"
      fi
    fi
  else
    echo "warning: $VENV_PY not found; skipping Python model downloads"
  fi

  local fuel_base="$FUEL_DIR/fuel.gazebosim.org"
  local fuel_models=(
    "openrobotics/models/chair|OpenRobotics/Chair"
    "openrobotics/models/minisofa|OpenRobotics/MiniSofa"
    "openrobotics/models/coffeetable|OpenRobotics/CoffeeTable"
    "openrobotics/models/desk|OpenRobotics/Desk"
    "openrobotics/models/lampandstand|OpenRobotics/LampAndStand"
    "openrobotics/models/trashbin|OpenRobotics/TrashBin"
    "openrobotics/models/cabinet|OpenRobotics/Cabinet"
    "openrobotics/models/fridge|OpenRobotics/Fridge"
    "openrobotics/models/jansport%20backpack%20red|OpenRobotics/JanSport%20Backpack%20Red"
    "openrobotics/models/standing%20person|OpenRobotics/Standing%20person"
    "openrobotics/models/cardboard%20box|OpenRobotics/Cardboard%20box"
    "makerspet/models/red_ball_10in|makerspet/red_ball_10in"
    "slhdn/models/plant|slhdn/plant"
  )
  for entry in "${fuel_models[@]}"; do
    local rel="${entry%%|*}"
    local uri_part="${entry##*|}"
    if [ ! -d "$fuel_base/$rel" ] && command -v gz >/dev/null 2>&1; then
      local owner="${uri_part%%/*}"
      local name="${uri_part#*/}"
      echo "downloading Gazebo Fuel model $owner/$name..."
      GZ_FUEL_CACHE="$FUEL_DIR" gz fuel download -v 4 -u "https://fuel.gazebosim.org/1.0/$owner/models/$name" || true
    fi
  done
}

export_bundle() {
  local dest="$1"
  verify_and_fetch
  mkdir -p "$dest"
  rm -rf "$dest/sprocket-models" "$dest/gz-fuel"
  mkdir -p "$dest/sprocket-models" "$dest/gz-fuel"
  echo "exporting $MODELS_DIR -> $dest/sprocket-models (dereferencing symlinks)..."
  cp -rL "$MODELS_DIR/." "$dest/sprocket-models/"
  if [ -d "$FUEL_DIR" ]; then
    echo "exporting $FUEL_DIR -> $dest/gz-fuel..."
    cp -rL "$FUEL_DIR/." "$dest/gz-fuel/"
  fi
  cat > "$dest/import.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${SPROCKET_MODELS_DIR:-$HOME/sprocket-models}"
FUEL_DIR="${GZ_FUEL_CACHE:-$HOME/.gz/fuel}"
mkdir -p "$MODELS_DIR" "$FUEL_DIR"
cp -rL "$HERE/sprocket-models/." "$MODELS_DIR/"
if [ -d "$HERE/gz-fuel" ]; then
  cp -rL "$HERE/gz-fuel/." "$FUEL_DIR/"
fi
echo "Restored Sprocket models to $MODELS_DIR and Gazebo Fuel cache to $FUEL_DIR"
EOF
  chmod +x "$dest/import.sh"
  cat > "$dest/README.txt" <<'EOF'
Sprocket v2 Portable 3D Simulation & AI Model Bundle
====================================================

Contents:
  sprocket-models/   LFM3-VL-2B GGUF + mmproj, YOLOE-26s-seg, YOLO26n-pose,
                     YOLO26n-depth, MobileCLIP2, Piper TTS voice, Faster-Whisper
  gz-fuel/           Gazebo Fuel 3D household object meshes and materials
  import.sh          Standalone restore script

Restore on target PC (inside WSL2):
  1) With repo checked out:
       bash tools/wsl/setup_models.sh --import /path/to/this/bundle
  2) Or standalone:
       bash /path/to/this/bundle/import.sh
EOF
  echo "bundle exported to $dest"
}

import_bundle() {
  local src="$1"
  [ -d "$src" ] || { echo "error: source directory '$src' not found" >&2; exit 1; }
  ensure_dirs
  if [ -d "$src/sprocket-models" ]; then
    echo "importing $src/sprocket-models -> $MODELS_DIR..."
    cp -rL "$src/sprocket-models/." "$MODELS_DIR/"
  else
    echo "warning: $src/sprocket-models not found" >&2
  fi
  if [ -d "$src/gz-fuel" ]; then
    echo "importing $src/gz-fuel -> $FUEL_DIR..."
    cp -rL "$src/gz-fuel/." "$FUEL_DIR/"
  else
    echo "warning: $src/gz-fuel not found" >&2
  fi
}

print_checklist() {
  local missing=0
  echo ""
  echo "=== Sprocket Models Status ($MODELS_DIR) ==="
  local files=(
    "llm/LFM3-VL-2B-QAD-Q4_0.gguf"
    "llm/mmproj-LFM3-VL-2B-F16.gguf"
    "yoloe-26s-seg.pt"
    "yolo26n-pose.pt"
    "yolo26n-depth.pt"
    "mobileclip2_b.ts"
    "voice/en_US-lessac-medium.onnx"
    "voice/en_US-lessac-medium.onnx.json"
    "whisper/faster-whisper-base.en/model.bin"
  )
  for rel in "${files[@]}"; do
    local p="$MODELS_DIR/$rel"
    if [ -e "$p" ]; then
      local sz
      sz="$(du -shL "$p" | awk '{print $1}')"
      printf "  [OK]   %-44s %s\n" "$rel" "$sz"
    else
      printf "  [MISS] %-44s -\n" "$rel"
      missing=$((missing + 1))
    fi
  done

  if [ "$WITH_D1" = "1" ] || [ -e "$MODELS_DIR/llm/d1-2B-Q4_0.gguf" ]; then
    for rel in "llm/d1-2B-Q4_0.gguf" "llm/mmproj-d1-2B-Q8_0.gguf"; do
      local p="$MODELS_DIR/$rel"
      if [ -e "$p" ]; then
        local sz
        sz="$(du -shL "$p" | awk '{print $1}')"
        printf "  [OK]   %-44s %s\n" "$rel" "$sz"
      elif [ "$WITH_D1" = "1" ]; then
        printf "  [MISS] %-44s -\n" "$rel"
        missing=$((missing + 1))
      fi
    done
  fi

  local fuel_base="$FUEL_DIR/fuel.gazebosim.org"
  local fuel_rels=(
    "openrobotics/models/chair"
    "openrobotics/models/minisofa"
    "openrobotics/models/coffeetable"
    "openrobotics/models/desk"
    "openrobotics/models/lampandstand"
    "openrobotics/models/trashbin"
    "openrobotics/models/cabinet"
    "openrobotics/models/fridge"
    "openrobotics/models/jansport%20backpack%20red"
    "openrobotics/models/standing%20person"
    "openrobotics/models/cardboard%20box"
    "makerspet/models/red_ball_10in"
    "slhdn/models/plant"
  )
  local fuel_ok=0
  for rel in "${fuel_rels[@]}"; do
    if [ -d "$fuel_base/$rel" ]; then
      fuel_ok=$((fuel_ok + 1))
    else
      printf "  [MISS] fuel:%s\n" "$rel"
      missing=$((missing + 1))
    fi
  done
  local fuel_sz="0"
  if [ -d "$FUEL_DIR" ]; then
    fuel_sz="$(du -shL "$FUEL_DIR" | awk '{print $1}')"
  fi
  printf "  [INFO] Gazebo Fuel models: %d/%d verified (%s in %s)\n" "$fuel_ok" "${#fuel_rels[@]}" "$fuel_sz" "$FUEL_DIR"
  return "$missing"
}

case "$MODE" in
  setup)
    verify_and_fetch
    ;;
  export)
    export_bundle "$BUNDLE_DIR"
    ;;
  import)
    import_bundle "$BUNDLE_DIR"
    ;;
esac

print_checklist
