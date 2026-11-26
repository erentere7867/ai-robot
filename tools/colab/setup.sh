#!/usr/bin/env bash
# Levels 1 and 2 on a Colab GPU runtime (docs/testing.md, "On a Colab GPU"): the venv, the language model and
# llama.cpp's CUDA server, then the model on port 8081. Run it in the clone on the runtime; safe to run again.
set -euo pipefail
cd "$(dirname "$0")/../.."
# Not --system-site-packages as on the robot: Colab's Python has a `tests` package of its own, which hid this repo's
# tests/ (3 errors at collection), and there is no ROS here to share. Its python3 has no ensurepip, so uv makes it.
[ -x ~/sprocket-venv/bin/python ] || uv venv -q ~/sprocket-venv
uv pip install -q --python ~/sprocket-venv/bin/python -e ".[eval,dev]" onnxruntime
bash tools/linux/setup_models.sh --llm
# llama.cpp's own CUDA build, unpacked in 3 minutes; compiling it on the T4 runtime's 2 cores was 22% done after 20.
[ -x ~/llama.cpp/build-cuda/bin/llama-server ] || python3 tools/colab/llama_image.py
bash tools/linux/brain.sh
