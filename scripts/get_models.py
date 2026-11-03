"""Download the candidate models (GGUF + vision projector) with the Hugging Face client (parallel Xet chunks).

    python scripts/get_models.py                 # all candidates
    python scripts/get_models.py qwen35-4b       # one of them

Files land in ~/robot-runtime/models/llm/candidates/<name>/, where config/models/*.yaml expect them.
None of these repositories is gated, so no token is needed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path("~/robot-runtime/models/llm/candidates").expanduser()
CANDIDATES = {
    "qwen35-2b": ("unsloth/Qwen3.5-2B-GGUF", ["Qwen3.5-2B-Q4_K_M.gguf", "mmproj-F16.gguf"], "q35-2b"),
    "qwen35-4b": ("unsloth/Qwen3.5-4B-GGUF", ["Qwen3.5-4B-Q4_K_M.gguf", "mmproj-F16.gguf"], "q35-4b"),
    "gemma4-e2b": ("ggml-org/gemma-4-E2B-it-GGUF", ["gemma-4-E2B-it-Q4_0.gguf", "mmproj-gemma-4-E2B-it-Q8_0.gguf"],
                   "gemma4"),
    "ministral3-3b": ("mistralai/Ministral-3-3B-Instruct-2512-GGUF",
                      ["Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", "Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf"],
                      "ministral"),
    "lfm3-1.7b": ("LiquidAI/LFM3-1.7B-GGUF", ["LFM3-1.7B-Q4_K_M.gguf"], "."),
}


def main(names: list[str]) -> int:
    for name in names or list(CANDIDATES):
        repo, files, sub = CANDIDATES[name]
        for filename in files:
            target = ROOT / sub
            if (target / filename).exists():
                print(f"{name}: {filename} already here")
                continue
            t0 = time.time()
            path = hf_hub_download(repo, filename, local_dir=target)
            size = Path(path).stat().st_size / 1e9
            print(f"{name}: {filename} {size:.2f} GB in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
