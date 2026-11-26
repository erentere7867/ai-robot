"""Unpack llama.cpp's official CUDA server image (ghcr.io/ggml-org/llama.cpp:server-cuda) without Docker, which Colab
does not have, into ~/llama-img, and put a llama-server where tools/linux/brain.sh looks for a CUDA build. Its CUDA
12.8 ran on Colab's T4 with driver 580."""

import io
import json
import tarfile
import urllib.request
from pathlib import Path

REPO, TAG = "ggml-org/llama.cpp", "server-cuda"
MANIFESTS = ("application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, "
             "application/vnd.docker.distribution.manifest.list.v2+json, "
             "application/vnd.docker.distribution.manifest.v2+json")
# Only these from the image: Colab's own loader runs the server, and the image's libc would not match it.
LIBS = ("libggml", "libllama", "libmtmd", "libcublas", "libcudart")


def main() -> None:
    url = f"https://ghcr.io/token?scope=repository:{REPO}:pull"
    token = json.load(urllib.request.urlopen(url))["token"]

    def get(path: str, accept: str = MANIFESTS) -> bytes:
        req = urllib.request.Request(f"https://ghcr.io/v2/{REPO}/{path}",
                                     headers={"Authorization": f"Bearer {token}", "Accept": accept})
        return urllib.request.urlopen(req).read()

    manifest = json.loads(get(f"manifests/{TAG}"))
    if "manifests" in manifest:  # a multi-platform index
        digest = next(m["digest"] for m in manifest["manifests"] if m["platform"]["architecture"] == "amd64")
        manifest = json.loads(get(f"manifests/{digest}"))
    root = Path.home() / "llama-img"
    for layer in manifest["layers"]:
        print(f"layer {layer['digest'][7:19]}: {layer['size'] / 1e6:.0f} MB", flush=True)
        with tarfile.open(fileobj=io.BytesIO(get(f"blobs/{layer['digest']}", "*/*")), mode="r:*") as tar:
            tar.extractall(root, members=[m for m in tar if not m.isdev() and ".wh." not in m.name], filter="tar")
    server = next(root.rglob("llama-server"))
    libs = sorted({str(p.parent) for p in root.rglob("*.so*") if p.name.startswith(LIBS)})
    wrapper = Path.home() / "llama.cpp/build-cuda/bin/llama-server"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(f'#!/bin/sh\nLD_LIBRARY_PATH={":".join(libs)}:$LD_LIBRARY_PATH exec {server} "$@"\n')
    wrapper.chmod(0o755)
    print(f"{wrapper} runs {server}")


if __name__ == "__main__":
    main()
