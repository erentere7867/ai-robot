"""The client of the GPU vision server (tools/vision_server.py): YOLOE and depth run on the Windows GPU, everything
around them (the words' text features, masks, feet, ranges, tracking) stays here."""

from __future__ import annotations

import io
import os
from typing import Any

import numpy as np

VISION_URL = os.environ.get("SPROCKET_VISION_URL", "http://127.0.0.1:8090")


class VisionServer:
    def __init__(self, url: str = VISION_URL, timeout_s: float = 20.0) -> None:
        import httpx

        self.url = url
        self.http = httpx.Client(base_url=url, timeout=timeout_s, trust_env=False)
        try:
            self.info = self.http.get("/health", timeout=3.0).json()
        except Exception as exc:  # noqa: BLE001 - said plainly: nothing sees without it
            raise RuntimeError(f"the GPU vision server is not answering on {url}: start it with "
                               r"tools\win\vision.ps1 on Windows") from exc
        self.slots = int(self.info["slots"])

    def _post(self, path: str, **arrays: np.ndarray) -> Any:
        # httpx responses can live until cyclic collection: 137 completed uploads still held 720p frames.
        # Close the upload buffer after the exchange and don't cache the response pixels on the HTTP object.
        with io.BytesIO() as buf:
            np.savez(buf, **arrays)
            buf.seek(0)
            with self.http.stream("POST", path, content=buf, headers={"Content-Type": "application/octet-stream"}) as r:
                if r.status_code != 200:
                    r.read()
                    raise RuntimeError(f"vision server {path}: {r.text}")
                body = b"".join(r.iter_bytes())
        return np.load(io.BytesIO(body))

    def detect(self, rgb: np.ndarray, words: np.ndarray, n: int, conf: float) -> Any:
        return self._post(f"/detect?n={n}&conf={conf}", picture=np.ascontiguousarray(rgb, np.uint8), words=words)

    def depth(self, picture: np.ndarray) -> np.ndarray:
        """Relative inverse depth of a picture already at the network's size (uint8 RGB)."""
        return self._post("/depth", picture=np.ascontiguousarray(picture, np.uint8))["inverse"].astype(np.float32)
