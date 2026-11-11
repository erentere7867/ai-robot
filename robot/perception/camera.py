"""Images for the vision model and the live page: a fast PNG encoder and a data URL."""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np

VLM_MAX_WIDTH = 512  # a wider image only costs tokens


def encode_png(rgb: np.ndarray) -> bytes:
    """Fast numpy/zlib PNG encoder for an (H, W, 3) uint8 RGB image (~2 ms)."""
    h, w, _ = rgb.shape
    filter_col = np.zeros((h, 1), dtype=np.uint8)
    scanlines = np.hstack([filter_col, np.ascontiguousarray(rgb).reshape(h, w * 3)]).tobytes()

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(scanlines, level=1)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def fit_width(rgb: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale so the image is at most `max_width` wide."""
    h, w, _ = rgb.shape
    if w <= max_width:
        return rgb
    nh = max(1, round(h * max_width / w))
    ys = (np.arange(nh) * h / nh).astype(int)
    xs = (np.arange(max_width) * w / max_width).astype(int)
    return rgb[ys][:, xs]


def data_url(rgb: np.ndarray) -> str:
    """An image as the vision model reads it."""
    return "data:image/png;base64," + base64.b64encode(encode_png(fit_width(rgb, VLM_MAX_WIDTH))).decode("ascii")
