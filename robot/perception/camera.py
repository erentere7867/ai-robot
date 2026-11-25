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
    scanlines = np.empty((h, w * 3 + 1), dtype=np.uint8)
    scanlines[:, 0] = 0
    scanlines[:, 1:] = rgb.reshape(h, w * 3)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(scanlines.tobytes(), level=1)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image array to `(height, width, 3)` uint8 using bilinear interpolation."""
    if width <= 0 or height <= 0:
        return np.zeros((max(0, height), max(0, width), 3), dtype=np.uint8)
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    h, w = arr.shape[:2]
    if h == height and w == width and arr.shape[2] == 3:
        return arr.copy()
    import cv2

    return cv2.resize(arr[:, :, :3], (width, height), interpolation=cv2.INTER_LINEAR)


def fit_width(rgb: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale smoothly (bilinear) so the image is at most `max_width` wide."""
    h, w, _ = rgb.shape
    if w <= max_width:
        return rgb
    return resize_rgb(rgb, max_width, max(1, round(h * max_width / w)))


def data_url(rgb: np.ndarray, max_width: int = VLM_MAX_WIDTH) -> str:
    """An image as the vision model reads it, at most `max_width` wide."""
    return "data:image/png;base64," + base64.b64encode(encode_png(fit_width(rgb, max_width))).decode("ascii")
