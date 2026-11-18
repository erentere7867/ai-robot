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


def resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image array to `(height, width, 3)` uint8 using vectorized bilinear interpolation."""
    if width <= 0 or height <= 0:
        return np.zeros((max(0, height), max(0, width), 3), dtype=np.uint8)
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    h, w = arr.shape[:2]
    if h == height and w == width and arr.shape[2] == 3:
        return arr.copy()
    arr_f = arr[:, :, :3].astype(np.float32)
    y_coords = np.clip((np.arange(height, dtype=np.float32) + 0.5) * (h / height) - 0.5, 0.0, h - 1.0)
    x_coords = np.clip((np.arange(width, dtype=np.float32) + 0.5) * (w / width) - 0.5, 0.0, w - 1.0)
    y0 = np.floor(y_coords).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.floor(x_coords).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy = (y_coords - y0)[:, None, None]
    fx = (x_coords - x0)[None, :, None]
    top = (1.0 - fx) * arr_f[y0[:, None], x0[None, :]] + fx * arr_f[y0[:, None], x1[None, :]]
    bottom = (1.0 - fx) * arr_f[y1[:, None], x0[None, :]] + fx * arr_f[y1[:, None], x1[None, :]]
    res = (1.0 - fy) * top + fy * bottom
    return np.clip(res + 0.5, 0, 255).astype(np.uint8)


def fit_width(rgb: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale smoothly (bilinear) so the image is at most `max_width` wide."""
    h, w, _ = rgb.shape
    if w <= max_width:
        return rgb
    return resize_rgb(rgb, max_width, max(1, round(h * max_width / w)))


def data_url(rgb: np.ndarray, max_width: int = VLM_MAX_WIDTH) -> str:
    """An image as the vision model reads it, at most `max_width` wide."""
    return "data:image/png;base64," + base64.b64encode(encode_png(fit_width(rgb, max_width))).decode("ascii")
