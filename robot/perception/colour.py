"""A colour name for an object's pixels: computed from the mask or box, never trusted to the detector (Fable).

The names are the planner's vocabulary (robot.sim.world.COLORS): red, orange, yellow, green, blue, purple, pink,
brown, black, white, grey.
"""

from __future__ import annotations

import numpy as np


def name(pixels: np.ndarray) -> str:
    """The colour of an (N, 3) array of RGB pixels in 0..255: the median pixel's hue, saturation and value; or, when
    some of the pixels are clearly coloured, the median of those. People name a bike by its frame: in the 3D home a
    red bike's frame was 18% of its pixels and a blue one's 9%, the rest black tyres and grey spokes."""
    if pixels.size == 0:
        return ""
    px = pixels.reshape(-1, 3)
    hi, lo = px.max(axis=1), px.min(axis=1)
    # Equivalent to (hi - lo) / hi > 0.35 & hi > 50 without casting the full uint8 crop to float64.
    w = np.uint16 if px.dtype == np.uint8 else np.float32
    colourful = (hi > 50) & ((hi - lo).astype(w, copy=False) * 20 > hi.astype(w, copy=False) * 7)
    n_col = int(np.count_nonzero(colourful))
    if n_col >= 20 and n_col >= 0.08 * len(px):
        px = px[colourful]
    rgb = np.median(px, axis=0) / 255.0
    r, g, b = rgb
    hi, lo = max(rgb), min(rgb)
    value, chroma = hi, hi - lo
    saturation = chroma / hi if hi > 1e-6 else 0.0
    if value < 0.18:
        return "black"
    if saturation < 0.18:
        return "white" if value > 0.75 else "grey"
    if hi == r:
        hue = (60 * ((g - b) / chroma)) % 360
    elif hi == g:
        hue = 60 * ((b - r) / chroma) + 120
    else:
        hue = 60 * ((r - g) / chroma) + 240
    if hue < 15 or hue >= 340:
        return "brown" if value < 0.45 else ("pink" if saturation < 0.45 else "red")
    if hue < 40:
        return "brown" if value < 0.6 else "orange"
    if hue < 70:
        return "yellow" if value > 0.5 else "brown"
    if hue < 170:
        return "green"
    if hue < 255:
        return "blue"
    if hue < 290:
        return "purple"
    return "pink"
