"""Visual grounding overlay: pure-NumPy id badges on the boxes the model sees."""

from __future__ import annotations

import numpy as np

# 5x7 bitmap font for ASCII characters
_FONT_5X7: dict[str, list[int]] = {
    " ": [0, 0, 0, 0, 0, 0, 0],
    "-": [0, 0, 0, 0b11111, 0, 0, 0],
    "[": [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
    "]": [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
    "(": [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
    ")": [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
    "0": [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    "1": [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "2": [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    "3": [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
    "4": [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    "5": [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    "6": [0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110],
    "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    "8": [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    "9": [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
    "A": [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "B": [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    "C": [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    "D": [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
    "E": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    "F": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    "G": [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
    "H": [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "I": [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "J": [0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b10001, 0b01110],
    "K": [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
    "L": [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    "M": [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
    "N": [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
    "O": [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "P": [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    "Q": [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
    "R": [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    "S": [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
    "T": [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    "U": [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "V": [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    "W": [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
    "X": [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
    "Y": [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
    "Z": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
    ".": [0, 0, 0, 0, 0, 0b01100, 0b01100],
    ":": [0, 0b01100, 0b01100, 0, 0b01100, 0b01100, 0],
    "_": [0, 0, 0, 0, 0, 0, 0b11111],
}

def _draw_char(img: np.ndarray, ch: str, x: int, y: int, color: tuple[int, int, int], scale: int = 1) -> None:
    glyph = _FONT_5X7.get(ch.upper(), _FONT_5X7[" "])
    h, w = img.shape[:2]
    for r, row in enumerate(glyph):
        for c in range(5):
            if row & (1 << (4 - c)):
                y0, y1 = y + r * scale, y + (r + 1) * scale
                x0, x1 = x + c * scale, x + (c + 1) * scale
                if 0 <= y0 < h and 0 <= x0 < w:
                    img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = color


def _draw_text(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: int = 1) -> None:
    cx = x
    for ch in text:
        _draw_char(img, ch, cx, y, color, scale)
        cx += (5 + 1) * scale


def _draw_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int],
               thickness: int = 2) -> None:
    h, w = img.shape[:2]
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    t = max(1, thickness)
    img[y1:min(h, y1 + t), x1:x2] = color
    img[max(0, y2 - t):y2, x1:x2] = color
    img[y1:y2, x1:min(w, x1 + t)] = color
    img[y1:y2, max(0, x2 - t):x2] = color


def _place_badge(bx0: int, by0: int, bw: int, bh: int, w: int, h: int,
                 placed: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    """Find a nearby non-overlapping pixel spot for a badge so adjacent tags never fuse (e.g. '19' + '20' -> '1920')."""
    def fits(x: int, y: int) -> bool:
        x1, y1 = x + bw, y + bh
        return all(x1 <= px0 or x >= px1 or y1 <= py0 or y >= py1 for px0, py0, px1, py1 in placed)

    step = bh + 2
    offsets = [(0, 0), (0, -step), (0, step), (0, -2 * step), (0, 2 * step),
               (0, -3 * step), (0, 3 * step), (bw + 2, 0), (-(bw + 2), 0)]
    for dx, dy in offsets:
        x = max(0, min(w - bw, bx0 + dx))
        y = max(0, min(h - bh, by0 + dy))
        if fits(x, y):
            return x, y
    return max(0, min(w - bw, bx0)), max(0, min(h - bh, by0))


Box = tuple[int, int, int, int]


def draw_set_of_mark(rgb: np.ndarray, boxes_and_ids: list[tuple[Box, int] | tuple[Box, int, str]]) -> np.ndarray:
    """Draw the tracker's id badges on the boxes of what was detected."""
    out = np.array(rgb, copy=True)
    h, w = out.shape[:2]
    placed: list[tuple[int, int, int, int]] = []

    # Thin black-and-white outlines and plain black badges with white numbers. The marks have no colour of their
    #    own: coloured boxes made the vision model call a red bike "blue", the colour of the box drawn round it.
    for item in boxes_and_ids:
        if len(item) == 3:
            box, track_id, label = item
            badge_str = f"{track_id} {label}" if label else f"{track_id}"
        else:
            box, track_id = item
            badge_str = f"{track_id}"
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        _draw_rect(out, x1, y1, x2, y2, (0, 0, 0), thickness=1)
        _draw_rect(out, x1 + 1, y1 + 1, x2 - 1, y2 - 1, (255, 255, 255), thickness=1)

        scale = 2
        bw = len(badge_str) * 6 * scale + 4
        bh = 7 * scale + 4

        # Place badge at top-left of box, nudged away from any earlier badge so adjacent IDs stay readable.
        bx0 = max(0, min(w - bw, x1))
        by0 = y1 - bh if y1 >= bh else y1
        bx0, by0 = _place_badge(bx0, by0, bw, bh, w, h, placed)
        placed.append((bx0, by0, bx0 + bw, by0 + bh))

        out[by0:by0 + bh, bx0:bx0 + bw] = (0, 0, 0)
        _draw_text(out, badge_str, bx0 + 2, by0 + 2, (255, 255, 255), scale=scale)

    return out
