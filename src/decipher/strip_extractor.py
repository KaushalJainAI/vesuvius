"""Strip extraction — find text lines and cut horizontal strips.

Logic is shared with scripts/extract_text_strips.py; this is the importable
library version that returns strip metadata + raw uint8 PNG bytes ready to
hand to the OpenRouter client.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile as tf
from PIL import Image


@dataclass
class Strip:
    index: int
    y0: int
    y1: int
    x0: int
    x1: int
    image_u8: np.ndarray   # (h, w) uint8 PNG-ready
    png_bytes: bytes       # encoded png

    @property
    def y_center(self) -> int:
        return (self.y0 + self.y1) // 2


def _load_label(path: Path) -> np.ndarray:
    arr = tf.imread(str(path))
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _detect_text_lines(
    label: np.ndarray,
    *,
    threshold: float = 0.55,
    smooth: int = 20,
    min_spacing_px: int = 120,
) -> List[int]:
    row_density = (label > threshold).mean(axis=1)
    k = smooth
    kernel = np.ones(2 * k + 1, dtype=np.float32) / (2 * k + 1)
    smoothed = np.convolve(row_density, kernel, mode="same")
    thresh = max(0.04, smoothed.mean() * 1.2)

    peaks: List[int] = []
    last = -10 ** 9
    for y in range(k, len(smoothed) - k):
        if smoothed[y] < thresh:
            continue
        if smoothed[y] >= smoothed[y - 1] and smoothed[y] >= smoothed[y + 1]:
            if y - last > min_spacing_px:
                peaks.append(int(y))
                last = y
    return peaks


def _pick_strip_columns(
    band: np.ndarray,
    *,
    max_width: int,
    threshold: float,
    margin: int = 80,
) -> Tuple[int, int]:
    """Pick a per-line x-range instead of one global centre crop.

    If the visible ink span is wider than ``max_width``, keep the densest
    window inside that span. This avoids silently discarding off-centre text
    while still keeping model images reasonably small.
    """
    H, W = band.shape
    if W <= max_width:
        return 0, W

    col_density = (band > threshold).mean(axis=0)
    active = np.where(col_density > max(0.01, col_density.max() * 0.08))[0]
    if active.size:
        left = max(0, int(active.min()) - margin)
        right = min(W, int(active.max()) + margin + 1)
    else:
        left, right = 0, W

    if right - left <= max_width:
        return left, right

    window = np.ones(max_width, dtype=np.float32) / max_width
    scores = np.convolve(col_density[left:right], window, mode="same")
    cx = left + int(np.argmax(scores))
    x0 = max(left, min(cx - max_width // 2, right - max_width))
    x1 = min(W, x0 + max_width)
    return x0, x1


def _encode_png(strip_f32: np.ndarray) -> Tuple[np.ndarray, bytes]:
    u8 = (np.clip(strip_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8, mode="L").save(buf, format="PNG", optimize=True)
    return u8, buf.getvalue()


def extract_strips(
    label_path: Path,
    *,
    strip_width: int = 3600,
    strip_height: int = 384,
    n_strips: int = 8,
    threshold: float = 0.55,
) -> List[Strip]:
    """Return up to ``n_strips`` evenly-spaced text-line strips."""
    label = _load_label(Path(label_path))
    H, W = label.shape

    lines = _detect_text_lines(label, threshold=threshold)

    if len(lines) > n_strips:
        idx = np.linspace(0, len(lines) - 1, n_strips).astype(int)
        lines = [lines[i] for i in idx]

    strips: List[Strip] = []
    for i, ym in enumerate(lines):
        y0 = max(0, ym - strip_height // 2)
        y1 = min(H, y0 + strip_height)
        y0 = max(0, y1 - strip_height)
        x0, x1 = _pick_strip_columns(
            label[y0:y1, :],
            max_width=strip_width,
            threshold=threshold,
        )
        strip_f32 = label[y0:y1, x0:x1]
        image_u8, png = _encode_png(strip_f32)
        strips.append(Strip(
            index=i, y0=y0, y1=y1, x0=x0, x1=x1,
            image_u8=image_u8, png_bytes=png,
        ))
    return strips
