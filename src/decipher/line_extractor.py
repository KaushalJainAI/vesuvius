"""Full-coverage line + crop extractor for the decipher pipeline.

Replaces the legacy ``strip_extractor`` for the whole-segment pipeline. Key
differences:

* Covers the entire label width (no center-column crop).
* Sizes each text-line strip from the segment's own blob statistics so the
  LLM receives images where letters fill a usable fraction of the frame.
* Emits ALL detected lines (no top-K subsample).
* Tiles each line strip into overlapping letter-scale crops so the LLM sees
  ~18 letters per image instead of a hairline ribbon of pixels.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile as tf
from PIL import Image
from skimage import measure, morphology


@dataclass
class Blob:
    blob_id: int
    cx: float
    cy: float
    x0: int
    y0: int
    x1: int
    y1: int
    area: int


@dataclass
class Crop:
    """One letter-scale image sent to the LLM."""
    crop_id: int          # unique within segment
    line_index: int
    crop_index: int       # within line
    y_band: Tuple[int, int]
    x_range: Tuple[int, int]
    image_u8: np.ndarray
    png_bytes: bytes

    @property
    def y_center(self) -> int:
        return (self.y_band[0] + self.y_band[1]) // 2


@dataclass
class LineStrip:
    """One full-width text line."""
    line_index: int
    y_band: Tuple[int, int]
    x_range: Tuple[int, int]
    crops: List[Crop] = field(default_factory=list)


@dataclass
class SegmentExtraction:
    label_path: Path
    height: int
    width: int
    line_pitch_px: int
    h_med: float
    blobs: List[Blob]
    lines: List[LineStrip]

    @property
    def all_crops(self) -> List[Crop]:
        return [c for line in self.lines for c in line.crops]


# ----------------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------------


def _load_label(path: Path) -> np.ndarray:
    arr = tf.imread(str(path))
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _detect_blobs(
    label: np.ndarray,
    threshold: float = 0.55,
    min_area: int = 200,
) -> List[Blob]:
    binary = label > threshold
    binary = morphology.remove_small_objects(binary, min_size=80)
    binary = morphology.binary_closing(binary, morphology.disk(2))
    lab = measure.label(binary)
    blobs: List[Blob] = []
    for p in measure.regionprops(lab):
        if p.area < min_area:
            continue
        y0, x0, y1, x1 = p.bbox
        blobs.append(Blob(
            blob_id=int(p.label),
            cx=float(p.centroid[1]),
            cy=float(p.centroid[0]),
            x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1),
            area=int(p.area),
        ))
    return blobs


def _estimate_line_pitch(blobs: List[Blob]) -> Tuple[float, int]:
    """Returns (median_blob_height, line_pitch_px)."""
    if not blobs:
        return 50.0, 110
    heights = sorted(b.y1 - b.y0 for b in blobs)
    h_med = float(heights[len(heights) // 2])
    pitch = max(60, round(h_med * 2.2))
    return h_med, pitch


def _detect_line_midlines(
    label: np.ndarray,
    threshold: float,
    h_med: float,
) -> List[int]:
    """Detect text-line midlines across the full label width."""
    row_density = (label > threshold).mean(axis=1)
    smooth_k = max(6, round(h_med / 4))
    kernel = np.ones(2 * smooth_k + 1, dtype=np.float32) / (2 * smooth_k + 1)
    smoothed = np.convolve(row_density, kernel, mode="same")
    if smoothed.max() <= 0:
        return []
    thresh = max(0.02, smoothed.mean() * 1.1)
    min_spacing = max(40, round(h_med * 1.8))
    peaks: List[int] = []
    last = -10 ** 9
    H = len(smoothed)
    for y in range(smooth_k, H - smooth_k):
        if smoothed[y] < thresh:
            continue
        if smoothed[y] >= smoothed[y - 1] and smoothed[y] >= smoothed[y + 1]:
            if y - last > min_spacing:
                peaks.append(int(y))
                last = y
    return peaks


def _encode_png(arr_f32: np.ndarray) -> Tuple[np.ndarray, bytes]:
    u8 = (np.clip(arr_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8, mode="L").save(buf, format="PNG", optimize=True)
    return u8, buf.getvalue()


# ----------------------------------------------------------------------------
# Public entry
# ----------------------------------------------------------------------------


def extract_segment(
    label_path: Path,
    *,
    threshold: float = 0.55,
    crop_letters: float = 18.0,
    crop_overlap: float = 0.20,
    max_lines: int = 0,            # 0 = unlimited
    max_crops_per_line: int = 0,   # 0 = unlimited
) -> SegmentExtraction:
    """Extract every text line + letter-scale crops from a label."""
    label_path = Path(label_path)
    label = _load_label(label_path)
    H, W = label.shape

    blobs = _detect_blobs(label, threshold=threshold)
    h_med, pitch = _estimate_line_pitch(blobs)

    crop_w = max(round(h_med * crop_letters), pitch * 3)
    crop_h = pitch

    midlines = _detect_line_midlines(label, threshold, h_med)
    if max_lines and len(midlines) > max_lines:
        idx = np.linspace(0, len(midlines) - 1, max_lines).astype(int)
        midlines = [midlines[i] for i in idx]

    lines: List[LineStrip] = []
    crop_uid = 0
    for li, ym in enumerate(midlines):
        y0 = max(0, ym - crop_h // 2)
        y1 = min(H, y0 + crop_h)
        y0 = max(0, y1 - crop_h)

        # Tile horizontally with overlap.
        step = max(1, round(crop_w * (1.0 - crop_overlap)))
        crops: List[Crop] = []
        ci = 0
        x = 0
        while x < W:
            x1 = min(W, x + crop_w)
            x0 = max(0, x1 - crop_w)
            slab = label[y0:y1, x0:x1]
            # Skip crops with effectively no ink — keeps cost down.
            if (slab > threshold).mean() < 0.005:
                x += step
                continue
            image_u8, png = _encode_png(slab)
            crops.append(Crop(
                crop_id=crop_uid, line_index=li, crop_index=ci,
                y_band=(y0, y1), x_range=(x0, x1),
                image_u8=image_u8, png_bytes=png,
            ))
            crop_uid += 1
            ci += 1
            if max_crops_per_line and ci >= max_crops_per_line:
                break
            if x1 >= W:
                break
            x += step
        if crops:
            lines.append(LineStrip(
                line_index=li,
                y_band=(y0, y1),
                x_range=(0, W),
                crops=crops,
            ))

    return SegmentExtraction(
        label_path=label_path,
        height=H, width=W,
        line_pitch_px=pitch,
        h_med=h_med,
        blobs=blobs,
        lines=lines,
    )


def snap_bbox_to_blob(
    bbox_label_px: Tuple[int, int, int, int],
    blobs: List[Blob],
    max_dist_factor: float = 1.5,
    h_med: float = 50.0,
) -> Blob | None:
    """Return the closest blob whose centroid is within ``max_dist_factor*h_med``
    of the predicted bbox centroid, or None."""
    if not blobs:
        return None
    x0, y0, x1, y1 = bbox_label_px
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    max_dist = max_dist_factor * h_med
    best: Blob | None = None
    best_d = float("inf")
    for b in blobs:
        d = ((b.cx - cx) ** 2 + (b.cy - cy) ** 2) ** 0.5
        if d < best_d and d <= max_dist:
            best = b
            best_d = d
    return best
