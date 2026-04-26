"""
src/label_filters.py

Stage 1 of the text deciphering pipeline — image enhancement filters.

Main entry points:
  load_label(path)              → float32 [0,1] array from .npy / .tif / .png
  apply_filters(prob)           → enhanced float32 probability map
  adaptive_local_threshold(prob)→ per-region binary threshold map
  visualize_filter_stages(...)  → matplotlib Figure (6-panel comparison)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import tifffile as tf
from matplotlib.figure import Figure
from skimage import exposure, morphology


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_label(path: Path | str) -> np.ndarray:
    """Load a probability map from .npy, .tif, or .png.  Always returns float32 [0,1]."""
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        arr = tf.imread(str(path))
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Filter pipeline
# ---------------------------------------------------------------------------

def apply_filters(
    prob: np.ndarray,
    *,
    median_radius: int = 1,
    clahe_clip: float = 0.02,
    clahe_tile: int = 8,
    close_disk: int = 2,
    open_disk: int = 1,
    threshold: float = 0.40,
    clahe_blend: float = 0.70,
) -> np.ndarray:
    """Full filter pipeline. Returns enhanced float32 map in [0, 1].

    Steps:
      1. Median denoising — suppress CT salt-and-pepper noise
      2. CLAHE — local contrast normalisation for uneven scan brightness
      3. Soft blend — preserve probability calibration (clahe_blend × CLAHE + rest × original)
      4. Morphological closing — bridge 1–2 px stroke gaps
      5. Morphological opening — remove isolated noise specks
      6. Mask: zero out regions that are below threshold in the cleaned binary
    """
    # 1. Median
    smoothed = ndi.median_filter(prob, size=2 * median_radius + 1) if median_radius > 0 else prob.copy()

    # 2. CLAHE — skimage expects [0,1] float; kernel_size = tile size in pixels
    clahe_out = exposure.equalize_adapthist(
        smoothed,
        clip_limit=clahe_clip,
        nbins=256,
        kernel_size=clahe_tile,
    ).astype(np.float32)

    # 3. Soft blend: keeps probability scale from original
    blended = clahe_blend * clahe_out + (1.0 - clahe_blend) * prob

    # 4–5. Morphological cleanup on binarised version
    binary = blended > threshold
    binary = morphology.binary_closing(binary, morphology.disk(close_disk))
    binary = morphology.binary_opening(binary, morphology.disk(open_disk))

    # 6. Apply cleaned mask: zero out regions that morphology removed
    enhanced = blended * binary.astype(np.float32)
    return np.clip(enhanced, 0.0, 1.0)


def adaptive_local_threshold(
    prob: np.ndarray,
    *,
    window: int = 31,
    offset: float = 0.05,
) -> np.ndarray:
    """Per-region adaptive threshold: pixel is ink if prob > local_mean + offset.

    Useful when scroll surface brightness varies significantly (uneven CT illumination).
    Returns float32 binary {0.0, 1.0}.
    """
    local_mean = ndi.uniform_filter(prob, size=window)
    return (prob > (local_mean + offset)).astype(np.float32)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_filter_stages(
    original: np.ndarray,
    filtered: np.ndarray,
    *,
    crop: Optional[Tuple[int, int, int, int]] = None,
    figsize: Tuple[float, float] = (18, 8),
    title_suffix: str = "",
) -> Figure:
    """6-panel figure comparing original vs filtered probability map.

    Row 1: images — original | filtered | signed diff (RdBu)
    Row 2: stats  — original histogram | filtered histogram | pixel scatter
    """
    if crop is not None:
        y0, y1, x0, x1 = crop
        orig = original[y0:y1, x0:x1]
        filt = filtered[y0:y1, x0:x1]
    else:
        orig, filt = original, filtered

    diff = filt - orig

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f"Stage 1 — Filter pipeline{title_suffix}", fontsize=13, fontweight="bold")

    kw = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    axes[0, 0].imshow(orig, **kw)
    axes[0, 0].set_title("Original probability map", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(filt, **kw)
    axes[0, 1].set_title("After median → CLAHE → morphology", fontsize=10)
    axes[0, 1].axis("off")

    im_diff = axes[0, 2].imshow(diff, cmap="RdBu_r", vmin=-0.4, vmax=0.4, interpolation="nearest")
    axes[0, 2].set_title("Diff (filtered − original)", fontsize=10)
    axes[0, 2].axis("off")
    fig.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].hist(orig.flatten(), bins=64, range=(0, 1), color="steelblue", alpha=0.8, edgecolor="none")
    axes[1, 0].set_xlabel("Probability"); axes[1, 0].set_ylabel("Pixel count")
    axes[1, 0].set_title("Original distribution", fontsize=10)

    axes[1, 1].hist(filt.flatten(), bins=64, range=(0, 1), color="darkorange", alpha=0.8, edgecolor="none")
    axes[1, 1].set_xlabel("Probability")
    axes[1, 1].set_title("Filtered distribution", fontsize=10)

    # Scatter: original vs filtered (sampled for speed)
    flat_o, flat_f = orig.flatten(), filt.flatten()
    idx = np.random.default_rng(0).choice(len(flat_o), size=min(8_000, len(flat_o)), replace=False)
    axes[1, 2].scatter(flat_o[idx], flat_f[idx], s=0.4, alpha=0.25, color="mediumpurple", rasterized=True)
    axes[1, 2].plot([0, 1], [0, 1], "k--", lw=0.8, label="identity")
    axes[1, 2].set_xlabel("Original"); axes[1, 2].set_ylabel("Filtered")
    axes[1, 2].set_title("Pixel-level scatter", fontsize=10)
    axes[1, 2].legend(fontsize=8)

    fig.tight_layout()
    return fig
