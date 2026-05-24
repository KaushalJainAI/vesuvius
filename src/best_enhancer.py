"""
src/best_enhancer.py

The winning label-enhancement pipeline from plan/label_enhancement_results.md:

    TV denoise → Sato vesselness boost → hysteresis threshold → morph cleanup

Designed to:
  * Operate on arbitrary-size float prob maps (tiled for memory safety)
  * Return either SOFT output (gradient preserved, for training labels) or
    HARD output (binary-masked, for Stage 2 letter extraction)
  * Drop in as a replacement for src.label_filters.apply_filters

Entry points:
  enhance(prob, soft_output=True, ...)                  — single-array
  enhance_tiled(prob, tile=2048, overlap=64, ...)       — auto-tiled
  enhance_segment(seg_id, out_dir, ...)                 — disk I/O + viz
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tifffile as tf
from skimage import filters, morphology, restoration


# ---------------------------------------------------------------------------
# Default hyperparameters — chosen from plan/label_enhancement_results.md
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    tv_weight=0.05,             # higher = smoother; 0.03–0.10 useful range
    sato_sigmas=(2.0, 4.0, 6.0), # stroke half-widths in px
    sato_boost=0.5,             # multiplicative boost factor
    hyst_low=0.35,
    hyst_high=0.55,
    close_disk=2,
    open_disk=1,
)


# ---------------------------------------------------------------------------
# Single-array enhancement (the actual algorithm)
# ---------------------------------------------------------------------------

def enhance(
    prob: np.ndarray,
    *,
    soft_output: bool = True,
    tv_weight: float = DEFAULTS["tv_weight"],
    sato_sigmas=DEFAULTS["sato_sigmas"],
    sato_boost: float = DEFAULTS["sato_boost"],
    hyst_low: float = DEFAULTS["hyst_low"],
    hyst_high: float = DEFAULTS["hyst_high"],
    close_disk: int = DEFAULTS["close_disk"],
    open_disk: int = DEFAULTS["open_disk"],
) -> np.ndarray:
    """Enhance a float prob map in [0, 1] via TV → Sato → hysteresis → morph.

    Args:
      prob:         float32 array, values in [0, 1].
      soft_output:  True returns the boosted float map masked by the binary;
                    False applies the binary mask hard (zero outside ink).
                    For TRAINING labels prefer True (preserve gradient + ignore
                    band). For Stage 2 letter extraction use False.

    Returns float32 in [0, 1] with same shape as input.
    """
    if prob.dtype != np.float32:
        prob = prob.astype(np.float32)

    # 1. TV denoise — preserves edges, kills speckle
    tv = restoration.denoise_tv_chambolle(prob, weight=tv_weight,
                                          channel_axis=None).astype(np.float32)

    # 2. Sato vesselness — boost stroke-like structures
    sato = filters.sato(tv, sigmas=sato_sigmas, black_ridges=False).astype(np.float32)
    mx = float(sato.max())
    if mx > 0:
        sato /= mx

    # 3. Combine: multiplicative stroke boost
    boosted = np.clip(tv * (1.0 + sato_boost * sato), 0.0, 1.0)
    del tv, sato
    gc.collect()

    # 4. Hysteresis: keep > t_high, OR > t_low and connected to a > t_high pixel
    binary = filters.apply_hysteresis_threshold(boosted, hyst_low, hyst_high)

    # 5. Morphological cleanup
    if close_disk > 0:
        binary = morphology.closing(binary, morphology.disk(close_disk))
    if open_disk > 0:
        binary = morphology.opening(binary, morphology.disk(open_disk))

    if soft_output:
        # Preserve gradient: keep boosted values where binary is on,
        # let small confidence elsewhere through at half weight so the
        # training loss ignore-band can still apply.
        out = np.where(binary, boosted, 0.5 * boosted * (boosted > hyst_low)).astype(np.float32)
    else:
        out = (boosted * binary.astype(np.float32)).astype(np.float32)
    del boosted, binary
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# Tiled enhancement (for arrays that may not fit / be slow)
# ---------------------------------------------------------------------------

def enhance_tiled(
    prob: np.ndarray,
    *,
    tile: int = 2048,
    overlap: int = 64,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """Tile-wise enhance with overlap. Sato has a finite receptive field (~6σ);
    64 px overlap > 6 × σ_max(6) = 36 px, so seams are imperceptible.
    """
    H, W = prob.shape
    out = np.zeros_like(prob, dtype=np.float32)

    if H <= tile and W <= tile:
        return enhance(prob, **kwargs)

    y0s = list(range(0, H, tile - overlap))
    x0s = list(range(0, W, tile - overlap))
    n_tiles = len(y0s) * len(x0s)
    if verbose:
        print(f"    Tiling {H}×{W} into {n_tiles} tiles of {tile}px (overlap {overlap})")

    for ti, y0 in enumerate(y0s):
        y1 = min(y0 + tile, H)
        for tj, x0 in enumerate(x0s):
            x1 = min(x0 + tile, W)
            tile_in = prob[y0:y1, x0:x1]
            tile_out = enhance(tile_in, **kwargs)

            # Trim back to non-overlap region (except at boundaries)
            wy0 = overlap // 2 if y0 > 0 else 0
            wy1 = tile_out.shape[0] - (overlap // 2 if y1 < H else 0)
            wx0 = overlap // 2 if x0 > 0 else 0
            wx1 = tile_out.shape[1] - (overlap // 2 if x1 < W else 0)
            out[y0 + wy0:y0 + wy1, x0 + wx0:x0 + wx1] = tile_out[wy0:wy1, wx0:wx1]

            if verbose and (ti * len(x0s) + tj) % 10 == 0:
                done = ti * len(x0s) + tj + 1
                print(f"      tile {done}/{n_tiles}")
    return out


# ---------------------------------------------------------------------------
# I/O wrapper
# ---------------------------------------------------------------------------

def load_label(path: Path | str) -> np.ndarray:
    """Load uint8/uint16/float prob map → float32 in [0, 1]."""
    arr = tf.imread(str(path))
    if arr.dtype == np.uint8:
        return (arr.astype(np.float32) / 255.0).clip(0, 1)
    if arr.dtype == np.uint16:
        return (arr.astype(np.float32) / 65535.0).clip(0, 1)
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def save_label(arr: np.ndarray, path: Path | str) -> None:
    """Save a float [0,1] prob map as uint8 TIF.

    Chunked conversion: avoids a 2× peak (clip + multiply + cast temporaries)
    while NOT mutating the input — callers reuse `arr` for stats/viz.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    u8 = np.empty(arr.shape, dtype=np.uint8)
    chunk = 2048
    for y in range(0, arr.shape[0], chunk):
        sl = arr[y:y + chunk]
        u8[y:y + chunk] = (np.clip(sl, 0.0, 1.0) * 255.0).astype(np.uint8)
    tf.imwrite(str(path), u8)
    del u8
    gc.collect()


def compute_diff_stats(original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
    """Per-segment before/after summary."""
    diff = enhanced - original
    abs_diff = np.abs(diff)
    return {
        "orig_mean":          float(original.mean()),
        "orig_p95":           float(np.percentile(original, 95)),
        "enh_mean":           float(enhanced.mean()),
        "enh_p95":            float(np.percentile(enhanced, 95)),
        "mean_abs_diff":      float(abs_diff.mean()),
        "pct_increased":      float((diff > 0.05).mean() * 100),
        "pct_decreased":      float((diff < -0.05).mean() * 100),
        "ink_pct_orig_t55":   float((original > 0.55).mean() * 100),
        "ink_pct_enh_t55":    float((enhanced  > 0.55).mean() * 100),
    }


def enhance_segment(
    label_path: Path | str,
    out_path: Path | str,
    *,
    soft_output: bool = True,
    tile: int = 2048,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Enhance one segment end-to-end.

    Returns (original, enhanced, stats).
    """
    label = load_label(label_path)
    if verbose:
        print(f"  Loaded {label.shape} from {label_path}")

    enhanced = enhance_tiled(label, tile=tile, soft_output=soft_output,
                             verbose=verbose)
    save_label(enhanced, out_path)
    if verbose:
        print(f"  Saved enhanced -> {out_path}")

    stats = compute_diff_stats(label, enhanced)
    return label, enhanced, stats
