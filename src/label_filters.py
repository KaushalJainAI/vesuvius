"""
src/label_filters.py

Stage 1 of the text deciphering pipeline — image enhancement filters.

Main entry points:
  load_label(path)              → float32 [0,1] array from .npy / .tif / .png
  apply_filters(prob)           → enhanced float32 probability map (legacy)
  STRATEGIES                    → dict {name: callable(prob) → prob}
  run_strategy(name, prob)      → run a named strategy
  adaptive_local_threshold(prob)→ per-region binary threshold map
  visualize_filter_stages(...)  → matplotlib Figure (6-panel comparison)

Strategy registry covers:
  S0 raw, S1 current, S2 median+morph, S3 tv+threshold,
  S4 tv+sauvola, S5 tv+hysteresis, S6 sato-only,
  S7 sato+hysteresis, S8 bilateral+sauvola, S9 nlm+sauvola,
  S10 anisotropic+sauvola, S11 tv+sato+hysteresis (full proposed).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import tifffile as tf
from matplotlib.figure import Figure
from skimage import exposure, filters, morphology, restoration


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


# ---------------------------------------------------------------------------
# New filter primitives (Stage-1 strategies)
# ---------------------------------------------------------------------------

def tv_denoise(prob: np.ndarray, weight: float = 0.05) -> np.ndarray:
    """Total-variation (Chambolle) denoising — preserves edges, kills speckle.

    Weight controls smoothness: higher = smoother. 0.03–0.10 is the useful range
    for probability maps.
    """
    out = restoration.denoise_tv_chambolle(prob, weight=weight, channel_axis=None)
    return out.astype(np.float32)


def sato_vesselness(prob: np.ndarray, sigmas=(2.0, 4.0, 6.0),
                    black_ridges: bool = False) -> np.ndarray:
    """Multi-scale Sato tubeness filter — boosts stroke-like tubular structures.

    Sigmas should bracket the expected stroke half-widths. For PHercParis4
    pseudo-labels: strokes are ~10-15 px wide → use sigmas (2, 4, 6).
    Output is rescaled to [0, 1].
    """
    out = filters.sato(prob, sigmas=sigmas, black_ridges=black_ridges)
    out = out.astype(np.float32)
    mx = float(out.max())
    if mx > 0:
        out /= mx
    return out


def sauvola_binarize(prob: np.ndarray, window: int = 51, k: float = 0.2,
                     r: float = 0.5) -> np.ndarray:
    """Sauvola adaptive binarization — gold standard for degraded documents.

    Per-window threshold = mean × (1 + k × (std/r − 1)).
    Returns float32 binary mask {0.0, 1.0}.
    """
    if window % 2 == 0:
        window += 1
    thr = filters.threshold_sauvola(prob, window_size=window, k=k, r=r)
    return (prob > thr).astype(np.float32)


def hysteresis_threshold(prob: np.ndarray, t_low: float = 0.35,
                         t_high: float = 0.55) -> np.ndarray:
    """Canny-style hysteresis: keep pixels > t_high, OR > t_low AND connected
    to a > t_high region. Recovers broken strokes without admitting noise.
    """
    return filters.apply_hysteresis_threshold(prob, t_low, t_high).astype(np.float32)


def bilateral_smooth(prob: np.ndarray, sigma_color: float = 0.10,
                     sigma_spatial: float = 3.0) -> np.ndarray:
    """Edge-preserving bilateral smoothing — cheaper than NLM.

    sigma_color is in the value scale [0,1]; sigma_spatial is in pixels.
    """
    out = restoration.denoise_bilateral(
        prob, sigma_color=sigma_color, sigma_spatial=sigma_spatial,
        channel_axis=None,
    )
    return out.astype(np.float32)


def nlm_denoise(prob: np.ndarray, h: float = 0.08, patch_size: int = 5,
                patch_distance: int = 6) -> np.ndarray:
    """Non-local means — best classical denoiser, but slow O(N²) on full images.

    Recommended only for tiles ≤ 1024². For full-segment use, switch to bilateral.
    """
    out = restoration.denoise_nl_means(
        prob, h=h, patch_size=patch_size, patch_distance=patch_distance,
        fast_mode=True, channel_axis=None,
    )
    return out.astype(np.float32)


def anisotropic_diffusion(prob: np.ndarray, n_iter: int = 12,
                          kappa: float = 0.10, gamma: float = 0.2) -> np.ndarray:
    """Perona–Malik anisotropic diffusion (edge-preserving smoothing).

    Conduction coefficient c = exp(-(|∇I|/kappa)²) — smooths along edges,
    not across them. Iterates explicit Euler steps.
    """
    img = prob.astype(np.float32).copy()
    for _ in range(n_iter):
        # Forward differences in 4 directions
        dN = np.roll(img, -1, axis=0) - img
        dS = np.roll(img,  1, axis=0) - img
        dE = np.roll(img, -1, axis=1) - img
        dW = np.roll(img,  1, axis=1) - img
        cN = np.exp(-(dN / kappa) ** 2)
        cS = np.exp(-(dS / kappa) ** 2)
        cE = np.exp(-(dE / kappa) ** 2)
        cW = np.exp(-(dW / kappa) ** 2)
        img += gamma * (cN * dN + cS * dS + cE * dE + cW * dW)
    return np.clip(img, 0.0, 1.0)


def morph_cleanup(binary: np.ndarray, close_disk: int = 2,
                  open_disk: int = 1) -> np.ndarray:
    """Closing then opening on a binary mask."""
    b = binary > 0.5
    if close_disk > 0:
        b = morphology.binary_closing(b, morphology.disk(close_disk))
    if open_disk > 0:
        b = morphology.binary_opening(b, morphology.disk(open_disk))
    return b.astype(np.float32)


# ---------------------------------------------------------------------------
# Strategy registry — one callable per ID from plan/label_enhancement.md
# ---------------------------------------------------------------------------

def _s0_raw(prob: np.ndarray) -> np.ndarray:
    return prob.copy()


def _s1_current(prob: np.ndarray) -> np.ndarray:
    return apply_filters(prob)


def _s2_median_morph(prob: np.ndarray) -> np.ndarray:
    sm = ndi.median_filter(prob, size=3)
    binary = sm > 0.40
    binary = morph_cleanup(binary, close_disk=2, open_disk=1)
    return (sm * binary).astype(np.float32)


def _s3_tv_threshold(prob: np.ndarray) -> np.ndarray:
    tv = tv_denoise(prob, weight=0.05)
    binary = tv > 0.40
    binary = morph_cleanup(binary)
    return (tv * binary).astype(np.float32)


def _s4_tv_sauvola(prob: np.ndarray) -> np.ndarray:
    tv = tv_denoise(prob, weight=0.05)
    binary = sauvola_binarize(tv, window=51, k=0.2)
    binary = morph_cleanup(binary)
    return (tv * binary).astype(np.float32)


def _s5_tv_hysteresis(prob: np.ndarray) -> np.ndarray:
    tv = tv_denoise(prob, weight=0.05)
    binary = hysteresis_threshold(tv, t_low=0.35, t_high=0.55)
    binary = morph_cleanup(binary)
    return (tv * binary).astype(np.float32)


def _s6_sato_only(prob: np.ndarray) -> np.ndarray:
    return sato_vesselness(prob)


def _s7_sato_hysteresis(prob: np.ndarray) -> np.ndarray:
    sato = sato_vesselness(prob)
    # Combine sato score with original prob (sato is noisy at low intensities)
    combined = 0.5 * sato + 0.5 * prob
    binary = hysteresis_threshold(combined, t_low=0.35, t_high=0.55)
    binary = morph_cleanup(binary)
    return (combined * binary).astype(np.float32)


def _s8_bilateral_sauvola(prob: np.ndarray) -> np.ndarray:
    sm = bilateral_smooth(prob)
    binary = sauvola_binarize(sm, window=51, k=0.2)
    binary = morph_cleanup(binary)
    return (sm * binary).astype(np.float32)


def _s9_nlm_sauvola(prob: np.ndarray) -> np.ndarray:
    # NLM is O(N²); guard against full-image runs
    if prob.size > 1_500_000:
        raise ValueError(
            f"NLM is too slow for arrays > 1.5M px (got {prob.size}). "
            "Apply S9 to crops only."
        )
    sm = nlm_denoise(prob)
    binary = sauvola_binarize(sm, window=51, k=0.2)
    binary = morph_cleanup(binary)
    return (sm * binary).astype(np.float32)


def _s10_anisotropic_sauvola(prob: np.ndarray) -> np.ndarray:
    sm = anisotropic_diffusion(prob, n_iter=12, kappa=0.10, gamma=0.2)
    binary = sauvola_binarize(sm, window=51, k=0.2)
    binary = morph_cleanup(binary)
    return (sm * binary).astype(np.float32)


def s11_tv_sato_hysteresis(prob: np.ndarray, *, soft_output: bool = True) -> np.ndarray:
    """TV → Sato boost → optional hysteresis mask.

    ``soft_output=True`` is for training/refinement labels: it preserves the
    probability gradient instead of forcing every non-mask pixel to zero.
    ``soft_output=False`` is for downstream letter/blob extraction.
    """
    tv = tv_denoise(prob, weight=0.05)
    sato = sato_vesselness(tv)
    boosted = np.clip(tv * (1.0 + 0.5 * sato), 0.0, 1.0)
    if soft_output:
        return boosted.astype(np.float32)
    binary = hysteresis_threshold(boosted, t_low=0.35, t_high=0.55)
    binary = morph_cleanup(binary, close_disk=2, open_disk=1)
    return (boosted * binary).astype(np.float32)


def _s11_full(prob: np.ndarray) -> np.ndarray:
    """Registry default for S11: hard-masked output for extraction/visual checks."""
    return s11_tv_sato_hysteresis(prob, soft_output=False)


STRATEGIES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "S0_raw":                    _s0_raw,
    "S1_current":                _s1_current,
    "S2_median_morph":           _s2_median_morph,
    "S3_tv_threshold":           _s3_tv_threshold,
    "S4_tv_sauvola":             _s4_tv_sauvola,
    "S5_tv_hysteresis":          _s5_tv_hysteresis,
    "S6_sato_only":              _s6_sato_only,
    "S7_sato_hysteresis":        _s7_sato_hysteresis,
    "S8_bilateral_sauvola":      _s8_bilateral_sauvola,
    "S9_nlm_sauvola":            _s9_nlm_sauvola,
    "S10_anisotropic_sauvola":   _s10_anisotropic_sauvola,
    "S11_tv_sato_hysteresis":    _s11_full,
}


def run_strategy(name: str, prob: np.ndarray) -> np.ndarray:
    """Run a named strategy from STRATEGIES."""
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy {name!r}. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name](prob)
