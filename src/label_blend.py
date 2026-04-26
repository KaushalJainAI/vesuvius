"""
src/label_blend.py

Stage 4 of the text deciphering pipeline — blend updated probability map
with original labels, save outputs, and track convergence.

Main entry points:
  blend_labels(original, updated, alpha)   → blended float32 label map
  compute_diff_stats(old, new)             → convergence statistics dict
  save_iteration_labels(labels, ...)       → (.npy path, .tif path)
  log_iteration(entry, log_path)           → appends to JSON log
  visualize_diff(original, updated, ...)  → 6-panel matplotlib Figure
  visualize_iteration_history(log_path)   → convergence plot Figure
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from matplotlib.figure import Figure

# Per-iteration blend weight toward the LLM-updated map.
# Cap at 0.50 — LLM is complementary, never the sole authority.
ALPHA_SCHEDULE: List[float] = [0.30, 0.40, 0.50, 0.50]


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

def blend_labels(
    original: np.ndarray,
    updated: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Weighted blend: alpha × updated + (1 − alpha) × original.

    Both inputs should be float32 in [0, 1]. Alpha is clamped to [0, 0.5]
    to prevent the LLM output from fully overriding the visual signal.
    """
    alpha = float(np.clip(alpha, 0.0, 0.5))
    blended = alpha * updated.astype(np.float32) + (1.0 - alpha) * original.astype(np.float32)
    return np.clip(blended, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Convergence statistics
# ---------------------------------------------------------------------------

def compute_diff_stats(old: np.ndarray, new: np.ndarray) -> Dict[str, float]:
    """Return per-iteration convergence statistics.

    Keys: mean_abs_diff, max_abs_diff, pixels_increased_pct, pixels_decreased_pct,
          pixels_unchanged_pct, rms_diff.
    """
    diff = new.astype(np.float32) - old.astype(np.float32)
    abs_diff = np.abs(diff)
    return {
        "mean_abs_diff":        float(abs_diff.mean()),
        "max_abs_diff":         float(abs_diff.max()),
        "rms_diff":             float(np.sqrt((diff ** 2).mean())),
        "pixels_increased_pct": float((diff > 0.05).mean() * 100),
        "pixels_decreased_pct": float((diff < -0.05).mean() * 100),
        "pixels_unchanged_pct": float((abs_diff <= 0.05).mean() * 100),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_iteration_labels(
    labels: np.ndarray,
    seg_id: str,
    iteration: int,
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Save float32 label map as .npy and uint8 .tif.

    Files: {out_dir}/{seg_id}_labels_v{iteration}.npy/.tif
    Returns (npy_path, tif_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{seg_id}_labels_v{iteration}"
    npy_path = out_dir / f"{stem}.npy"
    tif_path = out_dir / f"{stem}.tif"
    np.save(npy_path, labels.astype(np.float32))
    tf.imwrite(str(tif_path), (labels * 255).astype(np.uint8))
    return npy_path, tif_path


def log_iteration(entry: Dict, log_path: Path) -> None:
    """Append one iteration entry to the JSON refinement log.

    Creates the file on first call. Entries are a JSON array.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history: List[Dict] = []
    if log_path.exists():
        try:
            with open(log_path) as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            history = []
    history.append({**entry, "timestamp": datetime.utcnow().isoformat()})
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)


def load_log(log_path: Path) -> List[Dict]:
    """Load iteration history from JSON log. Returns [] if missing or invalid."""
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    try:
        with open(log_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_diff(
    original: np.ndarray,
    updated: np.ndarray,
    blended: np.ndarray,
    *,
    crop: Optional[Tuple[int, int, int, int]] = None,
    figsize: Tuple[float, float] = (18, 10),
    iteration: int = 0,
    alpha: float = 0.0,
) -> Figure:
    """6-panel figure: original / LLM-updated / blended labels (row 1)
    and ink-added / ink-removed / absolute-change heatmaps (row 2).
    """
    if crop is not None:
        y0, y1, x0, x1 = crop
        original = original[y0:y1, x0:x1]
        updated  = updated[y0:y1, x0:x1]
        blended  = blended[y0:y1, x0:x1]

    diff = updated.astype(np.float32) - original.astype(np.float32)
    stats = compute_diff_stats(original, blended)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    title = f"Stage 4 — Label blend  (iteration {iteration},  α={alpha:.2f})"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    kw = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 0].imshow(original, **kw); axes[0, 0].set_title("Original labels"); axes[0, 0].axis("off")
    axes[0, 1].imshow(updated,  **kw); axes[0, 1].set_title("LLM-updated map"); axes[0, 1].axis("off")
    axes[0, 2].imshow(blended,  **kw); axes[0, 2].set_title(f"Blended (final)  mean|Δ|={stats['mean_abs_diff']:.4f}"); axes[0, 2].axis("off")

    im_add = axes[1, 0].imshow(np.clip(diff, 0, 1),  cmap="Greens", vmin=0, vmax=0.5, interpolation="nearest")
    axes[1, 0].set_title("Ink added (↑ green)"); axes[1, 0].axis("off")
    fig.colorbar(im_add, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_rem = axes[1, 1].imshow(np.clip(-diff, 0, 1), cmap="Reds",   vmin=0, vmax=0.5, interpolation="nearest")
    axes[1, 1].set_title("Ink removed (↓ red)"); axes[1, 1].axis("off")
    fig.colorbar(im_rem, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im_abs = axes[1, 2].imshow(np.abs(diff), cmap="hot", vmin=0, vmax=0.3, interpolation="nearest")
    axes[1, 2].set_title("|updated − original|"); axes[1, 2].axis("off")
    fig.colorbar(im_abs, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def visualize_iteration_history(
    log_path: Path,
    *,
    figsize: Tuple[float, float] = (13, 5),
) -> Optional[Figure]:
    """Plot mean |Δ| convergence and pixels-changed bars per iteration."""
    history = load_log(log_path)
    if not history:
        return None

    iters      = [e.get("iteration", i + 1) for i, e in enumerate(history)]
    mean_diffs = [e.get("mean_abs_diff", 0) for e in history]
    pct_add    = [e.get("pixels_increased_pct", 0) for e in history]
    pct_rem    = [e.get("pixels_decreased_pct", 0) for e in history]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Refinement loop convergence", fontsize=12, fontweight="bold")

    axes[0].plot(iters, mean_diffs, "o-", color="mediumpurple", linewidth=2, markersize=6)
    axes[0].axhline(0.01, color="tomato", linestyle="--", lw=1.2, label="convergence threshold (0.01)")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Mean |Δ| label change")
    axes[0].set_title("Label convergence"); axes[0].legend(fontsize=9)
    axes[0].set_ylim(bottom=0)

    w = 0.35
    xi = np.array(iters, dtype=float)
    axes[1].bar(xi - w / 2, pct_add, width=w, color="seagreen",  alpha=0.8, label="% pixels ↑ (ink added)")
    axes[1].bar(xi + w / 2, pct_rem, width=w, color="tomato",    alpha=0.8, label="% pixels ↓ (ink removed)")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("% pixels changed")
    axes[1].set_title("Pixels changed per iteration"); axes[1].legend(fontsize=9)
    axes[1].set_xticks(iters)

    fig.tight_layout()
    return fig
