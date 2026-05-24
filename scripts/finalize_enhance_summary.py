"""
scripts/finalize_enhance_summary.py

Walk predictions/enhanced_labels/{id}/stats.json across all segments and
produce:
  predictions/enhanced_labels/summary.json     — aggregated stats table
  predictions/enhanced_labels/all_segments_grid.png — contact sheet

Use this when enhance_all_labels.py was interrupted before its end-of-run
summary.

Usage:
  python scripts/finalize_enhance_summary.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import gc

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from best_enhancer import compute_diff_stats, load_label  # noqa: E402


def downsample(arr: np.ndarray, target: int = 900) -> np.ndarray:
    H, W = arr.shape
    step = max(1, max(H, W) // target)
    return arr[::step, ::step]


def load_thumb(path: Path, target: int = 900) -> np.ndarray:
    """Load as uint8 (no float32 expansion), downsample, then convert."""
    arr = tf.imread(str(path))   # uint8, shape HxW
    H, W = arr.shape
    step = max(1, max(H, W) // target)
    arr = arr[::step, ::step]
    return arr.astype(np.float32) / 255.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/labelled_segments")
    p.add_argument("--out-dir", default="predictions/enhanced_labels")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    summary: Dict[str, Dict] = {}
    thumbs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        orig_p = d / "ink_labels.tif"
        enh_p = d / "ink_labels_enhanced.tif"
        if not (orig_p.exists() and enh_p.exists()):
            print(f"skip {d.name}: missing tif")
            continue

        seg_id = d.name
        seg_out = out_dir / seg_id
        stats_p = seg_out / "stats.json"

        # Prefer cached stats.json (avoids reloading huge arrays just for stats)
        if stats_p.exists():
            with stats_p.open() as f:
                stats = json.load(f)
            print(f"{seg_id}: cached stats")
        else:
            print(f"{seg_id}: recomputing stats from disk")
            orig = load_label(orig_p)
            enh = load_label(enh_p)
            stats = compute_diff_stats(orig, enh)
            with stats_p.open("w") as f:
                json.dump(stats, f, indent=2)
            del orig, enh

        summary[seg_id] = stats

        # Build thumbnail from disk — uint8 path, no full-size float32 load
        thumbs[seg_id] = (load_thumb(orig_p, 900), load_thumb(enh_p, 900))
        gc.collect()

    # Contact sheet
    seg_ids = sorted(thumbs)
    n = len(seg_ids)
    fig, axes = plt.subplots(n, 2, figsize=(9, 2.6 * n), dpi=110)
    axes = np.atleast_2d(axes)
    for i, sid in enumerate(seg_ids):
        o, e = thumbs[sid]
        axes[i, 0].imshow(o, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 0].set_title(f"{sid}  orig", fontsize=8)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(e, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 1].set_title(f"{sid}  enhanced", fontsize=8)
        axes[i, 1].axis("off")
    fig.suptitle(f"All {n} labelled segments — original vs S11 enhanced",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    sheet_path = out_dir / "all_segments_grid.png"
    fig.savefig(sheet_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nContact sheet -> {sheet_path}")

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Table
    print("\n" + "=" * 100)
    print(f"{'segment':16s} {'mean_d':>8s} {'orig_ink%':>10s} {'enh_ink%':>10s} "
          f"{'+px%':>7s} {'-px%':>7s}")
    print("-" * 100)
    for sid in seg_ids:
        s = summary[sid]
        print(f"{sid:16s} {s['mean_abs_diff']:8.3f} "
              f"{s['ink_pct_orig_t55']:9.1f}% {s['ink_pct_enh_t55']:9.1f}% "
              f"{s['pct_increased']:6.1f}% {s['pct_decreased']:6.1f}%")


if __name__ == "__main__":
    main()
