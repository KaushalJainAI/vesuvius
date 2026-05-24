"""
scripts/enhance_all_labels.py

Run src/best_enhancer.enhance_segment across all 11 labelled PHercParis4
segments and produce:

  data/labelled_segments/{id}/ink_labels_enhanced.tif   — enhanced uint8 label
  predictions/enhanced_labels/{id}/
    full_compare.png      — side-by-side (orig | enhanced | diff), downsampled
    crop_compare.png      — 3 dense crops × {orig, enhanced}, full res
    stats.json            — before/after numeric summary
  predictions/enhanced_labels/
    all_segments_grid.png — contact sheet of all 11 (orig vs enhanced thumbs)
    summary.json          — aggregated stats

Usage:
  python scripts/enhance_all_labels.py
  python scripts/enhance_all_labels.py --only 20231221180251 20231016151002
  python scripts/enhance_all_labels.py --tile 1536 --skip-existing
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from best_enhancer import enhance_segment, load_label  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def downsample(arr: np.ndarray, target: int = 1600) -> np.ndarray:
    """Block-mean downsample so the long edge is at most `target` px."""
    H, W = arr.shape
    step = max(1, max(H, W) // target)
    return arr[::step, ::step]


def find_text_crops(label: np.ndarray, crop: int, n: int,
                    threshold: float = 0.55,
                    min_density: float = 0.04,
                    max_density: float = 0.15) -> List[Tuple[int, int]]:
    H, W = label.shape
    step = crop
    candidates = []
    for y in range(0, H - crop, step):
        for x in range(0, W - crop, step):
            tile = label[y:y + crop, x:x + crop]
            density = float((tile > threshold).mean())
            if not (min_density < density < max_density):
                continue
            row_density = (tile > threshold).mean(axis=1)
            stripe_score = float(row_density.std())
            score = stripe_score - 0.02 * abs(density - 0.08)
            candidates.append((score, y, x))
    candidates.sort(reverse=True)
    top = candidates[: n * 4]
    rng = np.random.default_rng(0)
    rng.shuffle(top)
    picked: List[Tuple[int, int]] = []
    for _, y, x in top:
        if all(abs(y - py) > crop or abs(x - px) > crop for py, px in picked):
            picked.append((y, x))
        if len(picked) >= n:
            break
    return picked


def make_full_compare(orig: np.ndarray, enh: np.ndarray, out_path: Path,
                      seg_id: str, stats: Dict[str, float]) -> None:
    """1×3 grid: original | enhanced | diff (downsampled)."""
    o = downsample(orig, 1600)
    e = downsample(enh, 1600)
    d = e - o

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=130)
    fig.suptitle(
        f"{seg_id}   - mean|d|={stats['mean_abs_diff']:.3f}   "
        f"ink@0.55  orig {stats['ink_pct_orig_t55']:.1f}%  →  "
        f"enh {stats['ink_pct_enh_t55']:.1f}%",
        fontsize=12, fontweight="bold",
    )
    axes[0].imshow(o, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Original pseudo-label", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(e, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Enhanced  (TV → Sato → hyst)", fontsize=10)
    axes[1].axis("off")
    im = axes[2].imshow(d, cmap="RdBu_r", vmin=-0.5, vmax=0.5, interpolation="nearest")
    axes[2].set_title("Diff  (enhanced − original)", fontsize=10)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_crop_compare(orig: np.ndarray, enh: np.ndarray, out_path: Path,
                      seg_id: str, crop: int = 512, n: int = 3) -> None:
    """Per-crop original vs enhanced side-by-side."""
    coords = find_text_crops(orig, crop, n)
    if not coords:
        return
    fig, axes = plt.subplots(len(coords), 2, figsize=(8, 4 * len(coords)), dpi=140)
    axes = np.atleast_2d(axes)
    for i, (y, x) in enumerate(coords):
        axes[i, 0].imshow(orig[y:y + crop, x:x + crop], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"orig  y={y} x={x}", fontsize=8)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(enh[y:y + crop, x:x + crop], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"enhanced", fontsize=8)
        axes[i, 1].axis("off")
    fig.suptitle(f"{seg_id}   -   text-density {crop}×{crop} crops",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(thumbs: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       out_path: Path) -> None:
    """11×2 grid: each row is (orig thumb | enhanced thumb) for a segment."""
    seg_ids = sorted(thumbs.keys())
    n = len(seg_ids)
    fig, axes = plt.subplots(n, 2, figsize=(8, 2.5 * n), dpi=110)
    axes = np.atleast_2d(axes)
    for i, sid in enumerate(seg_ids):
        o, e = thumbs[sid]
        axes[i, 0].imshow(o, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 0].set_title(f"{sid}  orig", fontsize=8)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(e, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 1].set_title(f"{sid}  enhanced", fontsize=8)
        axes[i, 1].axis("off")
    fig.suptitle("All 11 labelled segments — original vs S11 enhanced",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/labelled_segments")
    p.add_argument("--out-dir", default="predictions/enhanced_labels")
    p.add_argument("--only", nargs="*", default=None,
                   help="Optional: process only these segment ids")
    p.add_argument("--tile", type=int, default=2048)
    p.add_argument("--skip-existing", action="store_true",
                   help="If the enhanced .tif already exists, skip the enhancement step "
                        "(still regenerates the comparison PNGs from disk)")
    p.add_argument("--soft", action="store_true", default=True,
                   help="Soft output (preserve gradient) for training labels")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_dirs = sorted([d for d in data_dir.iterdir()
                       if d.is_dir() and (d / "ink_labels.tif").exists()])
    if args.only:
        wanted = set(args.only)
        seg_dirs = [d for d in seg_dirs if d.name in wanted]

    print(f"Enhancing {len(seg_dirs)} segments  tile={args.tile}  soft={args.soft}\n")

    summary: Dict[str, Dict] = {}
    thumbs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for d in seg_dirs:
        seg_id = d.name
        seg_out = out_dir / seg_id
        seg_out.mkdir(exist_ok=True)
        enhanced_tif = d / "ink_labels_enhanced.tif"

        print(f"=== {seg_id} ===")
        t0 = time.perf_counter()

        if args.skip_existing and enhanced_tif.exists():
            print(f"  Skipping enhance (existing {enhanced_tif.name})")
            orig = load_label(d / "ink_labels.tif")
            enh = load_label(enhanced_tif)
            from best_enhancer import compute_diff_stats
            stats = compute_diff_stats(orig, enh)
        else:
            orig, enh, stats = enhance_segment(
                label_path=d / "ink_labels.tif",
                out_path=enhanced_tif,
                soft_output=args.soft,
                tile=args.tile,
                verbose=True,
            )

        dt = time.perf_counter() - t0
        stats["seconds"] = round(dt, 1)
        summary[seg_id] = stats

        print(f"  stats: mean|d|={stats['mean_abs_diff']:.3f}  "
              f"ink% {stats['ink_pct_orig_t55']:.1f}->{stats['ink_pct_enh_t55']:.1f}  "
              f"({dt:.1f}s)")

        # Visualisations
        make_full_compare(orig, enh, seg_out / "full_compare.png", seg_id, stats)
        make_crop_compare(orig, enh, seg_out / "crop_compare.png", seg_id)
        thumbs[seg_id] = (downsample(orig, 900), downsample(enh, 900))

        with (seg_out / "stats.json").open("w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Wrote {seg_out / 'full_compare.png'}\n")

        # Aggressive memory release between segments
        del orig, enh
        gc.collect()

    # Contact sheet across all segments
    if len(thumbs) > 1:
        sheet_path = out_dir / "all_segments_grid.png"
        make_contact_sheet(thumbs, sheet_path)
        print(f"Contact sheet: {sheet_path}")

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Summary table
    print("\n" + "=" * 96)
    print(f"{'segment':16s} {'mean_d':>9s} {'orig%':>8s} {'enh%':>8s} "
          f"{'+px%':>7s} {'-px%':>7s} {'time s':>8s}")
    print("-" * 96)
    for sid in sorted(summary):
        s = summary[sid]
        print(f"{sid:16s} {s['mean_abs_diff']:9.3f} "
              f"{s['ink_pct_orig_t55']:7.1f}% {s['ink_pct_enh_t55']:7.1f}% "
              f"{s['pct_increased']:6.1f}% {s['pct_decreased']:6.1f}% "
              f"{s['seconds']:8.1f}")
    print()


if __name__ == "__main__":
    main()
