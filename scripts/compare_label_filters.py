"""
scripts/compare_label_filters.py

Run every strategy in src/label_filters.STRATEGIES on the same set of crops
and produce:
  - per-crop side-by-side grids (one image per strategy)
  - per-strategy full-crop PNGs (so we can zoom in)
  - per-strategy metrics in results.json

Outputs land under: predictions/filter_strategy_benchmark/{seg}/

Usage:
  python scripts/compare_label_filters.py --seg 20231221180251
  python scripts/compare_label_filters.py --seg 20231221180251 --skip S9_nlm_sauvola
  python scripts/compare_label_filters.py --seg 20231221180251 --crop 768
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from label_filters import STRATEGIES, load_label  # noqa: E402
from skimage import measure, morphology              # noqa: E402


# ---------------------------------------------------------------------------
# Crop selection (same logic as scripts/visualize_labels.py)
# ---------------------------------------------------------------------------

def find_text_crops(label: np.ndarray, crop: int, n: int,
                    threshold: float = 0.55,
                    min_density: float = 0.04,
                    max_density: float = 0.15) -> List[Tuple[int, int]]:
    """Pick crops in the text-density sweet spot.

    Maximally-dense regions are solid ink blobs (bad for testing letter-level
    filters). Maximally-sparse are noise. Text lives in between.

    Also scores by horizontal-line structure: tiles with strong row-density
    variance (text-like stripes) are preferred.
    """
    H, W = label.shape
    step = crop // 2
    candidates = []
    for y in range(0, H - crop, step):
        for x in range(0, W - crop, step):
            tile = label[y:y + crop, x:x + crop]
            density = float((tile > threshold).mean())
            if not (min_density < density < max_density):
                continue
            # Row-density variance: text lines have stripes → high row variance
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


def find_dense_crops(label: np.ndarray, crop: int, n: int,
                     threshold: float = 0.5) -> List[Tuple[int, int]]:
    """Pick the densest crops (kept for backward compat — biased to solid blobs)."""
    H, W = label.shape
    step = crop // 2
    candidates = []
    for y in range(0, H - crop, step):
        for x in range(0, W - crop, step):
            tile = label[y:y + crop, x:x + crop]
            density = float((tile > threshold).mean())
            if density > 0.02:
                candidates.append((density, y, x))
    candidates.sort(reverse=True)
    top = candidates[: n * 3]
    rng = np.random.default_rng(0)
    rng.shuffle(top)
    picked: List[Tuple[int, int]] = []
    for _, y, x in top:
        if all(abs(y - py) > crop or abs(x - px) > crop for py, px in picked):
            picked.append((y, x))
        if len(picked) >= n:
            break
    return picked


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(prob: np.ndarray, binarize_threshold: float = 0.5) -> Dict:
    """Compute structural metrics on a strategy's output.

    All metrics assume the output is a float prob map; we binarize at threshold
    for CC analysis. Strategies that already binarize will have values {0,1}.
    """
    flat = prob.ravel()
    sorted_vals = np.sort(flat)
    n = len(sorted_vals)

    # Letter-vs-background separation
    top5 = sorted_vals[int(0.95 * n):]
    bottom50 = sorted_vals[: int(0.50 * n)]
    separation = float(top5.mean() - bottom50.mean())

    # Binarize + CC analysis
    binary = prob > binarize_threshold
    binary = morphology.binary_opening(binary, morphology.disk(1))
    labels = measure.label(binary, connectivity=2)
    props = measure.regionprops(labels)

    if not props:
        return {
            "separation":      separation,
            "n_components":    0,
            "median_cc_area":  0.0,
            "mean_compactness": 0.0,
            "speckle_index":   0.0,
            "ink_fraction":    float(binary.mean()),
        }

    areas = np.array([p.area for p in props], dtype=np.float32)
    perimeters = np.array([max(p.perimeter, 1.0) for p in props], dtype=np.float32)
    compactness = 4 * np.pi * areas / (perimeters ** 2)

    # Filter to letter-sized CCs for "useful" stats
    letter_mask = (areas >= 50) & (areas <= 5000)
    letter_areas = areas[letter_mask]
    letter_compact = compactness[letter_mask]

    speckle_index = float((areas < 30).sum() / len(areas))

    return {
        "separation":       separation,
        "n_components":     int(len(props)),
        "n_letter_sized":   int(letter_mask.sum()),
        "median_cc_area":   float(np.median(letter_areas)) if letter_areas.size else 0.0,
        "mean_compactness": float(np.mean(letter_compact)) if letter_compact.size else 0.0,
        "speckle_index":    speckle_index,
        "ink_fraction":     float(binary.mean()),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_strategy_grid(crops_by_strategy: Dict[str, np.ndarray],
                       out_path: Path, suptitle: str) -> None:
    """One subplot per strategy, grayscale."""
    names = list(crops_by_strategy.keys())
    n = len(names)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), dpi=140)
    axes = np.atleast_2d(axes)
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.imshow(crops_by_strategy[name], cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_solo(crop: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(crop.shape[1] / 200, crop.shape[0] / 200), dpi=200)
    ax.imshow(crop, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=7)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seg", default="20231221180251")
    p.add_argument("--data-dir", default="data/labelled_segments")
    p.add_argument("--out-dir", default="predictions/filter_strategy_benchmark")
    p.add_argument("--n-crops", type=int, default=3)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--skip", nargs="*", default=[],
                   help="Strategy names to skip (e.g. S9_nlm_sauvola for speed)")
    p.add_argument("--mode", choices=["text", "dense"], default="text",
                   help="text: moderate-density text-like crops; dense: solid-ink regions")
    args = p.parse_args()

    label_path = Path(args.data_dir) / args.seg / "ink_labels.tif"
    out_root = Path(args.out_dir) / args.seg
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading {label_path}")
    label = load_label(label_path)
    print(f"  shape={label.shape}, mean={label.mean():.3f}")

    picker = find_text_crops if args.mode == "text" else find_dense_crops
    coords = picker(label, args.crop, args.n_crops)
    print(f"Picked {len(coords)} {args.mode}-mode crops: {coords}")

    skip = set(args.skip)
    active_strategies = [s for s in STRATEGIES if s not in skip]
    print(f"Running {len(active_strategies)} strategies "
          f"(skipping {sorted(skip)})\n")

    # Per-strategy metrics aggregated across crops
    metrics_per_strategy: Dict[str, List[Dict]] = {s: [] for s in active_strategies}
    timings: Dict[str, float] = {s: 0.0 for s in active_strategies}

    # Process each crop
    for crop_idx, (y, x) in enumerate(coords):
        crop = label[y:y + args.crop, x:x + args.crop].copy()
        crop_dir = out_root / f"crop_{crop_idx:02d}_y{y}_x{x}"
        crop_dir.mkdir(exist_ok=True)
        print(f"--- Crop {crop_idx}  (y={y}, x={x}) ---")

        outputs: Dict[str, np.ndarray] = {}
        for name in active_strategies:
            t0 = time.perf_counter()
            try:
                out = STRATEGIES[name](crop)
            except Exception as e:
                print(f"  {name}: FAILED ({e})")
                outputs[name] = np.zeros_like(crop)
                continue
            dt = time.perf_counter() - t0
            timings[name] += dt

            outputs[name] = out
            m = compute_metrics(out)
            metrics_per_strategy[name].append(m)
            print(f"  {name:30s} {dt:5.2f}s   "
                  f"sep={m['separation']:.3f}  "
                  f"ccs={m['n_components']:5d}  "
                  f"compact={m['mean_compactness']:.3f}  "
                  f"speckle={m['speckle_index']:.3f}")

            save_solo(out, crop_dir / f"{name}.png",
                      f"{args.seg} y={y} x={x} — {name}")

        # Side-by-side grid for this crop
        save_strategy_grid(
            outputs, crop_dir / "grid.png",
            f"{args.seg}  crop y={y} x={x}  —  {args.crop}×{args.crop}",
        )

    # Aggregate metrics across crops
    summary: Dict[str, Dict] = {}
    for name in active_strategies:
        rows = metrics_per_strategy[name]
        if not rows:
            continue
        agg: Dict[str, float] = {}
        for k in rows[0]:
            agg[k] = float(np.mean([r[k] for r in rows]))
        agg["total_seconds"] = round(timings[name], 3)
        summary[name] = agg

    results_path = out_root / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "segment":    args.seg,
                "n_crops":    len(coords),
                "crop_size":  args.crop,
                "coords":     coords,
                "strategies": summary,
            },
            f, indent=2,
        )

    # Print summary table
    print("\n" + "=" * 110)
    print(f"{'strategy':30s} {'sep':>6s} {'ccs':>6s} {'med_area':>9s} "
          f"{'compact':>8s} {'speckle':>8s} {'ink_pct':>8s} {'time_s':>7s}")
    print("-" * 110)
    for name in active_strategies:
        if name not in summary:
            continue
        s = summary[name]
        print(f"{name:30s} {s['separation']:6.3f} {s['n_components']:6.0f} "
              f"{s['median_cc_area']:9.1f} {s['mean_compactness']:8.3f} "
              f"{s['speckle_index']:8.3f} {s['ink_fraction']*100:7.2f}% "
              f"{s['total_seconds']:7.2f}")

    print(f"\nDone. Outputs in {out_root}")
    print(f"Metrics JSON: {results_path}")


if __name__ == "__main__":
    main()
