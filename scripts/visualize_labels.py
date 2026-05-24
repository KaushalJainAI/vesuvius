"""
scripts/visualize_labels.py

Produce inspection images of a segment's pseudo-label map so we can SEE what
the filter pipeline has to work with.

Outputs (under predictions/raw_label_inspection/{segment_id}/):
  full.png         — whole label map, viridis
  hist.png         — histogram of label values (shows ignore-band density)
  binary_t40.png   — binary @ threshold 0.40
  binary_t55.png   — binary @ threshold 0.55
  crops/           — 6 random 512x512 crops with text-likely density
  zooms/           — 3 zoom-ins (256x256) on the densest letter regions

Usage:
  python scripts/visualize_labels.py --seg 20231221180251
  python scripts/visualize_labels.py --seg 20231221180251 --n-crops 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf


def load_label(path: Path) -> np.ndarray:
    arr = tf.imread(str(path))
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32)
    return np.clip(arr, 0.0, 1.0)


def find_dense_crops(label: np.ndarray, crop: int, n: int, threshold: float = 0.5,
                    rng: np.random.Generator | None = None) -> list[tuple[int, int]]:
    """Pick n top-left coords for crop x crop tiles, ranked by ink density."""
    rng = rng or np.random.default_rng(0)
    H, W = label.shape
    step = crop // 2
    candidates = []
    for y in range(0, H - crop, step):
        for x in range(0, W - crop, step):
            tile = label[y:y + crop, x:x + crop]
            density = float((tile > threshold).mean())
            if density > 0.02:  # at least 2% ink-like pixels
                candidates.append((density, y, x))
    candidates.sort(reverse=True)
    # take top 3n, sample n with some randomness so we don't get adjacent tiles
    top = candidates[: max(n * 3, n)]
    rng.shuffle(top)
    picked: list[tuple[int, int]] = []
    for _, y, x in top:
        if all(abs(y - py) > crop or abs(x - px) > crop for py, px in picked):
            picked.append((y, x))
        if len(picked) >= n:
            break
    return picked


def save_panel(fig_path: Path, image: np.ndarray, title: str, cmap: str = "viridis",
               vmin: float = 0.0, vmax: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(image.shape[1] / 200, image.shape[0] / 200), dpi=200)
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_histogram(fig_path: Path, label: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    ax.hist(label.ravel(), bins=64, color="steelblue", edgecolor="none")
    ax.axvspan(0.40, 0.55, alpha=0.15, color="orange", label="ignore band")
    ax.axvline(0.40, color="orange", linestyle="--", linewidth=0.7)
    ax.axvline(0.55, color="orange", linestyle="--", linewidth=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("label value")
    ax.set_ylabel("pixel count (log)")
    ax.set_title("label value distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def save_crop_grid(fig_path: Path, label: np.ndarray, coords: list[tuple[int, int]],
                   crop: int, title: str) -> None:
    n = len(coords)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=150)
    axes = np.atleast_2d(axes)
    for i, (y, x) in enumerate(coords):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.imshow(label[y:y + crop, x:x + crop], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"y={y}, x={x}", fontsize=7)
        ax.axis("off")
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seg", default="20231221180251", help="segment id")
    p.add_argument("--data-dir", default="data/labelled_segments")
    p.add_argument("--out-dir", default="predictions/raw_label_inspection")
    p.add_argument("--n-crops", type=int, default=6)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--zoom", type=int, default=256)
    args = p.parse_args()

    label_path = Path(args.data_dir) / args.seg / "ink_labels.tif"
    out = Path(args.out_dir) / args.seg
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "zooms").mkdir(parents=True, exist_ok=True)

    print(f"Loading {label_path}")
    label = load_label(label_path)
    print(f"  shape={label.shape}, dtype={label.dtype}, "
          f"min={label.min():.3f}, max={label.max():.3f}, mean={label.mean():.3f}")

    # Whole-segment views (downsampled for matplotlib speed if huge)
    H, W = label.shape
    if max(H, W) > 4000:
        step = max(H, W) // 2000
        small = label[::step, ::step]
    else:
        small = label
    save_panel(out / "full.png", small, f"{args.seg} — full label (viridis)")
    save_panel(out / "full_gray.png", small, f"{args.seg} — full label (gray)", cmap="gray")
    save_panel(out / "binary_t40.png", (small > 0.40).astype(np.float32),
               "binary @ t=0.40", cmap="gray")
    save_panel(out / "binary_t55.png", (small > 0.55).astype(np.float32),
               "binary @ t=0.55", cmap="gray")

    save_histogram(out / "hist.png", label)

    # Dense crops
    coords = find_dense_crops(label, args.crop, args.n_crops)
    print(f"Found {len(coords)} dense crops")
    save_crop_grid(out / "crops" / "grid.png", label, coords, args.crop,
                   f"{args.seg} — {args.crop}x{args.crop} dense crops")
    for i, (y, x) in enumerate(coords):
        save_panel(out / "crops" / f"crop_{i:02d}_y{y}_x{x}.png",
                   label[y:y + args.crop, x:x + args.crop],
                   f"y={y} x={x}", cmap="gray")

    # Zoom-ins (smaller, on densest 3)
    zoom_coords = find_dense_crops(label, args.zoom, 3, threshold=0.55)
    save_crop_grid(out / "zooms" / "grid.png", label, zoom_coords, args.zoom,
                   f"{args.seg} — {args.zoom}x{args.zoom} zooms (densest)")
    for i, (y, x) in enumerate(zoom_coords):
        save_panel(out / "zooms" / f"zoom_{i:02d}_y{y}_x{x}.png",
                   label[y:y + args.zoom, x:x + args.zoom],
                   f"y={y} x={x}", cmap="gray")

    print(f"Done. Outputs in {out}")


if __name__ == "__main__":
    main()
