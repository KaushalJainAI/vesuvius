"""
scripts/extract_text_strips.py

Extract wide horizontal text-line strips from an enhanced label so a human
(or model) can attempt to read the Greek letter sequence.

Each strip: 1800 x 200 px, centred on a detected text-line midline.
Saved as plain PNG under predictions/text_strips/{seg}/strip_NN.png plus a
side-by-side strips_grid.png contact sheet.

Usage:
  python scripts/extract_text_strips.py --seg 20231221180251
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_label(path: Path) -> np.ndarray:
    arr = tf.imread(str(path))
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0, 1)


def detect_text_lines(label: np.ndarray, threshold: float = 0.55,
                      smooth: int = 12) -> list[int]:
    """Return y-coords of text-line midlines by row-wise ink density peaks."""
    row_density = (label > threshold).mean(axis=1)
    # smooth with a 1d boxcar
    k = smooth
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    smooth_d = np.convolve(row_density, kernel, mode="same")

    # Find local maxima above the global mean
    thresh = max(0.04, smooth_d.mean() * 1.2)
    peaks: list[int] = []
    last = -100
    for y in range(k, len(smooth_d) - k):
        if smooth_d[y] < thresh:
            continue
        if smooth_d[y] >= smooth_d[y - 1] and smooth_d[y] >= smooth_d[y + 1]:
            if y - last > 60:        # min line spacing
                peaks.append(y)
                last = y
    return peaks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seg", default="20231221180251")
    p.add_argument("--data-dir", default="data/labelled_segments")
    p.add_argument("--out-dir", default="predictions/text_strips")
    p.add_argument("--width", type=int, default=2400)
    p.add_argument("--height", type=int, default=200)
    p.add_argument("--n-strips", type=int, default=8)
    p.add_argument("--use-enhanced", action="store_true", default=True)
    args = p.parse_args()

    seg_dir = Path(args.data_dir) / args.seg
    label_p = (seg_dir / "ink_labels_enhanced.tif"
               if args.use_enhanced else seg_dir / "ink_labels.tif")
    out = Path(args.out_dir) / args.seg
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {label_p}")
    label = load_label(label_p)
    H, W = label.shape

    # Choose horizontal center column with high overall ink density
    col_density = (label > 0.55).mean(axis=0)
    cx = int(np.argmax(np.convolve(col_density,
                                   np.ones(args.width) / args.width,
                                   mode="same")))
    x0 = max(0, cx - args.width // 2)
    x1 = min(W, x0 + args.width)
    x0 = max(0, x1 - args.width)
    print(f"  centre column = {cx}; strip x range = [{x0}, {x1}]")

    # Restrict line detection to that column range
    sub = label[:, x0:x1]
    lines = detect_text_lines(sub)
    print(f"  detected {len(lines)} candidate text lines")

    # Pick evenly-spaced lines if too many
    if len(lines) > args.n_strips:
        idx = np.linspace(0, len(lines) - 1, args.n_strips).astype(int)
        lines = [lines[i] for i in idx]

    strip_paths = []
    for i, ym in enumerate(lines):
        y0 = max(0, ym - args.height // 2)
        y1 = min(H, y0 + args.height)
        y0 = max(0, y1 - args.height)
        strip = label[y0:y1, x0:x1]

        # Save a low-DPI version (viewable in the chat) AND a high-DPI version
        for suffix, dpi in [("", 90), ("_hi", 200)]:
            fig, ax = plt.subplots(figsize=(12, 1.3), dpi=dpi)
            ax.imshow(strip, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"{args.seg}  y={y0}..{y1}  x={x0}..{x1}", fontsize=8)
            ax.axis("off")
            fig.tight_layout(pad=0)
            out_p = out / f"strip_{i:02d}_y{y0}{suffix}.png"
            fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        out_p = out / f"strip_{i:02d}_y{y0}.png"
        strip_paths.append(out_p)
        print(f"  strip {i}: y={ym}  -> {out_p.name}")

    # Contact sheet
    fig, axes = plt.subplots(len(lines), 1, figsize=(args.width / 200,
                                                     len(lines) * 1.6), dpi=140)
    if len(lines) == 1:
        axes = [axes]
    for ax, ym in zip(axes, lines):
        y0 = max(0, ym - args.height // 2)
        y1 = min(H, y0 + args.height)
        y0 = max(0, y1 - args.height)
        ax.imshow(label[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(f"y={y0}", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"{args.seg}  -  {len(lines)} text-line strips  "
                 f"({args.width}x{args.height} each)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    sheet = out / "strips_grid.png"
    fig.savefig(sheet, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGrid -> {sheet}")


if __name__ == "__main__":
    main()
