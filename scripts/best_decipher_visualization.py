"""
scripts/best_decipher_visualization.py

Produce the most readable image of each segment's ink map for human/LLM
deciphering. Auto-detects input source:

  1. predictions/**/{seg_id}_prob.npy / *.tif   (model output)
  2. data/labelled_segments/{seg_id}/ink_labels_enhanced.tif
  3. data/labelled_segments/{seg_id}/ink_labels.tif

For each input it runs SIX rendering strategies and picks the one with the
highest readability score (gradient contrast on a high-density text crop).

Strategies:
  raw          - input as-is, just rescaled to 0-255
  gamma_ir     - gamma=0.7 + percentile stretch (mimics IR photo look)
  clahe_sharp  - CLAHE local contrast + unsharp mask
  unsharp      - light Gaussian + unsharp at two scales
  tv_sato      - reuses src.best_enhancer (TV + Sato vesselness, soft)
  hyst_clean   - hysteresis + morph + small-CC removal (crisp binary-ish)

Outputs (under predictions/best_decipher_vis/{seg_id}/):
  raw.png, gamma_ir.png, clahe_sharp.png, unsharp.png, tv_sato.png, hyst_clean.png
  best.png            -> copy of the winning strategy at full resolution
  best_overview.png   -> long-edge <= 2400 px overview of the winner
  grid.png            -> 2x3 thumbnail grid of all 6 strategies with scores
  scores.json         -> readability score for each strategy

Usage:
  python scripts/best_decipher_visualization.py
  python scripts/best_decipher_visualization.py --only 20231012184424 20231221180251
  python scripts/best_decipher_visualization.py --crop 1024     # center crop for speed
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from best_enhancer import enhance_tiled  # noqa: E402
from skimage import filters, morphology  # noqa: E402


STRATEGIES = ["raw", "gamma_ir", "clahe_sharp", "unsharp", "tv_sato", "hyst_clean"]


# ---------------------------------------------------------------------------
# I/O + input discovery
# ---------------------------------------------------------------------------

def find_input(seg_id: str) -> Tuple[Path, str]:
    """Locate the best available input for a segment.

    Returns (path, source_tag).
    """
    pred_root = REPO / "predictions"
    if pred_root.exists():
        for npy in pred_root.rglob(f"{seg_id}_prob.npy"):
            return npy, "prediction_npy"
        for tif in pred_root.rglob(f"{seg_id}_prob.tif"):
            return tif, "prediction_tif"

    enh = REPO / "data" / "labelled_segments" / seg_id / "ink_labels_enhanced.tif"
    if enh.exists():
        return enh, "enhanced_label"

    lbl = REPO / "data" / "labelled_segments" / seg_id / "ink_labels.tif"
    if lbl.exists():
        return lbl, "raw_label"

    raise FileNotFoundError(f"No prediction or label found for {seg_id}")


def load_prob(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        arr = tf.imread(str(path))
    arr = arr.astype(np.float32)
    if arr.dtype != np.float32 or arr.max() > 1.5:
        if arr.max() > 255.5:
            arr = arr / 65535.0
        elif arr.max() > 1.5:
            arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def to_u8(arr01: np.ndarray) -> np.ndarray:
    return (np.clip(arr01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def strat_raw(p: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(p, (1, 99))
    if hi <= lo:
        return p.copy()
    return np.clip((p - lo) / (hi - lo), 0.0, 1.0)


def strat_gamma_ir(p: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(p, (2, 99.5))
    if hi <= lo:
        return p.copy()
    s = np.clip((p - lo) / (hi - lo), 0.0, 1.0)
    return np.power(s, 0.7).astype(np.float32)


def strat_clahe_sharp(p: np.ndarray) -> np.ndarray:
    u8 = to_u8(p)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    eq = clahe.apply(u8)
    blur = cv2.GaussianBlur(eq, (0, 0), sigmaX=1.5)
    sharp = cv2.addWeighted(eq, 1.6, blur, -0.6, 0)
    return sharp.astype(np.float32) / 255.0


def strat_unsharp(p: np.ndarray) -> np.ndarray:
    u8 = to_u8(p)
    b1 = cv2.GaussianBlur(u8, (0, 0), sigmaX=1.0)
    s1 = cv2.addWeighted(u8, 1.5, b1, -0.5, 0)
    b2 = cv2.GaussianBlur(s1, (0, 0), sigmaX=3.0)
    s2 = cv2.addWeighted(s1, 1.3, b2, -0.3, 0)
    return s2.astype(np.float32) / 255.0


def strat_tv_sato(p: np.ndarray) -> np.ndarray:
    return enhance_tiled(p, tile=2048, overlap=64, soft_output=True, verbose=False)


def strat_hyst_clean(p: np.ndarray) -> np.ndarray:
    binary = filters.apply_hysteresis_threshold(p, 0.30, 0.55)
    binary = morphology.closing(binary, morphology.disk(2))
    binary = morphology.opening(binary, morphology.disk(1))
    binary = morphology.remove_small_objects(binary, min_size=40)
    blend = np.where(binary, np.maximum(p, 0.85), p * 0.25)
    return blend.astype(np.float32)


STRATEGY_FNS = {
    "raw":          strat_raw,
    "gamma_ir":     strat_gamma_ir,
    "clahe_sharp":  strat_clahe_sharp,
    "unsharp":      strat_unsharp,
    "tv_sato":      strat_tv_sato,
    "hyst_clean":   strat_hyst_clean,
}


# ---------------------------------------------------------------------------
# Readability scoring
# ---------------------------------------------------------------------------

def pick_text_crop(p: np.ndarray, size: int = 1024) -> Tuple[int, int]:
    """Find a high-ink-density square crop for scoring."""
    H, W = p.shape
    if H <= size or W <= size:
        return 0, 0
    step = max(128, size // 4)
    best, best_score = (H // 2 - size // 2, W // 2 - size // 2), -1.0
    for y in range(0, H - size, step):
        for x in range(0, W - size, step):
            tile = p[y:y + size, x:x + size]
            d = float((tile > 0.5).mean())
            if 0.05 < d < 0.20:
                row_var = float((tile > 0.5).mean(axis=1).std())
                score = row_var - 0.02 * abs(d - 0.10)
                if score > best_score:
                    best_score, best = score, (y, x)
    return best


def readability_score(img01: np.ndarray, y: int, x: int, size: int = 1024) -> float:
    """Higher = sharper, higher dynamic range, more bimodal."""
    H, W = img01.shape
    size = min(size, H, W)
    tile = img01[y:y + size, x:x + size]
    u8 = to_u8(tile)

    # Laplacian variance = sharpness
    lap_var = float(cv2.Laplacian(u8, cv2.CV_64F).var())

    # Bimodality via histogram peakiness
    hist, _ = np.histogram(u8, bins=32, range=(0, 255))
    hist = hist / max(hist.sum(), 1)
    bimod = float(hist[:8].sum() + hist[24:].sum())  # mass in dark + light tails

    # Dynamic range
    p1, p99 = np.percentile(u8, (1, 99))
    dyn = float(p99 - p1) / 255.0

    return 0.0006 * lap_var + 0.6 * bimod + 0.4 * dyn


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_overview(u8: np.ndarray, out_path: Path, max_side: int = 2400) -> None:
    h, w = u8.shape[:2]
    s = min(max_side / max(h, w), 1.0)
    if s < 1.0:
        u8 = cv2.resize(u8, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), u8)


def make_grid(images: Dict[str, np.ndarray], scores: Dict[str, float],
              winner: str, out_path: Path, seg_id: str, source: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=120)
    for ax, name in zip(axes.flat, STRATEGIES):
        img = images[name]
        h, w = img.shape
        s = min(1200 / max(h, w), 1.0)
        thumb = cv2.resize(img, (int(w * s), int(h * s)),
                           interpolation=cv2.INTER_AREA) if s < 1.0 else img
        ax.imshow(thumb, cmap="gray", vmin=0, vmax=255)
        marker = "  <-- BEST" if name == winner else ""
        ax.set_title(f"{name}   score={scores[name]:.3f}{marker}",
                     fontsize=10, fontweight="bold" if name == winner else "normal")
        ax.axis("off")
    fig.suptitle(f"{seg_id}  (source: {source})  -  best: {winner}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-segment runner
# ---------------------------------------------------------------------------

def process_segment(seg_id: str, out_root: Path, crop: int | None = None) -> Dict:
    path, source = find_input(seg_id)
    print(f"=== {seg_id} ===  input: {path.relative_to(REPO)} ({source})")
    t0 = time.perf_counter()
    prob = load_prob(path)
    print(f"  loaded {prob.shape} dtype={prob.dtype}  mean={prob.mean():.3f}")

    if crop:
        H, W = prob.shape
        cy, cx = H // 2, W // 2
        h = min(crop, H) // 2
        w = min(crop, W) // 2
        prob = prob[cy - h:cy + h, cx - w:cx + w]
        print(f"  center-cropped to {prob.shape}")

    cy, cx = pick_text_crop(prob, size=min(1024, *prob.shape))

    out_dir = out_root / seg_id
    out_dir.mkdir(parents=True, exist_ok=True)

    images: Dict[str, np.ndarray] = {}
    scores: Dict[str, float] = {}

    for name in STRATEGIES:
        ts = time.perf_counter()
        try:
            out01 = STRATEGY_FNS[name](prob)
            u8 = to_u8(out01)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            continue
        score = readability_score(out01, cy, cx, size=min(1024, *prob.shape))
        images[name] = u8
        scores[name] = score
        cv2.imwrite(str(out_dir / f"{name}.png"), u8)
        print(f"  [{name:11s}] score={score:.3f}  ({time.perf_counter() - ts:.1f}s)")
        del out01
        gc.collect()

    if not scores:
        print(f"  ! no strategies succeeded for {seg_id}")
        return {"seg_id": seg_id, "error": "all strategies failed"}

    winner = max(scores, key=scores.get)
    print(f"  -> winner: {winner}  score={scores[winner]:.3f}")

    cv2.imwrite(str(out_dir / "best.png"), images[winner])
    save_overview(images[winner], out_dir / "best_overview.png")
    make_grid(images, scores, winner, out_dir / "grid.png", seg_id, source)

    result = {
        "seg_id":   seg_id,
        "source":   source,
        "input":    str(path.relative_to(REPO)),
        "shape":    list(prob.shape),
        "winner":   winner,
        "scores":   scores,
        "seconds":  round(time.perf_counter() - t0, 1),
    }
    with (out_dir / "scores.json").open("w") as f:
        json.dump(result, f, indent=2)

    del images, prob
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/labelled_segments")
    p.add_argument("--out-dir",  default="predictions/best_decipher_vis")
    p.add_argument("--only", nargs="*", default=None,
                   help="Process only these segment ids")
    p.add_argument("--crop", type=int, default=None,
                   help="Optional center-crop side length for quick previews")
    args = p.parse_args()

    out_root = REPO / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    if args.only:
        seg_ids = list(args.only)
    else:
        seg_ids = sorted([
            d.name for d in (REPO / args.data_dir).iterdir()
            if d.is_dir() and (d / "ink_labels.tif").exists()
        ])

    print(f"Processing {len(seg_ids)} segments -> {out_root}\n")

    summary = {}
    for sid in seg_ids:
        try:
            summary[sid] = process_segment(sid, out_root, crop=args.crop)
        except Exception as e:
            print(f"  ! {sid} failed: {e}")
            summary[sid] = {"seg_id": sid, "error": str(e)}
        print()

    with (out_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print(f"{'segment':16s} {'winner':14s} {'score':>8s} {'time s':>8s}")
    print("-" * 80)
    for sid, r in summary.items():
        if "error" in r:
            print(f"{sid:16s} ERROR: {r['error']}")
            continue
        print(f"{sid:16s} {r['winner']:14s} "
              f"{r['scores'][r['winner']]:8.3f} {r['seconds']:8.1f}")


if __name__ == "__main__":
    main()
