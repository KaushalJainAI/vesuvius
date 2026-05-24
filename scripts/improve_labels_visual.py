"""Conservative visual-only label improvement for labelled segments.

This script does not call an LLM and does not invent missing characters. It
uses the Stage 1 filter pipeline to suppress noise and repair small stroke
breaks, then softly blends the result with the original pseudo-labels.

Example:
    python scripts/improve_labels_visual.py --seg 20231221180251
    python scripts/improve_labels_visual.py --all --alpha 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import tifffile as tf
from skimage import morphology

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ALL_SEGMENTS = [
    "20231221180251", "20231031143852", "20231016151002", "20231106155351",
    "20230702185753", "20231210121321", "20230929220926", "20231022170901",
    "20231005123336", "20231012184424", "20231007101619",
]


def load_label(path: Path) -> np.ndarray:
    arr = tf.imread(str(path)).astype(np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def blend_labels(original: np.ndarray, updated: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 0.5))
    return np.clip(alpha * updated + (1.0 - alpha) * original, 0.0, 1.0).astype(np.float32)


def compute_diff_stats(old: np.ndarray, new: np.ndarray) -> dict:
    diff = new.astype(np.float32) - old.astype(np.float32)
    abs_diff = np.abs(diff)
    return {
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
        "rms_diff": float(np.sqrt((diff ** 2).mean())),
        "pixels_increased_pct": float((diff > 0.05).mean() * 100),
        "pixels_decreased_pct": float((diff < -0.05).mean() * 100),
        "pixels_unchanged_pct": float((abs_diff <= 0.05).mean() * 100),
    }


def save_iteration_labels(labels: np.ndarray, seg_id: str, iteration: int, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{seg_id}_labels_v{iteration}"
    npy_path = out_dir / f"{stem}.npy"
    tif_path = out_dir / f"{stem}.tif"
    np.save(npy_path, labels.astype(np.float32))
    tf.imwrite(str(tif_path), (labels * 255).astype(np.uint8))
    return npy_path, tif_path


def fast_filter(prob: np.ndarray, *, threshold: float, median_radius: int) -> np.ndarray:
    smoothed = ndi.median_filter(prob, size=2 * median_radius + 1) if median_radius > 0 else prob
    binary = smoothed > threshold
    binary = morphologyg.closing(binary, morphology.disk(2))
    binary = morphology.opening(binary, morphology.disk(1))
    repaired = np.maximum(smoothed, threshold + 0.10) * binary.astype(np.float32)
    return np.clip(repaired, 0.0, 1.0)


def improve_one(
    seg_id: str,
    *,
    data_root: Path,
    out_dir: Path,
    alpha: float,
    threshold: float,
    median_radius: int,
    clahe_clip: float,
    method: str,
) -> dict:
    label_path = data_root / seg_id / "ink_labels.tif"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    original = load_label(label_path)
    if method == "clahe":
        from label_filters import apply_filters
        filtered = apply_filters(
            original,
            median_radius=median_radius,
            clahe_clip=clahe_clip,
            threshold=threshold,
        )
    else:
        filtered = fast_filter(original, threshold=threshold, median_radius=median_radius)

    improved = blend_labels(original, filtered, alpha=alpha)
    npy_path, tif_path = save_iteration_labels(improved, seg_id, 1, out_dir)

    stats = compute_diff_stats(original, improved)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "seg_id": seg_id,
        "input": str(label_path),
        "npy": str(npy_path),
        "tif": str(tif_path),
        "alpha": alpha,
        "threshold": threshold,
        "median_radius": median_radius,
        "clahe_clip": clahe_clip,
        "method": method,
        **stats,
    }
    return entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seg", action="append", help="Segment ID to improve. Repeat for multiple.")
    group.add_argument("--all", action="store_true", help="Improve all known labelled segments.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "labelled_segments")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "predictions" / "legacy_label_enhancement")
    parser.add_argument("--alpha", type=float, default=0.15, help="Blend weight toward filtered labels, clamped by blend_labels.")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--median-radius", type=int, default=1)
    parser.add_argument("--clahe-clip", type=float, default=0.02)
    parser.add_argument("--method", choices=["fast", "clahe"], default="fast")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segs = ALL_SEGMENTS if args.all else args.seg
    args.out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for seg_id in segs:
        print(f"[labels] improving {seg_id}...")
        entry = improve_one(
            seg_id,
            data_root=args.data_root,
            out_dir=args.out_dir,
            alpha=args.alpha,
            threshold=args.threshold,
            median_radius=args.median_radius,
            clahe_clip=args.clahe_clip,
            method=args.method,
        )
        history.append(entry)
        print(
            "  saved",
            Path(entry["tif"]).name,
            "mean|diff|=",
            f"{entry['mean_abs_diff']:.4f}",
            "changed=",
            f"{100.0 - entry['pixels_unchanged_pct']:.2f}%",
        )

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(history, indent=2))
    print(f"[labels] summary -> {summary_path}")


if __name__ == "__main__":
    main()
