"""
scripts/build_decipher_demo.py

Build label-aligned deciphering assets for one segment:
  - label_full.png   - the entire enhanced label rendered at a screen-readable
                       width with light contrast boost
  - result.skeleton.json - per-line metadata (y-band + per-blob bounding
                           boxes in native pixel coords) so a downstream
                           process can assign Greek characters 1:1.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile as tf
from PIL import Image
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
DETECTOR_VERSION = 3


def load_label(path: Path) -> np.ndarray:
    arr = tf.imread(str(path))
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0, 1)


def render_full_label(label: np.ndarray, out: Path,
                      target_w: int = 1600,
                      threshold: float = 0.55,
                      gain: float = 1.25) -> tuple[int, int]:
    """Render the entire label as a single PNG at target width."""
    H, W = label.shape
    new_h = int(H * (target_w / W))
    img = Image.fromarray((np.clip(label, 0, 1) * 255).astype(np.uint8), mode="L")
    img = img.resize((target_w, new_h), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = np.where(a >= threshold, a, 0.0)
    a = np.clip(a * gain, 0, 1)
    Image.fromarray((a * 255).astype(np.uint8), mode="L").save(out, optimize=True)
    return target_w, new_h


def detect_all_lines(label: np.ndarray,
                     ink_threshold: float = 0.55,
                     smooth: int = 20,
                     min_spacing: int = 120,
                     band_pad: int = 28) -> list[tuple[int, int]]:
    """Return a list of (y0, y1) bands, one per detected text line.

    Algorithm: per-row ink density -> boxcar smooth -> local maxima above
    a density threshold with min spacing. The band extent around each peak
    is found by walking outward until density drops below half of peak.
    """
    H, W = label.shape
    row_density = (label > ink_threshold).mean(axis=1)
    kernel = np.ones(2 * smooth + 1) / (2 * smooth + 1)
    smooth_d = np.convolve(row_density, kernel, mode="same")
    thresh = max(0.04, smooth_d.mean() * 1.15)

    peaks: list[int] = []
    last = -10_000
    for y in range(smooth, H - smooth):
        if smooth_d[y] < thresh:
            continue
        if smooth_d[y] >= smooth_d[y - 1] and smooth_d[y] >= smooth_d[y + 1]:
            if y - last > min_spacing:
                peaks.append(y)
                last = y

    bands: list[tuple[int, int]] = []
    for y in peaks:
        peak_v = smooth_d[y]
        half = peak_v * 0.5
        y0 = y
        while y0 > 0 and smooth_d[y0] > half:
            y0 -= 1
        y1 = y
        while y1 < H - 1 and smooth_d[y1] > half:
            y1 += 1
        y0 = max(0, y0 - band_pad)
        y1 = min(H - 1, y1 + band_pad)
        bands.append((int(y0), int(y1)))
    return bands


def detect_chars_in_line(label: np.ndarray,
                         y_band: tuple[int, int],
                         x_range: tuple[int, int] | None = None,
                         threshold: float = 0.55,
                         min_char_w: int = 70,
                         max_char_w: int = 700,
                         min_char_h: int = 45,
                         max_char_h: int = 420,
                         min_area: int = 1_200,
                         max_area: int = 120_000) -> list[tuple[int, int, int, int]]:
    """Return larger ink-component boxes within a text band.

    The older column-peak detector happily boxed small bright specks. For the
    web/LLM path we want actual ink glyph/blob regions, so this works directly
    from connected components after thresholding.
    """
    y0, y1 = y_band
    if x_range is None:
        x0, x1 = 0, label.shape[1]
    else:
        x0, x1 = x_range
    band = label[y0:y1, x0:x1]
    if band.size == 0:
        return []
    ink = band >= threshold
    ink = ndimage.binary_closing(ink, structure=np.ones((3, 3), dtype=bool))
    labels, n = ndimage.label(ink)
    objects = ndimage.find_objects(labels)
    boxes: list[tuple[int, int, int, int]] = []
    for label_idx in range(1, n + 1):
        sl = objects[label_idx - 1]
        if sl is None:
            continue
        sy, sx = sl
        bh = int(sy.stop - sy.start)
        bw = int(sx.stop - sx.start)
        area = int((labels[sl] == label_idx).sum())
        if (
            bw < min_char_w or bw > max_char_w
            or bh < min_char_h or bh > max_char_h
            or area < min_area or area > max_area
        ):
            continue
        boxes.append((int(sx.start + x0), int(sy.start + y0), bw, bh))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="20230702185753")
    ap.add_argument("--data-dir", default="data/labelled_segments")
    ap.add_argument("--web-dir", default="web/public/assets/decipher")
    ap.add_argument("--target-w", type=int, default=1600,
                    help="rendered display width of label_full.png")
    args = ap.parse_args()

    label_p = REPO / args.data_dir / args.seg / "ink_labels_enhanced.tif"
    if not label_p.exists():
        label_p = REPO / args.data_dir / args.seg / "ink_labels.tif"
    print(f"Loading {label_p}")
    label = load_label(label_p)
    H, W = label.shape
    print(f"  native shape {H}x{W}")

    out_dir = REPO / args.web_dir / args.seg
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale outputs from previous schema versions
    for stale in ["crops", "strips"]:
        d = out_dir / stale
        if d.exists():
            shutil.rmtree(d)
    for stale in ["overview.png", "result.skeleton.json"]:
        p = out_dir / stale
        if p.exists():
            p.unlink()

    label_full_p = out_dir / "label_full.png"
    disp_w, disp_h = render_full_label(label, label_full_p,
                                       target_w=args.target_w)
    print(f"  label_full -> {label_full_p.relative_to(REPO)} ({disp_w}x{disp_h})")

    bands = detect_all_lines(label)
    print(f"  detected {len(bands)} text lines")

    lines_meta: list[dict] = []
    total_chars = 0
    for i, (y0, y1) in enumerate(bands, start=1):
        boxes = detect_chars_in_line(label, (y0, y1))
        if not boxes:
            continue
        xs = [b[0] for b in boxes] + [b[0] + b[2] for b in boxes]
        x0 = min(xs)
        x1 = max(xs)
        total_chars += len(boxes)
        lines_meta.append({
            "line_no": i,
            "y_band": [int(y0), int(y1)],
            "x_range": [int(x0), int(x1)],
            "char_bboxes": [list(b) for b in boxes],
        })
    print(f"  total blob/char candidates: {total_chars}")

    skeleton = {
        "seg_id": args.seg,
        "detector_version": DETECTOR_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "label_path": str(label_p),
        "label_image": {
            "path": "label_full.png",
            "width": int(disp_w),
            "height": int(disp_h),
            "native_width": int(W),
            "native_height": int(H),
        },
        "lines": lines_meta,
    }
    skel_p = out_dir / "result.skeleton.json"
    skel_p.write_text(json.dumps(skeleton, indent=2), encoding="utf-8")
    print(f"Skeleton -> {skel_p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
