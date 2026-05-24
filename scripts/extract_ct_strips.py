"""Generate per-strip CT crops to use as the background of the transcription view.

Reads each segment's existing decipher result.json, takes a mid-Z slice of the
surface volume, crops to each strip's (y_range, x_range), and writes
`web/public/assets/decipher/{seg_id}/ct_strips/strip_NN.png`.

Run after `scripts/decipher_all_segments.py`.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import tifffile as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "labelled_segments"
WEB_ROOT = ROOT / "web" / "public" / "assets" / "decipher"


def _load_mid_slice(seg_id: str, z_band: int = 4) -> np.ndarray:
    sv_dir = DATA_ROOT / seg_id / "surface_volume"
    slices = sorted(sv_dir.glob("*.tif"))
    if not slices:
        raise FileNotFoundError(f"No surface volume slices for {seg_id} at {sv_dir}")
    mid = len(slices) // 2
    lo = max(0, mid - z_band // 2)
    hi = min(len(slices), lo + z_band)
    acc = None
    for p in slices[lo:hi]:
        a = tf.imread(str(p)).astype(np.float32)
        acc = a if acc is None else acc + a
    avg = acc / float(hi - lo)
    avg = np.clip(avg, 0, 255).astype(np.uint8)
    return avg


def _stretch_contrast(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    lo = np.percentile(img, p_lo)
    hi = np.percentile(img, p_hi)
    if hi <= lo:
        return img
    out = (img.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def process_segment(seg_id: str) -> bool:
    web_dir = WEB_ROOT / seg_id
    result_path = web_dir / "result.json"
    if not result_path.exists():
        print(f"[skip] {seg_id}: no result.json")
        return False

    result = json.loads(result_path.read_text(encoding="utf-8"))
    strips = result.get("strips") or []
    if not strips:
        print(f"[skip] {seg_id}: no strips")
        return False

    try:
        ct = _load_mid_slice(seg_id)
    except FileNotFoundError as e:
        print(f"[skip] {seg_id}: {e}")
        return False

    out_dir = web_dir / "ct_strips"
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in strips:
        sid = int(s["strip_id"])
        y0, y1 = s["y_range"]
        x0, x1 = s["x_range"]
        y0 = max(0, int(y0)); y1 = min(ct.shape[0], int(y1))
        x0 = max(0, int(x0)); x1 = min(ct.shape[1], int(x1))
        crop = ct[y0:y1, x0:x1]
        crop = _stretch_contrast(crop)
        out_path = out_dir / f"strip_{sid:02d}.png"
        Image.fromarray(crop, mode="L").save(out_path, format="PNG", optimize=True)
        print(f"  -> {out_path.relative_to(WEB_ROOT)}  ({crop.shape[1]}x{crop.shape[0]})")

    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", help="Single segment id")
    args = p.parse_args()

    if args.only:
        segs = [args.only]
    else:
        segs = sorted(d.name for d in WEB_ROOT.iterdir()
                      if d.is_dir() and (d / "result.json").exists())

    for s in segs:
        print(f"[ct] {s}")
        try:
            process_segment(s)
        except Exception as e:
            print(f"  FAILED: {e!r}")


if __name__ == "__main__":
    main()
