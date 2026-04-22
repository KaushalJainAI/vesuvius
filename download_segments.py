"""
Download selected PHercParis4 scroll segments from dl.ash2txt.org.
Segments chosen to maximise coverage within a 16 GB budget (all 65 layers each).

Directory output layout:
  data/scroll/
    {segment_id}/
      surface_volume/
        00.tif ... 64.tif
      mask.png
      meta.json
"""

import os
import sys
import time
import requests
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths"
OUT_DIR  = Path("data/scroll")

# 7 segments, all 65 layers — total ~15.08 GB
# (Grand Prize segment first, then neighbours by size)
SEGMENTS = [
    ("20230827161847", "5.73 GB",  "Grand Prize — first readable text"),
    ("20230826170124", "2.35 GB",  "Nearby region"),
    ("20230820174948", "1.87 GB",  "Nearby region"),
    ("20230826135043", "1.55 GB",  "Nearby region"),
    ("20230828154913", "1.40 GB",  "Day after Grand Prize"),
    ("20230819093803", "1.15 GB",  "Nearby region"),
    ("20230504093154", "1.03 GB",  "Smaller early segment"),
]

N_LAYERS = 65   # download all 65 z-slices per segment


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Download url → dest, resume if partially downloaded, skip if complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    for attempt in range(1, retries + 1):
        try:
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            r = requests.get(url, headers=headers, stream=True, timeout=60)

            if r.status_code == 416:
                # Range not satisfiable → file already complete
                return True
            if r.status_code not in (200, 206):
                print(f"  HTTP {r.status_code} for {url}")
                return False

            total = int(r.headers.get("content-length", 0)) + existing
            mode  = "ab" if existing else "wb"

            with open(dest, mode) as f, tqdm(
                total=total,
                initial=existing,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
                leave=False,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB chunks
                    f.write(chunk)
                    bar.update(len(chunk))
            return True

        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"  Attempt {attempt}/{retries} failed: {e}")
            existing = dest.stat().st_size if dest.exists() else 0
            if attempt < retries:
                time.sleep(5 * attempt)

    return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR.resolve()}\n")

    total_segments = len(SEGMENTS)
    for seg_idx, (seg_id, approx_size, note) in enumerate(SEGMENTS, 1):
        seg_dir = OUT_DIR / seg_id
        sv_dir  = seg_dir / "surface_volume"
        sv_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{seg_idx}/{total_segments}] Segment {seg_id}  (~{approx_size})  — {note}")

        # --- mask.png ---
        # Server stores it as {id}_mask.png; we normalise to mask.png for the notebook.
        mask_dest = seg_dir / "mask.png"
        mask_url  = f"{BASE_URL}/{seg_id}/{seg_id}_mask.png"
        if not mask_dest.exists():
            print("  Downloading mask.png...")
            ok = download_file(mask_url, mask_dest)
            if not ok:
                print(f"  WARNING: could not download mask for {seg_id}")
        else:
            print("  mask.png already exists, skipping.")

        # --- meta.json ---
        meta_dest = seg_dir / "meta.json"
        meta_url  = f"{BASE_URL}/{seg_id}/meta.json"
        if not meta_dest.exists():
            download_file(meta_url, meta_dest)

        # --- TIFF layers ---
        failed = []
        for z in range(N_LAYERS):
            layer_name = f"{z:02d}.tif"
            dest       = sv_dir / layer_name
            url        = f"{BASE_URL}/{seg_id}/layers/{layer_name}"

            if dest.exists() and dest.stat().st_size > 0:
                continue  # already downloaded

            ok = download_file(url, dest)
            if not ok:
                failed.append(layer_name)

        if failed:
            print(f"  WARNING: {len(failed)} layer(s) failed: {failed}")
        else:
            print(f"  All {N_LAYERS} layers complete.")

    print("\n=== Download complete ===")
    total_gb = sum(
        f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()
    ) / 1e9
    print(f"Total data on disk: {total_gb:.2f} GB")


if __name__ == "__main__":
    main()
