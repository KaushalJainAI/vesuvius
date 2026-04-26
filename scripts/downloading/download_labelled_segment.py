"""
Download surface_volume Z-stack for a labelled segment.

Usage:
    python download_labelled_segment.py --seg 20231221180251
    python download_labelled_segment.py --seg 20231221180251 20231031143852  # multiple
    python download_labelled_segment.py --all                                # all 11

Labels are already downloaded to data/labelled_segments/{seg}/ink_labels.tif
Surface volumes go to              data/labelled_segments/{seg}/surface_volume/
"""
import argparse
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Server uses a self-signed cert chain — disable verification for this trusted host
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

BASE_URL = "https://data.aws.ash2txt.org/samples/PHercParis4/segments"
VOLUME   = "7.91um-54keV-volume-20230205180739.tifs"
NUM_LAYERS = 33  # 00–32

ALL_SEGMENTS = [
    "20231221180251",  # 3.5 GB  ← smallest
    "20231031143852",  # 3.7 GB
    "20231016151002",  # 4.0 GB
    "20231106155351",  # 4.5 GB
    "20230702185753",  # 4.6 GB
    "20231210121321",  # 5.1 GB
    "20230929220926",  # 6.1 GB
    "20231022170901",  # 6.3 GB
    "20231005123336",  # 8.6 GB
    "20231012184424",  # 9.8 GB
    "20231007101619",  # 14.2 GB ← largest
]

DATA_ROOT = Path(__file__).parent / "data" / "labelled_segments"


def download_layer(seg: str, layer: int, out_dir: Path, retries: int = 3) -> tuple[int, bool]:
    fname = f"{layer:02d}.tif"
    url   = f"{BASE_URL}/{seg}/surface-volumes/{VOLUME}/{fname}"
    dest  = out_dir / fname

    if dest.exists() and dest.stat().st_size > 0:
        return layer, True  # already downloaded

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            if dest.exists():
                req.add_header("Range", f"bytes={dest.stat().st_size}-")
            with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp, open(dest, "ab") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            return layer, True
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAILED layer {layer:02d}: {e}")
                return layer, False
            time.sleep(2 ** attempt)
    return layer, False


def download_segment(seg: str, workers: int = 4):
    out_dir = DATA_ROOT / seg / "surface_volume"
    out_dir.mkdir(parents=True, exist_ok=True)

    label = DATA_ROOT / seg / "ink_labels.tif"
    if not label.exists():
        print(f"WARNING: ink_labels.tif not found for {seg} — label was not downloaded")

    print(f"\n{'='*60}")
    print(f"Segment: {seg}")
    print(f"Label:   {label}  ({'OK' if label.exists() else 'MISSING'})")
    print(f"Output:  {out_dir}")
    print(f"Layers:  {NUM_LAYERS} (00–{NUM_LAYERS-1:02d})")

    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_layer, seg, i, out_dir): i for i in range(NUM_LAYERS)}
        for fut in as_completed(futures):
            layer, ok = fut.result()
            if ok:
                done += 1
                print(f"  [{done:2d}/{NUM_LAYERS}] layer {layer:02d} OK", end="\r")
            else:
                failed += 1

    print(f"\nDone: {done}/{NUM_LAYERS} layers downloaded, {failed} failed")
    if failed == 0:
        sz = sum(f.stat().st_size for f in out_dir.glob("*.tif"))
        print(f"Total size: {sz/1024**3:.2f} GB")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Download labelled segment surface volumes")
    parser.add_argument("--seg", nargs="+", help="Segment ID(s) to download")
    parser.add_argument("--all", action="store_true", help="Download all 11 segments")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download threads per segment")
    args = parser.parse_args()

    if args.all:
        segs = ALL_SEGMENTS
    elif args.seg:
        segs = args.seg
        invalid = [s for s in segs if s not in ALL_SEGMENTS]
        if invalid:
            print(f"Unknown segment IDs: {invalid}")
            print(f"Valid: {ALL_SEGMENTS}")
            sys.exit(1)
    else:
        parser.print_help()
        print("\nAvailable segments (smallest to largest):")
        sizes = [3.5, 3.7, 4.0, 4.5, 4.6, 5.1, 6.1, 6.3, 8.6, 9.8, 14.2]
        for seg, sz in zip(ALL_SEGMENTS, sizes):
            label_ok = (DATA_ROOT / seg / "ink_labels.tif").exists()
            print(f"  {seg}  ~{sz:4.1f} GB  label={'OK' if label_ok else 'MISSING'}")
        sys.exit(0)

    success = all(download_segment(seg, args.workers) for seg in segs)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
