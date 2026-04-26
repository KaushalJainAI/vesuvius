"""
Download Kaggle competition fragments from the Vesuvius Challenge data server.

Source: https://dl.ash2txt.org/fragments/
Downloads surface volumes (65 Z-layers), inklabels.png, and mask.png for
Frag1, Frag2, and Frag3.

Output layout:
  data/train/
    fragment1/
      surface_volume/00.tif ... 64.tif
      inklabels.png
      mask.png
    fragment2/
    fragment3/

Usage:
    python download_fragments.py
"""

import time
from pathlib import Path

import requests
from tqdm import tqdm

OUT_DIR  = Path("data/train")
BASE_URL = "https://dl.ash2txt.org/fragments"

FRAGMENTS = [
    ("Frag1", "PHercParis2Fr47.volpkg",  "fragment1"),
    ("Frag2", "PHercParis2Fr143.volpkg", "fragment2"),
    ("Frag3", "PHercParis1Fr34.volpkg",  "fragment3"),
]

N_LAYERS = 65  # 00.tif to 64.tif


def surface_url(frag_id: str, volpkg: str) -> str:
    return f"{BASE_URL}/{frag_id}/{volpkg}/working/54keV_exposed_surface"


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    for attempt in range(1, retries + 1):
        try:
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            r = requests.get(url, headers=headers, stream=True, timeout=60)

            if r.status_code == 416:
                return True  # already complete
            if r.status_code not in (200, 206):
                print(f"  HTTP {r.status_code} for {url}")
                return False

            total = int(r.headers.get("content-length", 0)) + existing
            mode  = "ab" if existing else "wb"

            with open(dest, mode) as f, tqdm(
                total=total, initial=existing,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc=dest.name, leave=False, ncols=80,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
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

    total_failed = []

    for frag_id, volpkg, local_name in FRAGMENTS:
        frag_dir = OUT_DIR / local_name
        sv_dir   = frag_dir / "surface_volume"
        sv_dir.mkdir(parents=True, exist_ok=True)
        base     = surface_url(frag_id, volpkg)

        print(f"\n=== {local_name} ({frag_id}) ===")

        # mask.png and inklabels.png
        for fname in ("mask.png", "inklabels.png"):
            dest = frag_dir / fname
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  {fname} already present, skipping.")
                continue
            print(f"  Downloading {fname}...")
            if not download_file(f"{base}/{fname}", dest):
                total_failed.append(f"{local_name}/{fname}")

        # 65 Z-layer TIFFs
        layer_failed = []
        print(f"  Downloading {N_LAYERS} Z-layers...")
        for z in range(N_LAYERS):
            name = f"{z:02d}.tif"
            dest = sv_dir / name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            if not download_file(f"{base}/surface_volume/{name}", dest):
                layer_failed.append(name)

        if layer_failed:
            print(f"  WARNING: {len(layer_failed)} layers failed.")
            total_failed.extend(f"{local_name}/surface_volume/{n}" for n in layer_failed)
        else:
            size_gb = sum(f.stat().st_size for f in frag_dir.rglob("*") if f.is_file()) / 1e9
            print(f"  All {N_LAYERS} layers complete. ({size_gb:.2f} GB)")

    # Summary
    print("\n=== Summary ===")
    for _, _, local_name in FRAGMENTS:
        frag_dir = OUT_DIR / local_name
        sv       = frag_dir / "surface_volume"
        layers   = len(list(sv.glob("*.tif"))) if sv.exists() else 0
        labels   = (frag_dir / "inklabels.png").exists()
        mask     = (frag_dir / "mask.png").exists()
        size_gb  = sum(f.stat().st_size for f in frag_dir.rglob("*") if f.is_file()) / 1e9
        status   = "OK" if layers == 65 and labels and mask else "INCOMPLETE"
        print(f"  {local_name}  layers={layers}/65  "
              f"inklabels={'yes' if labels else 'NO'}  "
              f"mask={'yes' if mask else 'NO'}  "
              f"{size_gb:.2f} GB  [{status}]")

    if total_failed:
        print(f"\n{len(total_failed)} file(s) failed — re-run to retry.")
    else:
        print("\nAll fragments downloaded. Run fragment_ink_detection.ipynb to train.")


if __name__ == "__main__":
    main()
