"""Add full-image character boxes to existing decipher result.json files.

This is a no-API post-process. It reads each segment's strip transcription,
snaps recognized characters to visible ink components in the filtered strip,
and writes bbox_full/bbox_strip metadata back into result.json so the ink-map
UI can draw squares for recognized letters.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decipher.alignment import align_result_characters, load_detection_lines  # noqa: E402

PRED_ROOT = ROOT / "predictions" / "decipher"
WEB_ROOT = ROOT / "web" / "public" / "assets" / "decipher"


def align_one(seg_id: str) -> bool:
    pred_dir = PRED_ROOT / seg_id
    web_dir = WEB_ROOT / seg_id
    result_path = pred_dir / "result.json"
    if not result_path.exists():
        result_path = web_dir / "result.json"
    if not result_path.exists():
        print(f"[skip] {seg_id}: no result.json")
        return False

    strips_dir = result_path.parent / "strips"
    if not strips_dir.exists():
        print(f"[skip] {seg_id}: no strips directory")
        return False

    data = json.loads(result_path.read_text(encoding="utf-8"))
    detection_lines = load_detection_lines(web_dir / "result.detection.json")
    align_result_characters(data, strips_dir=strips_dir, detection_lines=detection_lines)
    result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    if result_path.parent != web_dir:
        web_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_path, web_dir / "result.json")

    summary = data.get("alignment_summary", {})
    print(
        f"[align] {seg_id}: {summary.get('n_character_boxes', 0)} chars, "
        f"{summary.get('n_component_snapped', 0)} snapped, "
        f"{summary.get('n_linked_to_blobs', 0)} linked"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Align one segment id")
    args = parser.parse_args()

    if args.only:
        segs = [args.only]
    else:
        segs = sorted(d.name for d in WEB_ROOT.iterdir() if d.is_dir() and not d.name.startswith("sample"))
    for seg_id in segs:
        align_one(seg_id)


if __name__ == "__main__":
    main()
