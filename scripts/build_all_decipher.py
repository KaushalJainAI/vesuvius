"""
scripts/build_all_decipher.py

Build honest, detection-only decipher artifacts for every labelled segment.

For each segment we:
  1. Run build_decipher_demo.py (if its skeleton.json is missing) to render
     label_full.png and detect text-line bands + per-blob bounding boxes.
  2. Convert that skeleton into a result.json the frontend can read.

result.detection.json contains ONLY data that is actually derived from the enhanced
ink-label image: the rendered label PNG, per-line y-bands, per-line x-ranges,
and per-line connected-component bounding boxes (one box per detected ink
blob). No Greek characters, transliterations, English translations,
"topics", or "confidences" are produced here -- transcription has not been
run on these segments and the frontend renders a clear notice to that
effect.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web/public/assets/decipher"
LABELLED = REPO / "data/labelled_segments"

SEGMENT_IDS = [
    "20230702185753",
    "20230929220926",
    "20231005123336",
    "20231007101619",
    "20231012184424",
    "20231016151002",
    "20231022170901",
    "20231031143852",
    "20231106155351",
    "20231210121321",
    "20231221180251",
]

DETECTOR_VERSION = 3


SOURCE_NOTE = (
    "Ink-blob bounding boxes are derived from the enhanced label image via "
    "connected-component analysis. No transcription model has been run on "
    "this segment yet, so no Greek characters, transliterations, or "
    "translations are attached."
)


def build_crops_for_segment(seg_id: str) -> bool:
    """Run build_decipher_demo.py when skeleton output is missing or stale."""
    skel_p = WEB / seg_id / "result.skeleton.json"
    if skel_p.exists():
        try:
            skel = json.loads(skel_p.read_text(encoding="utf-8"))
            if skel.get("detector_version") == DETECTOR_VERSION:
                return True
            print(f"  {seg_id}: stale skeleton - rebuilding")
        except (OSError, json.JSONDecodeError):
            print(f"  {seg_id}: unreadable skeleton - rebuilding")
    label_p = LABELLED / seg_id / "ink_labels_enhanced.tif"
    if not label_p.exists():
        label_p = LABELLED / seg_id / "ink_labels.tif"
    if not label_p.exists():
        print(f"  {seg_id}: NO LABEL - skipping")
        return False
    print(f"  {seg_id}: building label + skeleton")
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/build_decipher_demo.py"),
         "--seg", seg_id],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"    FAILED:\n{r.stderr}")
        return False
    return True


def build_result_for_segment(seg_id: str) -> bool:
    skel_p = WEB / seg_id / "result.skeleton.json"
    if not skel_p.exists():
        print(f"  {seg_id}: missing skeleton - skipping")
        return False
    skel = json.loads(skel_p.read_text(encoding="utf-8"))

    out_lines: list[dict] = []
    total_blobs = 0
    native_w = int(skel["label_image"]["native_width"])
    native_h = int(skel["label_image"]["native_height"])
    for raw in skel["lines"]:
        bboxes = [
            b for b in raw.get("char_bboxes", [])
            if _valid_bbox(b, native_w, native_h)
        ]
        if not bboxes:
            continue
        total_blobs += len(bboxes)
        out_lines.append({
            "line_no": raw["line_no"],
            "y_band": raw["y_band"],
            "x_range": raw["x_range"],
            "blob_bboxes": [list(b) for b in bboxes],
            "n_blobs": len(bboxes),
        })

    result = {
        "seg_id": seg_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "label_image": skel["label_image"],
        "scroll": "PHercParis4 (PHerc. 1497) - Philodemus",
        "source": "ink_labels_enhanced",
        "transcription_available": False,
        "note": SOURCE_NOTE,
        "n_lines": len(out_lines),
        "n_blobs": total_blobs,
        "lines": out_lines,
    }
    out_p = WEB / seg_id / "result.detection.json"
    out_p.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    print(f"  {seg_id}: {len(out_lines)} lines, {total_blobs} blobs "
          f"-> result.detection.json")
    return True


def _valid_bbox(box: list[int], native_w: int, native_h: int) -> bool:
    if len(box) != 4:
        return False
    x, y, w, h = [int(v) for v in box]
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return False
    if x + w > native_w or y + h > native_h:
        return False
    # These are blob candidates, not line or word boxes. Oversized boxes are
    # usually stale output from older detector parameters and overwhelm the UI.
    return 45 <= h <= 420 and 70 <= w <= 700


def write_kaggle_placeholder() -> None:
    seg_id = "sample_readable_kaggle"
    out_dir = WEB / seg_id
    if not out_dir.exists():
        return
    result = {
        "seg_id": seg_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "scroll": "Kaggle fragment - readable sample",
        "source": "kaggle_sample",
        "label_image": None,
        "transcription_available": False,
        "note": ("Demo Kaggle fragment; no enhanced-label rendering or "
                 "blob detection is attached to this sample."),
        "n_lines": 0,
        "n_blobs": 0,
        "lines": [],
    }
    (out_dir / "result.detection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {seg_id}: placeholder result.detection.json written")


def main() -> None:
    for seg_id in SEGMENT_IDS:
        build_crops_for_segment(seg_id)
    print()
    for seg_id in SEGMENT_IDS:
        build_result_for_segment(seg_id)
    print()
    write_kaggle_placeholder()
    print("\nDone.")


if __name__ == "__main__":
    main()
