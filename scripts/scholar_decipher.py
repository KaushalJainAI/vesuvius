"""Run the Scholar stage on one or all segments.

Usage:
    python scripts/scholar_decipher.py --only 20231221180251
    python scripts/scholar_decipher.py                            # all
    python scripts/scholar_decipher.py --only X --dry-run         # print, don't save

Requires `predictions/decipher/{seg_id}/result.json` to exist
(run scripts/decipher_all_segments.py first).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar.runner import run_scholar_for_segment  # noqa: E402
from scholar.mock import build_mock_scholar  # noqa: E402

ALL_SEGMENTS = [
    "20231221180251", "20231031143852", "20231016151002", "20231106155351",
    "20230702185753", "20231210121321", "20230929220926", "20231022170901",
    "20231005123336", "20231012184424", "20231007101619",
]


PRED_ROOT = ROOT / "predictions" / "decipher"
WEB_ROOT = ROOT / "web" / "public" / "assets" / "decipher"
AGENT_DECODING_ROOT = ROOT / "agent_decoding"


def run_mock(seg_id: str) -> None:
    seg_dir = PRED_ROOT / seg_id
    result_path = seg_dir / "result.json"
    if not result_path.exists():
        print(f"[skip] {seg_id}: no result.json")
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    scholar = build_mock_scholar(result)
    result["scholar"] = scholar
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (seg_dir / "scholar.json").write_text(
        json.dumps(scholar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    AGENT_DECODING_ROOT.mkdir(parents=True, exist_ok=True)
    (AGENT_DECODING_ROOT / f"{seg_id}.json").write_text(
        json.dumps(scholar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    web_dir = WEB_ROOT / seg_id
    web_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result_path, web_dir / "result.json")
    shutil.copy2(seg_dir / "scholar.json", web_dir / "scholar.json")
    print(f"[mock] {seg_id}: wrote scholar ({len(scholar['strips'])} strips) -> agent_decoding/{seg_id}.json + web mirror")


async def main_async(args) -> None:
    segs = [args.only] if args.only else ALL_SEGMENTS
    for seg in segs:
        if args.mock:
            run_mock(seg)
            continue
        try:
            await run_scholar_for_segment(seg, dry_run=args.dry_run)
        except FileNotFoundError as e:
            print(f"[skip] {seg}: {e}")
        except Exception as e:
            print(f"[error] {seg}: {e!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", help="Run on a single segment id")
    p.add_argument("--dry-run", action="store_true", help="Print output, don't write files")
    p.add_argument("--mock", action="store_true",
                   help="Skip API calls; synthesize a realistic scholar block from existing consensus")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
