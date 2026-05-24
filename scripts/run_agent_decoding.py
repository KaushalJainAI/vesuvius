"""Run the local agent-decoding pipeline (no external API) across segments.

Outputs land in `agent_decoding/{seg_id}/` and mirror into
`web/public/assets/decipher/{seg_id}/agent/`.

Usage:
    python scripts/run_agent_decoding.py              # all 11 segments
    python scripts/run_agent_decoding.py --only ID    # single segment
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar.agent_pipeline import run_all, run_segment  # noqa: E402
from scholar.claude_pipeline import run_segment_sync as run_claude_segment, claude_available  # noqa: E402

ALL_SEGMENTS = [
    "20231221180251", "20231031143852", "20231016151002", "20231106155351",
    "20230702185753", "20231210121321", "20230929220926", "20231022170901",
    "20231005123336", "20231012184424", "20231007101619",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", help="Run a single segment id")
    p.add_argument("--with-claude", action="store_true",
                   help="Also run Claude Agent SDK pathway for side-by-side comparison")
    args = p.parse_args()

    segs = [args.only] if args.only else ALL_SEGMENTS
    for seg in segs:
        run_segment(seg)
    if args.with_claude:
        avail = claude_available()
        print(f"[claude] availability: {avail}")
        for seg in segs:
            try:
                run_claude_segment(seg)
            except Exception as e:
                print(f"[claude] {seg}: error {e!r}")


if __name__ == "__main__":
    main()
