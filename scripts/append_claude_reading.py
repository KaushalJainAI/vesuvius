"""Append a manual Claude reading to a segment's result.json.

I (Claude) looked at the 6 strips of segment 20231221180251 and recorded
my honest reading. This script injects those readings under the slug
`manual/claude-opus-4.7` in the per_model dict and recomputes the
consensus so the webapp shows my vote alongside the open-source models'.

Run:
    python scripts/append_claude_reading.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decipher.consensus import build_consensus  # noqa: E402
from decipher.model_registry import MANUAL_MODELS  # noqa: E402

SEG_ID = "20231221180251"
PRED_FILE = ROOT / "predictions" / "decipher" / SEG_ID / "result.json"
WEB_FILE = ROOT / "web" / "public" / "assets" / "decipher" / SEG_ID / "result.json"

# ---------------------------------------------------------------------------
# My honest reading per strip, after inspecting each strip image.
# Confidence is INTENTIONALLY LOW — these are heavily filtered ink probability
# blobs, not photographs of letters. I refuse to commit to specific letterforms
# where the stroke geometry is missing. This is the same answer the
# open-source models gave for the same reason.
# ---------------------------------------------------------------------------
MY_READINGS = {
    0: {
        "line_text": "",
        "characters": [],
        "translation_en": "[uncertain]",
        "notes": (
            "Strip 0 (y=346-546): widely-spaced soft puffs along the baseline. "
            "Stroke geometry not preserved by the upstream ink CNN — I can see "
            "blobs at letter-like spacing (~80-100 px apart) but cannot commit "
            "to specific letterforms. Refusing to guess."
        ),
        "overall_confidence": "low",
    },
    1: {
        "line_text": "",
        "characters": [
            {"char": "·", "x_norm": 0.08, "confidence": 0.30, "alternatives": []},
            {"char": "·", "x_norm": 0.55, "confidence": 0.30, "alternatives": []},
            {"char": "·", "x_norm": 0.78, "confidence": 0.30, "alternatives": []},
        ],
        "translation_en": "[uncertain]",
        "notes": (
            "Strip 1 (y=2142-2342): densest of the 6, with what appears to be a "
            "row of merged blobs. Letter spacing visible but no internal stroke "
            "definition. I can see three candidate positions where the blob mass "
            "looks letter-shaped; none specific enough to name."
        ),
        "overall_confidence": "low",
    },
    2: {
        "line_text": "",
        "characters": [
            {"char": "·", "x_norm": 0.10, "confidence": 0.35, "alternatives": []},
            {"char": "·", "x_norm": 0.50, "confidence": 0.40, "alternatives": []},
        ],
        "translation_en": "[uncertain]",
        "notes": (
            "Strip 2 (y=3894-4094): sparse left half, denser right half. A small "
            "isolated blob at x~0.10 could be a Ι or Ρ stem. Around x~0.50 there is "
            "a shape that resembles a Σ/Ε hook but the upstream resolution prevents "
            "commitment. Qwen 35B suggested ΕΝΘΑΔΕ here at low confidence — possible "
            "but not verifiable from the image alone."
        ),
        "overall_confidence": "low",
    },
    3: {
        "line_text": "",
        "characters": [
            {"char": "·", "x_norm": 0.65, "confidence": 0.35, "alternatives": []},
            {"char": "·", "x_norm": 0.88, "confidence": 0.35, "alternatives": []},
        ],
        "translation_en": "[uncertain]",
        "notes": (
            "Strip 3 (y=5631-5831): left two-thirds essentially empty. Right side "
            "shows two distinct blob clusters at x~0.65 and x~0.88. Both have "
            "letter-like extent but no internal topology — could be Ο/Θ/Φ/Ε family "
            "but I cannot distinguish."
        ),
        "overall_confidence": "low",
    },
    4: {
        "line_text": "",
        "characters": [],
        "translation_en": "[uncertain]",
        "notes": (
            "Strip 4 (y=7613-7813): clearly an inter-line gap, not a text line. "
            "Sparse scattered specks. Strip extractor placed this row between "
            "actual text lines; no characters to read."
        ),
        "overall_confidence": "low",
    },
    5: {
        "line_text": "",
        "characters": [
            {"char": "·", "x_norm": 0.12, "confidence": 0.30, "alternatives": []},
            {"char": "·", "x_norm": 0.32, "confidence": 0.35, "alternatives": []},
            {"char": "·", "x_norm": 0.58, "confidence": 0.30, "alternatives": []},
            {"char": "·", "x_norm": 0.82, "confidence": 0.30, "alternatives": []},
        ],
        "translation_en": "[uncertain]",
        "notes": (
            "Strip 5 (y=9120-9320): row of merged blob clusters with apparent "
            "regular spacing — most line-like of the six strips. Four candidate "
            "positions are visible, but the blob shapes are too smooth to "
            "identify. This is the position where I would expect actual text "
            "if the ink CNN had finer resolution."
        ),
        "overall_confidence": "low",
    },
}


def claude_spec():
    return next(m for m in MANUAL_MODELS if m.slug == "manual/claude-opus-4.7")


def inject_claude(data: dict) -> dict:
    spec = claude_spec()
    slug = spec.slug
    for strip in data.get("strips", []):
        sid = strip["strip_id"]
        parsed = MY_READINGS.get(sid)
        if not parsed:
            continue
        strip["per_model"][slug] = {
            "tier": spec.tier,
            "display_name": spec.display_name,
            "raw": json.dumps(parsed, ensure_ascii=False),
            "parsed": parsed,
            "error": None,
            "finish_reason": "manual_claude",
            "usage": None,
        }
        # Recompute consensus across all per_model entries (including me)
        per_model_parsed = {
            k: v.get("parsed") for k, v in strip["per_model"].items()
        }
        cons = build_consensus(per_model_parsed)
        strip["consensus"] = cons.to_dict()

    # Update top-level manual list
    manual = list(data.get("models_used_manual", []))
    if slug not in manual:
        manual.append(slug)
    data["models_used_manual"] = manual
    return data


def main() -> None:
    if not PRED_FILE.exists():
        print(f"[error] {PRED_FILE} not found. Run the decipher pipeline first.")
        sys.exit(1)
    data = json.loads(PRED_FILE.read_text(encoding="utf-8"))
    data = inject_claude(data)
    PRED_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] updated {PRED_FILE}")
    WEB_FILE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PRED_FILE, WEB_FILE)
    print(f"[done] mirrored to {WEB_FILE}")


if __name__ == "__main__":
    main()
