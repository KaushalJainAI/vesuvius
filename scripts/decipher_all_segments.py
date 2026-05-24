"""Orchestrator — run the deciphering pipeline across labelled segments.

Usage:
    # extract strips only, no LLM calls (use this to populate the webapp
    # with mock-quality readings before you have an API key set up)
    python scripts/decipher_all_segments.py --mock --only 20231221180251

    # real run, all 11 segments, open-source models only
    python scripts/decipher_all_segments.py

    # real run, single segment
    python scripts/decipher_all_segments.py --only 20231221180251

Outputs go to predictions/decipher/{seg_id}/ AND mirror into
web/public/assets/decipher/{seg_id}/ so the webapp can pick them up.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decipher.model_registry import OPEN_SOURCE_MODELS, MANUAL_MODELS, ModelSpec, by_slug  # noqa: E402
from decipher.pipeline import decipher_segment_async, _segment_summary  # noqa: E402

ALL_SEGMENTS = [
    "20231221180251", "20231031143852", "20231016151002", "20231106155351",
    "20230702185753", "20231210121321", "20230929220926", "20231022170901",
    "20231005123336", "20231012184424", "20231007101619",
]


def _model_specs(slugs: Optional[List[str]]) -> Optional[List[ModelSpec]]:
    if not slugs:
        return None
    specs: List[ModelSpec] = []
    for slug in slugs:
        try:
            specs.append(by_slug(slug))
        except KeyError:
            specs.append(ModelSpec(
                slug=slug,
                display_name=slug.split("/")[-1],
                tier=1,
                provider=slug.split("/", 1)[0] if "/" in slug else "openrouter",
                notes="Ad-hoc CLI model not listed in model_registry.py",
            ))
    return specs

DATA_ROOT = ROOT / "data" / "labelled_segments"
PRED_ROOT = ROOT / "predictions" / "decipher"
WEB_ROOT = ROOT / "web" / "public" / "assets" / "decipher"


# ---------------------------------------------------------------------------
# Mock generator
# ---------------------------------------------------------------------------

GREEK_UPPER = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
LUNATE = "Ϲ"
COMMON_BIGRAMS = ["ΕΝ", "ΟΝ", "ΟΥ", "ΑΙ", "ΕΙ", "ΗΣ", "ΟΣ", "ΤΟ", "ΤΑ", "ΕΣ"]

# Plausible Greek words from Herculaneum / Epicurean texts (Philodemus).
# Mock "readings" pull from this dictionary so the UI shows recognisable
# words rather than random garbage. Each entry: (greek_uncial, english).
MOCK_VOCAB: List[tuple[str, str]] = [
    ("ΠΟΡΦΥΡΑϹ",  "of purple"),
    ("ΦΙΛΟΣΟΦΙΑ", "philosophy"),
    ("ΗΔΟΝΗ",     "pleasure"),
    ("ΑΡΕΤΗ",     "virtue"),
    ("ΕΠΙΚΟΥΡΟϹ", "Epicurus"),
    ("ΘΕΟϹ",      "god"),
    ("ΣΟΦΙΑ",     "wisdom"),
    ("ΦΥΣΙϹ",     "nature"),
    ("ΛΟΓΟϹ",     "reason / word"),
    ("ΨΥΧΗ",      "soul"),
    ("ΑΛΗΘΕΙΑ",   "truth"),
    ("ΑΓΑΘΟΝ",    "the good"),
    ("ΔΙΚΑΙΟΝ",   "the just"),
    ("ΚΟΣΜΟϹ",    "world / order"),
    ("ΕΛΕΥΘΕΡΙΑ", "freedom"),
    ("ΠΟΛΙΤΕΙΑ",  "constitution"),
    ("ΑΝΘΡΩΠΟϹ",  "human"),
    ("ΖΩΗ",       "life"),
]
# Tiny per-word-translation jitter so different models don't return
# byte-identical strings (more realistic ensemble).
ENG_VARIATIONS: Dict[str, List[str]] = {
    "of purple":    ["of purple",      "purple [gen.]",    "purple (genitive)"],
    "philosophy":   ["philosophy",     "love of wisdom",   "philosophia"],
    "pleasure":     ["pleasure",       "delight",          "pleasure (hedone)"],
    "virtue":       ["virtue",         "excellence",       "arete"],
    "Epicurus":     ["Epicurus",       "Epikouros",        "Epicurus (name)"],
    "god":          ["god",            "deity",            "the god"],
    "wisdom":       ["wisdom",         "sophia",           "wisdom / skill"],
    "nature":       ["nature",         "physis",           "natural order"],
    "reason / word":["reason",         "word",             "logos"],
    "soul":         ["soul",           "psyche",           "soul / mind"],
    "truth":        ["truth",          "aletheia",         "truthfulness"],
    "the good":     ["the good",       "good (n.)",        "the good thing"],
    "the just":     ["the just",       "justice",          "the right"],
    "world / order":["the cosmos",     "world-order",      "kosmos"],
    "freedom":      ["freedom",        "liberty",          "eleutheria"],
    "constitution": ["polity",         "state",            "constitution"],
    "human":        ["human being",    "man",              "human"],
    "life":         ["life",           "living",           "zoe (life)"],
}


def _build_chars(greek: str, rng: random.Random) -> List[dict]:
    n = len(greek)
    if n == 0:
        return []
    chars = []
    for i, ch in enumerate(greek):
        chars.append({
            "char": ch,
            "x_norm": round((i + 0.5) / n, 4),
            "confidence": round(rng.uniform(0.45, 0.92), 3),
            "alternatives": [rng.choice(GREEK_UPPER)] if rng.random() < 0.3 else [],
        })
    return chars


def mock_for_strip(seg_id: str, strip_idx: int, rng: random.Random) -> dict:
    # Pick 1–2 vocab words for variety; concatenate with optional space
    n_words = 1 if rng.random() < 0.55 else 2
    chosen = [rng.choice(MOCK_VOCAB) for _ in range(n_words)]
    greek = chosen[0][0]
    eng_parts = [rng.choice(ENG_VARIATIONS.get(chosen[0][1], [chosen[0][1]]))]
    for g, e in chosen[1:]:
        greek = greek + g
        eng_parts.append(rng.choice(ENG_VARIATIONS.get(e, [e])))
    chars = _build_chars(greek, rng)
    return {
        "line_text": greek,
        "characters": chars,
        "translation_en": " · ".join(eng_parts),
        "notes": "MOCK reading (no API call made)",
        "overall_confidence": rng.choice(["low", "low", "medium"]),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_segment(seg_id: str, *, mock: bool, n_strips: int, api_key: Optional[str],
                      skip_api: bool = False, model_slugs: Optional[List[str]] = None) -> None:
    label_path = DATA_ROOT / seg_id / "ink_labels_enhanced.tif"
    if not label_path.exists():
        # fall back to original labels if enhanced run hasn't been done
        alt = DATA_ROOT / seg_id / "ink_labels.tif"
        if alt.exists():
            label_path = alt
        else:
            print(f"[skip] {seg_id}: no labels at {label_path} or {alt}")
            return

    out_dir = PRED_ROOT / seg_id
    out_dir.mkdir(parents=True, exist_ok=True)
    detection_src = WEB_ROOT / seg_id / "result.detection.json"
    if detection_src.exists():
        shutil.copy2(detection_src, out_dir / "result.detection.json")

    if mock:
        # extract strips then synthesise per-model readings
        from decipher.strip_extractor import extract_strips
        (out_dir / "strips").mkdir(parents=True, exist_ok=True)

        print(f"[mock] {seg_id}: extracting strips")
        strips = extract_strips(label_path, n_strips=n_strips)
        for s in strips:
            (out_dir / "strips" / f"strip_{s.index:02d}.png").write_bytes(s.png_bytes)

        rng = random.Random(hash(seg_id) & 0xFFFFFFFF)
        strips_out = []
        models = OPEN_SOURCE_MODELS
        for s in strips:
            per_model = {}
            for m in models:
                parsed = mock_for_strip(seg_id, s.index, rng)
                per_model[m.slug] = {
                    "tier": m.tier,
                    "display_name": m.display_name,
                    "raw": json.dumps(parsed, ensure_ascii=False),
                    "parsed": parsed,
                    "error": None,
                    "finish_reason": "mock",
                    "usage": None,
                }
            from decipher.consensus import build_consensus
            consensus = build_consensus({k: v["parsed"] for k, v in per_model.items()})
            strips_out.append({
                "strip_id": s.index,
                "image_path": f"strips/strip_{s.index:02d}.png",
                "y_range": [s.y0, s.y1],
                "x_range": [s.x0, s.x1],
                "y_center": s.y_center,
                "per_model": per_model,
                "consensus": consensus.to_dict(),
            })

        segment_summary = _segment_summary(strips_out)
        result = {
            "seg_id": seg_id,
            "created": "mock",
            "label_path": str(label_path),
            "models_used_open_source": [m.slug for m in models],
            "models_used_manual": [],
            "n_strips": len(strips),
            "mock": True,
            "segment_summary": segment_summary,
            "segment_text": segment_summary["text"],
            "segment_meaning": segment_summary["probable_summary"],
            "strips": strips_out,
        }
        (out_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"[mock] wrote {out_dir / 'result.json'}")
    else:
        await decipher_segment_async(
            seg_id=seg_id, label_path=label_path,
            out_dir=out_dir, n_strips=n_strips,
            api_key=api_key,
            skip_api=skip_api,
            models=_model_specs(model_slugs),
        )

    # Mirror to webapp public assets
    web_dir = WEB_ROOT / seg_id
    web_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_dir / "result.json", web_dir / "result.json")
    web_strips = web_dir / "strips"
    if web_strips.exists():
        shutil.rmtree(web_strips)
    shutil.copytree(out_dir / "strips", web_strips)
    plates_dir = out_dir / "plates"
    if plates_dir.exists():
        web_plates = web_dir / "plates"
        if web_plates.exists():
            shutil.rmtree(web_plates)
        shutil.copytree(plates_dir, web_plates)
    print(f"  -> mirrored to {web_dir}")


def write_segments_index() -> None:
    """Write a small JSON listing every segment with a result.json — used by
    the webapp to populate its dropdown."""
    entries = []
    for d in sorted(WEB_ROOT.glob("*/")):
        r = d / "result.json"
        if r.exists():
            try:
                data = json.loads(r.read_text(encoding="utf-8"))
            except Exception:
                continue
            entries.append({
                "seg_id": data["seg_id"],
                "n_strips": data.get("n_strips", 0),
                "mock": data.get("mock", False),
            })
    WEB_ROOT.mkdir(parents=True, exist_ok=True)
    (WEB_ROOT / "index.json").write_text(
        json.dumps({"segments": entries}, indent=2), encoding="utf-8",
    )
    print(f"[index] wrote {WEB_ROOT / 'index.json'} with {len(entries)} segments")


async def main_async(args) -> None:
    segs = [args.only] if args.only else ALL_SEGMENTS
    for seg in segs:
        await run_segment(
            seg, mock=args.mock, n_strips=args.n_strips, api_key=args.api_key,
            skip_api=args.skip_api,
            model_slugs=args.models,
        )
    write_segments_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", help="Run on a single segment id")
    p.add_argument("--mock", action="store_true",
                   help="Skip API calls; generate placeholder readings")
    p.add_argument("--n-strips", type=int, default=6)
    p.add_argument("--api-key", default=None,
                   help="Override OPENROUTER_API_KEY env var")
    p.add_argument("--skip-api", action="store_true",
                   help="Skip OpenRouter and write local template-only fallback readings")
    p.add_argument("--models", nargs="+",
                   help="Optional OpenRouter model slugs to use instead of the full registry")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
