"""Template-matching hints for the LLM prompt.

Runs the existing Greek-uncial template matcher (src/letter_candidates.py)
on each strip's float32 probability array and produces a compact textual
"dictionary" the LLM can consult while reading the strip image.

Two pieces of context per strip:

  1. PROBABLE-LETTER DICTIONARY — for each connected component in the strip
     (above a confidence floor), the top-3 most-likely Greek letters from
     template matching, with calibrated softmax probabilities and the
     position normalised to strip width.

  2. HISTORICAL / LEXICAL CONTEXT — the Herculaneum corpus is dominated by
     Epicurean philosophical Greek (Philodemus). We include a small
     vocabulary of frequent words/morphology so the LLM can bias its
     reading toward plausible Greek over random letter combinations.

The output is plain text appended to the LLM user prompt. The strip image
is still the primary signal — these hints are bias, not ground truth.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

# Allow letter_candidates to be imported when this module is loaded via the
# `decipher` package without further sys.path manipulation by callers.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from letter_candidates import (  # noqa: E402
    Candidate,
    GreekTemplateMatcher,
    assign_confidence_tier,
    extract_candidates,
    softmax_scores,
)


# ---------------------------------------------------------------------------
# Historical / lexical context (Herculaneum / Epicurean Greek)
# ---------------------------------------------------------------------------

HISTORICAL_CONTEXT = """\
The text comes from a carbonised papyrus scroll recovered at Herculaneum \
(buried 79 CE). The vast majority of these scrolls preserve Greek prose by \
Epicurean philosophers, especially Philodemus of Gadara (c. 110-30 BCE). \
The script is majuscule (uncial), spaceless, written in continuous lines. \
Sigma typically appears as a lunate Ϲ.

High-frequency vocabulary in this corpus includes:
  - philosophical: φιλοσοφια, ηδονη, αρετη, λογος, αληθεια, φυσις, ψυχη, \
αγαθον, κακον, δικαιον, σοφια
  - philodeman keywords: επικουρος, μετροδωρος, ερμαρχος, πολυαινος, διογενης
  - common function words: και, η, ος, μεν, δε, γαρ, τε, ει, ως
  - common morphemes: -ος / -ων / -ης / -οις / -ται / -μενος / -μενοι

Prefer readings that yield real Greek words or morpheme sequences over \
random letter combinations. NEVER invent letters that are not visually \
supported by the strip image.
"""


# ---------------------------------------------------------------------------
# Image → float prob array
# ---------------------------------------------------------------------------

def _png_bytes_to_prob(png_bytes: bytes) -> np.ndarray:
    """Decode a strip's PNG back to a float32 [0,1] probability map."""
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


# ---------------------------------------------------------------------------
# Per-strip hint builder
# ---------------------------------------------------------------------------

_MATCHER: Optional[GreekTemplateMatcher] = None


def _matcher() -> GreekTemplateMatcher:
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = GreekTemplateMatcher()
    return _MATCHER


def build_dictionary_for_strip(
    prob_strip: np.ndarray,
    *,
    threshold: float = 0.55,
    min_area: int = 1_200,
    max_area: int = 120_000,
    top_k: int = 3,
    skip_low_confidence: bool = True,
    max_records: int = 15,
) -> List[dict]:
    """Run template matching on one strip → list of probable-letter records.

    Each record:
        {
          "x_norm": <float 0..1, centroid x relative to strip width>,
          "tier":   "HIGH" | "MEDIUM" | "LOW",
          "mean_prob": <float>,
          "area_px": <int>,
          "candidates": [(char, prob_softmax), ...]
        }
    """
    H, W = prob_strip.shape
    cands: List[Candidate] = extract_candidates(
        prob_strip, threshold=threshold,
        min_area=min_area, max_area=max_area,
    )
    matcher = _matcher()
    records: List[dict] = []
    # Sort left → right by centroid x
    cands.sort(key=lambda c: c.centroid[1])
    for cand in cands:
        tier = assign_confidence_tier(cand)
        if skip_low_confidence and tier == "LOW":
            continue
        matches = matcher.match(cand.patch, top_k=top_k)
        probs = softmax_scores(matches[:top_k])
        x_norm = cand.centroid[1] / max(W, 1)
        records.append({
            "x_norm": round(float(x_norm), 4),
            "tier": tier,
            "mean_prob": round(float(cand.mean_prob), 3),
            "area_px": int(cand.area),
            "candidates": [(ch, round(p, 3)) for ch, p in probs],
        })
    # Cap to keep the prompt block short — prefer HIGH over MEDIUM,
    # then keep evenly-spaced subset across x_norm for the rest.
    if len(records) > max_records:
        high = [r for r in records if r["tier"] == "HIGH"]
        med  = [r for r in records if r["tier"] == "MEDIUM"]
        if len(high) >= max_records:
            records = high[:max_records]
        else:
            slots_left = max_records - len(high)
            step = max(1, len(med) // slots_left) if slots_left > 0 else 1
            records = sorted(high + med[::step][:slots_left], key=lambda r: r["x_norm"])
    return records


def format_dictionary_text(records: List[dict]) -> str:
    """Render the probable-letter records as a compact prompt block."""
    if not records:
        return "  (no candidates above confidence floor)"
    lines: List[str] = []
    for r in records:
        cands = " | ".join(f"{ch} {p:.2f}" for ch, p in r["candidates"])
        lines.append(
            f"  x={r['x_norm']:.3f}  tier={r['tier']:6s}  "
            f"prob={r['mean_prob']:.2f}  cands: {cands}"
        )
    return "\n".join(lines)


def build_prompt_hints(png_bytes: bytes) -> tuple[List[dict], str]:
    """Top-level helper: PNG bytes → (records, formatted block).

    The pipeline calls this once per strip and concatenates the formatted
    block into the LLM user prompt alongside the JSON schema instructions.
    """
    prob = _png_bytes_to_prob(png_bytes)
    records = build_dictionary_for_strip(prob)
    text = (
        "PROBABLE-LETTER DICTIONARY for this strip "
        "(from convolution-based template matching against a 24-letter "
        "Greek uncial bank, 3 styles × 3 sizes):\n"
        + format_dictionary_text(records)
        + "\n\nHISTORICAL CONTEXT\n"
        + HISTORICAL_CONTEXT
    )
    return records, text
