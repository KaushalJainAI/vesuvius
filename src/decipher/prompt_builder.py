"""Prompt assembly for the deciphering pipeline.

The same prompt is sent to every model so their outputs are comparable.
Output is constrained to a strict JSON schema so the consensus stage can
parse all models uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass


SYSTEM_PROMPT = """\
You are an expert papyrologist specialising in carbonised Herculaneum scrolls
(1st century BC, Epicurean philosophical texts in ancient Greek uncial /
majuscule script).

You will be shown a multi-panel reading plate made from the same horizontal
strip cut from a CT-derived ink probability map. Panels include the raw ink
probability, denoised local contrast, a high-contrast ink mask, and a
letter-scale component view. BRIGHT pixels probably contain ink; DARK pixels
are background or noise. The image is NOT a photograph; it is the output of a
machine-learning ink detector, so faint or partial letters are expected.
Strokes are typically 10-15 px wide; letter height is around 50-80 px.

Your job is to read whatever Greek letters you can see and return them in a
structured JSON form. Be honest about uncertainty. If you cannot make out a
letter, say so. NEVER invent letters that are not visually present.

Conventions:
  - Use uppercase Greek (Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω).
  - Use the lunate sigma Ϲ where the form is clearly a C-shape.
  - Use '.' for a visible mark that you cannot identify as a specific letter.
  - Use '_' for a gap where a letter probably exists but is illegible.
"""


_JSON_INSTRUCTIONS = """\
The image may contain ONE OR SEVERAL lines of text. Read everything you can.
Use the RAW panel as ground truth. Use the denoised/mask/component panels only
to improve signal-to-noise and separate letters from speckle.

Return ONLY a JSON object with this exact schema:

{
  "line_text": "<full text you read, in reading order. Use \\n between lines. No spaces unless visible in the image. Use uppercase Greek throughout.>",
  "characters": [
    {
      "char": "<one Greek letter, or . or _>",
      "x_norm": <float 0..1, visual horizontal centre of this character in the image crop. 0.0 is the left edge, 1.0 is the right edge.>,
      "confidence": <float 0..1>,
      "alternatives": ["<other plausible letter>"]
    }
  ],
  "translation_en": "<English translation. If you cannot parse meaningful Greek, say '[uncertain]'. Short - a phrase, not speculation.>",
  "probable_summary": "<one sentence explaining the probable meaning/content if any Greek is readable; otherwise '[uncertain]'>",
  "notes": "<one short sentence about what you actually saw>",
  "overall_confidence": "<low | medium | high>"
}

Rules:
  - characters must be in reading order (top line first, left to right; then next line).
  - x_norm must be the character's visual x-position in the image, not merely
    its ordinal index in the word. This is used to draw boxes on the ink map.
  - Empty characters array is fine if nothing is legible.
  - Treat the template-hint dictionary below as A WEAK PRIOR. Only use a hinted
    letter if YOU can verify it in the image. Reading template suggestions
    blindly produces gibberish.
  - translation_en must reflect line_text; do not translate text you did not read.
  - probable_summary may be broader than translation_en, but it must still be
    grounded in visible/read text.
  - Do NOT include any prose outside the JSON.
"""


@dataclass
class StripPromptContext:
    """Optional extra context for a strip (passed to every model)."""

    seg_id: str
    strip_index: int
    y_center: int
    expected_line_height_px: int = 80
    hint_block: str = ""


def build_user_prompt(ctx: StripPromptContext) -> str:
    parts = [
        f"Segment: {ctx.seg_id}",
        f"Strip: {ctx.strip_index} (line ~{ctx.y_center}px from top of label)",
        f"Expected letter height: ~{ctx.expected_line_height_px} px",
        "",
    ]
    if ctx.hint_block:
        parts.append(ctx.hint_block)
        parts.append("")
    parts.append(_JSON_INSTRUCTIONS)
    return "\n".join(parts)
