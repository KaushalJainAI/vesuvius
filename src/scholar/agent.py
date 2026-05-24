"""Agent definitions for the Scholar stage.

Two agents:

- StripScholarAgent — vision-enabled. Takes one strip PNG plus the existing
  consensus + template hints + per-model votes; returns a structured
  StripScholar (letter table, word divisions, recognised words, paraphrase).

- SegmentScholarAgent — text-only. Aggregates per-strip readings into a
  segment-level scholar synthesis (genre, lexical signals, historical
  context, candidate authors).

Both run via the openai-agents SDK pointed at OpenRouter. The strip agent
defaults to a strong vision model (Claude Opus 4 via OpenRouter); the
segment agent defaults to GPT-4o. Override with env vars
SCHOLAR_STRIP_MODEL and SCHOLAR_SEGMENT_MODEL.
"""
from __future__ import annotations

import base64
import os
from typing import Any, List, Optional

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


def _load_env() -> None:
    if _DOTENV_AVAILABLE:
        root = Path(__file__).resolve().parents[2]
        env_file = root / ".env"
        if env_file.exists():
            load_dotenv(env_file)


def _resolve_openai_key() -> str:
    _load_env()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Export it or add to .env. "
            "Scholar now uses the OpenAI API directly (no OpenRouter)."
        )
    return key


# ---------------------------------------------------------------------------
# Pydantic output schemas
# ---------------------------------------------------------------------------

class LetterReading(BaseModel):
    """One letter position. Lenient on field naming — `char`/`letter`/`principal` all accepted."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    index: int = Field(default=-1)
    principal: str = Field(default="")
    alternates: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    is_word_boundary: bool = Field(False)
    note: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # Map common alias keys -> canonical `principal`.
        if "principal" not in data:
            for k in ("char", "letter", "reading", "glyph"):
                if k in data and isinstance(data[k], str):
                    data["principal"] = data[k]
                    break
        # Accept singular "alternate" or "alts" too.
        if "alternates" not in data:
            for k in ("alts", "alternate", "alternatives"):
                if k in data:
                    v = data[k]
                    data["alternates"] = v if isinstance(v, list) else [v]
                    break
        return data


class RecognisedWord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    greek: str = Field(default="")
    english: str = Field(default="")
    certainty: str = Field(default="low")
    # Position of the word within the strip image, 0..1 left-to-right. Used by
    # the webapp to render translation labels above the correct ink span.
    x_norm_start: float = Field(default=0.0, ge=0.0, le=1.0)
    x_norm_end: float = Field(default=0.0, ge=0.0, le=1.0)
    # Per-word confidence (0..1). Distinct from `certainty`, which is a
    # qualitative bucket the model chooses. Webapp uses this for the bar.
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "greek" not in data:
            for k in ("word", "greek_word", "uncial"):
                if k in data and isinstance(data[k], str):
                    data["greek"] = data[k]
                    break
        if "english" not in data:
            for k in ("gloss_en", "gloss", "english_gloss", "meaning", "translation"):
                if k in data and isinstance(data[k], str):
                    data["english"] = data[k]
                    break
        # Map qualitative certainty -> numeric confidence if model omitted it
        if "confidence" not in data:
            cert = str(data.get("certainty", "")).strip().lower()
            if cert in ("high", "supported"):
                data["confidence"] = 0.85
            elif cert in ("medium", "plausible"):
                data["confidence"] = 0.6
            elif cert in ("low", "speculative"):
                data["confidence"] = 0.35
        return data


class StripScholar(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    strip_id: int = Field(default=-1)
    letters: List[LetterReading] = Field(default_factory=list)
    word_divisions: List[str] = Field(
        default_factory=list,
        description="Interpretive word-split of the strip, e.g. ['ΤΗΣ','ΣΠΟΥΔΗΣ','ΠΑΡΑΛΟΓΟΝ']",
    )
    recognized_words: List[RecognisedWord] = Field(default_factory=list)
    paraphrase_en: str = Field(default="", description="1–3 sentence probable meaning")
    caveats: str = Field("", description="Honest uncertainty note")

    @field_validator("word_divisions", mode="before")
    @classmethod
    def _flatten_word_divisions(cls, v: Any) -> Any:
        """Accept either ['ΤΗΣ','ΣΠΟΥΔΗΣ'] or [{'text': 'ΤΗΣ'}, ...]."""
        if not isinstance(v, list):
            return v
        out: List[str] = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("word") or item.get("greek")
                if isinstance(t, str):
                    out.append(t)
        return out

    @model_validator(mode="after")
    def _autofill_indices(self) -> "StripScholar":
        for i, l in enumerate(self.letters):
            if l.index < 0:
                l.index = i
        return self


class SegmentScholar(BaseModel):
    probable_genre: str = Field(..., description="e.g. 'Athenian forensic speech', 'Epicurean philosophical prose'")
    lexical_signals: List[str] = Field(
        default_factory=list,
        description="Key Greek words that drove the genre inference",
    )
    historical_context: str = Field(..., description="~150-word paragraph of historical context")
    candidate_authors: List[str] = Field(default_factory=list)
    overall_paraphrase: str = Field(..., description="What the whole segment is plausibly about")
    confidence_band: str = Field(..., description="one of: speculative, plausible, supported")


# ---------------------------------------------------------------------------
# SDK setup
# ---------------------------------------------------------------------------

# Scholar runs through the OpenAI API directly (NOT OpenRouter — that key
# is exhausted). gpt-4o-mini is the cheap default (multimodal-capable, ~$0.15
# per million input tokens). Override with SCHOLAR_STRIP_MODEL=gpt-4o for a
# stronger vision read, or SCHOLAR_SEGMENT_MODEL=gpt-4o for richer synthesis.
DEFAULT_STRIP_MODEL = "gpt-4o-mini"
DEFAULT_SEGMENT_MODEL = "gpt-4o-mini"


def _strip_model_name() -> str:
    return os.environ.get("SCHOLAR_STRIP_MODEL", DEFAULT_STRIP_MODEL)


def _segment_model_name() -> str:
    return os.environ.get("SCHOLAR_SEGMENT_MODEL", DEFAULT_SEGMENT_MODEL)


def _make_openai_client():
    """AsyncOpenAI client pointed at the OpenAI API directly (no OpenRouter)."""
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=_resolve_openai_key())


def build_strip_agent():
    """Vision-enabled strip-reading agent."""
    from agents import Agent, ModelSettings
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

    client = _make_openai_client()
    return Agent(
        model_settings=ModelSettings(max_tokens=8000, temperature=0.2),
        name="StripScholar",
        instructions=(
            "OUTPUT: ONE raw JSON object. No prose. No markdown fences. First "
            "char `{`, last char `}`. Use EXACTLY these field names:\n"
            "{\n"
            '  "letters": [\n'
            '    {"principal": "Τ", "alternates": ["Π"], "confidence": 0.82, "is_word_boundary": false, "note": null},\n'
            '    {"principal": "Η", "alternates": [], "confidence": 0.91, "is_word_boundary": true, "note": null}\n'
            "  ],\n"
            '  "word_divisions": ["ΤΗΣ", "ΣΠΟΥΔΗΣ"],\n'
            '  "recognized_words": [\n'
            '    {"greek": "ΤΗΣ", "english": "of the (fem. gen.)", "certainty": "high",\n'
            '     "x_norm_start": 0.12, "x_norm_end": 0.18, "confidence": 0.84}\n'
            "  ],\n"
            '  "paraphrase_en": "Probable meaning of the strip.",\n'
            '  "caveats": "Honest one-line uncertainty note."\n'
            "}\n\n"
            "You are a classical-Greek paleographer specialising in Herculaneum "
            "papyri and uncial (majuscule) bookhand. You will be shown one "
            "horizontal strip extracted from a CT-derived ink-probability map of "
            "a carbonised scroll, together with: (a) the current ensemble "
            "consensus reading, (b) template-matched letter candidates with "
            "probabilities, and (c) per-model votes from up to five vision "
            "models.\n\n"
            "Produce a structured scholar reading:\n"
            "  - letters[]: for each visible letter position, give principal "
            "    reading plus up to 3 ranked alternates and a 0–1 confidence. "
            "    Flag positions where damage or shape ambiguity makes alternates "
            "    competitive (e.g. Ω/Η, Π/Γ, Σ/Ε). Mark is_word_boundary=true on "
            "    the final letter of each interpretive word.\n"
            "  - word_divisions[]: Greek uncials run continuously; propose the "
            "    most defensible word-split.\n"
            "  - recognized_words[]: any Greek words you can identify with "
            "    high/medium/low certainty, with English gloss. For each word, "
            "    also give x_norm_start / x_norm_end (the left and right edges "
            "    of the word's ink span within the strip, 0=left, 1=right) and "
            "    a 0..1 numeric confidence consistent with the certainty bucket.\n"
            "  - paraphrase_en: 1–3 sentence probable meaning of the strip. "
            "    Use square brackets for uncertainty: 'on behalf of [someone]'.\n"
            "  - caveats: honest one-line note about damage / how much of this "
            "    is interpretive.\n\n"
            "Prefer letters that the template hints support; deviate from the "
            "consensus reading only when the image clearly favours another. "
            "Use proper Greek uncial characters (ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ and "
            "lunate Ϲ where appropriate). Never invent text the image cannot "
            "support — empty letters[] is acceptable if the strip is unreadable."
        ),
        model=OpenAIChatCompletionsModel(
            model=_strip_model_name(),
            openai_client=client,
        ),
    )


def build_segment_agent():
    """Text-only segment-synthesis agent."""
    from agents import Agent, ModelSettings
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

    client = _make_openai_client()
    return Agent(
        model_settings=ModelSettings(max_tokens=6000, temperature=0.3),
        name="SegmentScholar",
        instructions=(
            "OUTPUT: ONE raw JSON object. No prose. No markdown fences. First "
            "char `{`, last char `}`. Use EXACTLY these field names:\n"
            "{\n"
            '  "probable_genre": "Athenian forensic speech",\n'
            '  "lexical_signals": ["ΣΥΝΑΛΛΑΓΩΝ", "ΑΤΤΙΚΗΝ", "ΑΓΟΡΑ"],\n'
            '  "historical_context": "~150-word paragraph...",\n'
            '  "candidate_authors": ["Demosthenes", "Hyperides"],\n'
            '  "overall_paraphrase": "1-2 sentence drift of the segment.",\n'
            '  "confidence_band": "plausible"\n'
            "}\n\n"
            "You are a senior classicist synthesising a multi-strip scholar "
            "reading of one Herculaneum scroll segment. You receive a JSON "
            "array of per-strip readings (letter tables, recognised words, "
            "per-strip paraphrases). Produce:\n"
            "  - probable_genre: short label, e.g. 'Athenian forensic speech', "
            "    'Epicurean ethical prose', 'Stoic physics fragment', "
            "    'historical narrative'.\n"
            "  - lexical_signals: 3–8 key Greek words across the strips that "
            "    drive the genre inference.\n"
            "  - historical_context: ~150 words explaining what these lexical "
            "    signals + script style imply about period, milieu, and likely "
            "    intellectual tradition (Epicurean Garden, Athenian law-court, "
            "    Stoic school, etc.).\n"
            "  - candidate_authors: up to 4 plausible authors / schools.\n"
            "  - overall_paraphrase: 1–2 sentence drift of the whole segment, "
            "    using square brackets for uncertain spans.\n"
            "  - confidence_band: 'speculative' if <30% of letters look "
            "    anchored; 'plausible' if 30–60%; 'supported' if >60% with "
            "    recognised words present.\n\n"
            "Herculaneum context to weight heavily: PHercParis4 is from the "
            "Villa of the Papyri library and the largest single corpus there "
            "is Philodemus of Gadara (Epicurean). So Epicurean ethics, theology, "
            "music, rhetoric, and history of philosophy are prior-likely. "
            "Do NOT default to Epicurean if the lexical signals point elsewhere."
        ),
        model=OpenAIChatCompletionsModel(
            model=_segment_model_name(),
            openai_client=client,
        ),
    )


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------

def build_strip_input(
    *,
    strip_id: int,
    image_bytes: bytes,
    consensus: dict,
    template_hints: list,
    per_model: dict,
) -> list:
    """Build the multimodal `input` list for Runner.run on a strip."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    consensus_lite = {
        "text": consensus.get("text", ""),
        "translation_en": consensus.get("translation_en", ""),
        "characters": [
            {
                "char": c.get("char"),
                "x_norm": c.get("x_norm"),
                "confidence": c.get("confidence"),
                "alternatives": c.get("alternatives", []),
            }
            for c in (consensus.get("characters") or [])
        ],
    }
    votes_lite = {
        slug: {
            "text": (entry.get("parsed") or {}).get("line_text") or "",
            "translation_en": (entry.get("parsed") or {}).get("translation_en") or "",
        }
        for slug, entry in (per_model or {}).items()
        if entry.get("parsed")
    }
    hints_lite = [
        {
            "x_norm": h.get("x_norm"),
            "candidates": [(c[0], round(float(c[1]), 3)) for c in (h.get("candidates") or [])[:3]],
        }
        for h in (template_hints or [])
    ]

    import json
    payload = (
        f"Strip {strip_id} of a Herculaneum scroll segment.\n\n"
        f"## Ensemble consensus\n```json\n{json.dumps(consensus_lite, ensure_ascii=False, indent=2)}\n```\n\n"
        f"## Template-match hints (top candidates per detected ink blob)\n"
        f"```json\n{json.dumps(hints_lite, ensure_ascii=False, indent=2)}\n```\n\n"
        f"## Per-model line readings (for cross-check)\n"
        f"```json\n{json.dumps(votes_lite, ensure_ascii=False, indent=2)}\n```\n\n"
        f"Now look at the image carefully and produce the scholar reading."
    )

    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": payload},
                {"type": "input_image", "image_url": data_url, "detail": "auto"},
            ],
        }
    ]


def build_segment_input(strip_scholars: List[StripScholar]) -> str:
    import json
    arr = [s.model_dump() for s in strip_scholars]
    return (
        "Per-strip scholar readings for this segment (in order):\n\n"
        f"```json\n{json.dumps(arr, ensure_ascii=False, indent=2)}\n```\n\n"
        "Synthesise the segment-level scholar reading."
    )
