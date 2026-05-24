"""Model registry for the deciphering pipeline.

Two groups:
  - OPEN_SOURCE_MODELS   : queried automatically via OpenRouter
  - MANUAL_MODELS        : filled by the user via Claude Code (no API call)

Add / remove / swap models here. The pipeline reads this file at runtime.

Notes on selection (Jan 2026 OpenRouter catalog):
  - Open-source vision models with strong performance are Qwen and Gemma
    families. We pick three Tier-1 (latest, largest) and two Tier-2
    (older / MoE-variant) for ensemble diversity.
  - Greek-specialised LLMs (Meltemi-7B, Llama-Krikri-8B) are TEXT-ONLY —
    they cannot read strip images. They appear as an optional polish step,
    not as voting models. See `greek_polish.py` (not yet implemented).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ModelSpec:
    slug: str                    # OpenRouter model id, e.g. "qwen/qwen3.6-35b-a3b"
    display_name: str            # short label for UI
    tier: int                    # 1 = stronger / latest, 2 = diversity
    provider: str                # "qwen", "google", ...
    notes: str = ""              # one-line rationale

    @property
    def safe_id(self) -> str:
        """Filename-safe id (no slashes)."""
        return self.slug.replace("/", "__")


# -----------------------------------------------------------------------------
# Open-source / open-weights — auto-queried via OpenRouter
# -----------------------------------------------------------------------------
OPEN_SOURCE_MODELS: List[ModelSpec] = [
    ModelSpec(
        slug="qwen/qwen3.6-35b-a3b",
        display_name="Qwen 3.6 35B (MoE)",
        tier=1,
        provider="qwen",
        notes="Largest open-weights MoE vision model; non-Western training mix.",
    ),
    ModelSpec(
        slug="qwen/qwen3.6-27b",
        display_name="Qwen 3.6 27B (dense)",
        tier=1,
        provider="qwen",
        notes="Dense complement to the MoE; different inductive bias.",
    ),
    ModelSpec(
        slug="google/gemma-4-31b-it",
        display_name="Gemma 4 31B",
        tier=1,
        provider="google",
        notes="Strongest open-weights vision model from Google.",
    ),
    ModelSpec(
        slug="google/gemma-4-26b-a4b-it",
        display_name="Gemma 4 26B (MoE)",
        tier=2,
        provider="google",
        notes="Gemma MoE variant — fourth independent reading at near-zero cost.",
    ),
    ModelSpec(
        slug="qwen/qwen3.5-27b",
        display_name="Qwen 3.5 27B",
        tier=2,
        provider="qwen",
        notes="Previous-gen Qwen — disagreement tells us about generation gap.",
    ),
    ModelSpec(
        slug="moonshotai/kimi-k2.6",
        display_name="Kimi K2.6",
        tier=1,
        provider="moonshotai",
        notes="Moonshot AI vision-capable model; 262k context, distinct Chinese-lab training mix.",
    ),
]


# -----------------------------------------------------------------------------
# Text-only reviewer models (post-consensus polish, no image input)
# DeepSeek does NOT have a vision model on OpenRouter (Jan 2026). We add it
# here so the consensus text can be reviewed by a strong reasoner for Greek
# linguistic plausibility — wired but disabled by default in pipeline.py.
# -----------------------------------------------------------------------------
TEXT_REVIEWER_MODELS: List[ModelSpec] = [
    ModelSpec(
        slug="deepseek/deepseek-v4-pro",
        display_name="DeepSeek V4 Pro (text-only reviewer)",
        tier=3,
        provider="deepseek",
        notes="Text-only — reviews consensus reading for Greek plausibility.",
    ),
]


# -----------------------------------------------------------------------------
# Manual slots — filled by user via Claude Code subscription, no API call
# -----------------------------------------------------------------------------
MANUAL_MODELS: List[ModelSpec] = [
    ModelSpec(
        slug="manual/claude-opus-4.7",
        display_name="Claude Opus 4.7",
        tier=0,  # frontier — distinct from tier 1/2 OS
        provider="anthropic",
        notes="Pasted from Claude Code session.",
    ),
    ModelSpec(
        slug="manual/gpt-5.5-pro",
        display_name="GPT-5.5 Pro",
        tier=0,
        provider="openai",
        notes="Optional manual slot.",
    ),
    ModelSpec(
        slug="manual/gemini-pro-latest",
        display_name="Gemini Pro",
        tier=0,
        provider="google",
        notes="Optional manual slot.",
    ),
]


def all_models() -> List[ModelSpec]:
    return [*OPEN_SOURCE_MODELS, *MANUAL_MODELS]


def by_slug(slug: str) -> ModelSpec:
    for m in all_models():
        if m.slug == slug:
            return m
    raise KeyError(slug)
