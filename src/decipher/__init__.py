"""Vesuvius Decipher — multi-LLM Greek-text deciphering pipeline.

Entry points:
    from decipher.pipeline import decipher_segment
    from decipher.strip_extractor import extract_strips
    from decipher.model_registry import OPEN_SOURCE_MODELS

The pipeline takes an enhanced ink-probability label, cuts horizontal
text-line strips, queries several vision LLMs in parallel via OpenRouter,
optionally merges manually-pasted readings (from Claude Code / GPT / Gemini),
and produces a per-segment JSON with per-model outputs plus a consensus
reading.
"""
from __future__ import annotations

__all__ = [
    "decipher_segment",
    "extract_strips",
    "OPEN_SOURCE_MODELS",
]


def __getattr__(name):
    # lazy imports so optional deps (httpx) don't load unless used
    if name == "decipher_segment":
        from .pipeline import decipher_segment
        return decipher_segment
    if name == "extract_strips":
        from .strip_extractor import extract_strips
        return extract_strips
    if name == "OPEN_SOURCE_MODELS":
        from .model_registry import OPEN_SOURCE_MODELS
        return OPEN_SOURCE_MODELS
    raise AttributeError(name)
