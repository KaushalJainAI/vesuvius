"""Scholar stage — OpenAI Agents SDK over OpenRouter.

Reads an existing decipher result.json and produces a richer paleographer-
style reading (letter alternates, recognised words, paraphrase, historical
context) attached as `result["scholar"]`.
"""
from .runner import run_scholar_for_segment

__all__ = ["run_scholar_for_segment"]
