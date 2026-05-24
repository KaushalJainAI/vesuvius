"""Consensus voting across multiple model readings of the same strip.

Inputs : list of (model_slug, parsed_json) from per-model queries.
Outputs: a `ConsensusResult` with one entry per consensus position.

Alignment algorithm: greedy 1-D nearest-neighbour on `x_norm`. For each
character position from the first model, we look up the closest character
in every other model within a tolerance window (default 0.04 of strip
width), gather votes, then pick the majority.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


X_TOLERANCE = 0.04  # 4% of strip width


@dataclass
class CharVote:
    char: str
    confidence: float
    model: str
    x_norm: float


@dataclass
class ConsensusChar:
    char: str
    x_norm: float
    confidence: float
    tier: str                                 # "HIGH" | "MED" | "LOW"
    votes: List[CharVote] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)


@dataclass
class ConsensusResult:
    text: str
    characters: List[ConsensusChar]
    n_models_used: int
    notes: str = ""
    translation_en: str = ""
    probable_summary: str = ""
    translation_votes: List[Dict[str, str]] = field(default_factory=list)
    summary_votes: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "translation_en": self.translation_en,
            "probable_summary": self.probable_summary,
            "translation_votes": self.translation_votes,
            "summary_votes": self.summary_votes,
            "n_models_used": self.n_models_used,
            "notes": self.notes,
            "characters": [
                {
                    "char": c.char,
                    "x_norm": c.x_norm,
                    "confidence": round(c.confidence, 3),
                    "tier": c.tier,
                    "alternatives": c.alternatives,
                    "votes": [
                        {
                            "model": v.model,
                            "char": v.char,
                            "confidence": round(v.confidence, 3),
                            "x_norm": round(v.x_norm, 4),
                        }
                        for v in c.votes
                    ],
                }
                for c in self.characters
            ],
        }


def _extract_chars(parsed: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not parsed or not isinstance(parsed, dict):
        return []
    chars = parsed.get("characters") or []
    out: List[Dict[str, Any]] = []
    for c in chars:
        if not isinstance(c, dict):
            continue
        ch = c.get("char")
        x = c.get("x_norm")
        conf = c.get("confidence", 0.5)
        if not isinstance(ch, str) or not ch:
            continue
        try:
            x = float(x)
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        out.append({"char": ch, "x_norm": x, "confidence": conf})
    out.sort(key=lambda d: d["x_norm"])
    return out


def _tier_for(top_count: int, n_models: int) -> Tuple[str, float]:
    """Return (tier, confidence_multiplier)."""
    if n_models == 0:
        return "LOW", 0.0
    frac = top_count / n_models
    if frac >= 0.8:
        return "HIGH", 1.0
    if frac >= 0.5:
        return "MED", 0.75
    return "LOW", 0.4


def build_consensus(
    per_model: Dict[str, Optional[Dict[str, Any]]],
    *,
    x_tolerance: float = X_TOLERANCE,
) -> ConsensusResult:
    """Cluster characters across models by x position and vote.

    ``per_model`` maps model slug → parsed JSON (None if that model failed).
    """
    # Collect (x_norm, char, conf, model) tuples
    points: List[Tuple[float, str, float, str]] = []
    for model_slug, parsed in per_model.items():
        for c in _extract_chars(parsed):
            points.append((c["x_norm"], c["char"], c["confidence"], model_slug))

    n_models = sum(1 for v in per_model.values() if v is not None)

    if not points:
        return ConsensusResult(text="", characters=[], n_models_used=n_models,
                               notes="No legible characters from any model.")

    # Sort by x and cluster within tolerance
    points.sort(key=lambda t: t[0])

    clusters: List[List[Tuple[float, str, float, str]]] = []
    current: List[Tuple[float, str, float, str]] = [points[0]]
    for p in points[1:]:
        ref_x = sum(q[0] for q in current) / len(current)
        if abs(p[0] - ref_x) <= x_tolerance:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)

    out: List[ConsensusChar] = []
    for cluster in clusters:
        by_model: Dict[str, Tuple[float, str, float, str]] = {}
        for x, c, conf, m in cluster:
            prev = by_model.get(m)
            if prev is None or conf > prev[2]:
                by_model[m] = (x, c, conf, m)

        votes = [CharVote(char=c, confidence=conf, model=m, x_norm=x)
                 for x, c, conf, m in by_model.values()]
        counter = Counter(v.char for v in votes)
        top_char, top_count = counter.most_common(1)[0]
        tier, mult = _tier_for(top_count, n_models)

        # Confidence = mean of confidences for the winning character
        winning_confs = [v.confidence for v in votes if v.char == top_char]
        conf = (sum(winning_confs) / len(winning_confs)) * mult

        alts = [c for c, _ in counter.most_common() if c != top_char]
        x_centroid = sum(v.x_norm for v in votes if v.char == top_char) / len(winning_confs)

        out.append(ConsensusChar(
            char=top_char, x_norm=x_centroid, confidence=conf,
            tier=tier, votes=votes, alternatives=alts,
        ))

    text = "".join(c.char for c in out)

    # ---- translation consensus
    translation_votes: List[Dict[str, str]] = []
    summary_votes: List[Dict[str, str]] = []
    candidates: Counter = Counter()
    summary_candidates: Counter = Counter()
    for model_slug, parsed in per_model.items():
        if not parsed or not isinstance(parsed, dict):
            continue
        tr = parsed.get("translation_en")
        if isinstance(tr, str) and tr.strip():
            tr_clean = tr.strip()
            translation_votes.append({"model": model_slug, "translation": tr_clean})
            # treat case-insensitive identical strings as the same vote
            candidates[tr_clean.lower()] += 1
        sm = parsed.get("probable_summary")
        if isinstance(sm, str) and sm.strip():
            sm_clean = sm.strip()
            summary_votes.append({"model": model_slug, "summary": sm_clean})
            summary_candidates[sm_clean.lower()] += 1
    chosen_translation = ""
    if candidates:
        top_norm, _ = candidates.most_common(1)[0]
        # return the first verbatim form matching the normalised key
        for v in translation_votes:
            if v["translation"].lower() == top_norm:
                chosen_translation = v["translation"]
                break
    chosen_summary = ""
    if summary_candidates:
        top_norm, _ = summary_candidates.most_common(1)[0]
        for v in summary_votes:
            if v["summary"].lower() == top_norm:
                chosen_summary = v["summary"]
                break

    return ConsensusResult(
        text=text, characters=out, n_models_used=n_models,
        notes=f"{n_models} model(s); {len(out)} consensus positions",
        translation_en=chosen_translation,
        probable_summary=chosen_summary,
        translation_votes=translation_votes,
        summary_votes=summary_votes,
    )
