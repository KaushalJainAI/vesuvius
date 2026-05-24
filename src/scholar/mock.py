"""Realistic-feeling mock scholar output for offline / no-credits demos.

Builds a `scholar` block in the same shape the live agents produce, using:
  - the per-strip consensus text already in result.json
  - hand-curated Greek vocab and Epicurean / classical historical context
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

# Recurring Greek words present in PHercParis4 / Philodemus corpus.
WORD_BANK = [
    ("ΗΔΟΝΗ", "pleasure"),
    ("ΑΡΕΤΗ", "virtue"),
    ("ΦΥΣΙΣ", "nature"),
    ("ΛΟΓΟΣ", "reason / discourse"),
    ("ΨΥΧΗ", "soul"),
    ("ΘΕΟΣ", "god"),
    ("ΦΙΛΟΣΟΦΙΑ", "philosophy"),
    ("ΣΟΦΙΑ", "wisdom"),
    ("ΑΛΗΘΕΙΑ", "truth"),
    ("ΑΓΑΘΟΝ", "the good"),
    ("ΕΠΙΚΟΥΡΟΣ", "Epicurus"),
    ("ΑΠΟΛΛΩΝ", "Apollo"),
    ("ΚΟΣΜΟΣ", "world / order"),
]

ALT_FOR = {
    "Α": ["Λ", "Δ"], "Β": ["Ρ"], "Γ": ["Π", "Τ"], "Δ": ["Α", "Λ"], "Ε": ["Σ"],
    "Ζ": ["Ξ"], "Η": ["Ω", "Ν"], "Θ": ["Ο"], "Ι": ["Τ"], "Κ": ["Χ"],
    "Λ": ["Α", "Δ"], "Μ": ["Ν"], "Ν": ["Η", "Μ"], "Ξ": ["Ζ"], "Ο": ["Θ", "Ε"],
    "Π": ["Γ", "Τ"], "Ρ": ["Β"], "Σ": ["Ε"], "Τ": ["Π", "Γ"], "Υ": ["Ψ"],
    "Φ": ["Ψ"], "Χ": ["Κ"], "Ψ": ["Υ"], "Ω": ["Η"], "Ϲ": ["Σ"],
}


def _letters_from_consensus(consensus: dict, rng: random.Random) -> List[dict]:
    chars = consensus.get("characters") or []
    out: List[dict] = []
    for i, c in enumerate(chars):
        ch = str(c.get("char") or "")
        if not ch:
            continue
        conf = float(c.get("confidence") or 0.5)
        # Use existing alternatives if present, else pick visually-similar set.
        alts = list(c.get("alternatives") or [])
        if not alts:
            alts = ALT_FOR.get(ch.upper(), [])[:1]
        out.append({
            "index": i,
            "principal": ch,
            "alternates": alts,
            "confidence": round(min(0.95, max(0.2, conf + rng.uniform(-0.05, 0.10))), 3),
            "is_word_boundary": (i > 0 and rng.random() < 0.18) or i == len(chars) - 1,
            "note": "shape ambiguous; alternates plausible" if conf < 0.5 else None,
        })
    return out


def _word_divisions(letters: List[dict]) -> List[str]:
    words: List[str] = []
    buf = ""
    for l in letters:
        buf += l["principal"]
        if l["is_word_boundary"]:
            if buf:
                words.append(buf)
            buf = ""
    if buf:
        words.append(buf)
    return words


def _recognised_words(words: List[str], rng: random.Random) -> List[dict]:
    out: List[dict] = []
    used = set()
    for w in words:
        # If a real word from the bank substring-matches, claim it.
        for greek, eng in WORD_BANK:
            if greek in w and greek not in used:
                out.append({"greek": greek, "english": eng, "certainty": rng.choice(["medium", "low"])})
                used.add(greek)
                break
    # Always salt with at least one bank word for demo richness if list empty.
    if not out and words:
        greek, eng = rng.choice(WORD_BANK)
        out.append({"greek": greek, "english": eng, "certainty": "low"})
    return out


def _strip_paraphrase(words: List[str], rng: random.Random) -> str:
    if not words:
        return "[Strip too damaged for a continuous reading; only scattered ink traces survive.]"
    snippets = [
        "Probable reference to [a moral subject] developing the contrast between {x} and {y}.",
        "A clause appearing to describe {x}; the surrounding context likely treats {y}.",
        "Fragmentary clause; the visible Greek suggests {x}, possibly within a discussion of {y}.",
        "The letters cluster around vocabulary of {x}, with traces of {y} in the right half.",
    ]
    x = rng.choice(WORD_BANK)[1]
    y = rng.choice(WORD_BANK)[1]
    return rng.choice(snippets).format(x=x, y=y)


def _strip_caveats(letters: List[dict], rng: random.Random) -> str:
    n_low = sum(1 for l in letters if l["confidence"] < 0.45)
    if n_low > len(letters) * 0.4:
        return "Carbonisation damage is severe; the majority of letters here are interpretive."
    return "Moderate damage; major letter forms are anchored but word boundaries are interpretive."


def build_mock_scholar(result: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a `scholar` block from an existing decipher result.json."""
    seg_id = result.get("seg_id", "?")
    rng = random.Random(hash(seg_id) & 0xFFFFFFFF)

    strip_scholars: List[Dict[str, Any]] = []
    for strip in result.get("strips") or []:
        sid = int(strip.get("strip_id", 0))
        letters = _letters_from_consensus(strip.get("consensus") or {}, rng)
        words = _word_divisions(letters)
        strip_scholars.append({
            "strip_id": sid,
            "letters": letters,
            "word_divisions": words,
            "recognized_words": _recognised_words(words, rng),
            "paraphrase_en": _strip_paraphrase(words, rng),
            "caveats": _strip_caveats(letters, rng),
        })

    # Segment-level synthesis. Pull lexical signals from recognised words.
    lex_signals: List[str] = []
    for s in strip_scholars:
        for w in s["recognized_words"]:
            if w["greek"] not in lex_signals:
                lex_signals.append(w["greek"])
    lex_signals = lex_signals[:8]

    # Confidence band based on average letter confidence.
    all_letters = [l for s in strip_scholars for l in s["letters"]]
    if all_letters:
        avg = sum(l["confidence"] for l in all_letters) / len(all_letters)
    else:
        avg = 0.0
    band = "supported" if avg >= 0.65 else "plausible" if avg >= 0.45 else "speculative"

    segment = {
        "probable_genre": "Epicurean ethical / philosophical prose (Philodemus tradition)",
        "lexical_signals": lex_signals,
        "historical_context": (
            "PHercParis4 belongs to the carbonised library recovered from the Villa "
            "of the Papyri in Herculaneum, the bulk of which transmits writings of "
            "Philodemus of Gadara — an Epicurean philosopher resident in Italy in the "
            "first century BCE who studied under Zeno of Sidon at the Athenian Garden. "
            "The recurrence of vocabulary such as ἡδονή (pleasure), ἀρετή (virtue), "
            "ψυχή (soul), φύσις (nature) is consistent with treatises on ethics, "
            "rhetoric, music, or the history of philosophy. Uncial bookhand, "
            "scriptio continua, and the absence of consistent word-division all "
            "match a luxury philosophical roll of the late Republic / early Empire."
        ),
        "candidate_authors": ["Philodemus of Gadara", "Epicurus (in citation)", "Zeno of Sidon"],
        "overall_paraphrase": (
            "The visible passages form part of an ethical or natural-philosophical "
            "argument concerning [pleasure / virtue / the soul]; surviving lexical "
            "anchors suggest a continuous Epicurean prose discussion rather than a "
            "documentary or poetic text."
        ),
        "confidence_band": band,
    }

    return {
        "strip_model": "mock/offline",
        "segment_model": "mock/offline",
        "mock": True,
        "segment": segment,
        "strips": strip_scholars,
    }
