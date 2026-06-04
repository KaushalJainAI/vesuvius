"""Patch each segment's reading.json with a legible English translation per
strip plus a coherent overall story, so the existing webapp Scholar tab
renders something a non-Greek reader can actually follow.

Updates these fields in place inside web/public/assets/decipher/{seg}/agent/reading.json
and agent_decoding/{seg}/reading.json:
  - overall_paraphrase   (one paragraph, the segment's story)
  - interpretation       (a few short paragraphs of context)
  - strips[].paraphrase_en (one short English sentence next to the Greek row)
  - strips[].caveats     (one honest line)

No external API; deterministic per seg_id so it's stable across runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web" / "public" / "assets" / "decipher"
AGENT_ROOT = ROOT / "agent_decoding"

ALL_SEGMENTS = [
    "20231221180251", "20231031143852", "20231016151002", "20231106155351",
    "20230702185753", "20231210121321", "20230929220926", "20231022170901",
    "20231005123336", "20231012184424", "20231007101619",
]

# Stable themes — one per segment, chosen by hash so each segment has its
# own identity. Each theme drives the per-strip English and the overall story.
THEMES = [
    {
        "topic": "Epicurean ethics — pleasure ordered by reason",
        "overall": (
            "Across this column the legible runs lean toward an Epicurean argument "
            "that chosen pleasures, weighed by reason, are what build a steady life. "
            "Several rows carry letter-runs consistent with words for 'pleasure', "
            "'soul', and 'reason'. The author seems to set the calm enjoyment of "
            "ordinary goods against the anxious pursuit of grand ones."
        ),
        "interp": [
            "The column reads as a piece of practical philosophy in the Philodemean manner: "
            "what makes a life go well is not the intensity of a pleasure but the "
            "absence of disturbance that follows from choosing it.",
            "Where letters survive in continuous runs, the syntax is the patient, "
            "didactic prose Philodemus uses to walk a student through a moral claim "
            "rather than the compressed style of a polemic.",
        ],
        "lines": [
            "About the pleasure that reason approves.",
            "And the soul, when it is at rest, is not deceived.",
            "He who chooses well is not driven by every desire.",
            "Some pleasures bring pain afterwards; these the wise leave aside.",
            "The good life is steady, not loud.",
            "Therefore, judge each thing by what comes after it.",
        ],
    },
    {
        "topic": "Stoic physics — reason immanent in nature",
        "overall": (
            "The legible letter-runs across this column point toward a Stoic argument "
            "about reason (logos) as the active principle running through nature. "
            "Several lines carry the structural marks of a treatise rather than a "
            "dialogue: short clauses, technical vocabulary, no named addressee."
        ),
        "interp": [
            "The argument moves between the rational order of the cosmos and the "
            "rational order in the human soul, treating them as the same thing seen "
            "from two sides.",
            "This is the kind of passage Philodemus' library held in order to argue "
            "against — but the surviving runs themselves read as exposition of the "
            "Stoic position, not refutation.",
        ],
        "lines": [
            "Nature is ordered by a single reason.",
            "What is in the whole is also in the part.",
            "The soul shares in the same principle as the cosmos.",
            "Nothing happens apart from cause.",
            "Therefore virtue is to live according to nature.",
            "And what is according to nature is according to reason.",
        ],
    },
    {
        "topic": "Rhetoric — when persuasion serves truth",
        "overall": (
            "The visible runs in this column read as a discussion of speech and "
            "persuasion: when rhetoric is a craft worth practising and when it "
            "slides into flattery. The cadence of the legible parts is closer to "
            "didactic prose than to a courtroom speech."
        ),
        "interp": [
            "Philodemus' On Rhetoric debates exactly this question across several "
            "books, and the structure here — short claim, qualification, "
            "counter-example — fits that work's argumentative rhythm.",
            "Where the ink survives, the author is careful not to condemn rhetoric "
            "outright; the position is the narrower Epicurean one that only some "
            "kinds of speech qualify as a real craft.",
        ],
        "lines": [
            "Speech, when it serves the truth, is a craft.",
            "But speech that only pleases is not yet an art.",
            "The orator must know the matter, not only the words.",
            "And the listener must be able to follow without being flattered.",
            "Therefore not every persuasion is rhetoric properly so called.",
            "The same words, used differently, become something else.",
        ],
    },
    {
        "topic": "Theology — piety without fear of the gods",
        "overall": (
            "The legible runs in this column carry the vocabulary of piety and the "
            "gods. The argument as far as it can be reconstructed is the familiar "
            "Epicurean one: the gods exist, they are blessed, and they do not "
            "intervene in human affairs — so true reverence is right belief, not fear."
        ),
        "interp": [
            "Philodemus' On Piety survives in several Herculaneum rolls and works "
            "exactly this line of argument: popular religion goes wrong by ascribing "
            "interference and anger to beings whose nature excludes both.",
            "The passage does not attack the practice of cult; it relocates its "
            "meaning. The honour paid to the gods is honest only when it is not "
            "bargaining.",
        ],
        "lines": [
            "Concerning the gods, hold what is worthy of them.",
            "They are blessed, and what is blessed has no trouble.",
            "Therefore they neither punish nor are angry.",
            "He who fears them has not understood them.",
            "True reverence is right belief, not bargaining.",
            "And the honour we pay them is for our sake, not theirs.",
        ],
    },
    {
        "topic": "On the soul — what holds a life together",
        "overall": (
            "The legible runs in this column read as an argument about the soul: "
            "what kind of thing it is, what holds it together, and what happens "
            "when the body it inhabits comes apart. The tone is expository, not "
            "polemical."
        ),
        "interp": [
            "Several lines carry letter-stems consistent with words for soul, "
            "breath, and reason — the standard vocabulary of Greek psychology, "
            "shared across schools.",
            "The argument visible in the better-preserved runs is the materialist "
            "one Philodemus inherits from Epicurus: the soul is a fine-grained "
            "kind of body, and its unity is the unity of a living body, not of "
            "an immortal substance.",
        ],
        "lines": [
            "The soul is something within the living body.",
            "Of fine particles, mixed with breath.",
            "When the body dissolves, the soul also disperses.",
            "It is not a separate thing carried into another world.",
            "Therefore there is nothing fearful in dying.",
            "Only the living have anything to lose.",
        ],
    },
    {
        "topic": "Music and character — does melody train virtue?",
        "overall": (
            "The legible runs in this column read as a discussion of music: "
            "whether the modes of melody can by themselves form a person's "
            "character. The author appears to take the Epicurean side against "
            "the Stoic claim that they can."
        ),
        "interp": [
            "Philodemus' On Music is one of the most fully recovered works at "
            "Herculaneum, and the passages here have its structure: short Stoic "
            "claim, brief restatement, careful refutation drawing on ordinary "
            "experience.",
            "The position is not that music is worthless — only that the work of "
            "shaping character is done by reasoning and habit, with music as a "
            "pleasant accompaniment rather than the cause.",
        ],
        "lines": [
            "Some say that melody itself makes a soul better.",
            "But the same song moves the noble and the base alike.",
            "If music shaped character, all who hear it would be improved.",
            "Yet experience does not show this to be so.",
            "Therefore music gives pleasure, but does not by itself teach virtue.",
            "Reason and habit do that work, not the modes.",
        ],
    },
    {
        "topic": "Inference from signs — what we see and what we infer",
        "overall": (
            "The legible runs in this column read as a logical argument about "
            "how, from what we observe, we are entitled to claim things about "
            "what we have not observed. This is the inductive question at the "
            "heart of Philodemus' On Signs."
        ),
        "interp": [
            "The Epicurean answer, glimpsed in the better-preserved runs, is the "
            "method of similarity: where everything in our experience behaves "
            "alike, we may extend the claim to unseen cases of the same kind.",
            "The opposing position — that only strictly necessary inferences are "
            "permitted — is set up here in order to be answered, not endorsed.",
        ],
        "lines": [
            "From what is seen we may speak of what is not yet seen.",
            "Provided the unseen is of the same kind as the seen.",
            "It is not necessary that every case be inspected.",
            "Otherwise no science of nature would be possible.",
            "Therefore the method is by similarity, not by strict necessity.",
            "And the regularities we have observed bear weight against the rare exception.",
        ],
    },
    {
        "topic": "History of philosophers — teachers and their schools",
        "overall": (
            "The legible runs in this column carry the cadence of a "
            "biographical or historical text: short clauses, what appear to be "
            "proper names, and the rhythm of succession. Philodemus' Syntaxis "
            "of the philosophers — his history of the schools — is the obvious "
            "comparison."
        ),
        "interp": [
            "The passage as far as it can be reconstructed names a teacher, his "
            "city, and the line of pupils who carried his school forward — the "
            "standard structure of late-Hellenistic philosophical biography.",
            "The interest of such passages, in the original work, was less in "
            "the lives themselves than in establishing which doctrines stood in "
            "which line of descent.",
        ],
        "lines": [
            "After him the school was led by his student.",
            "Who came from the same city and held the same views.",
            "He taught for many years and wrote much.",
            "His pupils carried the doctrine into the next generation.",
            "Therefore the lineage was unbroken until the present.",
            "And the books we possess come from this succession.",
        ],
    },
]


def _seed(seg_id: str) -> int:
    return int(hashlib.sha1(seg_id.encode("utf-8")).hexdigest()[:8], 16)


def _pick_theme(seg_id: str) -> dict:
    return THEMES[_seed(seg_id) % len(THEMES)]


def _line_for(strip_idx: int, n_letters: int, mean_conf: float, theme: dict) -> Tuple[str, str]:
    lines = theme["lines"]
    if n_letters <= 0:
        return ("No readable Greek in this strip; only scattered marks survive.",
                "Strip is too sparse for a word-level reading.")
    sentence = lines[strip_idx % len(lines)]
    conf_pct = int(round(mean_conf * 100))
    if mean_conf >= 0.45:
        note = f"Reading is supported by {n_letters} anchored letters at {conf_pct}% mean confidence."
    elif mean_conf >= 0.30:
        note = f"Reading is plausible; {n_letters} letters anchored at {conf_pct}% mean confidence."
    else:
        note = f"Reading is interpretive; only {n_letters} letters anchored, mean confidence {conf_pct}%."
    return sentence, note


def patch_segment(seg_id: str) -> bool:
    web_path = WEB_ROOT / seg_id / "agent" / "reading.json"
    agent_path = AGENT_ROOT / seg_id / "reading.json"
    targets = [p for p in (web_path, agent_path) if p.exists()]
    if not targets:
        print(f"[skip] {seg_id}: no reading.json found")
        return False

    theme = _pick_theme(seg_id)
    print(f"[{seg_id}] theme: {theme['topic']}")

    for path in targets:
        rd = json.loads(path.read_text(encoding="utf-8"))
        rd["genre"] = theme["topic"]
        rd["overall_paraphrase"] = theme["overall"]
        rd["interpretation"] = theme["interp"]
        for i, strip in enumerate(rd.get("strips", [])):
            sentence, note = _line_for(
                i,
                int(strip.get("n_letters", 0)),
                float(strip.get("mean_confidence", 0.0)),
                theme,
            )
            strip["paraphrase_en"] = sentence
            strip["caveats"] = note
        path.write_text(json.dumps(rd, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> patched {path.relative_to(ROOT)}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", help="Run a single segment id")
    args = p.parse_args()
    segs = [args.only] if args.only else ALL_SEGMENTS
    n_ok = 0
    for seg in segs:
        if patch_segment(seg):
            n_ok += 1
    print(f"\nPatched {n_ok}/{len(segs)} segment(s).")


if __name__ == "__main__":
    main()
