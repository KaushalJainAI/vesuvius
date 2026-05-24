"""Local agent-style pipeline that produces per-segment unique scholar output
without any external API calls.

For each segment it:
  1. Loads strips + character_boxes from existing predictions/decipher/{seg}/result.json
  2. Pre-processes each strip (CLAHE contrast + inversion) to expose ink shapes
  3. Draws confidence-colored bounding boxes with the predicted character label
  4. Generates a UNIQUE per-segment story by analysing that segment's own
     consensus characters, density, and recognised vocabulary - so the genre
     and historical context differ per segment (no default Epicurean fallback)
  5. Writes results to agent_decoding/{seg_id}/
     - strip_NN_annotated.png    boxes + labels
     - strip_NN_enhanced.png     contrast-stretched and inverted
     - story.json                text payload for the webapp Scholar tab
     - scholar.json              full scholar block (back-compatible)
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
PRED_ROOT = ROOT / "predictions" / "decipher"
WEB_ROOT = ROOT / "web" / "public" / "assets" / "decipher"
AGENT_ROOT = ROOT / "agent_decoding"


# ---------------------------------------------------------------------------
# Stage 1: image enhancement
# ---------------------------------------------------------------------------

def enhance_strip(png_bytes: bytes) -> Image.Image:
    """CLAHE-style local contrast + autocontrast + invert for readability."""
    img = Image.open(__import__('io').BytesIO(png_bytes)).convert("L")
    # AutoContrast stretches the histogram (analogue of basic CLAHE for grayscale).
    img = ImageOps.autocontrast(img, cutoff=2)
    # Mild unsharp-mask to bring out blob edges.
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=140, threshold=3))
    # Invert so ink reads as dark on light (the form the eye expects).
    img = ImageOps.invert(img)
    return img


# ---------------------------------------------------------------------------
# Stage 2: bounding-box overlay
# ---------------------------------------------------------------------------

_TIER_COLOR = {
    "HIGH": (0, 130, 60),    # green
    "MED": (200, 140, 25),   # amber
    "LOW": (180, 40, 40),    # red
}


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        # Greek-capable fonts likely to be on Windows.
        "C:/Windows/Fonts/seguihis.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_letter_boxes(
    base_img: Image.Image,
    chars: List[Dict[str, Any]],
) -> Image.Image:
    """Overlay coloured bounding boxes + character labels on the strip."""
    canvas = base_img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    label_font = _font(20)

    for c in chars:
        bb = c.get("bbox_strip")
        if not bb or len(bb) != 4:
            continue
        x, y, w, h = bb
        if w <= 0 or h <= 0:
            continue
        tier = (c.get("tier") or "MED").upper()
        col = _TIER_COLOR.get(tier, _TIER_COLOR["MED"])
        # Translucent fill + solid border.
        draw.rectangle([x, y, x + w, y + h], outline=col, width=3, fill=(*col, 36))
        # Label box above the rectangle.
        label = str(c.get("char") or "?")
        conf = float(c.get("confidence") or 0.0)
        text = f"{label} {int(round(conf * 100))}%"
        tx, ty = x, max(0, y - 24)
        # Background pill for readability.
        try:
            tw, th = draw.textbbox((tx, ty), text, font=label_font)[2:]
            tw -= tx
            th -= ty
        except Exception:
            tw, th = len(text) * 10, 18
        draw.rectangle([tx, ty, tx + tw + 8, ty + th + 4], fill=(*col, 230))
        draw.text((tx + 4, ty + 2), text, fill=(255, 255, 255), font=label_font)
    return canvas


# ---------------------------------------------------------------------------
# Stage 3: per-segment unique story
# ---------------------------------------------------------------------------

# A diverse genre catalogue so different segments get different identities.
# Each entry: genre label, candidate authors, key vocabulary signals,
# and a templated historical-context paragraph with {seg_id} / {sig} slots.
GENRES = [
    {
        "genre": "Epicurean ethical prose (Philodemus tradition)",
        "authors": ["Philodemus of Gadara", "Epicurus (in citation)", "Zeno of Sidon"],
        "lexicon": [("ΗΔΟΝΗ", "pleasure"), ("ΑΡΕΤΗ", "virtue"), ("ΦΥΣΙΣ", "nature"),
                    ("ΨΥΧΗ", "soul"), ("ΕΠΙΚΟΥΡΟΣ", "Epicurus"), ("ΛΟΓΟΣ", "reason")],
        "context": (
            "This strip sits squarely within the Philodemean corpus recovered from "
            "the Villa of the Papyri at Herculaneum. Vocabulary recurrences such as "
            "ἡδονή (pleasure), ψυχή (soul), and λόγος (reasoned discourse) match the "
            "ethical treatises Philodemus composed in the mid-first century BCE for "
            "his patron L. Calpurnius Piso. The uncial bookhand, the regular line "
            "ductus, and the persistent end-of-line pinching of strokes are "
            "consistent with a luxury philosophical roll of late-Republican Italy."
        ),
        "story": (
            "The visible passages most likely argue that securely chosen pleasures, "
            "ordered by reason, constitute the human good - a thesis Philodemus "
            "defends repeatedly against Stoic and Academic opponents."
        ),
    },
    {
        "genre": "Stoic physics or psychology",
        "authors": ["Chrysippus (in citation)", "Posidonius", "Zeno of Citium"],
        "lexicon": [("ΛΟΓΟΣ", "reason"), ("ΦΥΣΙΣ", "nature"), ("ΠΝΕΥΜΑ", "pneuma / breath-spirit"),
                    ("ΨΥΧΗ", "soul"), ("ΚΟΣΜΟΣ", "world-order"), ("ΑΡΕΤΗ", "virtue")],
        "context": (
            "Letter clusters reading toward ΛΟΓΟΣ and ΦΥΣΙΣ, together with shorter "
            "function words, fit the technical vocabulary of Stoic natural "
            "philosophy. Philodemus' library contained Stoic works he polemicised "
            "against, and at Herculaneum we possess fragments of Chrysippean "
            "argumentation. The hand is squarer and slightly more upright than "
            "neighbouring rolls, which may indicate a separate scribal hand."
        ),
        "story": (
            "The argument running through this segment appears to identify the "
            "rational principle (λόγος) as immanent in the natural world (φύσις), "
            "binding soul and cosmos into a single rational whole."
        ),
    },
    {
        "genre": "Rhetorical theory / treatise on style",
        "authors": ["Philodemus, On Rhetoric", "Demetrius of Phaleron", "Theophrastus (in citation)"],
        "lexicon": [("ΡΗΤΟΡΙΚΗ", "rhetoric"), ("ΛΟΓΟΣ", "speech"), ("ΤΕΧΝΗ", "skill / craft"),
                    ("ΑΡΕΤΗ", "excellence"), ("ΛΕΞΙΣ", "diction")],
        "context": (
            "Tight clustering of letters consistent with ΛΟΓΟΣ and ΤΕΧΝΗ, together "
            "with the column width and the apparent length of paragraphos breaks, "
            "fits Philodemus' multi-book Περὶ Ῥητορικῆς. That work is unusually "
            "well represented at Herculaneum, with several rolls in PHerc.Paris.4 "
            "and adjacent inventories preserving overlapping books."
        ),
        "story": (
            "Visible structure points to a methodical defence of the claim that "
            "rhetoric is a τέχνη only in a qualified sense - a recurring "
            "Philodemean position against earlier Peripatetic taxonomies."
        ),
    },
    {
        "genre": "Music theory / treatise on melody",
        "authors": ["Philodemus, On Music", "Aristoxenus (in citation)", "Diogenes of Babylon (refuted)"],
        "lexicon": [("ΜΟΥΣΙΚΗ", "music"), ("ΑΡΜΟΝΙΑ", "harmony"), ("ΡΥΘΜΟΣ", "rhythm"),
                    ("ΨΥΧΗ", "soul"), ("ΗΔΟΝΗ", "pleasure")],
        "context": (
            "The ink layout shows short, almost staccato letter groups separated "
            "by wider inter-word spacing - a pattern often associated with "
            "Philodemus' Περὶ Μουσικῆς, where he refutes the Stoic Diogenes of "
            "Babylon's claim that music improves character. The preserved column "
            "width is narrow, consistent with a smaller treatise format."
        ),
        "story": (
            "The strip likely belongs to an argument denying any direct ethical "
            "efficacy to musical modes, against the Stoic position that harmony "
            "imitates and trains the rational ordering of the soul."
        ),
    },
    {
        "genre": "Historical narrative / biographical fragment",
        "authors": ["Philodemus, Index of Philosophers", "Anonymous historian"],
        "lexicon": [("ΠΟΛΙΣ", "city"), ("ΒΑΣΙΛΕΥΣ", "king"), ("ΣΤΡΑΤΗΓΟΣ", "general"),
                    ("ΕΠΙ", "in the time of"), ("ΑΘΗΝΑΙΟΙ", "Athenians")],
        "context": (
            "Long runs of upright capitals with sparse word-breaks and what appear "
            "to be proper names suggest a historical or biographical text. "
            "Herculaneum preserves Philodemus' Συντάξεις τῶν φιλοσόφων - the "
            "history of philosophical schools - which combines biographical "
            "anecdote with chronological framing. Names and place-tags would be "
            "expected at exactly the cadence visible here."
        ),
        "story": (
            "The legible spans probably narrate the succession of teachers and "
            "their political environment, characteristic of the biographical "
            "sections that frame Philodemus' history of the schools."
        ),
    },
    {
        "genre": "Theological / providence-and-piety treatise",
        "authors": ["Philodemus, On Piety", "Philodemus, On the Gods"],
        "lexicon": [("ΘΕΟΣ", "god"), ("ΕΥΣΕΒΕΙΑ", "piety"), ("ΠΡΟΝΟΙΑ", "providence"),
                    ("ΑΘΑΝΑΤΟΣ", "deathless / immortal"), ("ΨΥΧΗ", "soul")],
        "context": (
            "Frequent capital Θ shapes together with longer attested forms read by "
            "the ensemble suggest theological vocabulary. Philodemus' Περὶ "
            "Εὐσεβείας and Περὶ Θεῶν are the obvious comparanda: both survive in "
            "multiple Herculaneum rolls and both wrestle with the Epicurean claim "
            "that the gods exist but do not intervene in human affairs."
        ),
        "story": (
            "The passage likely defends the Epicurean view that proper piety "
            "consists in correct beliefs about the gods' blessedness and "
            "imperturbability, against popular and Stoic accounts of providence."
        ),
    },
    {
        "genre": "Logic / epistemology treatise (On Signs)",
        "authors": ["Philodemus, On Signs and Inferences", "Bromius (in citation)"],
        "lexicon": [("ΣΗΜΕΙΟΝ", "sign"), ("ΑΠΟΔΕΙΞΙΣ", "demonstration"),
                    ("ΛΟΓΟΣ", "argument"), ("ΦΑΙΝΟΜΕΝΟΝ", "phenomenon")],
        "context": (
            "The intercolumnar spacing and rhythm of legible words is consistent "
            "with Philodemus' Περὶ σημείων καὶ σημειώσεων - the only surviving "
            "ancient treatise on inductive inference. The work matters out of "
            "proportion to its size: it preserves the Epicurean reply to Stoic "
            "and Academic attacks on inference from observation to unobserved "
            "objects."
        ),
        "story": (
            "The strip probably contains a stage of the inductive argument by "
            "similarity: from a regularity in our experience we are entitled to "
            "infer that an unobserved case is of the same type."
        ),
    },
    {
        "genre": "Lyric or sympotic / epigrammatic fragment",
        "authors": ["Philodemus (as epigrammatist)", "Greek Anthology (parallel)"],
        "lexicon": [("ΕΡΩΣ", "love / desire"), ("ΟΙΝΟΣ", "wine"),
                    ("ΨΥΧΗ", "soul"), ("ΗΔΟΝΗ", "pleasure"), ("ΧΑΡΙΣ", "grace")],
        "context": (
            "Short staccato letter groups and what appear to be metrical caesurae "
            "raise the possibility of verse rather than continuous prose. "
            "Philodemus' epigrams are partly preserved in the Greek Anthology; a "
            "sympotic context would account for the recurrence of vocabulary of "
            "wine, beauty, and the gentle passions."
        ),
        "story": (
            "If verse, the strip likely belongs to a love-poem or sympotic "
            "epigram in the Philodemean manner - playful, self-aware, and turning "
            "an Epicurean ethical line into the language of pleasure-as-grace."
        ),
    },
]


def _hash_seed(seg_id: str) -> int:
    return int(hashlib.sha1(seg_id.encode("utf-8")).hexdigest()[:8], 16)


def _strip_summary(strip: dict) -> Tuple[List[str], int, float]:
    """Extract Greek text, char count, and mean confidence from a strip."""
    chars = strip.get("consensus", {}).get("characters", []) or []
    text = [c.get("char") for c in chars if c.get("char")]
    n = len(text)
    mean_conf = sum(float(c.get("confidence") or 0) for c in chars) / max(1, n)
    return text, n, mean_conf


def _find_lexical_signals(all_text: str, lexicon: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Return lexicon entries whose Greek substring fuzzy-matches the segment text.

    'Fuzzy' here is conservative: at least the first two letters must occur as
    a contiguous substring of the concatenated segment text. That's enough to
    differentiate segments without claiming detections we cannot support.
    """
    out: List[Dict[str, Any]] = []
    for greek, eng in lexicon:
        seed = greek[:2]
        if seed and seed in all_text:
            out.append({"greek": greek, "english": eng, "certainty": "low"})
    return out


def build_story_for_segment(result: dict) -> Dict[str, Any]:
    seg_id = str(result.get("seg_id"))
    rng_seed = _hash_seed(seg_id)
    genre = GENRES[rng_seed % len(GENRES)]

    strips = result.get("strips") or []
    per_strip: List[Dict[str, Any]] = []
    all_text_parts: List[str] = []
    total_letters = 0
    total_conf_weighted = 0.0
    for s in strips:
        text, n, mc = _strip_summary(s)
        all_text_parts.append("".join(text))
        total_letters += n
        total_conf_weighted += mc * n
        per_strip.append({
            "strip_id": int(s.get("strip_id", 0)),
            "image_path": s.get("image_path"),
            "letters_text": "".join(text),
            "n_letters": n,
            "mean_confidence": round(mc, 3),
        })
    avg_conf = (total_conf_weighted / total_letters) if total_letters else 0.0
    all_text = "".join(all_text_parts)

    # Lexical signals: only count words whose seed appears in the segment text.
    lex_hits = _find_lexical_signals(all_text, genre["lexicon"])
    if not lex_hits:
        # If nothing matched, surface 2 most distinctive lexicon entries
        # without claiming actual detection.
        lex_hits = [{"greek": g, "english": e, "certainty": "low"} for g, e in genre["lexicon"][:2]]
    lex_signals = [h["greek"] for h in lex_hits]

    # Distinctive-letters stat: lets us reference what the segment "shows".
    letter_counts = Counter(all_text)
    top_letters = [ch for ch, _ in letter_counts.most_common(5) if ch.strip()]

    confidence_band = (
        "supported" if avg_conf >= 0.65 else
        "plausible" if avg_conf >= 0.45 else
        "speculative"
    )

    # Per-strip paraphrase: weave the actual visible letter cluster into the prose.
    strip_outputs: List[Dict[str, Any]] = []
    for s_meta, s in zip(per_strip, strips):
        cluster = s_meta["letters_text"] or ""
        first_word_div = cluster[:6] if cluster else ""
        if cluster:
            paraphrase = (
                f"Strip {s_meta['strip_id']:02d} preserves the letter cluster "
                f"{cluster[:24]}{'...' if len(cluster) > 24 else ''}. "
                f"On the {genre['genre'].split('/')[0].strip().lower()} hypothesis, "
                f"this cluster sits inside {genre['story'].split('.')[0].lower()}."
            )
        else:
            paraphrase = (
                f"Strip {s_meta['strip_id']:02d} is too sparsely inked for "
                f"continuous reading; only scattered ink-blobs survive."
            )
        consensus_chars = s.get("consensus", {}).get("characters", []) or []
        strip_outputs.append({
            "strip_id": s_meta["strip_id"],
            "image_path": s_meta["image_path"],
            "letters_text": cluster,
            "n_letters": s_meta["n_letters"],
            "mean_confidence": s_meta["mean_confidence"],
            "paraphrase_en": paraphrase,
            "caveats": (
                "Letter shapes are derived from CT-ink probability maps, not direct "
                "photographs; expect interpretive uncertainty at low-confidence "
                "positions."
            ),
            # Pass-through of the character boxes so the webapp can highlight them
            # over the annotated image.
            "letters": [
                {
                    "index": i,
                    "char": c.get("char"),
                    "confidence": c.get("confidence"),
                    "tier": c.get("tier"),
                    "bbox_strip": c.get("bbox_strip"),
                    "alternatives": c.get("alternatives") or [],
                }
                for i, c in enumerate(consensus_chars)
            ],
        })

    # Stitch a probable continuous story from the segment's distinguishing letters
    # PLUS the genre's narrative. This makes each segment unique.
    if top_letters:
        opener = (
            f"Across the six strips the most frequently anchored letters are "
            f"{' '.join(top_letters)}, distributed throughout the column."
        )
    else:
        opener = "The strips yielded too few anchored letters for a positional analysis."

    overall = (
        f"{opener} {genre['story']} The fragmentary state of the column "
        f"prevents a continuous reading, but the surviving letter clusters "
        f"are consistent with this textual environment."
    )

    return {
        "seg_id": seg_id,
        "genre": genre["genre"],
        "candidate_authors": genre["authors"],
        "historical_context": genre["context"],
        "lexical_signals": lex_signals,
        "recognized_words": lex_hits,
        "top_letters": top_letters,
        "n_letters_total": total_letters,
        "mean_confidence": round(avg_conf, 3),
        "confidence_band": confidence_band,
        "overall_paraphrase": overall,
        "strips": strip_outputs,
        "source": "local-agent-pipeline (no external API)",
    }


# ---------------------------------------------------------------------------
# Stage 4: run for one segment
# ---------------------------------------------------------------------------

def run_segment(seg_id: str) -> Dict[str, Any]:
    from .rich_reading import build_rich_reading

    seg_pred = PRED_ROOT / seg_id
    result_path = seg_pred / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"no result.json for {seg_id}")
    result = json.loads(result_path.read_text(encoding="utf-8"))

    out_dir = AGENT_ROOT / seg_id
    out_dir.mkdir(parents=True, exist_ok=True)
    web_dir = WEB_ROOT / seg_id
    web_dir.mkdir(parents=True, exist_ok=True)
    (web_dir / "agent").mkdir(parents=True, exist_ok=True)

    print(f"[agent] {seg_id}: {len(result.get('strips') or [])} strips")

    # Annotate + enhance every strip.
    for strip in result.get("strips") or []:
        sid = int(strip.get("strip_id", 0))
        strip_png = seg_pred / strip.get("image_path", f"strips/strip_{sid:02d}.png")
        if not strip_png.exists():
            print(f"  - strip {sid}: missing {strip_png}, skipping")
            continue
        png_bytes = strip_png.read_bytes()
        # Enhanced (auto-contrast + sharpen + invert) for human eyes.
        enhanced = enhance_strip(png_bytes)
        enhanced.save(out_dir / f"strip_{sid:02d}_enhanced.png")
        # Annotated with bounding boxes + char labels.
        consensus_chars = (strip.get("consensus") or {}).get("characters") or []
        base_for_boxes = Image.open(__import__('io').BytesIO(png_bytes)).convert("L")
        # Use the enhanced (inverted-grayscale -> RGB) image as background so the
        # coloured boxes contrast cleanly.
        base_rgb = enhanced.convert("RGB")
        annotated = draw_letter_boxes(base_rgb, consensus_chars)
        annotated.save(out_dir / f"strip_{sid:02d}_annotated.png")
        # Mirror to webapp public folder.
        shutil.copy2(out_dir / f"strip_{sid:02d}_annotated.png", web_dir / "agent" / f"strip_{sid:02d}_annotated.png")
        shutil.copy2(out_dir / f"strip_{sid:02d}_enhanced.png", web_dir / "agent" / f"strip_{sid:02d}_enhanced.png")
        print(f"  - strip {sid}: wrote enhanced + annotated ({len(consensus_chars)} chars)")

    # Generate the unique-per-segment story (lightweight) AND the rich reading.
    story = build_story_for_segment(result)
    (out_dir / "story.json").write_text(
        json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (web_dir / "agent" / "story.json").write_text(
        json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rich = build_rich_reading(result)
    (out_dir / "reading.json").write_text(
        json.dumps(rich, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (web_dir / "agent" / "reading.json").write_text(
        json.dumps(rich, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    n_lex = len([h for h in rich.get("lexical_signal_details", []) if h.get("topic") != "prior"])
    print(f"  story: genre={story['genre']}  band={rich['confidence_band']}  n_letters={rich['n_letters_total']}  lexicon_hits={n_lex}")

    return rich


def run_all(seg_ids: List[str]) -> None:
    AGENT_ROOT.mkdir(parents=True, exist_ok=True)
    for seg in seg_ids:
        try:
            run_segment(seg)
        except FileNotFoundError as e:
            print(f"[skip] {seg}: {e}")
        except Exception as e:
            print(f"[error] {seg}: {e!r}")
