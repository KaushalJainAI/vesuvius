"""
src/llm_gap_fill.py

Stage 3 of the text deciphering pipeline — LLM gap-filling via Claude API.

Uses claude-opus-4-7 to:
  - Resolve ambiguous letter candidates (MEDIUM confidence tier)
  - Infer characters in spatial gaps between high-confidence anchors
  - Return per-position (char, confidence) predictions

Main entry points:
  LLMGapFiller.fill_lines(lines, W, H)       → List[LineFillResult]
  LLMGapFiller.apply_to_prob_map(prob, ...)  → updated float32 map
  visualize_llm_output(prob, line, result)   → matplotlib Figure
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CharPrediction:
    pos_x: float                                 # absolute pixel x
    pos_y: float                                 # absolute pixel y (line centre)
    char: str
    conf: float                                  # [0, 1]
    source: str                                  # "anchor" | "llm_med" | "llm_gap"
    bbox: Optional[Tuple[int, int, int, int]] = None  # from existing component


@dataclass
class LineFillResult:
    line_id: int
    predictions: List[CharPrediction] = field(default_factory=list)
    raw_llm_response: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert in ancient Greek palaeography specialising in Herculaneum \
papyrus scrolls (1st century BC Epicurean texts, majuscule/uncial script). \
You are deciphering text from a CT-derived ink probability map.

You will receive one or more numbered text lines. Each line shows character \
positions extracted from the scan with confidence tiers:
  HIGH → reliable anchor — treat as ground truth
  MED  → ambiguous; pick the best candidate from the listed options
  GAP  → no visual evidence; infer from Greek lexical/grammatical context

Rules:
1. Never change HIGH letters.
2. For MED positions: select one character from the listed options.
3. For GAP positions: infer the most plausible character from Epicurean Greek.
   Common words: περί, τοῦ, καί, τῶν, ἐν, τό, τήν, εἰ, ὅτι, γάρ, δέ, μέν.
4. Use uppercase Greek (the scroll uses majuscule; accents/breathings absent).
5. Confidence guide: HIGH anchor=1.0, clear MED=0.75-0.90, uncertain MED=0.60-0.75,
   inferred GAP=0.50-0.70.
6. Reply ONLY with valid JSON. No prose, no explanation.

Schema (one object per line):
{"line_id": <int>, "positions": [{"pos": <0-1 float>, "char": "<letter>", "conf": <float>, "source": "anchor|llm_med|llm_gap"}]}

Separate multiple line responses with a line containing only "---".\
"""


def _build_line_prompt(line, segment_width: int) -> str:
    """Lazy import to avoid circular dependency with letter_candidates."""
    from letter_candidates import build_llm_line_context
    ctx = build_llm_line_context(line, segment_width)
    return f"Line {line.line_id}:\n{ctx}"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_llm_response(text: str) -> List[Dict]:
    """Extract the `positions` array from a Claude JSON response.
    Tolerates markdown fences and stray whitespace.
    """
    text = text.strip()
    # Strip markdown code fences
    if "```" in text:
        text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```"))

    # Try direct parse
    try:
        data = json.loads(text)
        return data.get("positions", [])
    except json.JSONDecodeError:
        pass

    # Find the outermost { ... } and parse that
    start = text.find("{")
    end   = text.rfind("}") + 1
    if 0 <= start < end:
        try:
            data = json.loads(text[start:end])
            return data.get("positions", [])
        except json.JSONDecodeError:
            pass

    return []


# ---------------------------------------------------------------------------
# LLMGapFiller
# ---------------------------------------------------------------------------

class LLMGapFiller:
    """Resolves ambiguous candidates and fills text gaps using Claude.

    Usage:
        filler  = LLMGapFiller()
        results = filler.fill_lines(lines, segment_width=W, segment_height=H)
        updated = filler.apply_to_prob_map(prob, results)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-7",
    ) -> None:
        import anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model

    # ------------------------------------------------------------------
    # Line selection
    # ------------------------------------------------------------------

    def fill_lines(
        self,
        lines,
        segment_width: int,
        segment_height: int,
        *,
        batch_size: int = 5,
        skip_all_high: bool = True,
    ) -> List[LineFillResult]:
        """Run LLM gap-filling on all text lines.

        Lines that contain only HIGH-confidence candidates and have no large
        gaps are skipped to save API tokens.
        """
        from letter_candidates import assign_confidence_tier

        active = []
        for line in lines:
            has_med = any(
                assign_confidence_tier(c) == "MEDIUM" for c in line.candidates
            )
            has_gap = self._has_large_gap(line)
            if not skip_all_high or has_med or has_gap:
                active.append(line)

        print(f"  LLM: {len(active)}/{len(lines)} lines need filling…")

        results: List[LineFillResult] = []
        for i in range(0, len(active), batch_size):
            batch = active[i:i + batch_size]
            results.extend(self._fill_batch(batch, segment_width, segment_height))
            done = min(i + batch_size, len(active))
            if done % (batch_size * 5) == 0:
                print(f"    {done}/{len(active)} lines processed")

        return results

    @staticmethod
    def _has_large_gap(line, gap_factor: float = 2.5) -> bool:
        if len(line.candidates) < 2:
            return False
        xs = [c.centroid[1] for c in line.candidates]
        ws = [c.bbox[3] - c.bbox[1] for c in line.candidates]
        med_w = float(np.median(ws)) if ws else 20.0
        return any((xs[i + 1] - xs[i]) > gap_factor * med_w for i in range(len(xs) - 1))

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _fill_batch(
        self,
        lines,
        segment_width: int,
        segment_height: int,
    ) -> List[LineFillResult]:
        prompts = [_build_line_prompt(line, segment_width) for line in lines]
        user_msg = "\n\n---\n\n".join(prompts)

        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text
        except Exception as exc:
            return [
                LineFillResult(line_id=ln.line_id, error=str(exc)) for ln in lines
            ]

        # Split Claude's response by "---" separators (one block per line)
        raw_blocks = [b.strip() for b in raw.split("---")]
        # Pad with empty blocks if Claude gave fewer than expected
        while len(raw_blocks) < len(lines):
            raw_blocks.append("{}")

        results = []
        for line, block in zip(lines, raw_blocks):
            positions = _parse_llm_response(block)
            preds = self._to_predictions(positions, line, segment_width)
            results.append(LineFillResult(
                line_id=line.line_id,
                predictions=preds,
                raw_llm_response=block,
            ))
        return results

    # ------------------------------------------------------------------
    # Position → pixel coordinates
    # ------------------------------------------------------------------

    def _to_predictions(
        self,
        positions: List[Dict],
        line,
        segment_width: int,
    ) -> List[CharPrediction]:
        if not line.candidates:
            return []

        line_y = float(np.mean([c.centroid[0] for c in line.candidates]))
        char_h = float(np.median([c.bbox[2] - c.bbox[0] for c in line.candidates]))
        # Map normalised-x → nearest candidate for bbox lookup
        cand_by_norm_x = {
            c.centroid[1] / max(segment_width, 1): c for c in line.candidates
        }

        preds: List[CharPrediction] = []
        for pos_dict in positions:
            if not isinstance(pos_dict, dict):
                continue
            pos    = float(pos_dict.get("pos", 0))
            char   = str(pos_dict.get("char", "?"))
            conf   = float(pos_dict.get("conf", 0.5))
            source = str(pos_dict.get("source", "llm_gap"))

            px_x, px_y = pos * segment_width, line_y

            # Match to closest existing candidate bbox (within half a char width)
            bbox = None
            best_d = float("inf")
            for norm_x, cand in cand_by_norm_x.items():
                d = abs(norm_x - pos) * segment_width
                if d < char_h * 0.6 and d < best_d:
                    best_d, bbox = d, cand.bbox

            preds.append(CharPrediction(
                pos_x=px_x, pos_y=px_y,
                char=char, conf=conf,
                source=source, bbox=bbox,
            ))
        return preds

    # ------------------------------------------------------------------
    # Apply predictions to probability map
    # ------------------------------------------------------------------

    def apply_to_prob_map(
        self,
        prob: np.ndarray,
        fill_results: List[LineFillResult],
        *,
        gap_blob_radius: int = 12,
        gap_discount: float = 0.75,
    ) -> np.ndarray:
        """Write LLM predictions back into the probability map.

        llm_med with bbox  → raise existing component pixels to min(0.90, conf)
        llm_gap            → paint a Gaussian blob at predicted centroid × discount
        anchor             → untouched
        """
        updated = prob.copy()
        H, W = prob.shape

        for result in fill_results:
            if result.error:
                continue
            for pred in result.predictions:
                if pred.source == "anchor":
                    continue

                y = int(np.clip(round(pred.pos_y), 0, H - 1))
                x = int(np.clip(round(pred.pos_x), 0, W - 1))

                if pred.source == "llm_med" and pred.bbox is not None:
                    y0, x0, y1, x1 = pred.bbox
                    y0, y1 = max(0, y0), min(H, y1)
                    x0, x1 = max(0, x0), min(W, x1)
                    target = float(min(0.90, pred.conf))
                    updated[y0:y1, x0:x1] = np.maximum(updated[y0:y1, x0:x1], target)

                else:  # llm_gap — Gaussian blob, no visual evidence so discounted
                    target = float(pred.conf * gap_discount)
                    r = gap_blob_radius
                    y0, y1 = max(0, y - r), min(H, y + r + 1)
                    x0, x1 = max(0, x - r), min(W, x + r + 1)
                    ys = np.arange(y0, y1) - y
                    xs = np.arange(x0, x1) - x
                    yy, xx = np.meshgrid(ys, xs, indexing="ij")
                    sigma = r / 2.5
                    gaussian = (np.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2)) * target).astype(np.float32)
                    updated[y0:y1, x0:x1] = np.maximum(updated[y0:y1, x0:x1], gaussian)

        return np.clip(updated, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

_SRC_COLORS = {"anchor": "limegreen", "llm_med": "cyan", "llm_gap": "orange"}
_TIER_COLORS = {"HIGH": "limegreen", "MEDIUM": "yellow", "LOW": "tomato"}


def visualize_llm_output(
    prob: np.ndarray,
    line,
    fill_result: LineFillResult,
    *,
    context_pad: int = 50,
    figsize: Tuple[float, float] = (18, 5),
) -> Figure:
    """Side-by-side: raw candidates (left) vs LLM-annotated output (right).

    Left panel  — bounding boxes coloured by confidence tier, top-1 template match shown.
    Right panel — LLM predictions coloured by source (anchor / llm_med / llm_gap).
    """
    from letter_candidates import assign_confidence_tier

    if not line.candidates:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No candidates in line", ha="center", va="center")
        return fig

    ys = [c.centroid[0] for c in line.candidates]
    xs = [c.centroid[1] for c in line.candidates]
    char_h = float(np.median([c.bbox[2] - c.bbox[0] for c in line.candidates]))

    y_min = max(0, int(min(ys)) - context_pad)
    y_max = min(prob.shape[0], int(max(ys)) + context_pad)
    x_min = max(0, int(min(xs)) - context_pad)
    x_max = min(prob.shape[1], int(max(xs)) + context_pad)
    region = prob[y_min:y_max, x_min:x_max]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax in axes:
        ax.imshow(region, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")

    axes[0].set_title(f"Line {line.line_id} — raw candidates", fontsize=10)
    axes[1].set_title(f"Line {line.line_id} — LLM output", fontsize=10)

    # Left: bounding boxes
    for cand in line.candidates:
        tier = assign_confidence_tier(cand)
        color = _TIER_COLORS.get(tier, "white")
        y0, x0, y1, x1 = cand.bbox
        rect = mpatches.Rectangle(
            (x0 - x_min, y0 - y_min), x1 - x0, y1 - y0,
            linewidth=1.0, edgecolor=color, facecolor="none",
        )
        axes[0].add_patch(rect)
        if cand.matches:
            axes[0].text(x0 - x_min, y0 - y_min - 2,
                         cand.matches[0].char, color=color, fontsize=8, fontweight="bold")

    # Right: LLM predictions
    if fill_result and not fill_result.error:
        for pred in fill_result.predictions:
            px = pred.pos_x - x_min
            py = pred.pos_y - y_min
            color = _SRC_COLORS.get(pred.source, "white")
            axes[1].plot(px, py, "o", color=color, markersize=6, markeredgewidth=0.5, markeredgecolor="white")
            axes[1].text(px + 2, py - char_h * 0.55,
                         pred.char, color=color, fontsize=9, fontweight="bold")

    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in [
        ("Anchor (HIGH)", "limegreen"), ("LLM resolved (MED)", "cyan"), ("LLM gap fill", "orange"),
    ]]
    axes[1].legend(handles=legend_patches, fontsize=7, loc="upper right",
                   framealpha=0.6, facecolor="black", labelcolor="white")

    fig.tight_layout()
    return fig


def visualize_all_line_results(
    prob: np.ndarray,
    lines,
    fill_results: List[LineFillResult],
    *,
    n_lines: int = 5,
    figsize: Tuple[float, float] = (18, 4),
) -> List[Figure]:
    """Return one figure per line (up to n_lines) showing LLM output."""
    result_by_id = {r.line_id: r for r in fill_results}
    figs = []
    for line in lines[:n_lines]:
        fr = result_by_id.get(line.line_id, LineFillResult(line_id=line.line_id))
        figs.append(visualize_llm_output(prob, line, fr, figsize=figsize))
    return figs
