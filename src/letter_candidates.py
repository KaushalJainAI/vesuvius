"""
src/letter_candidates.py

High-quality Greek letter template matching for Herculaneum CT ink maps.

Improvements over the naive NCC matcher in experiment_filters.py:

  1. Multi-style templates — three variants per letter:
       - modern serif (high-quality font rendering)
       - ancient uncial approximation (thicker stroke, rounder forms, lunate sigma)
       - skeleton-only (stroke-width-agnostic structural template)
     Matching is run against all 3 variants; best score per letter is kept.

  2. Stroke-width normalisation — skeletonise candidate → dilate to 2 px before matching.
     This removes the dominant source of template mismatch (stroke weight variation
     across scroll regions and different scan conditions).

  3. Bidirectional Chamfer distance — measures mean distance from candidate pixels to
     nearest template pixels AND vice versa. Far more robust than NCC when strokes are
     broken or partially occluded (common in CT-derived ink maps).

  4. Hu moment distance — 7 log-scale Hu moments are rotation/scale-normalised shape
     descriptors. Provides a shape-level signal that complements pixel-level metrics.

  5. Topology bonus — estimates hole count (Euler number) per candidate and rewards
     templates with matching topology. Α, Θ, Ο, Ρ have enclosed loops; Ε, Ζ, Τ do not.
     Used as a soft bonus, not a hard filter, since ancient strokes often break loops.

  6. Multi-scale search — candidate is tested at three template sizes (32, 48, 64 px).
     Best score across scales is kept. Handles natural letter-size variation across
     segments without forcing a fixed size.

  7. Ensemble score — weighted combination of all metrics:
       0.40 × Chamfer  +  0.35 × NCC  +  0.25 × IoU  +  topology_bonus
     Chamfer is dominant because it is most robust to broken strokes.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage as ndi
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from skimage import measure, morphology
from skimage.measure import moments as sk_moments
from skimage.measure import moments_central, moments_hu
from skimage.morphology import binary_dilation, disk, skeletonize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GREEK_UPPER: List[str] = list("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")

GREEK_NAMES: List[str] = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho",
    "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
]

# In Herculaneum scrolls sigma is written as a C-shape (lunate sigma).
# We render Σ for the modern template, but also render 'Ϲ' as the ancient variant.
ANCIENT_VARIANTS: Dict[str, str] = {
    "Σ": "Ϲ",   # U+03F9 GREEK CAPITAL LUNATE SIGMA SYMBOL
}

# Approximate hole (enclosed loop) count for well-formed uppercase Greek letters.
# euler_number for a single connected component = 1 - holes.
# Used as topology bonus; values are approximate for clean renderings.
LETTER_HOLES: Dict[str, int] = {
    "Α": 1, "Β": 2, "Γ": 0, "Δ": 1, "Ε": 0, "Ζ": 0,
    "Η": 0, "Θ": 1, "Ι": 0, "Κ": 0, "Λ": 0, "Μ": 0,
    "Ν": 0, "Ξ": 0, "Ο": 1, "Π": 0, "Ρ": 1, "Σ": 0,
    "Τ": 0, "Υ": 0, "Φ": 2, "Χ": 0, "Ψ": 0, "Ω": 1,
}

# Template sizes to test (small / medium / large).
TEMPLATE_SIZES: Tuple[int, ...] = (32, 48, 64)

# Ensemble weights: (Chamfer, NCC, IoU)
ENSEMBLE_W = (0.40, 0.35, 0.25)

# Topology bonus magnitude (added to ensemble score when hole count matches)
TOPOLOGY_BONUS = 0.06


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LetterMatch:
    char: str
    score: float          # final ensemble score in [0, 1] (higher = better)
    chamfer: float        # bidirectional Chamfer (lower = better, stored as raw)
    ncc: float
    iou: float
    hu_dist: float        # Hu moment L2 distance (lower = better)
    best_size: int        # template size that gave best score
    best_style: str       # "serif" | "uncial" | "skeleton"


@dataclass
class Candidate:
    id: int
    bbox: Tuple[int, int, int, int]   # y0, x0, y1, x1
    centroid: Tuple[float, float]
    area: int
    mean_prob: float
    patch: np.ndarray                 # raw float32 binary crop
    matches: List[LetterMatch] = field(default_factory=list)


@dataclass
class TextLine:
    line_id: int
    candidates: List[Candidate]       # sorted left → right by centroid x


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def _render_letter_raw(ch: str, size: int, *, fontsize_scale: float = 0.72,
                       family: str = "serif") -> np.ndarray:
    """Render `ch` as a white-on-black binary (size, size) array."""
    fig = Figure(figsize=(1, 1), dpi=size)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, ch,
            ha="center", va="center",
            fontsize=size * fontsize_scale,
            family=family,
            color="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("black")
    ax.axis("off")
    fig.patch.set_facecolor("black")
    fig.subplots_adjust(0, 0, 1, 1)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[..., :3].mean(-1)
    img = (buf > 64).astype(np.float32)
    if img.shape[0] != size:
        zy, zx = size / img.shape[0], size / img.shape[1]
        img = ndi.zoom(img, (zy, zx), order=1)
        img = (img > 0.5).astype(np.float32)
    return img


def _render_uncial(ch: str, size: int) -> np.ndarray:
    """Ancient uncial approximation:
    - Uses lunate sigma variant for Σ
    - Slightly thicker strokes (dilated serif render)
    - Rounder overall form via larger fontsize and tighter crop
    """
    ch_render = ANCIENT_VARIANTS.get(ch, ch)
    base = _render_letter_raw(ch_render, size, fontsize_scale=0.82, family="serif")
    # Thicken strokes by one pixel to approximate ancient pen width
    base = binary_dilation(base > 0.5, disk(1)).astype(np.float32)
    # Re-centre after dilation (may have expanded slightly)
    base = _centre_crop(base, size)
    return base


def _render_skeleton(ch: str, size: int) -> np.ndarray:
    """Stroke-width-agnostic template: render serif then skeletonise + dilate to 2 px.
    Matches candidates that have already been stroke-normalised."""
    base = _render_letter_raw(ch, size, fontsize_scale=0.72)
    skel = skeletonize(base > 0.5)
    return binary_dilation(skel, disk(1)).astype(np.float32)


def _centre_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Crop or pad `img` to (size, size) keeping content centred."""
    H, W = img.shape
    if H == size and W == size:
        return img
    out = np.zeros((size, size), dtype=img.dtype)
    # compute paste coordinates
    y0 = max(0, (size - H) // 2)
    x0 = max(0, (size - W) // 2)
    ch = min(H, size - y0)
    cw = min(W, size - x0)
    sy = max(0, (H - size) // 2)
    sx = max(0, (W - size) // 2)
    out[y0:y0 + ch, x0:x0 + cw] = img[sy:sy + ch, sx:sx + cw]
    return out


# ---------------------------------------------------------------------------
# GreekTemplateMatcher
# ---------------------------------------------------------------------------

class GreekTemplateMatcher:
    """Pre-builds a bank of multi-style templates and efficiently scores candidates.

    Template bank structure:
      bank[size][style][char] = (binary_array, distance_transform)

    Distance transforms are pre-computed for fast Chamfer scoring.
    """

    def __init__(
        self,
        sizes: Sequence[int] = TEMPLATE_SIZES,
        letters: Sequence[str] = GREEK_UPPER,
    ) -> None:
        self.sizes = list(sizes)
        self.letters = list(letters)
        self._bank: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = {}
        self._hu_bank: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        styles = {
            "serif":    _render_letter_raw,
            "uncial":   _render_uncial,
            "skeleton": _render_skeleton,
        }
        for sz in self.sizes:
            self._bank[sz] = {}
            self._hu_bank[sz] = {}
            for style, fn in styles.items():
                self._bank[sz][style] = {}
                self._hu_bank[sz][style] = {}
                for ch in self.letters:
                    tpl = fn(ch, sz)
                    # Precompute distance transform: distance from each pixel to
                    # the nearest ink pixel in the template.
                    dt = ndi.distance_transform_edt(1.0 - (tpl > 0.5))
                    self._bank[sz][style][ch] = (tpl, dt)
                    self._hu_bank[sz][style][ch] = _hu_moments(tpl)

    # ------------------------------------------------------------------
    # Candidate preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _stroke_normalize(patch: np.ndarray) -> np.ndarray:
        """Skeletonise → dilate to uniform 2-px stroke width."""
        binary = patch > 0.5
        if binary.sum() < 5:
            return patch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skel = skeletonize(binary)
        return binary_dilation(skel, disk(1)).astype(np.float32)

    @staticmethod
    def _resize_aspect(patch: np.ndarray, size: int) -> np.ndarray:
        """Resize `patch` to fit within (size, size) preserving aspect ratio,
        then centre on a black canvas."""
        H, W = patch.shape
        if H == 0 or W == 0:
            return np.zeros((size, size), dtype=np.float32)
        margin = max(4, size // 10)
        s = size - margin * 2
        if H >= W:
            nh = s
            nw = max(1, int(round(W * s / H)))
        else:
            nw = s
            nh = max(1, int(round(H * s / W)))
        zy, zx = nh / H, nw / W
        resized = ndi.zoom(patch, (zy, zx), order=1)
        resized = (resized > 0.5).astype(np.float32)
        canvas = np.zeros((size, size), dtype=np.float32)
        y0 = (size - resized.shape[0]) // 2
        x0 = (size - resized.shape[1]) // 2
        canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        return canvas

    # ------------------------------------------------------------------
    # Scoring primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _chamfer(cand: np.ndarray, tpl: np.ndarray,
                 dt_tpl: np.ndarray, dt_cand: np.ndarray) -> float:
        """Bidirectional normalised Chamfer distance (lower = better).

        Measures:
          mean distance from candidate pixels to nearest template pixel  (forward)
        + mean distance from template pixels to nearest candidate pixel  (backward)

        Normalised by image diagonal so scores are size-independent.
        """
        size = cand.shape[0]
        diag = float(size) * 1.41421
        c_mask = cand > 0.5
        t_mask = tpl > 0.5
        if c_mask.sum() == 0 or t_mask.sum() == 0:
            return 1.0
        fwd = dt_tpl[c_mask].mean() / diag
        bwd = dt_cand[t_mask].mean() / diag
        return float((fwd + bwd) * 0.5)

    @staticmethod
    def _ncc(cand: np.ndarray, tpl: np.ndarray) -> float:
        """Normalised cross-correlation in [-1, 1], mapped to [0, 1]."""
        c = cand.flatten() - cand.mean()
        t = tpl.flatten() - tpl.mean()
        denom = np.sqrt((c ** 2).sum() * (t ** 2).sum()) + 1e-8
        return float((c * t).sum() / denom * 0.5 + 0.5)

    @staticmethod
    def _iou(cand: np.ndarray, tpl: np.ndarray) -> float:
        """Intersection over Union."""
        inter = float((cand * tpl).sum())
        union = float(cand.sum() + tpl.sum() - inter) + 1e-8
        return inter / union

    @staticmethod
    def _hu_dist(hu_cand: np.ndarray, hu_tpl: np.ndarray) -> float:
        """L1 distance between log-scale Hu moment vectors (lower = better)."""
        return float(np.abs(hu_cand - hu_tpl).mean())

    @staticmethod
    def _topology_bonus(cand: np.ndarray, char: str) -> float:
        """Reward templates whose expected hole count matches the candidate.
        Returns TOPOLOGY_BONUS if match, 0 otherwise."""
        expected_holes = LETTER_HOLES.get(char, 0)
        try:
            euler = measure.euler_number(cand > 0.5, connectivity=1)
            cand_holes = max(0, 1 - euler)
        except Exception:
            return 0.0
        # Allow off-by-one: ancient strokes often fail to close a loop
        if abs(cand_holes - expected_holes) <= 1:
            return TOPOLOGY_BONUS
        return 0.0

    # ------------------------------------------------------------------
    # Match
    # ------------------------------------------------------------------

    def match(self, patch: np.ndarray, top_k: int = 5) -> List[LetterMatch]:
        """Score all Greek letters against `patch`.

        Pipeline:
          For each size s:
            - Normalise patch (stroke-norm then resize to s)
            - Compute Hu moments once for this scale
            - For each style × letter:
                ensemble = w_ch × chamfer_score + w_ncc × ncc + w_iou × iou
              where chamfer_score = 1 - chamfer (lower chamfer → higher score)
            - Add topology bonus
            - Keep best (style, score) per letter across sizes

        Returns the top_k LetterMatch objects sorted by score descending.
        """
        wc, wn, wi = ENSEMBLE_W

        # Accumulate best score per letter
        best: Dict[str, LetterMatch] = {}

        for sz in self.sizes:
            # Two candidate versions to try at this scale:
            #   - original (for serif/uncial templates)
            #   - stroke-normalised (for skeleton templates)
            cand_raw = self._resize_aspect(patch, sz)
            cand_norm = self._resize_aspect(self._stroke_normalize(patch), sz)
            dt_raw = ndi.distance_transform_edt(1.0 - (cand_raw > 0.5))
            dt_norm = ndi.distance_transform_edt(1.0 - (cand_norm > 0.5))
            hu_raw = _hu_moments(cand_raw)
            hu_norm = _hu_moments(cand_norm)

            for style, style_bank in self._bank[sz].items():
                # Skeleton style compares against stroke-normalised candidate
                if style == "skeleton":
                    cand, dt_cand, hu_cand = cand_norm, dt_norm, hu_norm
                else:
                    cand, dt_cand, hu_cand = cand_raw, dt_raw, hu_raw

                for ch, (tpl, dt_tpl) in style_bank.items():
                    chamfer = self._chamfer(cand, tpl, dt_tpl, dt_cand)
                    ncc     = self._ncc(cand, tpl)
                    iou     = self._iou(cand, tpl)
                    hu_d    = self._hu_dist(hu_cand, self._hu_bank[sz][style][ch])

                    # Chamfer: convert to similarity (lower chamfer = higher score)
                    # Clip at 0.5 max Chamfer to avoid dominating on truly bad matches
                    chamfer_sim = max(0.0, 1.0 - min(chamfer, 0.5) * 2.0)

                    # Hu distance: convert to similarity; typical range is 0–3
                    hu_sim = max(0.0, 1.0 - min(hu_d, 3.0) / 3.0)

                    ensemble = (wc * chamfer_sim
                                + wn * ncc
                                + wi * iou
                                + (1 - wc - wn - wi) * hu_sim)
                    ensemble += self._topology_bonus(cand, ch)

                    if ch not in best or ensemble > best[ch].score:
                        best[ch] = LetterMatch(
                            char=ch,
                            score=ensemble,
                            chamfer=chamfer,
                            ncc=ncc,
                            iou=iou,
                            hu_dist=hu_d,
                            best_size=sz,
                            best_style=style,
                        )

        ranked = sorted(best.values(), key=lambda m: m.score, reverse=True)
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Hu moments helper (uses skimage, no cv2 dependency)
# ---------------------------------------------------------------------------

def _hu_moments(binary: np.ndarray) -> np.ndarray:
    """Compute 7 log-scale Hu moments from a binary image."""
    b = (binary > 0.5).astype(np.float64)
    if b.sum() == 0:
        return np.zeros(7)
    m = sk_moments(b)
    if m[0, 0] == 0:
        return np.zeros(7)
    cx = m[1, 0] / m[0, 0]
    cy = m[0, 1] / m[0, 0]
    mc = moments_central(b, center=(cx, cy))
    hu = moments_hu(mc)
    # Log scale with sign preservation (standard cv2.HuMoments convention)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-15)
    return np.nan_to_num(log_hu)


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def extract_candidates(
    prob: np.ndarray,
    *,
    threshold: float = 0.40,
    min_area: int = 50,
    max_area: int = 8000,
) -> List[Candidate]:
    """Threshold + morphological cleanup + connected components.

    Args:
        prob:       float32 [0,1] probability map.
        threshold:  binarisation threshold.
        min_area:   discard components smaller than this (noise).
        max_area:   discard components larger than this (merged blobs).

    Returns:
        List of Candidate objects, unsorted.
    """
    if prob.max() > 1.5:
        prob = prob / 255.0

    binary = (prob > threshold).astype(bool)
    binary = morphology.binary_closing(binary, morphology.disk(2))
    binary = morphology.remove_small_objects(binary, min_size=min_area)

    labelled, _ = ndi.label(binary)
    candidates: List[Candidate] = []

    for region in measure.regionprops(labelled, intensity_image=prob):
        if region.area < min_area or region.area > max_area:
            continue
        y0, x0, y1, x1 = region.bbox
        patch_mask = (labelled[y0:y1, x0:x1] == region.label).astype(np.float32)
        mean_prob = float(prob[y0:y1, x0:x1][labelled[y0:y1, x0:x1] == region.label].mean())
        candidates.append(Candidate(
            id=region.label,
            bbox=(y0, x0, y1, x1),
            centroid=(float(region.centroid[0]), float(region.centroid[1])),
            area=int(region.area),
            mean_prob=mean_prob,
            patch=patch_mask,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Text-line detection
# ---------------------------------------------------------------------------

def detect_text_lines(
    candidates: List[Candidate],
    *,
    line_eps_factor: float = 1.5,
) -> List[TextLine]:
    """Cluster candidates into horizontal text lines.

    Uses a simple single-linkage vertical clustering: two candidates belong
    to the same line if their centroid Y coordinates are within
    `line_eps_factor × median_char_height` of each other.

    Returns TextLine objects sorted top-to-bottom, candidates within each
    line sorted left-to-right.
    """
    if not candidates:
        return []

    heights = [c.bbox[2] - c.bbox[0] for c in candidates]
    median_h = float(np.median(heights)) if heights else 20.0
    eps = line_eps_factor * median_h

    centroids_y = np.array([c.centroid[0] for c in candidates])
    order = np.argsort(centroids_y)
    sorted_cands = [candidates[i] for i in order]
    sorted_y = centroids_y[order]

    lines: List[List[Candidate]] = []
    current: List[Candidate] = [sorted_cands[0]]
    current_y = sorted_y[0]

    for cand, cy in zip(sorted_cands[1:], sorted_y[1:]):
        if cy - current_y <= eps:
            current.append(cand)
        else:
            lines.append(current)
            current = [cand]
            current_y = cy

    lines.append(current)

    text_lines: List[TextLine] = []
    for i, line in enumerate(lines):
        # Sort left → right by centroid x within line
        line_sorted = sorted(line, key=lambda c: c.centroid[1])
        text_lines.append(TextLine(line_id=i, candidates=line_sorted))

    return text_lines


# ---------------------------------------------------------------------------
# Softmax calibration
# ---------------------------------------------------------------------------

def softmax_scores(
    matches: List[LetterMatch],
    temperature: float = 8.0,
) -> List[Tuple[str, float]]:
    """Convert raw ensemble scores → calibrated pseudo-probabilities.

    Higher temperature = more peaked distribution.
    Returns [(char, probability), ...] summing to ~1.
    """
    if not matches:
        return []
    scores = np.array([m.score for m in matches])
    scores = scores * temperature - scores.max() * temperature
    exp = np.exp(scores)
    probs = exp / (exp.sum() + 1e-9)
    return [(m.char, float(p)) for m, p in zip(matches, probs)]


# ---------------------------------------------------------------------------
# Confidence tier assignment
# ---------------------------------------------------------------------------

def assign_confidence_tier(cand: Candidate) -> str:
    """Assign a confidence tier based on mean probability.

    HIGH   (>0.70) — reliable anchor for LLM context
    MEDIUM (0.45–0.70) — ambiguous, send to LLM for resolution
    LOW    (<0.45) — likely noise, omit from LLM context
    """
    if cand.mean_prob > 0.70:
        return "HIGH"
    elif cand.mean_prob >= 0.45:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Full pipeline convenience function
# ---------------------------------------------------------------------------

def run_matching_pipeline(
    prob: np.ndarray,
    *,
    threshold: float = 0.40,
    min_area: int = 50,
    max_area: int = 8000,
    top_k: int = 3,
    matcher: Optional[GreekTemplateMatcher] = None,
    skip_low_confidence: bool = True,
) -> Tuple[List[TextLine], GreekTemplateMatcher]:
    """End-to-end: extract candidates → match templates → detect lines.

    Args:
        prob:                 float32 [0,1] probability map.
        threshold:            binarisation threshold.
        min_area/max_area:    CC area filter bounds.
        top_k:                number of letter matches to store per candidate.
        matcher:              pre-built GreekTemplateMatcher (built on first call if None).
        skip_low_confidence:  if True, skip template matching for LOW-tier candidates.

    Returns:
        (lines, matcher) — TextLine list sorted top-to-bottom, and the matcher
        for reuse across multiple segments.
    """
    if matcher is None:
        print("Building template bank (3 styles × 3 sizes × 24 letters)…")
        matcher = GreekTemplateMatcher()
        print("  Done.")

    print(f"Extracting candidates (threshold={threshold})…")
    candidates = extract_candidates(prob, threshold=threshold,
                                    min_area=min_area, max_area=max_area)
    print(f"  {len(candidates)} candidates found.")

    print("Matching against Greek templates…")
    for i, cand in enumerate(candidates):
        tier = assign_confidence_tier(cand)
        if skip_low_confidence and tier == "LOW":
            continue
        cand.matches = matcher.match(cand.patch, top_k=top_k)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(candidates)}")

    print("Detecting text lines…")
    lines = detect_text_lines(candidates)
    print(f"  {len(lines)} lines detected.")

    return lines, matcher


# ---------------------------------------------------------------------------
# LLM context builder (feeds Stage 3 of the deciphering pipeline)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_candidates_overlay(
    prob: np.ndarray,
    candidates: List[Candidate],
    *,
    figsize: Tuple[float, float] = (14, 9),
    max_labels: int = 300,
    crop: Optional[Tuple[int, int, int, int]] = None,
) -> "plt.Figure":
    """Probability map with colour-coded bounding boxes per confidence tier.

    GREEN  — HIGH confidence (mean_prob > 0.70), top-1 letter shown
    YELLOW — MEDIUM confidence (0.45 – 0.70), top-1 letter shown
    RED    — LOW confidence (< 0.45)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    tier_color = {"HIGH": "limegreen", "MEDIUM": "yellow", "LOW": "tomato"}

    if crop is not None:
        y0c, y1c, x0c, x1c = crop
        display = prob[y0c:y1c, x0c:x1c]
        oy, ox = y0c, x0c
    else:
        display = prob
        oy, ox = 0, 0

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"Candidate overlay  ({len(candidates)} candidates)", fontsize=11)
    ax.axis("off")

    labeled = 0
    for cand in candidates:
        tier = assign_confidence_tier(cand)
        color = tier_color[tier]
        y0, x0, y1, x1 = cand.bbox
        rect = mpatches.Rectangle(
            (x0 - ox, y0 - oy), x1 - x0, y1 - y0,
            linewidth=0.7, edgecolor=color, facecolor="none", alpha=0.9,
        )
        ax.add_patch(rect)
        if labeled < max_labels and tier != "LOW" and cand.matches:
            ax.text(
                x0 - ox, y0 - oy - 2,
                cand.matches[0].char,
                color=color, fontsize=6, fontweight="bold",
                clip_on=True,
            )
            labeled += 1

    legend = [
        mpatches.Patch(color="limegreen", label=f"HIGH (>{0.70:.0%})"),
        mpatches.Patch(color="yellow",    label=f"MEDIUM (0.45–0.70)"),
        mpatches.Patch(color="tomato",    label=f"LOW (<0.45)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8,
              framealpha=0.7, facecolor="black", labelcolor="white")

    fig.tight_layout()
    return fig


def visualize_template_matches(
    candidates: List[Candidate],
    matcher: "GreekTemplateMatcher",
    *,
    n: int = 20,
    figsize: Tuple[float, float] = (16, 10),
    seed: int = 0,
) -> "plt.Figure":
    """Grid showing N candidates with their top-3 template matches.

    Each row: [candidate patch | best template | 2nd template | 3rd template]
    Columns show letter name, score, and style used.
    Only HIGH/MEDIUM candidates are shown (LOW skipped by default).
    """
    import matplotlib.pyplot as plt

    eligible = [c for c in candidates if assign_confidence_tier(c) != "LOW" and c.matches]
    rng = np.random.default_rng(seed)
    sample = eligible if len(eligible) <= n else list(rng.choice(eligible, size=n, replace=False))  # type: ignore[arg-type]

    if not sample:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No candidates to display", ha="center", va="center")
        return fig

    n_show = len(sample)
    cols = 4  # candidate + 3 templates
    rows = n_show
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"Template matches — {n_show} sampled HIGH/MEDIUM candidates", fontsize=11)

    for row, cand in enumerate(sample):
        # Column 0: candidate patch
        axes[row, 0].imshow(
            GreekTemplateMatcher._resize_aspect(cand.patch, 48),
            cmap="gray", vmin=0, vmax=1,
        )
        tier = assign_confidence_tier(cand)
        axes[row, 0].set_title(f"{tier}\nprob={cand.mean_prob:.2f}", fontsize=6, color="white" if tier == "HIGH" else "gold")
        axes[row, 0].axis("off")
        axes[row, 0].set_facecolor("black")

        # Columns 1–3: top-3 template matches
        for col, match in enumerate(cand.matches[:3]):
            tpl_arr = matcher._bank[match.best_size][match.best_style][match.char][0]
            axes[row, col + 1].imshow(tpl_arr, cmap="gray", vmin=0, vmax=1)
            axes[row, col + 1].set_title(
                f"{match.char}  {match.score:.2f}\n{match.best_style} {match.best_size}px",
                fontsize=6,
            )
            axes[row, col + 1].axis("off")

    # Header row labels
    for i, lbl in enumerate(["Candidate", "Match #1", "Match #2", "Match #3"]):
        axes[0, i].set_xlabel(lbl, fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def visualize_text_lines(
    prob: np.ndarray,
    lines: List[TextLine],
    *,
    figsize: Tuple[float, float] = (14, 9),
    max_lines: int = 40,
    crop: Optional[Tuple[int, int, int, int]] = None,
) -> "plt.Figure":
    """Probability map with each text line highlighted in a unique colour.

    A coloured horizontal band is drawn behind each detected line. The line
    number is printed on the left edge.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    cmap = plt.get_cmap("tab20")

    if crop is not None:
        y0c, y1c, x0c, x1c = crop
        display = prob[y0c:y1c, x0c:x1c]
        oy, ox = y0c, x0c
    else:
        display = prob
        oy, ox = 0, 0

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"Text line detection — {len(lines)} lines", fontsize=11)
    ax.axis("off")

    shown_lines = lines[:max_lines]
    for line in shown_lines:
        if not line.candidates:
            continue
        ys = [c.centroid[0] for c in line.candidates]
        xs = [c.centroid[1] for c in line.candidates]
        char_h = float(np.median([c.bbox[2] - c.bbox[0] for c in line.candidates]))

        color = cmap(line.line_id % 20)
        band_y0 = min(ys) - char_h * 0.6 - oy
        band_y1 = max(ys) + char_h * 0.6 - oy
        band_h  = max(4.0, band_y1 - band_y0)
        x_left  = min(xs) - ox - 4

        ax.axhspan(band_y0, band_y0 + band_h, alpha=0.12, color=color, lw=0)
        ax.text(
            max(0, x_left), (band_y0 + band_y0 + band_h) / 2,
            str(line.line_id),
            color=color, fontsize=7, va="center", fontweight="bold",
        )

        # Mark each candidate centroid
        for cand in line.candidates:
            ax.plot(cand.centroid[1] - ox, cand.centroid[0] - oy,
                    ".", color=color, markersize=2, alpha=0.7)

    # Legend for first few lines
    patches = [
        mpatches.Patch(color=cmap(i % 20), label=f"Line {i}")
        for i in range(min(10, len(shown_lines)))
    ]
    if len(shown_lines) > 10:
        patches.append(mpatches.Patch(color="white", label=f"…+{len(shown_lines)-10} more"))
    ax.legend(handles=patches, fontsize=7, loc="upper right",
              framealpha=0.6, facecolor="black", labelcolor="white", ncol=2)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# LLM context builder (feeds Stage 3 of the deciphering pipeline)
# ---------------------------------------------------------------------------

def build_llm_line_context(line: TextLine, segment_width: int) -> str:
    """Format one text line as a structured string for the LLM prompt.

    HIGH-confidence candidates show their top-1 match as an anchor.
    MEDIUM candidates show their top-3 matches as hypotheses.
    GAPs between candidates are noted with estimated position.
    """
    entries: List[str] = []
    prev_x: Optional[float] = None
    median_char_w = float(np.median([c.bbox[3] - c.bbox[1] for c in line.candidates])) if line.candidates else 20.0

    for cand in line.candidates:
        pos = cand.centroid[1] / max(segment_width, 1)
        tier = assign_confidence_tier(cand)

        # Insert GAP marker if large horizontal gap before this candidate
        if prev_x is not None:
            gap_px = cand.centroid[1] - prev_x
            if gap_px > 2.5 * median_char_w:
                n_missing = max(1, round(gap_px / median_char_w) - 1)
                gap_pos = (prev_x + cand.centroid[1]) / 2.0 / max(segment_width, 1)
                entries.append(f"  pos {gap_pos:.3f}: GAP  (~{n_missing} char missing)")

        if tier == "HIGH" and cand.matches:
            top = cand.matches[0]
            entries.append(
                f"  pos {pos:.3f}: HIGH → {top.char}  (score {top.score:.2f})"
            )
        elif tier == "MEDIUM" and cand.matches:
            probs = softmax_scores(cand.matches[:3])
            choices = " | ".join(f"{ch} {p:.2f}" for ch, p in probs)
            entries.append(
                f"  pos {pos:.3f}: MED  → {choices}"
            )

        prev_x = cand.centroid[1]

    return "\n".join(entries)
