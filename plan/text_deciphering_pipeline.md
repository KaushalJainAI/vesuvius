# Text Deciphering Pipeline — Filters + LLM + Iterative Label Refinement

## Goal

Convert a raw ink probability map (or current pseudo-label) into high-quality, linguistically coherent Greek text labels through a self-improving loop:

```
probability map / labels
        ↓
  image enhancement (filters)
        ↓
  letter candidate extraction
        ↓
  LLM gap-filling (Greek linguistic priors)
        ↓
  updated label map
        ↓
  retrain model  ←──────────────────────────┐
        ↓                                    │
  new probability map ─────────────────────→┘
```

The same pipeline runs on both:
- **Pseudo-labels** (`ink_labels.tif`) — improve the training signal
- **Model predictions** (`{seg}_prob.npy`) — improve inference output

---

## Stage 0 — Input Normalisation

Accept either input format:
- `float32 [0,1]` probability map → threshold at `t=0.40` to get binary mask, keep float for blend weighting
- `uint8 [0,255]` TIFF label → normalise to `[0,1]` float

Always preserve the **float confidence** at every stage. Binary decisions are only made at the very end (final transcription). Blends use float weights so no information is destroyed mid-loop.

---

## Stage 1 — Image Enhancement Filters

Goal: make stroke boundaries sharper, bridge small gaps, suppress noise — without inventing ink that isn't there.

### 1.1 Contrast normalisation
```
CLAHE(clip_limit=2.0, tile_grid=(8,8))
```
Equates brightness variation across the segment. Faint ink in darker scan regions is brought up to comparable brightness as bright regions.

### 1.2 Morphological stroke repair
```
closing(disk=2)   → bridge 1–2 px gaps within a stroke
opening(disk=1)   → remove isolated noise specks
```
Applied to the binarised mask; result multiplied back onto the float map to avoid inventing probability.

### 1.3 Adaptive thresholding per text line
Standard global threshold loses faint strokes in dim regions. Alternative: compute local mean in a `31×31` window, threshold at `mean + 0.05`. Useful when the scroll surface has uneven illumination.

### 1.4 Skeletonisation (for letter candidate extraction only)
Thin all ink blobs to single-pixel skeletons before template matching. This makes stroke topology (loops, serifs, junctions) more reliable for classification.

### 1.5 Denoising
Apply `median_filter(radius=1)` before CLAHE to suppress salt-and-pepper CT noise without blurring stroke edges.

### Filter order
```
median → CLAHE → threshold → closing → opening → (skeleton for later stage)
```

---

## Stage 2 — Letter Candidate Extraction

### 2.1 Connected component analysis
- Label all connected components in the cleaned binary mask
- Filter by area: `50 px ≤ area ≤ 5000 px` — drops tiny noise and large merged blobs
- Compute per-component: bounding box, centroid, aspect ratio, area, mean float confidence

### 2.2 Confidence classification
Each component gets a confidence tier based on its mean float probability:

| Tier | Mean prob | Interpretation |
|---|---|---|
| **High** | > 0.70 | Likely real ink — use as LLM anchor |
| **Medium** | 0.45–0.70 | Ambiguous — send to LLM for resolution |
| **Low** | < 0.45 | Probably noise — omit from LLM context |

### 2.3 Text line detection
- Sort high-confidence centroids by Y coordinate
- Cluster into lines using 1D DBSCAN on Y (eps = `1.5 × median_char_height`)
- Within each line, sort by X → character sequence

### 2.4 Greek letter template matching

Implemented in `src/letter_candidates.py` via `GreekTemplateMatcher`. Key design:

**Multi-style template bank** — 3 variants per letter × 3 sizes (32/48/64 px) = 216 templates:
- `serif` — clean matplotlib serif rendering
- `uncial` — ancient approximation: thicker strokes (dilated 1 px), uses lunate sigma (Ϲ) for Σ to match Herculaneum scroll style
- `skeleton` — skeletonised + re-dilated to 2 px; stroke-width-agnostic, matched against a stroke-normalised candidate

**Candidate preprocessing:**
- Skeletonise → dilate to uniform 2 px stroke width
- Aspect-preserving resize to each template size, centred on black canvas

**4-metric ensemble** (per letter, best score across all style × size combos):

| Metric | Weight | What it captures |
|---|---|---|
| Bidirectional Chamfer | 0.40 | Broken/partial strokes — robust to gaps |
| NCC | 0.35 | Global intensity correlation |
| IoU | 0.25 | Overlap ratio |
| Hu moments | residual | Rotation/scale-invariant shape descriptor |

**Topology bonus** (+0.06): letters with enclosed loops (Α, Θ, Ο, Ρ, Β, Φ) are rewarded when the candidate Euler number matches. Soft — ancient strokes often break loops.

Output per candidate: top-3 `LetterMatch` objects → `softmax_scores()` → pseudo-probabilities for LLM prompt.

---

## Stage 3 — LLM Gap-Filling

### 3.1 Context window construction

For each text line, build a prompt showing:
- The line's high-confidence letters (anchors) with their spatial positions
- Medium-confidence letter candidates (each with top-3 template guesses)
- Gaps: positions between anchors where no component was detected

```
Line 4, L→R (positions normalised to [0,1]):
  pos 0.03: HIGH  → Π (score 0.91)
  pos 0.09: HIGH  → Ε (score 0.87)
  pos 0.16: MED   → Ρ|Κ|Π (scores 0.61, 0.44, 0.39)   ← ask LLM
  pos 0.22: GAP   → ??? (no component detected)          ← ask LLM
  pos 0.28: HIGH  → Ι (score 0.88)
```

### 3.2 Prompt design

```
You are deciphering ancient Greek text from the Herculaneum scrolls (Epicurean philosophy, ~1st century BC).
The following characters were extracted from CT scan line {n}.
HIGH-confidence letters are reliable anchors.
For MED positions give your best character guess with probability.
For GAP positions infer the missing character from Greek word/phrase priors.

Return JSON: {"positions": [{"pos": float, "char": str, "conf": float, "source": "anchor|llm_med|llm_gap"}]}

Context:
{line_context}
```

Model: `claude-opus-4-7` (largest context, best at ancient Greek).

### 3.3 LLM output → confidence update

The LLM returns per-position `(char, conf)` pairs. Convert to pixel-level label updates:
- **Anchor positions**: keep existing label unchanged
- **LLM-resolved MED**: set component pixels to `min(0.90, llm_conf)` in the float map
- **LLM-inferred GAP**: paint a 24×24 blob at predicted position with value `llm_conf × 0.75` (discounted because no visual evidence)

Gap positions are soft — they nudge the label map but don't override strong zeros.

### 3.4 Batching
Lines are processed in batches of 5 per API call to reduce latency. Each call costs ~500 input tokens + ~200 output tokens. A full segment (~500 lines) = ~100 API calls.

---

## Stage 4 — Label Map Update

Blend the LLM-updated map back with the original label:

```
updated = α × llm_refined_map + (1 - α) × original_map
```

| Iteration | α (blend weight toward LLM output) |
|---|---|
| 1 | 0.30 — conservative, trust original labels mostly |
| 2 | 0.40 |
| 3 | 0.50 |
| 4+ | 0.50 — cap here; LLM is complementary, not the ground truth |

The ignore band `[0.40, 0.55]` in the training loss still applies — LLM output in the ambiguous range won't be forced to one side.

Save updated map as:
- `{seg}_labels_v{n}.tif` — uint8, for inspection
- `{seg}_labels_v{n}.npy` — float32, for training

---

## Stage 5 — Iterative Training Loop

```
LOOP:
  1. Start with labels_v{n} (or probability map from last model run)
  2. Run filter pipeline (Stage 1)
  3. Extract letter candidates (Stage 2)
  4. LLM gap-fill (Stage 3)
  5. Blend → labels_v{n+1} (Stage 4)
  6. [Optional] Retrain SegmentInkNet on labels_v{n+1}
  7. Run inference → new probability map
  8. Evaluate: compute label diff stats, manually inspect 3 random lines
  9. If stopping criterion met → exit. Else n += 1, go to 1.
```

### 5.1 Two modes of the loop

**Label-only mode** (no retraining, fast):
- Skips step 6–7
- Each iteration is filters + LLM only (~1–2 hrs per segment)
- Useful for refining labels before investing GPU time

**Full loop mode** (retraining included):
- Completes all 9 steps
- Each iteration = LLM refinement + full training run (~8–12 hrs)
- Use after label-only mode has converged to a good starting point

### 5.2 Stopping criteria

Stop when ANY of these is met:
- Mean absolute diff between `labels_v{n}` and `labels_v{n-1}` < `0.01` (labels no longer changing)
- F0.5 on val segment does not improve by > 0.5% after retraining
- 4 iterations completed (label-only) or 3 retraining cycles
- Manual inspection shows no new letter recoveries

### 5.3 Loop bookkeeping

Track per-iteration:
```json
{
  "iteration": 2,
  "segment": "20231221180251",
  "mode": "full_loop",
  "label_diff_mean": 0.023,
  "label_diff_max": 0.81,
  "pixels_changed_pct": 4.2,
  "new_letters_added": 142,
  "llm_gaps_filled": 87,
  "val_f05": 0.712,
  "model_checkpoint": "models/best_segment_model_v2.pth"
}
```
Saved to `models/refinement_log.json`.

---

## Stage 6 — Final Transcription

After the loop converges:
1. Run final letter extraction on the converged label map
2. Sort all letters by (line, x_position)
3. Map character centroids → unicode Greek text
4. Assign per-character confidence (combined filter score + LLM confidence)
5. Output:
   - `predictions/{seg}_transcription.txt` — plain text, one line per scroll line
   - `predictions/{seg}_transcription.json` — structured, with per-char confidence + bounding box

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────┐
│                     INPUT                                   │
│   ink_labels.tif  OR  {seg}_prob.npy (model output)        │
└─────────────────────────┬──────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1 — FILTER PIPELINE                                  │
│  median → CLAHE → threshold → closing → opening             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2 — LETTER CANDIDATES                                │
│  CC analysis → confidence tiers → line clustering           │
│  → Greek template matching (top-3 per component)            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3 — LLM GAP-FILLING (claude-opus-4-7)                │
│  anchors + ambiguous + gaps → JSON char predictions          │
│  → pixel-level confidence updates                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4 — LABEL BLEND                                      │
│  α × llm_map + (1-α) × original → labels_v{n+1}.tif/.npy   │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┴──────────────┐
            │                            │
            ▼                            ▼
  [label-only mode]            [full loop mode]
  → next iteration              → retrain model
                                → new prob map
                                → next iteration
```

---

## Implementation Plan

| Step | File | Description | Effort |
|---|---|---|---|
| 1 | `src/label_filters.py` | Filter pipeline (Stage 1) | Low |
| 2 | `src/letter_candidates.py` | CC extraction + line detection + template matching (Stage 2) — **done** | Medium |
| 3 | `src/llm_gap_fill.py` | Prompt builder + Claude API calls + pixel update (Stage 3) | Medium |
| 4 | `src/label_blend.py` | Blend + save + bookkeeping (Stage 4) | Low |
| 5 | `src/refine_loop.py` | Orchestrator — runs stages 1–4 in sequence, loops | Medium |
| 6 | `notebooks/label_refinement.ipynb` | Interactive version: run one iteration, inspect diff | Medium |

### Dependencies
- `scikit-image` — CLAHE, morphology, CC analysis, skeletonisation
- `anthropic` — Claude API (already in requirements.txt)
- `scipy` — DBSCAN for line clustering
- `tifffile` — label I/O

---

## Key Design Decisions & Trade-offs

| Decision | Choice | Alternative | Reason |
|---|---|---|---|
| Blend weight cap | α ≤ 0.50 | Higher α | LLM is complementary, not ground truth; over-trusting LLM creates hallucination artifacts in labels |
| Gap fill discounting | × 0.75 | Full LLM confidence | No visual evidence for gaps — should not override strong model zeros |
| Template matching first | Yes — top-3 as LLM prior | LLM only | Gives LLM grounded hypotheses, reduces hallucination, saves tokens |
| Ignore band in loss | Keep [0.40, 0.55] | Widen/remove | LLM-refined labels are still noisy; ignore band prevents confident fitting of uncertain regions |
| Letter-level context window | Per-line | Per-page | Line context is sufficient for Greek word structure; page context would exceed token budget |
| Model: claude-opus-4-7 | Yes | claude-haiku-4-5 | Ancient Greek linguistic reasoning benefits from largest model; cost amortised over ~100 calls/segment |

---

## Expected Gains

| Stage | What it fixes | Expected label quality gain |
|---|---|---|
| Filters alone | Noise, broken strokes, low contrast | +5–10% pixel accuracy vs raw labels |
| + Template matching | Misclassified components (wrong letter shape) | Reduces ~30% of top-1 errors |
| + LLM gap-filling | Gaps where visual evidence is absent | Recovers 15–30% of gap characters |
| + Retraining on refined labels | Model learns from better signal | +4–8% F0.5 (see model_improvement.md §3.1) |
| + 2nd loop iteration | Model now generates better prob maps as starting point | Compounding gains, diminishing after iteration 3 |

---

## Relationship to model_improvement.md

This pipeline directly implements **§3.1 Iterative pseudo-label refinement** from the model improvement plan, with the addition of the LLM gap-filling stage (originally §4 / Stage 3 of the 4-stage pipeline). The two plans are complementary:

- `model_improvement.md` → improve the **model** that generates probability maps
- `text_deciphering_pipeline.md` → improve the **labels/output** through filters + LLM

Both feed each other: better labels → better model → better probability maps → better label refinement starting point.
