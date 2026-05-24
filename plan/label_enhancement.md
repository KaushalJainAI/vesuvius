# Label Enhancement — Strategy Plan

## What we're enhancing

`data/labelled_segments/{seg}/ink_labels.tif` — float32 ink probability maps produced by the GP-winner `tile64-stride16` model. Visualised in [predictions/raw_label_inspection/20231221180251/](../predictions/raw_label_inspection/20231221180251/).

**Observed properties of the raw labels (val segment 20231221180251):**

| Property | Value | Implication |
|---|---|---|
| Shape | 9800 × 12920 float32 | Need tiled processing or downsampled previews |
| Range | min 0.000, max 0.929, mean 0.243 | No `1.0` saturation, no clean `0.0` background |
| Distribution | Long-tailed, NOT bimodal | No natural global threshold exists |
| Ignore band 0.40–0.55 | Densely populated | Cannot be ignored — many real letter edges live here |
| Spatial coherence | Bottom rows near-legible, top/sides pure noise | Single global threshold cannot serve both |
| Letter morphology | Puffy blobs, stroke width ~10–15 px, letter height ~50–80 px | Filters must work at the stroke scale, not the pixel scale |
| Effective resolution | Coarser than CT (model used 64-px tiles) | Sharpening can't recover what the model didn't localise |

## Why the current `apply_filters` is suspect

[src/label_filters.py](../src/label_filters.py) currently does **median → CLAHE → blend → threshold → closing → opening → mask**. Two of these are theoretically wrong for THIS data:

1. **CLAHE on a probability map** — CLAHE corrects illumination variation in photos. A probability map has no illumination — values already are calibrated probabilities. CLAHE redistributes histogram bins per tile, which inflates noise in low-signal regions (top of segment) up to the brightness of high-signal regions (bottom). This is exactly backwards: we want to SUPPRESS the noisy top regions, not amplify them.
2. **Global threshold** at the end (`threshold=0.40` default) — the histogram has no bimodality, so any global value either drowns letters in speckle (low) or kills faint regions entirely (high). Sauvola or hysteresis is matched to this data.

The other steps (median radius 1, closing disk=2, opening disk=1) are fine and we keep them.

## Strategies to evaluate

Each strategy operates on the same float prob map in `[0, 1]` and outputs an enhanced float prob map in `[0, 1]`. We compare them on a fixed set of crops from the val segment so visual differences are directly comparable.

| ID | Strategy | Rationale |
|---|---|---|
| **S0** | Raw (no filter) | Baseline |
| **S1** | Current pipeline (median + CLAHE + morphology) | What's in `label_filters.py` now |
| **S2** | Median + morphology only (drop CLAHE) | Test whether CLAHE is harmful |
| **S3** | TV denoising + global threshold + morphology | TV preserves edges while killing speckle, designed for piecewise-smooth-with-edges signals |
| **S4** | TV + Sauvola binarization | Sauvola is the DIBCO-benchmark adaptive threshold for degraded documents |
| **S5** | TV + hysteresis (0.35, 0.55) + morphology | Hysteresis recovers broken strokes without admitting noise |
| **S6** | Sato vesselness (σ=2,4,6) → multiply | Vesselness boosts stroke-like tubular structures, suppresses blobs |
| **S7** | Sato + hysteresis + morphology | Combines structure-aware boost with hysteresis recovery |
| **S8** | Bilateral filter + Sauvola | Edge-preserving smoothing; cheaper than NLM |
| **S9** | Non-local means + Sauvola (small crops only — NLM is slow on full image) | Exploits self-similarity of strokes; SOTA for natural-image denoising |
| **S10** | Anisotropic (Perona–Malik) diffusion + Sauvola | Smooth along strokes, preserve cross-stroke edges |
| **S11** | TV → Sato → hysteresis → morphology (proposed full) | Combines all the strong ideas |

## Comparison metrics

For each strategy on the val segment we compute:

| Metric | What it measures | Better direction |
|---|---|---|
| **Letter-vs-background separation** | mean(top 5% pixels) − mean(bottom 50% pixels) | Higher = cleaner separation |
| **CC count** (after binarize at strategy-appropriate threshold) | Number of connected components in 50–5000 px range | Closer to expected letter count (~hundreds per crop) |
| **Mean CC size** | Median connected component area in px | Should be 100–600 px for individual letters |
| **CC compactness** | 4πA / P² | Higher = more letter-like (compact), lower = stringy noise |
| **Skeleton fragmentation** | mean stroke segment length after skeletonization | Higher = better-connected strokes |
| **Histogram bimodality** | KL-divergence of resulting histogram from a fitted 2-Gaussian | Higher = clearer ink/no-ink separation |
| **Speckle index** | Fraction of CCs with area < 30 px | Lower = less noise speckle |

Plus qualitative visual judgement — letters should be visible and distinguishable from background to a human reader.

## What we will NOT try (yet)

- **Deep-learning denoisers** (DocDiff, DE-GAN, DocEnTr) — training-cost prohibitive, only justified if classical methods saturate
- **Wiener deconvolution** — would require modelling the GP-winner's effective PSF, possible but high-effort
- **MRF / graph-cut binarization** — only justified if Sauvola + hysteresis underperforms
- **Sliding-window per-region thresholding by ML model** — out of scope until classical methods are exhausted

## Implementation plan

1. Extend [src/label_filters.py](../src/label_filters.py): add `tv_denoise`, `sato_vesselness`, `sauvola_binarize`, `hysteresis_threshold`, `bilateral_smooth`, `anisotropic_diffusion` helpers + a `STRATEGIES` registry mapping the IDs above to callables. **Keep `apply_filters` backward-compatible.**
2. Write [scripts/compare_label_filters.py](../scripts/compare_label_filters.py) that:
   - Loads val segment ink_labels.tif
   - Picks 3 dense crops (same coords as `predictions/raw_label_inspection/`)
   - For each strategy: runs it, saves crop PNGs, computes metrics
   - Writes side-by-side PNG grids per crop
   - Writes `results.json` with per-strategy metrics
3. Run on segment 20231221180251.
4. Inspect outputs visually + numerically.
5. Write `plan/label_enhancement_results.md` with verdicts and the recommended pipeline.

## Success criterion

A clear winner emerges on at least 2 of 3 metrics (separation, CC compactness, speckle index) AND visually-cleaner letters than the current S1. If multiple strategies tie, pick the simpler one.
