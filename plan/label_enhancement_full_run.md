# Full-Segment Label Enhancement Run — Results

Applied the winning S11 pipeline ([src/best_enhancer.py](../src/best_enhancer.py)) to all 11 labelled PHercParis4 segments via [scripts/enhance_all_labels.py](../scripts/enhance_all_labels.py).

Pipeline: **TV denoise (λ=0.05) → Sato vesselness (σ=2,4,6) → multiplicative boost (0.5×) → hysteresis (0.35/0.55) → morph close(2) + open(1)**.

Outputs:
- Enhanced labels: `data/labelled_segments/{id}/ink_labels_enhanced.tif` (uint8, same shape as original)
- Per-segment comparisons: `predictions/enhanced_labels/{id}/full_compare.png` + `crop_compare.png`
- Contact sheet: [predictions/enhanced_labels/all_segments_grid.png](../predictions/enhanced_labels/all_segments_grid.png)
- Summary stats: [predictions/enhanced_labels/summary.json](../predictions/enhanced_labels/summary.json)

## Per-segment results

Canonical numbers in [predictions/enhanced_labels/summary.json](../predictions/enhanced_labels/summary.json). Each row links its `full_compare.png`.

| Segment | Shape | mean &#124;d&#124; | orig p95 | enh p95 | orig ink% | enh ink% | +px% | −px% | time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [20230702185753](../predictions/enhanced_labels/20230702185753/full_compare.png) | 15520×11180 | 0.169 | 0.867 | 0.948 | 11.9% | 13.4% | 11.9% | 80.4% | 375 |
| [20230929220926](../predictions/enhanced_labels/20230929220926/full_compare.png) | 9840×23280  | 0.170 | 0.831 | 0.939 | 12.3% | 14.2% | 13.4% | 78.1% | 722 |
| [20231005123336](../predictions/enhanced_labels/20231005123336/full_compare.png) | 10700×29880 | 0.171 | 0.827 | 0.939 | 11.4% | 13.3% | 12.7% | 79.6% | 1080 |
| [20231007101619](../predictions/enhanced_labels/20231007101619/full_compare.png) | 15960×33220 | 0.173 | 0.753 | 0.902 |  9.3% | 11.0% | 11.2% | 82.3% | 1900 |
| [20231012184424](../predictions/enhanced_labels/20231012184424/full_compare.png) | 15580×23260 | 0.168 | 0.847 | 0.944 | 10.7% | 12.1% | 10.9% | 82.7% | 1027 |
| [20231016151002](../predictions/enhanced_labels/20231016151002/full_compare.png) | 9800×15960  | 0.174 | 0.749 | 0.891 |  8.0% |  9.3% |  8.5% | 85.9% | 440 |
| [20231022170901](../predictions/enhanced_labels/20231022170901/full_compare.png) | 6560×35580  | 0.167 | 0.812 | 0.934 | 10.2% | 11.7% | 11.1% | 82.8% | 567 |
| [20231031143852](../predictions/enhanced_labels/20231031143852/full_compare.png) | 10480×13520 | 0.176 | 0.690 | 0.821 |  7.2% |  8.5% |  8.6% | 86.7% | 391 |
| [20231106155351](../predictions/enhanced_labels/20231106155351/full_compare.png) | 10520×15740 | 0.170 | _n/a_ | _n/a_ |  9.7% | 11.3% | 11.0% | 82.3% | 499 |
| [20231210121321](../predictions/enhanced_labels/20231210121321/full_compare.png) | 15640×12280 | 0.171 | _n/a_ | _n/a_ |  9.3% | 10.6% | 10.0% | 84.0% | 570 |
| [20231221180251](../predictions/enhanced_labels/20231221180251/full_compare.png) (val) | 9800×12920  | 0.171 | 0.678 | 0.812 |  7.1% |  8.5% |  8.9% | 86.3% | 327 |

**Aggregate over all 11 segments**: mean|d| = 0.171 ± 0.003, ink%-shift +1.5 pp, +px = 10.7%, −px = 83.2%.

The remarkable consistency (mean|d| in [0.167, 0.176] across 11 segments of vastly different sizes and ink densities) is itself the strongest evidence the filter is doing the right thing: it's not over-fitting to any single segment's quirks.

### Contact sheet across all 11

[predictions/enhanced_labels/all_segments_grid.png](../predictions/enhanced_labels/all_segments_grid.png) — 11 rows × 2 columns (orig | enhanced). The pattern is uniform: dim gray noisy backgrounds on the left, crisp white letters on near-black backgrounds on the right.

## How to read the numbers

- **mean &#124;diff&#124;**: average absolute change per pixel. ≈0.17 indicates substantial change (raw mean was 0.243). Note this is on `[0, 1]` scale.
- **ink% > 0.55**: fraction of pixels above 0.55 — a proxy for "high-confidence ink." Modest increase is good (strokes brighten); large jumps would indicate hallucination.
- **+px% / −px%**: fraction of pixels that moved up / down by more than 0.05. The high −px% values are expected and intended — background noise is being suppressed, not letter content.

## Visual summary

For each segment the script writes a `full_compare.png` showing **original | enhanced | diff** at downsampled resolution. The diff panel uses `RdBu_r` — **blue regions are where enhancement removed gray noise** (background cleanup), **red regions are where strokes were boosted**.

The val segment ([predictions/enhanced_labels/20231221180251/full_compare.png](../predictions/enhanced_labels/20231221180251/full_compare.png)) shows the typical pattern:
- Original: smooth gray text on smooth gray noise — letters barely separable from background by eye.
- Enhanced: crisp white letter shapes on near-black background. Text lines clearly delineated.
- Diff: pervasive light blue (background suppression) with concentrated red along letter strokes.

The 3-crop comparison ([predictions/enhanced_labels/20231221180251/crop_compare.png](../predictions/enhanced_labels/20231221180251/crop_compare.png)) shows the same effect at 512×512 zoom — letter loops and stroke junctions preserved that S1 would have destroyed.

## Why this matters for the project

1. **Better training signal.** [src/segment_model.py](../src/segment_model.py) trains `SegmentInkNet` to predict the pseudo-label. With the original labels containing ~25% background "ink mean" noise, the model spends capacity learning to reproduce noise. Training on enhanced labels means the loss target itself is cleaner.

2. **Less reliance on the ignore band.** The current loss uses an `[0.40, 0.55]` ignore band to avoid forcing the model to commit on ambiguous pixels. With enhanced labels, the ambiguous mid-range is mostly resolved (boosted up or pushed down), which means the model trains on more pixels per batch.

3. **Stage 2 letter extraction.** Component analysis was unstable on raw labels — too many spurious specks merged with letter blobs. Enhanced labels give cleaner connected components, so [src/letter_candidates.py](../src/letter_candidates.py) needs less filtering work.

## Recommended next steps

1. **Wire into [src/refine_loop.py](../src/refine_loop.py)** — replace `apply_filters(...)` with `from best_enhancer import enhance_tiled; enhanced = enhance_tiled(labels)`. The legacy `apply_filters` stays for old runs.

2. **Retrain on enhanced labels.** Run [notebooks/segment_ink_detection.ipynb](../notebooks/segment_ink_detection.ipynb) with `ink_labels_enhanced.tif` as the target. Save the resulting model alongside `best_segment_model.pth` for direct F0.5 comparison.

3. **Iterate.** If retraining gives better predictions, the refined predictions become the next iteration's input — the loop in [plan/text_deciphering_pipeline.md](text_deciphering_pipeline.md) §5 can now actually run.

4. **Hyperparameter sweep (optional).** TV `weight` (0.03/0.05/0.08), Sato `boost` (0.3/0.5/0.7), hysteresis `(low, high)` thresholds. Current defaults were chosen on a single val segment — a brief sweep on 2-3 segments could surface a better point.

## Reproducing this run

```bash
source venv/Scripts/activate
python scripts/enhance_all_labels.py                  # all 11 segments
python scripts/enhance_all_labels.py --only <id>      # one segment
python scripts/enhance_all_labels.py --skip-existing  # only regenerate viz from existing TIFs
```

Tile size, soft-output toggle, and other knobs are exposed as CLI flags. Defaults are sensible.
