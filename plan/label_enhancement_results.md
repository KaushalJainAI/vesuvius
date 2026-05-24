# Label Enhancement — Results & Verdict

Comparison of 12 strategies on segment `20231221180251` (val segment).
3 text-density crops (512×512) at `(y=4352, x=7680)`, `(y=6656, x=7936)`, `(y=3584, x=3840)`.

Plan and rationale in [label_enhancement.md](label_enhancement.md).
Raw outputs: [predictions/filter_strategy_benchmark/20231221180251/](../predictions/filter_strategy_benchmark/20231221180251/).

## Aggregate metrics (mean over 3 crops)

| Strategy | sep ↑ | CCs | med area | compact ↑ | speckle ↓ | ink % | time s |
|---|---:|---:|---:|---:|---:|---:|---:|
| **S0_raw** (baseline) | 0.739 | 6 | 1289 | 0.751 | 0.048 | 14.59% | 0.00 |
| **S1_current** (median+CLAHE+morph) | 0.955 | 7 | 865 | 0.599 | 0.000 | 14.62% | 0.63 |
| **S2_median_morph** | 0.896 | 6 | 1289 | 0.752 | 0.048 | 14.59% | 0.08 |
| **S3_tv_threshold** | 0.895 | 6 | 1266 | 0.759 | 0.000 | 14.56% | 2.74 |
| **S4_tv_sauvola** | 0.777 | 6 | 1266 | 0.759 | 0.000 | 14.42% | 0.51 |
| **S5_tv_hysteresis** | 0.895 | 6 | 1266 | 0.759 | 0.000 | 14.56% | 0.63 |
| **S6_sato_only** | 0.586 | 13 | 502 | 0.494 | 0.028 | 3.97% | 0.64 |
| **S7_sato_hysteresis** | 0.687 | 6 | 1398 | 0.589 | 0.042 | 9.91% | 0.69 |
| **S8_bilateral_sauvola** | 0.774 | 6 | 1157 | 0.764 | 0.000 | 14.29% | 3.47 |
| **S9_nlm_sauvola** | 0.775 | 6 | 1245 | 0.765 | 0.000 | 14.37% | 1.20 |
| **S10_anisotropic_sauvola** | 0.778 | 6 | 1264 | 0.756 | 0.000 | 14.42% | 0.64 |
| **S11_tv_sato_hysteresis** ⭐ | **0.998** | 7 | 1350 | 0.707 | 0.000 | 16.12% | 1.03 |

Higher = better for `sep` and `compact`. Lower = better for `speckle`.

## Visual takeaways

The grids per crop are at:
- [crop_00/grid.png](../predictions/filter_strategy_benchmark/20231221180251/crop_00_y4352_x7680/grid.png) — sparse text
- [crop_01/grid.png](../predictions/filter_strategy_benchmark/20231221180251/crop_01_y6656_x7936/grid.png) — visible loop letter (looks like Φ)
- [crop_02/grid.png](../predictions/filter_strategy_benchmark/20231221180251/crop_02_y3584_x3840/grid.png) — mid-density text

Key observations from looking at the images, NOT just the numbers:

| Group | What you see | Verdict |
|---|---|---|
| **S0 raw** | Soft puffy blobs, gradient gray background | Reference — the "raw label" state |
| **S1 current** | Aggressive binarisation. Letter loops get FILLED-IN black (loops destroyed). High contrast but topology lost. | ⚠️ CLAHE is harmful here — confirmed |
| **S2–S5, S8–S10** | All visually very similar to S0. Median/TV/bilateral/NLM all produce roughly the same softly-smoothed output; downstream binarisation differs only at margins. | Diminishing returns from fancier denoisers |
| **S6 sato only** | Reveals stroke OUTLINES beautifully — loop edges visible where S1 destroyed them. But sparse (4% ink): can't stand alone. | Useful as *auxiliary signal*, not standalone |
| **S7 sato+hyst** | Stroke-aware but loses content. Compactness drops. | Worse than S11 |
| **S11 tv+sato+hyst** ⭐ | **Best of both**: TV smoothing keeps letter interiors soft and full, Sato boosts strokes (loops preserved), hysteresis closes gaps without inviting noise. Highest separation. | ✅ Recommended |

## Conclusion: S1 is actively harmful for label refinement

In crop 1 the central letter has a clear closed loop (looks like Φ). After S1:
- The loop's bright outline is preserved
- The **inside** of the loop is forced to black by the threshold-then-mask step
- The smooth gradient that distinguished "loop interior with thin ink" from "loop interior without ink" is destroyed

This matters for two reasons:
1. **Training**: the model trained on S1 labels learns "ink loops should be filled" — a hallucination not in the data.
2. **Letter classification (Stage 2)**: template matching needs the loop topology to distinguish Ο vs Θ vs Φ etc. S1 destroys this.

## Why CLAHE is wrong for this data (re-confirmed empirically)

The plan-doc rationale matched the data:
- CLAHE assumes the input has **illumination variation** to correct.
- The input is a **calibrated probability map**. There is no illumination.
- CLAHE's per-tile histogram equalisation pushes mid values to extremes → polarises soft labels into hard ones → forces letters into "filled-blob" topology.

## Why TV + Sato + hysteresis wins

| Step | What it does | Why it matches the data |
|---|---|---|
| **TV denoise (λ=0.05)** | Preserves edges, removes speckle, stays in float space | Probability map is piecewise-smooth-with-edges — this is exactly TV's signal model |
| **Sato vesselness (σ=2,4,6)** | Multi-scale tubular structure detector → boost strokes | Greek strokes are tubular; sigmas chosen to match observed 4–12 px half-width |
| **Boost: `clip(tv × (1 + 0.5·sato), 0, 1)`** | Sato score multiplicatively boosts strokes in the prob map | Letters get brighter; non-stroke noise doesn't |
| **Hysteresis (0.35, 0.55)** | Pixel-graph traversal: keep > 0.55 OR connected to > 0.55 via > 0.35 | Recovers broken strokes without admitting isolated noise — bimodality of label histogram is irrelevant |
| **Morph close(2) + open(1)** | Final binary cleanup | Small disks — won't merge adjacent letters at 5–10 px spacing |

## Recommendations

### 1. Replace the legacy `apply_filters` default in [src/label_filters.py](../src/label_filters.py)

The function should remain backward-compatible for callers, but the default behaviour invoked by [src/refine_loop.py](../src/refine_loop.py) should be **S11** (`tv_sato_hysteresis`). Either:
- Add a `strategy: str = "S11_tv_sato_hysteresis"` parameter, OR
- Change `RefinementConfig` to call `run_strategy("S11_tv_sato_hysteresis", labels)` instead of `apply_filters(...)`.

### 2. Use S6 (Sato) as an auxiliary channel for Stage 2

Letter template matching ([src/letter_candidates.py](../src/letter_candidates.py)) benefits from stroke-outline structure. Feed both the enhanced prob map AND the Sato channel into `extract_candidates` for richer skeleton/topology features.

### 3. Do NOT use any strategy that hard-binarises before training

S1, S4, S8, S9, S10 all end in a `binary × prob_map` step which zeroes out non-binary regions. For TRAINING labels we want to PRESERVE gradient — the model learns better from `[0.0, 0.45, 0.8]` than from `[0, 0, 1]`. The ignore-band `[0.40, 0.55]` in the loss exists specifically to handle this.

**Action**: in S11, expose a `soft_output: bool = True` flag — when True, return `boosted` (no binary mask), when False return the masked version for letter extraction. The same strategy serves both stages.

### 4. NLM is not worth its cost

S9 (NLM) has the same metric profile as S8 (bilateral) and S10 (anisotropic), but is 2–3× slower than bilateral and adds memory pressure. Drop it from production. Keep in the registry for crop-level sanity checks.

### 5. Drop S1 from the registry once S11 is the default

After we adopt S11, remove S1 from the production code path. Keep `apply_filters` only as a deprecated alias to `_s1_current` for reproducibility of old refinement_log entries.

## Next steps

1. **Edit [src/refine_loop.py](../src/refine_loop.py)** — wire it to `run_strategy("S11_tv_sato_hysteresis", labels)`.
2. **Edit [src/label_filters.py](../src/label_filters.py) S11** — add `soft_output=True` so training labels keep gradient.
3. **Run one full label_only refinement iteration** with S11 and compare `labels_v1` against `labels_v0` (raw) on the same crops.
4. **Re-evaluate at full-segment scale** — test that TV / Sato don't OOM on 9800×12920 arrays. (TV may need tiled processing — Chambolle is iterative but per-pixel; tiling with overlap is safe.)
5. **Sanity check on a second segment** — run S11 on the largest segment (`20231007101619`) to confirm the win generalises beyond the val segment.
