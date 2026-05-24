# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Build a generalizable pipeline to decipher ancient Greek from CT data of **any** Herculaneum scroll — including undeciphered segments with no prior labels. PHercParis4 (Scroll 1) is the development target. Full plan: [docs/objective.md](docs/objective.md).

**4-stage pipeline:**
1. **Ink detection** — 3D CNN trained on labelled segments → per-pixel ink probability map
2. **Letter isolation** — threshold + morphology + connected components → high-confidence letter candidates + low-confidence candidates
3. **LLM gap-filling** — Claude receives high-confidence anchors + ambiguous patches → infers missing characters using Greek linguistic priors (skipped for now)
4. **Final transcription** — reconstruct line order from letter centroids → structured text with per-character confidence scores

Two separate training tracks:
- **Fragment track** (`notebooks/fragment_ink_detection.ipynb`) — trained on Kaggle fragments with manual ink labels (65-layer Z-stacks)
- **Segment track** (`notebooks/segment_ink_detection.ipynb`) — trained on up to 11 PHercParis4 segments with pseudo-labels (33-layer Z-stacks); uses all 11 locally, 3 on Kaggle
- **Kaggle-only segment track** (`notebooks/kaggle_segment_train.ipynb`) — self-contained version of the segment track for Kaggle (no `src/segment_model.py` needed); downloads 4 smallest segments, time-budgeted at 7.5 h

## Environment

```bash
# Activate venv before running anything
source venv/Scripts/activate          # Windows Git Bash / bash
# or
venv\Scripts\activate.bat             # Windows cmd

pip install -r requirements.txt
```

PyTorch CUDA install (if GPU available):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Widgets rendering in VS Code requires `ipywidgets` (already in requirements.txt). If widgets fail to render, restart the kernel after install.

## Key commands

```bash
# --- Fragment track (Kaggle labelled fragments, 65-layer) ---
# Download the 7 unlabelled scroll segments (~15 GB)
python scripts/downloading/download_segments.py

# Run fragment-trained notebook
jupyter notebook notebooks/fragment_ink_detection.ipynb

# Explore raw segment data interactively
jupyter notebook notebooks/visualize_segments.ipynb   # 3D viewer, MIPs, orthogonal projections
jupyter notebook notebooks/visualize_data.ipynb       # Z-stack browser, depth profiles, patch sampling

# --- Segment track (11 labelled PHercParis4 segments, 33-layer) ---
# Labels are already in data/labelled_segments/{id}/ink_labels.tif
# Download surface volumes for all 11 segments (~72 GB total):
python scripts/downloading/download_labelled_segment.py --all
# Or pick specific segments (sorted smallest → largest):
python scripts/downloading/download_labelled_segment.py --seg 20231221180251 20231031143852 20231016151002

# Run segment-trained notebook (also works on Kaggle — first cell auto-downloads)
jupyter notebook notebooks/segment_ink_detection.ipynb

# Standalone Kaggle version (no segment_model.py required, 4 segments, 7.5 h budget):
jupyter notebook notebooks/kaggle_segment_train.ipynb

# After inference, experiment with Greek letter template matching:
python src/experiment_filters.py --pred predictions/{segment_id}_prob.npy
```

## Notebook architecture

### `notebooks/fragment_ink_detection.ipynb` — Fragment track (65-layer inputs)

All hyperparameters live in the `CFG` dict in cell 1:

```python
CFG = dict(
    # Paths
    train_dir   = Path('data/train'),
    scroll_dir  = Path('data/scroll'),
    model_dir   = Path('models'),
    pred_dir    = Path('predictions'),

    # Surface volume
    z_start     = 0,
    z_end       = 65,

    # Patch extraction
    patch_size  = 224,
    stride      = 112,   # 50% overlap

    # Training
    batch_size  = 8,
    num_epochs  = 15,
    lr          = 1e-4,
    val_fragment = 'fragment1',
    seed        = 42,

    # Inference
    threshold   = 0.45,
    min_cc_size = 50,

    # Scroll segments for download/inference (~15 GB, 7 segments)
    scroll_segments = [
        '20230827161847',   # 5.73 GB — Grand Prize
        '20230826170124',   # 2.35 GB
        '20230820174948',   # 1.87 GB
        '20230826135043',   # 1.55 GB
        '20230828154913',   # 1.40 GB
        '20230819093803',   # 1.15 GB
        '20230504093154',   # 1.03 GB
    ],
)
```

| Cells | Purpose |
|---|---|
| 0–1 | Imports, `CFG` dict |
| 2 | `load_surface_volume`, `load_mask`, `load_labels`, `extract_patches` |
| 3 | `VesuviusDataset` + albumentations augmentation |
| 4 | `VesuviusNet` — 3D encoder → 2D decoder |
| 5 | `bce_dice_loss`, `fbeta_score` (F0.5) |
| 6 | DataLoaders |
| 7 | Training loop (AdamW + CosineAnnealingLR + AMP) → `models/best_model.pth` |
| 8 | Training curves |
| 9 | Sliding-window inference → `{id}_prob.png`, `{id}_ink.png` |
| 10 | Visualise probability + binarised mask |
| 11 | Optional: Claude API transcription |
| 12 | Optional: download pretrained weights |

### `segment_ink_detection.ipynb` — Segment track (33-layer inputs, Kaggle-compatible)

Uses `segment_model.py` for all model/dataset logic. Hyperparameters live in `CFG` inside `segment_model.py` (`patch_size=256`, `patches_per_seg=400`, `batch_size=4`, `grad_accum=4`, `num_epochs=8`). On Kaggle uses 3 smallest segments (~11 GB); locally uses all 11.

| Section | Purpose |
|---|---|
| 1 | Auto-detect Kaggle vs local; download missing segments from S3 |
| 2 | Imports + CFG overrides (sets `val_segment` to smallest) |
| 3 | Build `SegmentStreamDataset` (one segment in RAM at a time) |
| 4 | `SegmentInkNet` + AdamW + CosineAnnealingLR |
| 5 | Training loop (AMP + gradient accumulation) → `models/best_segment_model.pth` |
| 6 | Training curves |
| 7 | Tiled sliding-window inference → `predictions/{seg}_prob.npy`, `.tif` |
| 8 | Visualization: CT / prediction / label / overlay / diff |

### `kaggle_segment_train.ipynb` — Self-contained Kaggle segment training

Fully self-contained (no `segment_model.py` needed). Designed for Kaggle GPU sessions with a hard 7.5-hour training budget. Downloads 4 smallest segments (~15.7 GB). Saves checkpoints every epoch.

| Section | Purpose |
|---|---|
| 1 | Paths, CFG (`patch_size=256`, `patches_per_seg=800`, `num_epochs=60`, `max_train_hours=7.5`) |
| 2 | Download 4 smallest segments from S3 |
| 3 | Inline `SegmentInkNet` + `SegmentStreamDataset` + loss definitions |
| 4 | Time-budgeted training → `best_segment_model.pth`, `last_segment_model.pth`, `training_history.json` |
| 5 | Training curves (loss + F0.5 + LR) → `models/training_curves.png` |
| 6 | Inference on val segment → `predictions/{seg}_prob.npy`, `.tif` |
| 7 | Visualization: CT / prediction / label / overlay / diff → `predictions/{seg}_vis.png` |

## Models

### `VesuviusNet` (fragment track, 65-layer)
```
Input (B, 65, H, W)
→ enc3d_1: Conv3d(1→32)×2 + MaxPool3d(2,1,1)      # Z: 65→32
→ enc3d_2: Conv3d(32→64)×2 + MaxPool3d(2,1,1)     # Z: 32→16
→ enc3d_3: Conv3d(64→128) + AdaptiveAvgPool3d(1,*) # Z→1
→ dec2d: Conv2d 128→64→32→16 with BN+ReLU
→ head: Conv2d(16→1, k=1)
Output (B, 1, H, W) logits
```

### `SegmentInkNet` (segment track, 33-layer, 0.24 M params)
```
Input (B, 33, H, W)
→ enc3d_1: Conv3d(1→16)×2 + MaxPool3d(2,1,1)   # Z: 33→16
→ enc3d_2: Conv3d(16→32)×2 + MaxPool3d(2,1,1)  # Z: 16→8
→ enc3d_3: Conv3d(32→64)×2 + AdaptiveAvgPool3d  # Z→1
→ dec2d: Conv2d 64→32→16 with BN+ReLU
→ head: Conv2d(16→1, k=1)
Output (B, 1, H, W) logits
```
Z-adaptive — `AdaptiveAvgPool3d` means the same model accepts any Z depth.

## Inference post-processing

Applied after sigmoid thresholding:
- Binarise at `threshold` (default 0.45 fragment / 0.40 segment)
- Morphological closing with `disk=2`
- Remove connected components smaller than `min_cc_size=50` pixels

## Data layout

```
data/
  train/                          # Kaggle competition fragments (manual ink labels)
    fragment1/
      surface_volume/00–64.tif    # 65-slice Z-stack, uint8/uint16 TIFF
      mask.png                    # valid papyrus region
      inklabels.png               # ground-truth binary ink mask (0 or 255)
    fragment2/ fragment3/ ...
  scroll/                         # 7 PHercParis4 segments (inference only, 65-layer)
    20230827161847/               # Grand Prize — first readable text confirmed here
      surface_volume/00–64.tif
      mask.png
      meta.json
    20230826170124/ ...
  labelled_segments/              # 11 PHercParis4 segments WITH pseudo-labels (33-layer)
    20231221180251/               # smallest — use as val_segment
      surface_volume/00–32.tif    # 33-layer Z-stack
      ink_labels.tif              # uint8 pseudo-label (0–255 prob) from GP-winner model
    20231031143852/ ...           # 10 more segments
models/
  best_model.pth                  # fragment-track best (val F0.5)
  best_segment_model.pth          # segment-track best (val F0.5)
  last_segment_model.pth          # segment-track last epoch (kaggle_segment_train only)
  training_history.json           # per-epoch metrics (kaggle_segment_train only)
  training_curves.png             # loss + F0.5 + LR curves (kaggle_segment_train only)
predictions/                      # all model + filter outputs (artifacts only, no code)
  {segment_id}_prob.npy           # raw float32 probability map (from SegmentInkNet inference)
  {segment_id}_prob.tif           # same as uint8 TIFF
  {segment_id}_vis.png            # CT / pred / label / overlay visualization
  raw_label_inspection/{seg}/     # scripts/visualize_labels.py        — raw pseudo-label inspection
  filter_strategy_benchmark/{seg}/# scripts/compare_label_filters.py    — 12-strategy comparison
  enhanced_labels/{seg}/          # scripts/enhance_all_labels.py       — winning S11 enhanced outputs
  legacy_label_enhancement/{seg}/ # scripts/improve_labels_visual.py    — older simpler filter
filter_experiments/               # OUTPUT artifacts from src/experiment_filters.py (template matching)
  candidates.csv                  # per-component top-K Greek letter guesses
  candidates_grid.png             # visual grid of candidates
  templates.png                   # Greek uppercase template reference
```

## Data sources

**Fragment data (training ground truth):**
Kaggle Vesuvius Challenge competition — manual IR-photo ink annotations.

**Unlabelled scroll segments (inference, 65-layer):**
`https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/`

| Segment ID | Size |
|---|---|
| 20230827161847 | 5.73 GB (Grand Prize) |
| 20230826170124–20230504093154 | 1–2 GB each |
| **Total** | **~15.1 GB** |

**Labelled segments with pseudo-labels (segment track, 33-layer):**
`https://data.aws.ash2txt.org/samples/PHercParis4/segments/`

| Segment ID | Surface Vol | Label |
|---|---|---|
| 20231221180251 | 3.5 GB | 25 MB |
| 20231031143852 | 3.7 GB | 27 MB |
| 20231016151002 | 4.0 GB | 29 MB |
| 20231106155351 | 4.5 GB | 35 MB |
| 20230702185753 | 4.6 GB | 37 MB |
| 20231210121321 | 5.1 GB | 38 MB |
| 20230929220926 | 6.1 GB | 51 MB |
| 20231022170901 | 6.3 GB | 47 MB |
| 20231005123336 | 8.6 GB | 70 MB |
| 20231012184424 | 9.8 GB | 73 MB |
| 20231007101619 | 14.2 GB | 111 MB |
| **Total** | **~72 GB** | **~540 MB** |

Labels are pseudo-labels — outputs of a `tile64-stride16` ink-detection model, not manual annotation.
The training loss uses an ignore-band on values 0.4–0.6 to avoid fitting ambiguous label regions.

## Scholar stage (src/scholar/) — paleographer-style readings

A post-step on top of the decipher pipeline. Reads an existing
`predictions/decipher/{seg_id}/result.json` and produces a richer scholar
block: per-letter alternates with confidence, word-division reading,
recognised Greek words with glosses, probable per-strip meaning, and a
segment-level genre / historical-context paragraph.

Implemented with the **OpenAI Agents SDK** calling the **OpenAI API
directly** (OpenRouter dropped after the key cap was exhausted):
- `StripScholarAgent` — vision-enabled, default `gpt-4o-mini`
- `SegmentScholarAgent` — text-only, default `gpt-4o-mini`
- Override via `SCHOLAR_STRIP_MODEL` / `SCHOLAR_SEGMENT_MODEL` (e.g. `gpt-4o`)
- Billing flows through `OPENAI_API_KEY` (set in `.env` or env var)

```bash
# Live (needs OPENAI_API_KEY)
python scripts/scholar_decipher.py --only 20231221180251
python scripts/scholar_decipher.py                      # all 11

# Offline preview (no API calls; synthesises from existing consensus)
python scripts/scholar_decipher.py --mock
```

Output appears as a new top-level `"scholar"` field inside `result.json`
plus a sibling `scholar.json`, mirrored into
`web/public/assets/decipher/{seg_id}/`. The webapp renders it in a new
**Scholar** tab in `/viewer/{id}`
([web/src/components/viewer/ScholarReading.tsx](web/src/components/viewer/ScholarReading.tsx)).

## Claude API integration (cell 11 of fragment notebook)

Requires `ANTHROPIC_API_KEY` in the environment. Uses `claude-opus-4-7` to transcribe ancient Greek from binarised ink patches. Skipped in segment track for now — use `experiment_filters.py` for letter candidates instead.
