# Vesuvius Scroll Decipherment Pipeline

A generalizable pipeline to detect ink and decipher ancient Greek from CT scan data of Herculaneum scrolls (PHercParis4 / Scroll 1). Runs on local machines and Kaggle.

---

## What this does

The Herculaneum scrolls were carbonized by the eruption of Mount Vesuvius in 79 AD. Modern CT scanning can image the rolled papyrus non-destructively, but the ink is nearly invisible in X-ray. This pipeline uses deep learning to detect ink from the 3D CT voxel patterns, isolate individual letter candidates, and identify Greek characters.

![Ink detection result showing letter R / Ρ detected in a scroll segment](docs/example_detection.png)

---

## Documentation

Comprehensive guides and project details:

- [Project Objectives](docs/objective.md) — Success metrics, constraints, and long-term goals.
- [Data Description](docs/data_description.md) — Detailed breakdown of CT scans, surface volumes, and label formats.

---

## Pipeline overview

```
CT scan (65 or 33-layer Z-stack TIFF)
         │
         ▼
 Stage 1: Ink Detection
  └─ 3D CNN collapses Z → 2D ink probability map
         │
         ▼
 Stage 2: Letter Isolation
  └─ threshold + morphological closing + connected components
  └─ per-component bbox + area + centroid
         │
         ▼
 Stage 3: Letter Recognition (src/experiment_filters.py)
  └─ normalize component to 64×64
  └─ NCC + IoU vs 24 Greek uppercase templates
  └─ ranked top-K letter guesses + pseudo-confidence
         │
         ▼
 Stage 4: Transcription (planned)
  └─ line order from centroids
  └─ LLM gap-filling for ambiguous characters
```

---

## Quick start

### 1. Setup

```bash
git clone <repo>
cd Vesuvius

# Create venv and install dependencies
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash
# or: venv\Scripts\activate.bat   # Windows cmd

pip install -r requirements.txt

# Optional: GPU-accelerated PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2a. Fragment track (Kaggle labelled fragments, 65-layer Z-stacks)

Uses manual ink annotations from the [Vesuvius Challenge Kaggle competition](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection).

```bash
# Download 7 unlabelled PHercParis4 scroll segments for inference (~15 GB)
python scripts/downloading/download_segments.py

# Train + infer
jupyter notebook notebooks/fragment_ink_detection.ipynb
```

### 2b. Segment track (11 labelled PHercParis4 segments, 33-layer Z-stacks)

Uses pseudo-labels (outputs of prior Grand Prize winner model). Labels are already downloaded. Surface volumes must be downloaded separately.

```bash
# View available segments and their sizes:
python scripts/downloading/download_labelled_segment.py

# Download specific segments (start with the 3 smallest, ~11 GB):
python scripts/downloading/download_labelled_segment.py --seg 20231221180251 20231031143852 20231016151002

# Or download all 11 (~72 GB):
python scripts/downloading/download_labelled_segment.py --all

# Train + infer
jupyter notebook notebooks/segment_ink_detection.ipynb
```

### 3. Experiment with Greek letter filters

After inference produces a probability map:

```bash
python src/experiment_filters.py --pred predictions/20231221180251_prob.npy

# Restrict to a specific region:
python src/experiment_filters.py --pred predictions/20231221180251_prob.npy --crop 1000,3000,500,2500

# Adjust sensitivity:
python src/experiment_filters.py --pred predictions/20231221180251_prob.npy --threshold 0.45 --min-size 80
```

Outputs to `filter_experiments/`:
- `candidates.csv` — per-component top-5 Greek letter guesses with confidence scores
- `candidates_grid.png` — visual grid of all extracted letter candidates
- `templates.png` — the 24 Greek uppercase templates used for matching

---

## Running on Kaggle

The segment notebook auto-detects Kaggle and adjusts paths and download targets accordingly.

1. Upload `src/segment_model.py` to your Kaggle notebook as a utility script (or add it to `/kaggle/working/`).
2. Open `notebooks/segment_ink_detection.ipynb` — cell 1 will automatically download the 3 smallest segments (~11 GB, fits within Kaggle's 20 GB `/kaggle/working` limit).
3. To use more segments, edit `SEGMENTS_TO_USE` in cell 1.

> **Note:** Kaggle's `/kaggle/working` disk cap is ~20 GB. With 3 segments you get ~11 GB of surface volumes + ~81 MB labels. Training on all 11 requires a Kaggle Dataset upload (offline).

---

## File reference

| File | Description |
|---|---|
| `notebooks/fragment_ink_detection.ipynb` | Fragment track: train on Kaggle fragments (ground-truth labels), infer on PHercParis4 scroll segments → ink probability maps |
| `notebooks/segment_ink_detection.ipynb` | Segment-track notebook: train on 11 labelled PHercParis4 segments, Kaggle-compatible |
| `notebooks/kaggle_segment_train.ipynb` | Self-contained segment training for Kaggle (no `src/segment_model.py` needed) |
| `src/segment_model.py` | Shared module: `SegmentInkNet`, `SegmentStreamDataset`, loss, inference |
| `src/experiment_filters.py` | CLI: extract Greek letter candidates from a probability map and rank against templates |
| `scripts/downloading/download_segments.py` | Download 7 unlabelled PHercParis4 scroll segments (65-layer, ~15 GB) |
| `scripts/downloading/download_labelled_segment.py` | Download 11 labelled segment surface volumes (33-layer, ~72 GB total) |
| `scripts/downloading/download_fragments.py` | Download competition fragment data |
| `notebooks/visualize_segments.ipynb` | Interactive 3D viewer: MIPs, orthogonal projections for scroll segments |
| `notebooks/visualize_data.ipynb` | Z-stack browser, depth profiles, patch sampling for fragments |
| `docs/objective.md` | Full project goals, constraints, and success metrics |
| `docs/data_description.md` | Detailed explanation of the Vesuvius datasets |

---

## Data layout

```
data/
  train/                          # Kaggle fragments (manual labels, 65-layer)
    fragment1/
      surface_volume/00–64.tif
      mask.png
      inklabels.png
    fragment2/ fragment3/ ...
  scroll/                         # 7 segments, inference only (65-layer, no labels)
    20230827161847/               # Grand Prize — first confirmed readable text
      surface_volume/00–64.tif
      mask.png
    ...
  labelled_segments/              # 11 segments with pseudo-labels (33-layer)
    20231221180251/               # smallest ~3.5 GB — default val segment
      surface_volume/00–32.tif
      ink_labels.tif              # uint8 pseudo-label from GP-winner model
    ...
models/
  best_model.pth                  # fragment-track best weights
  best_segment_model.pth          # segment-track best weights
predictions/
  {id}_prob.npy                   # float32 probability map
  {id}_prob.tif                   # same as uint8 TIFF
  {id}_vis.png                    # visualization: CT / prediction / label / overlay
filter_experiments/
  candidates.csv                  # letter candidates with top-K Greek guesses
  candidates_grid.png             # visual grid of candidates
  templates.png                   # Greek uppercase template reference
```

---

## Models

### Fragment track — `VesuviusNet` (65-layer input)

3D conv encoder collapses Z, 2D decoder produces per-pixel logits.

```
Input (B, 65, H, W)
→ enc3d_1: Conv3d(1→32)×2 + MaxPool3d(2,1,1)       Z: 65→32
→ enc3d_2: Conv3d(32→64)×2 + MaxPool3d(2,1,1)      Z: 32→16
→ enc3d_3: Conv3d(64→128) + AdaptiveAvgPool3d       Z→1
→ dec2d: Conv2d 128→64→32→16 + BN + ReLU
→ head: Conv2d(16→1, kernel=1)
Output (B, 1, H, W) logits
```

Loss: 0.5 × BCE + 0.5 × Dice. Metric: F0.5 (precision-weighted).

### Segment track — `SegmentInkNet` (33-layer input, 0.24 M params)

Lighter architecture designed for the 33-layer labelled segment data.

```
Input (B, 33, H, W)
→ enc3d_1: Conv3d(1→16)×2 + MaxPool3d(2,1,1)       Z: 33→16
→ enc3d_2: Conv3d(16→32)×2 + MaxPool3d(2,1,1)      Z: 16→8
→ enc3d_3: Conv3d(32→64)×2 + AdaptiveAvgPool3d      Z→1
→ dec2d: Conv2d 64→32→16 + BN + ReLU
→ head: Conv2d(16→1, kernel=1)
Output (B, 1, H, W) logits
```

Loss: soft-label BCE + Dice with an **ignore-band on label values 0.4–0.6** (pseudo-labels have uncertain regions; these are excluded from supervision).

The `AdaptiveAvgPool3d` at the end means the model accepts any Z depth — swap `CFG['z_layers']` to use it on 65-layer data.

---

## Key hyperparameters (segment track)

Defined in `src/segment_model.py` → `CFG` dict:

| Parameter | Default | Description |
|---|---|---|
| `patch_size` | 256 | Spatial patch size (pixels) |
| `patches_per_seg` | 400 | Patches sampled per segment per epoch |
| `batch_size` | 4 | Keep low for cross-platform compatibility |
| `grad_accum` | 4 | Effective batch = `batch_size × grad_accum` = 16 |
| `num_epochs` | 8 | Training epochs |
| `lr` | 1e-4 | AdamW learning rate |
| `ignore_band` | (0.4, 0.6) | Pseudo-label confidence range to exclude from loss |
| `val_segment` | `20231221180251` | Held-out segment (smallest, fastest val) |
| `threshold` | 0.4 | Binarization threshold for F0.5 and visualization |

---

## Known limitations

- **Pseudo-labels are noisy.** The segment-track labels are outputs of a prior model — training ceiling is bounded by that model's quality. Treat trained weights as a starting point, not ground truth.
- **Filter confidences are not calibrated.** `scripts/experiment_filters.py` uses normalized cross-correlation against a single serif font. Scores rank letters within a component but are not probabilities — use them to shortlist candidates for human review or LLM disambiguation.
- **33-layer vs 65-layer mismatch.** The two training tracks use different Z depths. Mixing data requires either resampling Z or using the adaptive pooling path.
- **Scroll script vs modern Greek font.** Herculaneum scrolls use a specific majuscule writing style. Template matching against a modern font will miss highly stylized letterforms — this is a known gap for Stage 3.

---

## Background

- [Vesuvius Challenge](https://scrollprize.org/) — the competition that produced the Grand Prize transcription
- [Kaggle Ink Detection Competition](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection)
- [EduceLab Scrolls Dataset](https://dl.ash2txt.org/)
- [Grand Prize announcement](https://scrollprize.org/grandprize)
