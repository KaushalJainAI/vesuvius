# Vesuvius Scroll Decipherment Pipeline

A generalizable pipeline to detect ink and decipher ancient Greek from CT scan data of Herculaneum scrolls (PHercParis4 / Scroll 1). Runs on local machines and Kaggle.

---

## What this does

The Herculaneum scrolls were carbonized by the eruption of Mount Vesuvius in 79 AD. Modern CT scanning can image the rolled papyrus non-destructively, but the ink is nearly invisible in X-ray. This pipeline uses deep learning to detect ink from the 3D CT voxel patterns, isolate individual letter candidates, and identify Greek characters.

The Grand Prize winning team (Nader / Farritor / Schilliger, 2023) used a **TimeSformer-based model** — divided space-time attention over CT Z-stacks — to produce the first readable Herculaneum transcription. That model's outputs serve as pseudo-labels for our segment track. TimeSformer is the target end-state architecture for Stage 1 of this pipeline.

---

## Documentation

| Document | Description |
|---|---|
| [docs/objective.md](docs/objective.md) | Pipeline stages, success metrics, constraints, and improvement targets |
| [docs/data_description.md](docs/data_description.md) | CT scans, surface volumes, Z-layers, label formats — no prior knowledge assumed |
| [docs/initial_model.md](docs/initial_model.md) | Baseline `SegmentInkNet` architecture, known weaknesses, comparison to target |
| [plan/model_improvement.md](plan/model_improvement.md) | Full improvement roadmap: CNN V2 → TimeSformer, with rationale and priority table |

---

## Pipeline overview

```
CT scan (33 or 65-layer Z-stack TIFF)
         │
         ▼
 Stage 1: Ink Detection
  └─ 3D model (CNN baseline → TimeSformer target)
  └─ collapses Z + spatial attention → 2D ink probability map
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
  └─ LLM gap-filling for ambiguous characters (claude-opus-4-7)
```

---

## Quick start

### 1. Setup

```bash
git clone <repo>
cd Vesuvius

python -m venv venv
source venv/Scripts/activate      # Windows Git Bash
# or: venv\Scripts\activate.bat   # Windows cmd

pip install -r requirements.txt

# Optional: GPU-accelerated PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2a. Fragment track (Kaggle labelled fragments, 65-layer)

Uses manual ink annotations from the Vesuvius Challenge Kaggle competition.

```bash
# Download 7 unlabelled PHercParis4 scroll segments for inference (~15 GB)
python scripts/downloading/download_segments.py

# Train + infer
jupyter notebook notebooks/fragment_ink_detection.ipynb
```

### 2b. Segment track — local (all 11 labelled segments, 33-layer)

Uses pseudo-labels from the GP-winner TimeSformer model. Labels are pre-downloaded. Surface volumes must be downloaded separately.

```bash
# Download all 11 labelled segment surface volumes (~72 GB):
python scripts/downloading/download_labelled_segment.py --all

# Or start with the 3 smallest (~11 GB):
python scripts/downloading/download_labelled_segment.py --seg 20231221180251 20231031143852 20231016151002

# Train + infer (uses all downloaded segments automatically)
jupyter notebook notebooks/segment_ink_detection.ipynb
```

### 2c. Segment track — Kaggle (self-contained, 3–4 segments)

```bash
# Self-contained notebook, no src/segment_model.py needed
# Downloads 4 smallest segments automatically (~15.7 GB)
jupyter notebook notebooks/kaggle_segment_train.ipynb
```

### 3. Experiment with Greek letter filters

After inference produces a probability map:

```bash
python src/experiment_filters.py --pred predictions/20231221180251_prob.npy

# Restrict to a region:
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

The segment notebook auto-detects Kaggle and adjusts paths and download targets.

1. Upload `src/segment_model.py` to your Kaggle notebook as a utility script.
2. Open `notebooks/segment_ink_detection.ipynb` — cell 1 downloads the 3 smallest segments (~11 GB, fits the 20 GB `/kaggle/working` limit).
3. Alternatively use the fully self-contained `notebooks/kaggle_segment_train.ipynb`.

> Kaggle's `/kaggle/working` disk cap is ~20 GB. Training on all 11 segments requires a local setup or a Kaggle Dataset upload (offline).

---

## File reference

| File | Description |
|---|---|
| `notebooks/fragment_ink_detection.ipynb` | Fragment track: train on Kaggle fragments, infer on scroll segments |
| `notebooks/segment_ink_detection.ipynb` | Segment track: train on labelled segments, Kaggle-compatible |
| `notebooks/kaggle_segment_train.ipynb` | Self-contained segment training for Kaggle (no `src/segment_model.py` needed) |
| `src/segment_model.py` | Shared module: `SegmentInkNet`, `SegmentStreamDataset`, loss, inference |
| `src/experiment_filters.py` | CLI: extract Greek letter candidates from a probability map |
| `scripts/downloading/download_segments.py` | Download 7 unlabelled scroll segments (65-layer, ~15 GB) |
| `scripts/downloading/download_labelled_segment.py` | Download 11 labelled segment surface volumes (33-layer, ~72 GB total) |
| `docs/objective.md` | Full project goals, pipeline stages, success metrics |
| `docs/data_description.md` | Detailed explanation of CT data, labels, and data layout |
| `docs/initial_model.md` | Baseline `SegmentInkNet` architecture and known weaknesses |
| `plan/model_improvement.md` | Improvement roadmap: CNN V2 → TimeSformer |

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
      ink_labels.tif              # uint8 pseudo-label from GP-winner TimeSformer model
    ...
models/
  best_model.pth                  # fragment-track best weights
  best_segment_model.pth          # segment-track best weights
predictions/
  {id}_prob.npy                   # float32 probability map
  {id}_prob.tif                   # same as uint8 TIFF
  {id}_vis.png                    # CT / prediction / label / overlay visualization
filter_experiments/
  candidates.csv
  candidates_grid.png
  templates.png
```

---

## Models

### Fragment track — `VesuviusNet` (65-layer input, ~1 M params)

```
Input (B, 65, H, W)
→ enc3d_1: Conv3d(1→32)×2 + MaxPool3d(2,1,1)       Z: 65→32
→ enc3d_2: Conv3d(32→64)×2 + MaxPool3d(2,1,1)      Z: 32→16
→ enc3d_3: Conv3d(64→128) + AdaptiveAvgPool3d       Z→1
→ dec2d: Conv2d 128→64→32→16 + BN + ReLU
→ head: Conv2d(16→1, kernel=1)
Output (B, 1, H, W) logits
```

### Segment track — `SegmentInkNet` (33-layer input, ~0.24 M params)

```
Input (B, 33, H, W)
→ enc3d_1: Conv3d(1→16)×2 + MaxPool3d(2,1,1)       Z: 33→16
→ enc3d_2: Conv3d(16→32)×2 + MaxPool3d(2,1,1)      Z: 16→8
→ enc3d_3: Conv3d(32→64)×2 + AdaptiveAvgPool3d      Z→1
→ dec2d: Conv2d 64→32→16 + BN + ReLU
→ head: Conv2d(16→1, kernel=1)
Output (B, 1, H, W) logits
```

`AdaptiveAvgPool3d` makes both models Z-depth agnostic — the same weights work on any number of input layers.

Loss: soft-label BCE + Dice with **ignore band on label values 0.4–0.6** (pseudo-labels are uncertain there; excluded from supervision).

### Target architecture — TimeSformer

The Grand Prize winning model used divided space-time attention over CT Z-stacks: Z-depth attention per spatial token + spatial attention per Z layer. This is the end-state target for Stage 1. See [plan/model_improvement.md](plan/model_improvement.md) for the implementation path.

---

## Known limitations

- **Pseudo-label ceiling.** Segment-track labels are outputs of the GP-winner TimeSformer. Training ceiling is bounded by that model's accuracy. Iterative pseudo-label refinement (see improvement plan) can raise this ceiling.
- **No skip connections in baseline.** `SegmentInkNet` discards spatial detail in the encoder. U-Net skip connections are the first improvement target.
- **Filter confidences are not calibrated.** `experiment_filters.py` uses NCC against a single font. Scores rank letters but are not probabilities.
- **33-layer vs 65-layer mismatch.** The two tracks use different Z depths. `AdaptiveAvgPool3d` handles this but the two models cannot share weights directly.
- **Scroll script vs modern Greek font.** Template matching uses a modern majuscule font — highly stylized Herculaneum letterforms will be missed. Known gap for Stage 3.

---

## Background

- [Vesuvius Challenge](https://scrollprize.org/)
- [Grand Prize announcement](https://scrollprize.org/grandprize)
- [Kaggle Ink Detection Competition](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection)
- [EduceLab Scrolls Dataset](https://dl.ash2txt.org/)
