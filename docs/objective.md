# Project Objective

## Ultimate Goal

Build a generalizable pipeline that can be applied to **any undeciphered CT segment from any Herculaneum scroll** — including scrolls with no prior labels, no known text, and potentially different scanning conditions — and produce a readable ancient Greek transcription.

PHercParis4 (Scroll 1) is the development and validation target. The pipeline must transfer to unknown scrolls without retraining.

---

## Pipeline

### Stage 1 — Ink Detection

Train a deep learning model on labelled data to produce a per-pixel ink probability map from a 3D CT Z-stack.

Two training tracks exist, reflecting different data sources and model architectures:

**Fragment track** (`notebooks/fragment_ink_detection.ipynb`):
- Input: `(65, 224, 224)` surface volume patches from Kaggle fragments with manual ink labels
- Model: `VesuviusNet` — 3D CNN encoder + 2D decoder, ~1 M params
- Inference: sliding-window over 65-layer PHercParis4 scroll segments

**Segment track** (`notebooks/segment_ink_detection.ipynb`, `notebooks/kaggle_segment_train.ipynb`):
- Input: `(33, 256, 256)` surface volume patches from 11 labelled PHercParis4 segments
- Model: `SegmentInkNet` — 3D CNN encoder + 2D decoder, ~0.24 M params (baseline)
- Training data: **all 11 labelled segments** (~72 GB) with pseudo-labels from the Grand Prize winning model
- Target architecture: **TimeSformer** with divided space-time attention — the same class of model used by the GP-winner team to generate the pseudo-labels (see [Model Improvement Plan](../plan/model_improvement.md))

**Generalisation requirement:** Model must work on unseen scrolls with no fine-tuning. Training data should cover diverse fragment/segment conditions.

### Stage 2 — Letter Isolation

Apply image processing to the probability map to isolate individual letter candidates.

- Binarise at confidence threshold (default 0.40 for segment track, 0.45 for fragment track)
- Morphological closing to connect ink strokes within a letter
- Connected component analysis — retain high-confidence components as letter candidates
- Low-confidence components kept separately as candidates for Stage 3
- Output: high-confidence letter mask + low-confidence candidate regions with spatial coordinates

### Stage 3 — LLM Gap-filling

Feed isolated letter patches + surrounding context to a large language model to resolve ambiguous characters.

- High-confidence letters anchor the linguistic context
- LLM (`claude-opus-4-7`) receives: image patch + neighbouring high-confidence characters + line position
- Model infers the most probable character using ancient Greek vocabulary, grammar, and letter-shape priors
- Critical for unknown scrolls where no ground truth exists

### Stage 4 — Final Text Output

Reconstruct a continuous transcription from spatial letter positions.

- Line order recovered from letter centroid coordinates
- Output structured text with per-character confidence scores
- Flag regions for human review where confidence is below threshold

---

## Development Target

Segment `20230827161847` (Grand Prize — first readable text confirmed here, known transcription available for validation).

## Transfer Target

Any new CT segment from any Herculaneum scroll — the pipeline takes a directory of 33 or 65-layer TIFFs + a papyrus mask as input and produces a transcription. No scroll-specific configuration required.

---

## Constraints

- Local machine: ~72 GB disk for all 11 labelled segments + models + predictions
- Kaggle: ~20 GB `/kaggle/working` limit — use 3–4 segments
- Windows machine, PyTorch + CUDA (optional)
- Kaggle segment training hard time budget: 7.5 hours

## Success Metrics

- **Segment track F0.5**: precision-weighted detection on held-out segment `20231221180251`
- **Cross-segment F0.5**: average F0.5 across 3–5 rotating val segments (more robust)
- **Transfer**: legible transcription on a segment from a scroll not seen during training

## Improvement Roadmap

See [plan/model_improvement.md](../plan/model_improvement.md) for the full improvement plan, including:
- U-Net skip connections and increased model capacity (Tier 1)
- Focal loss, Z attention pool, cross-segment validation (Tier 2)
- Pseudo-label refinement, EfficientNet backbone, MAE pre-training (Tier 3)
- **TimeSformer** — the GP-winner architecture, target end-state for Stage 1 (Tier 4)
