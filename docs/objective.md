# Project Objective

## Ultimate Goal

Build a generalizable pipeline that can be applied to **any undeciphered CT segment from any Herculaneum scroll** — including scrolls with no prior labels, no known text, and potentially different scanning conditions — and produce a readable ancient Greek transcription.

PHercParis4 (Scroll 1) is the development and validation target. The pipeline must transfer to unknown scrolls without retraining.

---

## Pipeline

### Stage 1 — Ink Detection (3D CNN)

Train a UNet-style 3D CNN on labelled fragment data (Kaggle fragments 1–4, optionally Frag5/Frag6) to learn what CT ink signatures look like across different fragments.

- Input: `(65, 224, 224)` surface volume patches
- Output: `(1, 224, 224)` ink probability map per patch
- Sliding-window inference over full scroll segments → full-resolution probability map
- Output: `predictions/{segment_id}_prob.png`

**Generalization requirement:** Model must work on unseen scrolls with no fine-tuning. This means training data should cover diverse fragment conditions (different carbonization, scanning energy, papyrus thickness).

### Stage 2 — Letter Isolation (Post-processing filters)

Apply image processing to the probability map to isolate individual letter candidates.

- Binarise at confidence threshold (default `0.45`)
- Morphological closing to connect ink strokes within a letter
- Connected component analysis — retain components above confidence threshold as high-confidence letters
- Low-confidence components kept separately as candidates for Stage 3
- Output: high-confidence letter mask + low-confidence candidate regions with spatial coordinates

### Stage 3 — LLM Gap-filling

Feed isolated letter patches + surrounding context to a large language model to resolve low-confidence and ambiguous characters.

- High-confidence letters anchor the linguistic context (known positions in line)
- LLM (`claude-opus-4-7`) receives: image patch + neighbouring high-confidence characters + line position
- Model infers the most probable character using ancient Greek vocabulary, grammar, and letter-shape priors
- Critical for unknown scrolls where no ground truth exists at all

### Stage 4 — Final Text Output

Reconstruct a continuous transcription from spatial letter positions.

- Line order recovered from letter centroid coordinates
- Output structured text with per-character confidence scores
- Flag regions for human review where confidence is below threshold

---

## Development Target

Segment `20230827161847` (Grand Prize — first readable text confirmed here, known transcription available for validation).

## Transfer Target

Any new CT segment from any Herculaneum scroll — the pipeline takes a directory of 65-layer TIFFs + a papyrus mask as input and produces a transcription. No scroll-specific configuration required.

---

## Constraints

- ~15 GB disk budget — 7 pre-selected PHercParis4 segments for development
- Windows machine, PyTorch + CUDA (optional)
- Training data: Kaggle fragments 1–4 in `data/train/`

## Success Metric

- **Development:** F0.5 on held-out fragment (precision-weighted — fewer false positive ink detections, since Stage 3 handles gaps)
- **Transfer:** Legible transcription on a segment from a scroll not seen during training
