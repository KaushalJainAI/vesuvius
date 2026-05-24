# Production Research Pipeline

This project should treat notebooks as inspection and reporting surfaces, not
as the source of truth for training logic. The reproducible path is now:

```text
labelled segment data
    -> shared loader / patch sampler
    -> shared model factory
    -> shared trainer
    -> standardized checkpoint + history
    -> standardized inference output
    -> run comparison table
```

## Main Commands

Train the baseline on original pseudo-labels:

```powershell
.\venv\Scripts\python.exe scripts\train_segment_model.py --model baseline
```

Train the recommended U-Net V2 model:

```powershell
.\venv\Scripts\python.exe scripts\train_segment_model.py --model unet_v2
```

Train U-Net V2 on the conservative improved labels:

```powershell
.\venv\Scripts\python.exe scripts\train_segment_model.py --model unet_v2 --label-root predictions\improved_labels_visual
```

Run a quick smoke training run before spending GPU time:

```powershell
.\venv\Scripts\python.exe scripts\train_segment_model.py --model baseline --max-segments 2 --epochs 1 --patches-per-seg 4 --val-patches 2
```

Run inference from a standardized checkpoint:

```powershell
.\venv\Scripts\python.exe scripts\predict_segment_model.py --checkpoint models\runs\<run>\best.pth --seg 20231221180251
```

Compare standardized runs:

```powershell
.\venv\Scripts\python.exe scripts\summarize_runs.py
```

## What Belongs Where

- `src/research_config.py`: canonical segments, paths, train configuration.
- `src/vesuvius_data.py`: volume/label loading and patch sampling.
- `src/ink_models.py`: baseline and U-Net V2 model definitions.
- `src/train_utils.py`: losses, metrics, training loop, inference.
- `scripts/train_segment_model.py`: standardized training entry point.
- `scripts/predict_segment_model.py`: standardized inference entry point.
- `scripts/summarize_runs.py`: comparison table across runs.
- Notebooks: visualization, inspection, and ablation reporting.

## Research Rules

1. Do not compare notebook runs unless they used the shared trainer.
2. Always report the label source: original pseudo-labels or refined labels.
3. Always report validation segment, threshold, precision, recall, and F0.5.
4. Treat segment F0.5 as pseudo-label agreement, not proof of true ink.
5. LLM-assisted text should not become training ink unless visual evidence exists.

## Recommended Next Run

Use U-Net V2 with original labels, then U-Net V2 with
`predictions/improved_labels_visual`, keeping the same validation segment. If
the refined-label run improves F0.5 without a large precision collapse, the
label cleanup is probably helping rather than merely changing the target.

