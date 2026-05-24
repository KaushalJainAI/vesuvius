# Model Visual Assets

These images support the model and model-results section of the video content pack. They are based on local outputs from the five-segment ensemble experiment in `final/ensemble_5segments`.

## 01_model_input.png

- Shows: a CT middle-slice preview from segment `20231012184424`.
- Dataset/segment: `20231012184424`.
- Split/use: validation/demo visual from the five-segment experiment cache.
- Evaluator should notice: the ink signal is not obvious from the raw CT image alone.

## 02_ground_truth_label.png

- Shows: the binary ink label mask for segment `20231012184424`.
- Dataset/segment: `20231012184424`.
- Split/use: label used in the five-segment training/validation experiment.
- Evaluator should notice: the model is trained to predict pixel-level ink regions, not text characters directly.

## 03_model_prediction.png

- Shows: the soft-voting ensemble ink probability map for segment `20231012184424`.
- Dataset/segment: `20231012184424`.
- Split/use: model output/demo visual from the trained ensemble.
- Evaluator should notice: brighter regions indicate higher predicted ink probability.

## 04_prediction_overlay.png

- Shows: the ensemble prediction heatmap overlaid on the CT preview image.
- Dataset/segment: `20231012184424`.
- Split/use: model output/demo visual from the trained ensemble.
- Evaluator should notice: the overlay makes it easier to review candidate ink regions in context.

## 05_training_curve.png

- Shows: training and validation curves for all five models.
- Dataset/segment: all five segments in the ensemble experiment.
- Split/use: training/validation metric output.
- Evaluator should notice: I3D and ResNet improve more strongly than the transformer baselines, and the curves show this is a prototype training run.

## 06_result_comparison.png

- Shows: CT input, ground-truth label, ensemble probability map, and overlay side by side.
- Dataset/segment: `20231012184424`.
- Split/use: validation/demo comparison.
- Evaluator should notice: the result is easiest to understand visually as overlap between the prediction and the known ink label.

## 07_failure_case.png

- Shows: a review-needed example from segment `20231005123336`.
- Dataset/segment: `20231005123336`.
- Split/use: validation/demo qualitative example.
- Evaluator should notice: sparse or uncertain predicted regions still need human review; the model should not be treated as a final decipherment step.

## 08_model_pipeline.svg

- Shows: the model pipeline from CT/unwrapped image patches to candidate text regions.
- Dataset/segment: conceptual diagram, not a specific segment.
- Split/use: video explanation diagram.
- Evaluator should notice: the model produces candidate ink regions, not final readable text.

