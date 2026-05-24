# Model Video Content: Ink Detection

## Short Summary

For the model stage, we trained a prototype ink-detection ensemble on five Vesuvius/Herculaneum scroll segments. The model does not read text directly. It looks at CT slice data and predicts where ink-like pixels may be present.

The strongest single model was an I3D-style 3D CNN, and the best overall result came from a soft-voting ensemble of five small models: U-Net, ResNet-style CNN, I3D-style 3D CNN, Vision Transformer, and TimeSformer-style transformer.

The ensemble reached a provisional validation Dice score of **0.8142** in the provided run log. This means the predicted ink mask overlapped the known ink label much better than any single model, but it is still a baseline/prototype result and should not be described as fully solving scroll decipherment.

## Model Name Or Architecture

The current model system is a **five-model soft-voting ensemble**:

- **U-Net small**: a 2D encoder-decoder segmentation model.
- **ResNet small**: a 2D residual CNN.
- **I3D small**: a 3D CNN that learns across the z-slice stack.
- **Vision Transformer small**: a patch-based transformer over image patches.
- **TimeSformer small**: a transformer-style model that treats z-slices as frames.

The most important single model in the result is the **I3D-style 3D CNN**, because it achieved the best individual validation Dice score.

## Why This Model Was Used

Ink in CT scroll data is subtle. It may not be obvious in a single 2D image slice, so the model benefits from seeing a small stack of nearby CT slices.

The ensemble was used for a practical reason: different models notice different patterns.

- The U-Net and ResNet are good baseline image segmentation models.
- The I3D model can use 3D information across multiple CT slices.
- The ViT and TimeSformer give transformer-based comparisons.
- Averaging their predictions gives a more stable probability map than relying on one model alone.

This is best described as a **prototype/baseline ink detection model**, not a final decipherment system.

## Input Format

The model input is a small CT patch:

- Patch size: **64 x 64 pixels**
- Channels: **17 CT z-slices**
- Z-slices used: **slice 8 through slice 24**
- Input tensor shape: **batch x 17 x 64 x 64**

In simple terms: each training example is a small square region from a scroll segment, with 17 nearby depth slices stacked together so the model can see local 3D structure.

## Output Format

The model outputs a pixel-wise ink prediction:

- Output shape: **batch x 1 x 64 x 64**
- Output type: raw model logits converted to probabilities with sigmoid
- Visual output: **ink probability map**
- Final mask: probability map thresholded at **0.25**

In simple terms: for every pixel in the patch, the model estimates how likely that pixel is to be ink.

## Training Data Used

Training used five timestamp-named scroll segments:

- `20231012184424`
- `20231007101619`
- `20231005123336`
- `20230929220926`
- `20230702185753`

Each segment was converted into cached CT volumes and binary ink masks. The script sampled patches from these segments. It sampled more often around known ink pixels so the model saw enough positive examples.

Known setup:

- Training patches per segment: **650**
- Total training patches per epoch: **3250**
- Validation patches per segment: **140**
- Total validation patches per epoch: **700**

## Validation / Testing Data Used

The current run used validation patches sampled from the same five available segments, using a different random seed from training patch sampling. This gives a useful baseline check, but it is not the same as a fully independent held-out benchmark.

For evaluator-facing language, describe the result as:

> Provisional validation performance on sampled patches from the five-segment experiment.

Avoid saying that this proves the model generalizes to all scrolls.

## Training Settings

Known settings from `run_ensemble.py`:

- Epochs: **15**
- Batch size: **12**
- Optimizer: **AdamW**
- Learning rate: **4e-4**
- Scheduler: **CosineAnnealingLR**
- Minimum scheduler learning rate: **2e-5**
- Loss: **binary cross entropy with logits + Dice loss**
- Positive class weight in BCE: **14.0**
- Validation Dice threshold: **0.25**
- Device in provided run: **CUDA**

## Best Checkpoint / Best Epoch

From the provided run log:

| Model | Best epoch | Best validation Dice |
|---|---:|---:|
| U-Net | 15 | 0.5497 |
| ResNet | 14 | 0.6287 |
| I3D | 14 | 0.7598 |
| ViT | 2 or 15, tied after rounding | 0.4733 |
| TimeSformer | 8 or 15, tied after rounding | 0.4733 |

The **best single checkpoint** was the **I3D model at epoch 14**, with validation Dice **0.7598**.

The **best overall result** was the **soft-voting ensemble**, with validation Dice **0.8142**.

## Metrics

From the provided run log:

| Method | Validation loss | Dice score |
|---|---:|---:|
| U-Net best | 0.5539 | 0.5497 |
| ResNet best | 0.4267 | 0.6287 |
| I3D best | 0.2811 | 0.7598 |
| ViT best | about 0.4138 | 0.4733 |
| TimeSformer best | about 0.4189 | 0.4733 |
| Soft-voting ensemble | 0.3104 | 0.8142 |

Dice score measures overlap between the predicted ink mask and the ground-truth ink label. A higher Dice score means the prediction and the label overlap better.

## What The Result Means In Simple Language

The model is learning to find ink-like regions in CT data. The I3D model performed best among the individual models because it can use information across multiple CT slices, not just one image at a time.

The ensemble performed better than the individual models. This suggests that combining several model views produces a cleaner and more reliable ink probability map.

The result is promising for a prototype pipeline: it can produce candidate ink regions that can be reviewed visually and used downstream. However, it does not mean the scroll text is automatically recovered. The output is a probability map and candidate mask, not verified readable text.

## Limitations

- The validation setup is provisional and based on the current five-segment experiment.
- The model was trained on limited available labeled data.
- The output can include false positives where the model sees ink-like texture but no true ink is present.
- Thin or faint ink may still be missed.
- The current result should be described as baseline/prototype model performance.
- Human review and downstream surface/text interpretation are still required.

## Next Steps

- Test on a stronger held-out validation split or separate segment set.
- Add more labeled examples if available.
- Compare model predictions against expert-reviewed ink regions.
- Improve threshold selection for different segment types.
- Use the model output as a candidate-region generator for the virtual unwrapping and review pipeline.
- Track qualitative examples: strong predictions, missed ink, and false positives.

## 3-Minute Video Narration

For the model stage, our goal is not to read the scroll directly. The model's job is simpler and more focused: it looks at CT data and predicts where ink is likely to be.

The input to the model is a small 64 by 64 image patch, but instead of using only one image, we stack 17 nearby CT slices together. This gives the model a small 3D view of the scroll material. That matters because ink can be very subtle in this data, and sometimes the useful signal only appears when nearby slices are considered together.

We tested a small ensemble of five models. The ensemble includes a U-Net, a ResNet-style CNN, an I3D-style 3D CNN, a Vision Transformer, and a TimeSformer-style model. The U-Net and ResNet are strong image segmentation baselines. The I3D model is especially useful because it directly learns across the depth slices. The transformer models give another way to compare patch-level patterns.

Each model outputs an ink probability map. In other words, for every pixel, it estimates how likely that pixel is to belong to ink. We then threshold that probability map to create a binary ink mask. Finally, the ensemble averages the probability maps from all five models, which gives a more stable prediction than any one model alone.

In the current five-segment experiment, the best single model was the I3D model. It reached a provisional validation Dice score of 0.7598. The full soft-voting ensemble improved the result further, reaching a Dice score of 0.8142. Dice measures how much the predicted ink mask overlaps with the known ink label, so higher is better.

Visually, the important result is that the model can highlight candidate ink regions on top of the CT image. This gives us a practical review layer for the larger pipeline. It can help identify where ink may be present after the surface has been prepared and unwrapped.

The result is promising, but it is still a prototype. We should not claim that the model fully solves scroll decipherment. It produces candidate ink probability maps, and those predictions still need validation, review, and connection to the upstream surface reconstruction and downstream reading workflow.

## Suggested Visuals For The Video

Use the images in `docs/content/model/`:

- `01_model_input.png`: show the CT slice/patch input.
- `02_ground_truth_label.png`: show the known ink label used for training.
- `03_model_prediction.png`: show the model probability map.
- `04_prediction_overlay.png`: show prediction overlaid on the CT image.
- `05_training_curve.png`: show training/validation loss and Dice trends.
- `06_result_comparison.png`: show input, label, prediction, and overlay side by side.
- `07_failure_case.png`: show why model predictions still need review.
- `08_model_pipeline.svg`: use as the simple model pipeline diagram.

