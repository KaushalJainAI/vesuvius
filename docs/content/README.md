# NotebookLM Content Pack

Upload this whole `docs/content/` folder to NotebookLM along with `docs/video.md`.

This folder contains:

- Formal opening title card and PEC logo
- Problem, data, and pipeline diagrams
- Real preprocessing before/after screenshots from the PPT
- Real VC3D seeding, correction, scrubbing, and surface extension screenshots from the PPT
- Website review demo recording
- Model explanation and result assets
- Review interface and candidate-reading visuals

## Important Files

- `19_opening_title_card.svg`: opening title card with members, SIDs, mentor, and college
- `31_preprocessing_before_after.svg`: before/after denoising slide
- `32_vc3d_correction_before_after.svg`: before/after VC3D correction slide
- `34_unrolled_segment_slide.svg`: slide for the real unrolled segment image
- `20_website_demo_recording.webm`: short website review demo recording
- `model_video_content.md`: model explanation and metrics
- `model/`: model input, label, prediction, overlay, training curve, comparison, and failure case

## One File Still Needs To Be Added Manually

Place the attached real unrolled segment image here:

```text
docs/content/33_real_unrolled_segment.png
```

After that, `34_unrolled_segment_slide.svg` will automatically reference it.

