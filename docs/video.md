# 3-Minute Evaluator Demo Video Plan

This file is the main context document for creating a short evaluator-facing project video with NotebookLM or any script-writing assistant. The video should sell the project through a clear demonstration story, not a technical lecture.

## Project Positioning

**Project name:** Geometric Virtual Unwrapping and Volumetric Ink Detection of Carbonized Herculaneum Papyrus via X-ray Micro-Tomography

**Short title for video:** Vesuvius Scroll Decipherment Pipeline

**One-line pitch:** This project builds a workflow that takes high-resolution CT data from a carbonized Herculaneum scroll, prepares and virtually unwraps the papyrus surface, then moves toward ink detection and model-assisted review.

**Problem:** The Herculaneum scrolls were carbonized by the eruption of Mount Vesuvius in 79 AD. Physically opening them can crack or destroy them. X-ray micro-CT scans let researchers look inside, but the scroll is still curled, damaged, noisy, and geometrically complex. Before ink can be detected, the papyrus surface must be found, traced, corrected, and flattened into a usable form.

**Core contribution:** The project is not only about ink prediction. The main story is an end-to-end preparation pipeline:

```text
Scroll -> X-ray micro-CT volume -> preprocessing -> point cloud / normal grid / surface tensor -> VC3D segmentation -> seeding and tracing -> correction -> surface extension / virtual unwrapping -> ink detection model -> review interface
```

**Evaluator takeaway:** We are turning a fragile rolled scroll into reviewable digital evidence. The upstream geometric work makes later ink prediction meaningful.

## Model And Result Summary

The model stage should be explained as an **ink detection model**, not as a direct text-reading system.

**Model system:** Five-model soft-voting ensemble:

- U-Net small
- ResNet-style CNN small
- I3D-style 3D CNN small
- Vision Transformer small
- TimeSformer-style transformer small

**Input:** CT patches of size `64 x 64` pixels with `17` nearby z-slices, so the model sees a small 3D view of the scroll material.

**Output:** A pixel-wise ink probability map. After thresholding, this becomes a candidate ink mask.

**Training/validation experiment:** Five timestamp-named scroll segments:

- `20231012184424`
- `20231007101619`
- `20231005123336`
- `20230929220926`
- `20230702185753`

**Best individual model:** I3D-style 3D CNN, provisional validation Dice `0.7598`.

**Best overall result:** Soft-voting ensemble, provisional validation Dice `0.8142`.

**Simple explanation:** Dice measures how strongly the predicted ink mask overlaps the known ink label. A higher score means better overlap. The result is promising for candidate ink detection, but it does not mean the scroll text is fully deciphered.

Use `docs/model_video_content.md` and the files under `docs/content/model/` for the full model explanation and visual assets.

## Opening Title Card Details

Start the video with a formal title card for about **6-8 seconds**. Use `19_opening_title_card.svg` as the reference visual and include the official college logo from `18_pec_logo.png`.

**Institution:** Punjab Engineering College (Deemed to be University), Chandigarh

**Branch:** Computer Science Engineering (Data Science) Branch

**Project type:** Major Project II

**Project members:**

- Arnav Vikas Garg (SID: 22106060)
- Sunit Mehta (SID: 22106059)
- Sumeet Kumar Panda (SID: 22106048)
- Kaushal Jain (SID: 22106045)

**Mentor:** Dr. Trilok Chand, Professor, Department of Computer Science and Engineering

Keep this opening clean and formal. Do not read every detail slowly in the voiceover; let the names and IDs remain visible on screen while the narration introduces the project.

## Conflict-Safe Dataset Note

The project material currently references three related but distinct contexts:

- **PHerc. 172 / ID 20241024131838:** This is the upstream scroll pipeline story from the PPT. It includes high-resolution 3D X-ray micro-CT data, preprocessing, VC3D setup, seeding, tracing, correction, and surface extension.
- **Five-segment model experiment:** This is the model/results story from `docs/model_video_content.md`. It explains the five-model ensemble trained and validated on sampled patches from segments `20231012184424`, `20231007101619`, `20231005123336`, `20230929220926`, and `20230702185753`.
- **PHercParis4 / web demo assets:** The website and existing `docs/content/` images demonstrate the later review workflow: ink maps, candidate letters, extracted strips, provisional readings, and consensus-style review.

Do not describe these as the same dataset. In the video, present them as parts of the same overall project pipeline:

```text
Our upstream work focuses on preparing a new scroll volume such as PHerc. 172. The model experiment shows how ink-like pixels can be detected from CT patches. The web demo shows how later-stage ink evidence and provisional readings can be presented for review.
```

## What Images To Include

Use the images in `docs/content/` as the visual pack for NotebookLM, slides, or video editing.

| File | Use in video | Message |
|---|---|---|
| `01_ct_preview.jpg` | Problem/input visual | CT-style data is the starting evidence, but it is difficult to interpret directly. |
| `02_ink_probability_map.png` | Later ink detection result | After preparation, the model can highlight probable ink regions. |
| `03_real_ink_label_overlay.png` | Before/after or validation visual | Labelled or enhanced ink evidence is easier to inspect than raw data. |
| `04_ct_text_strip.png` | Later demo walkthrough | A strip cut from the CT view where text may exist. |
| `05_extracted_ink_strip.png` | Later demo walkthrough | The extracted ink strip makes likely writing more visible. |
| `06_reading_plate.png` | Results | A review-ready visual plate for one extracted text region. |
| `07_annotated_candidate_letters.png` | Results | Candidate letters are marked for human review. |
| `08_enhanced_strip.png` | Results | The enhanced strip improves visibility for review. |
| `09_pipeline_storyboard.svg` | Overall story slide | From scroll volume to reviewable text evidence. |
| `10_video_timeline.svg` | Planning slide | The 3-minute timing structure. |
| `11_project_fact_sheet.svg` | Intro or summary slide | Problem, contribution, demo, and evaluator takeaway. |
| `12_results_artifacts.svg` | Results slide | Concrete outputs produced by the project. |
| `13_upstream_pipeline_before_ink.svg` | Upstream pipeline slide | CT, preprocessing, geometry, and unwrapping before ink prediction. |
| `14_vc3d_unwrapping_workflow.svg` | VC3D slide | Seeding, tracing, correction, and surface extension. |
| `15_preprocessing_toolkit.svg` | Preprocessing slide | Gaussian, non-local means, wavelet, Frangi, and CLAHE. |
| `16_pherc172_data_card.svg` | Data slide | PHerc. 172 metadata and micro-CT scan context. |
| `17_updated_end_to_end_pipeline.svg` | Main pipeline slide | Preparation first, prediction second, review last. |
| `18_pec_logo.png` | Opening title card | Official college logo from the project PPT. |
| `19_opening_title_card.svg` | First frame/title slide | Project title, members, SIDs, mentor, branch, and college. |
| `20_website_demo_recording.webm` | Website demo clip | 14.8-second screen recording of the review workspace flow. |
| `21_before_denoising.png` | Real preprocessing screenshot | PPT image showing the scan before denoising. |
| `22_after_denoising.png` | Real preprocessing screenshot | PPT image showing the scan after denoising. |
| `23_vc3d_gui.png` | Real VC3D screenshot | VC3D environment/interface from the PPT. |
| `24_vc3d_initial_seed.png` | Real VC3D screenshot | Initial seeding/tracing view from the PPT. |
| `25_vc3d_before_correction.png` | Real VC3D screenshot | Trace before local correction. |
| `26_vc3d_after_correction.png` | Real VC3D screenshot | Trace after local correction. |
| `27_before_scrubbing.png` | Real VC3D screenshot | Flattened-surface correction before scrubbing. |
| `28_after_scrubbing.png` | Real VC3D screenshot | Flattened-surface correction after scrubbing. |
| `29_surface_extension_vc3d.png` | Real VC3D screenshot | Surface extension / auto-grown surface view. |
| `30_unwrapped_surface_view.png` | Real VC3D screenshot | Unwrapped or extended surface view from the PPT. |
| `31_preprocessing_before_after.svg` | Comparison slide | Before/after denoising combined into one video-ready slide. |
| `32_vc3d_correction_before_after.svg` | Comparison slide | Before/after VC3D correction combined into one video-ready slide. |
| `model/01_model_input.png` | Model section | Raw CT patch/slice input where ink is not obvious. |
| `model/02_ground_truth_label.png` | Model section | Known ink label used as training/validation target. |
| `model/03_model_prediction.png` | Model result | Ensemble ink probability map. |
| `model/04_prediction_overlay.png` | Model result | Prediction heatmap over CT input for easier review. |
| `model/05_training_curve.png` | Model result | Training/validation curves for the five models. |
| `model/06_result_comparison.png` | Model result | Side-by-side input, label, prediction, and overlay. |
| `model/07_failure_case.png` | Limitation slide | Example showing why human review is still needed. |
| `model/08_model_pipeline.svg` | Model explanation slide | CT patches -> model -> probability map -> mask -> candidates. |

## Updated 3-Minute Storyboard

### 0:00-0:08 - Formal Opening

**Show:** `19_opening_title_card.svg`.

**Voiceover:**  
"This is our Major Project II: a pipeline for geometric virtual unwrapping and volumetric ink detection of carbonized Herculaneum papyrus."

**On-screen text:**  
Project title, team members with SIDs, mentor name, department, branch, and college logo.

### 0:08-0:25 - Hook and Problem

**Show:** Carbonized scroll/CT visual, `01_ct_preview.jpg`, or `16_pherc172_data_card.svg`.

**Voiceover:**  
"The Herculaneum scrolls preserve ancient writing, but they were carbonized almost two thousand years ago. Opening them physically can destroy them. X-ray micro-CT lets us look inside, but the scroll is still curled, noisy, and extremely hard to read."

**On-screen text:**  
`Reading a scroll without opening it`

### 0:25-0:45 - Why The Problem Is Hard

**Show:** `31_preprocessing_before_after.svg`, `13_upstream_pipeline_before_ink.svg`, or `15_preprocessing_toolkit.svg`.

**Voiceover:**  
"The challenge is not only detecting ink. Before that, we have to prepare the CT volume, reduce noise, enhance faint papyrus structures, understand the surface geometry, and turn a curled sheet into a flat segment that can be inspected."

**On-screen text:**  
`Before ink prediction, the surface must be reconstructed`

### 0:45-1:08 - Our Upstream Contribution

**Show:** `23_vc3d_gui.png`, `24_vc3d_initial_seed.png`, `32_vc3d_correction_before_after.svg`, and `29_surface_extension_vc3d.png` or `30_unwrapped_surface_view.png`.

**Voiceover:**  
"Our workflow uses geometric virtual unwrapping. We identify reliable seed points, trace the papyrus surface, correct tracing errors, and extend the surface into a usable unwrapped segment. This makes the later ink-detection stage possible."

**On-screen text:**  
`Seed -> trace -> correct -> extend`

### 1:08-1:25 - Complete Pipeline

**Show:** `17_updated_end_to_end_pipeline.svg`.

**Voiceover:**  
"The full pipeline moves from micro-CT slices to preprocessing, geometry reconstruction, virtual unwrapping, ink prediction, candidate letter extraction, and finally a review interface. This is important because each stage produces evidence for the next one."

**On-screen text:**  
`Raw CT volume -> unwrapped surface -> ink evidence -> review`

### 1:25-1:58 - Model And Ink Detection Results

**Show:** `model/08_model_pipeline.svg`, then `model/06_result_comparison.png`, `model/04_prediction_overlay.png`, and optionally `model/05_training_curve.png`.

**Voiceover:**  
"For the ink detection stage, the model does not read the text directly. It predicts where ink-like pixels may be present. We trained a small five-model ensemble using 64 by 64 CT patches with 17 nearby depth slices. The best individual model was the I3D-style 3D CNN, and the soft-voting ensemble reached a provisional validation Dice score of 0.8142. This means the predicted ink mask overlaps well with the known ink label, but it is still a prototype result that needs review."

**On-screen text:**  
`Model output: ink probability map, not final transcription`

### 1:58-2:35 - Website / Review Demonstration

**Show:** `20_website_demo_recording.webm` or a fresh screen recording of the website with the same flow.

Recommended order:

1. Show the segment catalogue or homepage.
2. Open one segment.
3. Show CT preview.
4. Switch to ink probability or label overlay.
5. Show candidate character regions.
6. Open the decipher/review panel.
7. Show extracted strips and provisional consensus output.

**Voiceover:**  
"This interface demonstrates the later review stage. A reviewer can inspect the scan view, compare it with ink-focused evidence, examine candidate letters, and view provisional model-assisted readings. The goal is not to hide uncertainty, but to make the evidence visible and reviewable."

**On-screen text:**  
`Interactive review of model-assisted evidence`

### 2:35-2:50 - Results And Outputs

**Show:** `12_results_artifacts.svg`, `model/06_result_comparison.png`, `05_extracted_ink_strip.png`, `07_annotated_candidate_letters.png`, or `08_enhanced_strip.png`.

**Voiceover:**  
"The current outputs include unwrapped evidence, ink probability maps, enhanced strips, annotated candidate regions, and provisional readings. Together, these show how the system moves from raw scan data toward inspectable historical evidence."

**On-screen text:**  
`Output: unwrapped evidence, ink maps, candidates, and provisional review`

### 2:50-3:00 - Closing Impact

**Show:** `17_updated_end_to_end_pipeline.svg` or the strongest before/after result.

**Voiceover:**  
"In short, our project connects the hard geometric preparation stage with ink detection and human review, helping turn unreadable scroll data into evidence that can be studied, questioned, and improved."

**Closing line:**  
`From a sealed scroll to reviewable digital evidence.`

## NotebookLM Prompt

Paste this into NotebookLM after uploading this file and the images in `docs/content/`.

```text
Create a 3-minute evaluator-facing demo video script for my project.

The project is called "Geometric Virtual Unwrapping and Volumetric Ink Detection of Carbonized Herculaneum Papyrus via X-ray Micro-Tomography." The short video title can be "Vesuvius Scroll Decipherment Pipeline."

The video should sell the project clearly to an evaluator. It should be demonstration-focused and not too technical.

Important: do not skip the pipeline before ink prediction. The story should explain that before ink can be detected, the CT volume must be prepared and the scroll surface must be virtually unwrapped.

Use this pipeline:
Scroll -> X-ray micro-CT volume -> preprocessing -> point cloud / normal grid / surface tensor -> VC3D segmentation -> seeding and tracing -> correction -> surface extension / virtual unwrapping -> ink detection model -> review interface.

Use these facts:
- The video should start with a formal title card that includes the college logo, project title, project members with SIDs, mentor, branch, college, and Major Project II.
- Project members: Arnav Vikas Garg (22106060), Sunit Mehta (22106059), Sumeet Kumar Panda (22106048), Kaushal Jain (22106045).
- Mentor: Dr. Trilok Chand, Professor, Department of Computer Science and Engineering.
- Institution: Punjab Engineering College (Deemed to be University), Chandigarh.
- Branch: Computer Science Engineering (Data Science) Branch.
- The Herculaneum scrolls were carbonized by Mount Vesuvius in 79 AD.
- Physically opening the scrolls can destroy them.
- X-ray micro-CT gives 3D scan data, but the scroll remains curled, noisy, and hard to interpret.
- The upstream work focuses on PHerc. 172 / ID 20241024131838, scanned at Diamond Light Source, with 7.91 micrometer resolution and 53 keV energy.
- Preprocessing includes Gaussian denoising, non-local means denoising, wavelet denoising, Frangi filtering, and CLAHE.
- VC3D is used for virtual unwrapping: seeding, tracing, correction, and surface extension.
- Use the real PPT screenshots for this part: before/after denoising, VC3D GUI, initial seed, before/after correction, before/after scrubbing, and surface extension.
- After a usable surface exists, ink detection and candidate review become meaningful.
- The ink detection stage uses a five-model soft-voting ensemble: U-Net small, ResNet-style CNN small, I3D-style 3D CNN small, Vision Transformer small, and TimeSformer-style transformer small.
- The model input is a 64 x 64 CT patch with 17 nearby z-slices.
- The model output is a pixel-wise ink probability map, which can be thresholded into a candidate ink mask.
- In the five-segment experiment, the best individual model was the I3D-style 3D CNN with provisional validation Dice 0.7598.
- The soft-voting ensemble reached provisional validation Dice 0.8142.
- Explain Dice simply as overlap between the predicted ink mask and the known ink label.
- Say clearly that the model predicts candidate ink regions, not final text.
- The website/demo assets show the later review stage: CT view, ink probability map, candidate letters, extracted strips, enhanced strips, and provisional model-assisted readings.

Important dataset wording:
Do not claim PHerc. 172, the five-segment model experiment, and PHercParis4 are the same dataset. Say that the upstream work focuses on preparing a scroll volume such as PHerc. 172, the model experiment demonstrates candidate ink detection on sampled CT patches, and the web demo shows how later-stage ink evidence and provisional review can be presented.

Use careful wording:
- Say "candidate", "provisional", "model-assisted", "sample-derived", and "not yet verified."
- Do not claim final verified transcription.
- Do not say the project fully solves Vesuvius scroll decipherment.

Create:
1. A scene-by-scene storyboard with timestamps
2. Voiceover narration for each scene
3. What to show on screen
4. Short slide text
5. A strong opening and closing
6. A list of website screens or visuals I should record

Use this timing:
0:00-0:08 Formal title card with college logo, member names, SIDs, mentor, branch, and college
0:08-0:25 Hook and problem
0:25-0:45 Why the problem is hard before ink prediction
0:45-1:08 Upstream contribution: preprocessing and virtual unwrapping
1:08-1:25 Complete end-to-end pipeline
1:25-1:58 Model and ink detection results
1:58-2:35 Website/review demonstration
2:35-2:50 Results and outputs
2:50-3:00 Impact and closing

Make the script confident, simple, natural, and evaluator-friendly.
```

## Things To Avoid Saying

- Do not imply ink prediction is the only contribution.
- Do not merge PHerc. 172, the five-segment model experiment, and PHercParis4 into one dataset.
- Do not describe the model as reading the scroll directly; it predicts candidate ink regions.
- Do not say provisional readings are verified final translations.
- Do not spend too long explaining model architecture, loss functions, or code.
- Do not show only code or terminal output.
- Do not describe hardcoded demo assets as live production inference.
- Do not hide uncertainty; evaluators usually trust the project more when uncertainty is labelled clearly.

## Recording Checklist

- Start with the formal title card for 6-8 seconds.
- Make sure member names, SIDs, mentor name, branch, college name, and logo are readable.
- Record the website in full-screen browser mode.
- Keep mouse movement slow and deliberate.
- Show one clear upstream pipeline slide before showing ink prediction.
- Include one visual for preprocessing or virtual unwrapping.
- Show one clear before/after comparison.
- Show at least one annotated candidate-letter result.
- Show the provisional reading or consensus panel.
- End on impact, not on limitations.
