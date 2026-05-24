# Hardcoded Demo Flow

This frontend intentionally keeps several values hardcoded so the application can demonstrate the PHercParis4 archive-review flow without requiring the full local data pipeline, large segment downloads, or model inference.

## User-facing wording policy

- Present the app as an archive catalogue and record workspace, not as a finished transcription product.
- Use restrained terms such as `provisional`, `candidate`, `model-assisted`, `sample-derived`, and `not yet verified`.
- Do not surface raw implementation labels such as `mock`, `placeholder`, `AI reading`, or `demo transcription` in user-facing copy.
- Upload/intake copy should describe a prototype local-file interaction, not a production guarantee about data handling.
- Reading and translation text should be labelled as provisional unless a later verified source is added.

## Demo Data

- `public/manifest.json` is the main hardcoded segment catalog.
- Each segment has fixed metadata: id, label, size, dimensions, layer count, archive-style description, preview image paths, and letter bounding boxes.
- The `letters` arrays are candidate character records. They are not loaded from a live model endpoint.
- Each segment currently has at least eight candidate characters so the viewer can show a useful review flow.

## Shared Sample Assets

- Most segments reuse `/assets/samples/ct_preview.jpg`.
- Most segments reuse `/assets/samples/ink_prob.png`.
- Segment A uses `/assets/samples/ink_labels_real.png` as a label preview.
- These assets stand in for real per-segment generated previews.

## Homepage Demo Values

- Segment count, total data size, maximum Z-layer count, and total candidate characters are derived from `manifest.json`.
- The upload interaction is simulated with a fixed delay in `src/pages/Home.tsx`.
- Skeleton loading uses a fixed card count to keep the layout stable while the manifest loads.

## Viewer Demo Values

- `src/components/viewer/InteractiveDecoder.tsx` uses a fixed 512 x 512 virtual coordinate space because the demo letter boxes are normalized to that canvas.
- The inspection window uses fixed sizing and sweep behavior.
- The characters-in-view text is computed by checking which hardcoded candidate positions overlap the inspection window.
- `src/components/viewer/DecipherAnimation.tsx` uses fixed pipeline phase labels and animation timings.
- `src/hooks/useAnimationSequence.ts` contains fixed phase durations.

## Multi-LLM Decipher Demo

- `public/assets/decipher/20231221180251/result.json` is prototype reading data.
- The result shows the intended model-comparison and consensus UI without requiring API calls.
- The app labels these records as prototype/provisional reading data when that flag is present.

## Hardcoded values intentionally surfaced

- Accession-like label: `PHercParis4`.
- Segment ids from `public/manifest.json`.
- Sample-derived segment sizes, dimensions, z-layer counts, and candidate character counts.
- Label-overlay availability and source volume links.

## Hardcoded values not surfaced as raw implementation language

- The internal `mock` flag in result JSON.
- Placeholder/prototype generation details.
- Raw demo labels such as `Full demo transcription`, `AI reading`, or `Live decoded text`.
- Any claim that uploaded files are processed by a complete local production pipeline.

## External Data Link

- `src/pages/Viewer.tsx` still has a raw data link template for PHercParis4 segment data.
- This is kept as part of the real application flow, but it can be removed if the frontend should not link to external data at all.

## Why Keep These Hardcoded

The hardcoded layer lets the frontend communicate the prototype archive-review flow:

1. Choose a segment.
2. Inspect CT and ink probability views.
3. Move across candidate characters.
4. Review confidence and provisional reading text.
5. Compare model-assisted consensus output.

Replacing these with live data should happen behind the same component contracts: `Manifest`, `SegmentMeta`, and `LetterBBox`.
