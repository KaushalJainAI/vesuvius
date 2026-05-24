# Deciphering Pipeline

Status note: the extraction strategy is being revised because the current
six-strip production path under-extracts visible text. See
[`best_text_extraction_next.md`](best_text_extraction_next.md) for the current
audit, failure diagnosis, and next implementation order.

This document describes the current deciphering flow and the alternate
script-driven pipeline now available under `scripts/`. The training and label
enhancement work still produces the best available ink map; this pipeline takes
that map and turns it into strip images, model readings, consensus JSON, and
static web assets.

## Inputs and Outputs

Primary input, in priority order:

1. `data/labelled_segments/{seg_id}/ink_labels_enhanced.tif`
2. `data/labelled_segments/{seg_id}/ink_labels.tif`
3. For visualization-only scripts, `predictions/**/{seg_id}_prob.npy` or
   `predictions/**/{seg_id}_prob.tif` can also be used.

Primary outputs:

- `predictions/decipher/{seg_id}/strips/strip_NN.png`
- `predictions/decipher/{seg_id}/result.json`
- `web/public/assets/decipher/{seg_id}/strips/strip_NN.png`
- `web/public/assets/decipher/{seg_id}/result.json`
- `web/public/assets/decipher/{seg_id}/result.detection.json`
- `web/public/assets/decipher/index.json`

The web app reads the mirrored files from `web/public/assets/decipher/`, so the
browser never needs an API key. `result.json` is reserved for strip readings and
model consensus. `result.detection.json` is reserved for detection-only line and
blob overlays.

## Current End-to-End Pipeline

The current production path is implemented in `src/decipher/` and orchestrated
by `scripts/decipher_all_segments.py`.

```text
ink_labels_enhanced.tif or ink_labels.tif
    -> src/decipher/strip_extractor.py
    -> strip PNGs
    -> src/decipher/template_hints.py
    -> OpenRouter vision model calls
    -> optional manual readings from predictions/decipher/{seg_id}/manual/*.json
    -> src/decipher/consensus.py
    -> predictions/decipher/{seg_id}/result.json
    -> web/public/assets/decipher/{seg_id}/result.json
```

### Stage 1: Strip Extraction

`src/decipher/strip_extractor.py` detects row-density peaks in the label image
and cuts wide horizontal strips around likely text lines. The default
orchestrator extracts six strips per segment.

The same strip metadata is saved into `result.json`:

- `strip_id`
- `image_path`
- `y_range`
- `x_range`
- `y_center`

### Stage 2: Template Hints

`src/decipher/template_hints.py` runs the existing Greek uncial template matcher
on each strip before any LLM call. The hint block contains:

- probable connected-component positions as normalized `x_norm`
- confidence tier for each component
- top candidate Greek letters and soft probabilities
- a compact Herculaneum / Epicurean Greek context block

These hints are prompt context only. The strip image remains the primary visual
evidence.

### Stage 3: Vision-LLM Reading

`src/decipher/openrouter_client.py` sends each strip image to the configured
OpenRouter vision models. The model registry is data-driven in
`src/decipher/model_registry.py`.

Required environment:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
```

Expected model output schema:

```json
{
  "line_text": "PORFYRA",
  "characters": [
    {
      "char": "P",
      "x_norm": 0.05,
      "confidence": 0.9,
      "alternatives": ["G"]
    }
  ],
  "translation_en": "",
  "notes": "",
  "overall_confidence": "medium"
}
```

Actual Greek characters are preserved in UTF-8 JSON. The ASCII example above is
only for readability in this document.

### Stage 4: Manual Readings

Manual readings can be dropped into:

```text
predictions/decipher/{seg_id}/manual/{model_name}.json
```

Expected shape:

```json
{
  "model": "manual/claude-opus-4.7",
  "strips": {
    "0": {
      "line_text": "",
      "characters": [],
      "translation_en": "",
      "meaning": "",
      "notes": "",
      "overall_confidence": "low"
    }
  }
}
```

`src/decipher/pipeline.py` merges these manual readings into the same consensus
pass as the OpenRouter readings.

### Stage 5: Consensus

`src/decipher/consensus.py` aligns per-model characters by normalized x-position
and votes per character. Each model contributes at most one vote to a consensus
position, so duplicate characters from one model cannot overweight agreement.
The output keeps both the consensus and the dissenting model alternatives so
uncertainty is visible in the UI.

Each strip output contains:

- `template_hints`
- `per_model`
- `consensus.text`
- `consensus.translation_en`
- `consensus.meaning`
- `consensus.characters`

### Stage 6: Static Web Mirror

After each segment finishes, `scripts/decipher_all_segments.py` mirrors the
result into `web/public/assets/decipher/{seg_id}/` and refreshes
`web/public/assets/decipher/index.json`. `DecipherPanel.tsx` reads this static
asset tree.

## Main Commands

Extract strips and generate mock readings for one segment:

```powershell
.\venv\Scripts\python.exe scripts\decipher_all_segments.py --mock --only 20231221180251
```

Run the real OpenRouter pipeline for one segment:

```powershell
.\venv\Scripts\python.exe scripts\decipher_all_segments.py --only 20231221180251
```

Run the real pipeline across all configured labelled segments:

```powershell
.\venv\Scripts\python.exe scripts\decipher_all_segments.py
```

Control how many strips are extracted:

```powershell
.\venv\Scripts\python.exe scripts\decipher_all_segments.py --only 20231221180251 --n-strips 8
```

## Alternate Pipeline in `scripts/`

The alternate pipeline is useful when the full multi-model path is too noisy,
too slow, or when the goal is to create inspectable demo assets rather than
automated readings.

```text
best available ink map
    -> scripts/best_decipher_visualization.py
    -> best.png / best_overview.png / grid.png / scores.json
    -> scripts/build_decipher_demo.py
    -> strip PNGs + overview.png + result.skeleton.json
    -> manual reading fill-in
    -> web/public/assets/decipher/{seg_id}/result.json
```

### Alternate Step A: Pick the Most Readable Rendering

`scripts/best_decipher_visualization.py` searches for the best available input
for each segment, then renders six readability variants:

- `raw`
- `gamma_ir`
- `clahe_sharp`
- `unsharp`
- `tv_sato`
- `hyst_clean`

It scores each variant on a high-density text crop and writes:

- `predictions/best_decipher_vis/{seg_id}/best.png`
- `predictions/best_decipher_vis/{seg_id}/best_overview.png`
- `predictions/best_decipher_vis/{seg_id}/grid.png`
- `predictions/best_decipher_vis/{seg_id}/scores.json`

Commands:

```powershell
.\venv\Scripts\python.exe scripts\best_decipher_visualization.py
.\venv\Scripts\python.exe scripts\best_decipher_visualization.py --only 20231012184424 20231221180251
.\venv\Scripts\python.exe scripts\best_decipher_visualization.py --crop 1024
```

Use this when you need a human-readable or LLM-readable image before committing
to strip extraction and consensus.

### Alternate Step B: Build Static Demo Assets

`scripts/build_decipher_demo.py` creates web-ready assets for a single segment:

- six crisp strip PNGs
- an overview thumbnail with strip bands overlaid
- `result.skeleton.json` with coordinates and empty manual-reading fields

Command:

```powershell
.\venv\Scripts\python.exe scripts\build_decipher_demo.py --seg 20230702185753
```

After the skeleton is created, `scripts/build_all_decipher.py` converts the
line/blob metadata into the detection-only web overlay file:

```text
web/public/assets/decipher/{seg_id}/result.detection.json
```

This path is intentionally manual-first. It is best for demos, presentations,
and cases where visual inspection is more trustworthy than automated model
agreement.

### Alternate Step C: Readable Kaggle Sanity Check

`scripts/decipher_readable_sample.py` runs the same model and consensus stack on
`web/public/assets/samples/ink_labels_real.png`, a manually annotated Kaggle
fragment image that is actually readable. It sends the whole upscaled image as
one strip and mirrors the result to:

```text
web/public/assets/decipher/sample_readable_kaggle/
```

Command:

```powershell
.\venv\Scripts\python.exe scripts\decipher_readable_sample.py
```

Use this to separate pipeline failures from upstream image-quality limitations:
if the readable sample works but PHercParis4 segment strips do not, the
bottleneck is the ink map resolution rather than the JSON, consensus, or web
asset plumbing.

### Alternate Step D: Simple Strip Extraction

`scripts/extract_text_strips.py` is the lightweight inspection tool. It extracts
wide text-line strips and a contact sheet to `predictions/text_strips/{seg_id}/`
without model calls or web mirroring.

Command:

```powershell
.\venv\Scripts\python.exe scripts\extract_text_strips.py --seg 20231221180251
```

Use this for quick visual triage.

## Choosing a Path

Use the current end-to-end pipeline when:

- you want model readings and consensus JSON
- you have `OPENROUTER_API_KEY` configured
- the output should be immediately available in the web app
- you want mock data to exercise UI states

Use the alternate script pipeline when:

- the segment needs visual enhancement before reading
- you want a manually curated deciphering demo
- you need a sanity check against a clearly readable labelled image
- you only need strip/contact-sheet inspection

## Result JSON Contract

Every web-consumable result should preserve this shape:

```json
{
  "seg_id": "20231221180251",
  "created": "2026-05-23T00:00:00+00:00",
  "label_path": "data/labelled_segments/20231221180251/ink_labels_enhanced.tif",
  "models_used_open_source": [],
  "models_used_manual": [],
  "n_strips": 6,
  "strips": [
    {
      "strip_id": 0,
      "image_path": "strips/strip_00.png",
      "y_range": [0, 200],
      "x_range": [0, 2400],
      "y_center": 100,
      "template_hints": [],
      "per_model": {},
      "consensus": {
        "text": "",
        "translation_en": "",
        "meaning": "",
        "characters": []
      }
    }
  ]
}
```

Additional fields such as `mock`, `overview_path`, `overview_dims`,
`source_note`, and `segment_meaning` are allowed. The web app should tolerate
missing optional fields but expects `strips`, `image_path`, and `consensus`.

## Notes and Limitations

- LLM readings are hypotheses, not ground truth.
- Manual and automated readings should keep raw model output when available.
- LLM-assisted text should not become training ink unless visual evidence exists.
- Low agreement is still useful: it tells us which strips are not visually
  recoverable with the current ink map.
- The deciphering pipeline improves as the upstream ink detection and label
  enhancement pipelines produce sharper maps.
