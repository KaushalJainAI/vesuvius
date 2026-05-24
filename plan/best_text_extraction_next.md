# Best Text Extraction Plan

Updated after auditing the current outputs, UI assets, and code paths.

## Current Results

The current web-facing `result.json` files are no longer empty, but the amount
of extracted text is still too low for the visible ink:

| Segment | Strips | Consensus chars | Linked chars | API/model errors | Current summary state |
| --- | ---: | ---: | ---: | ---: | --- |
| 20230702185753 | 6 | 63 | 52 | 0 | uncertain fragment |
| 20230929220926 | 6 | 30 | 29 | 6 | fallback only |
| 20231005123336 | 6 | 35 | 31 | 6 | fallback only |
| 20231007101619 | 6 | 35 | 33 | 6 | fallback only |
| 20231012184424 | 6 | 37 | 31 | 6 | fallback only |
| 20231016151002 | 6 | 63 | 52 | 0 | uncertain fragment |
| 20231022170901 | 6 | 33 | 27 | 6 | fallback only |
| 20231031143852 | 6 | 73 | 70 | 0 | uncertain fragment |
| 20231106155351 | 6 | 72 | 61 | 0 | possible MALISTA-like reading |
| 20231210121321 | 6 | 33 | 30 | 6 | fallback only |
| 20231221180251 | 6 | 63 | 55 | 0 | uncertain participle/function-word fragment |

For comparison, the unused full-line extractor already detects much more
coverage. On `20231031143852`, using `label_full.png` converted to TIFF, it
found:

- 12 text lines
- 48 letter-scale crops
- 215 connected ink blobs
- median blob height 31 px
- line pitch 68 px

This confirms the main problem: the production path still samples 6 broad
strips, while the better full-coverage line/crop extractor exists but is not
integrated.

## What Is Working

- The S11 label enhancement path is the best current signal source.
- Reading plates are generated and served in the web assets.
- The UI can show transcription, translation, summary, and mapped boxes.
- Character-to-blob snapping works when characters are produced.
- The pipeline now avoids overwriting useful results when all API calls fail.

## Main Failures

1. **Coverage is too low.**
   Six strips cannot cover the whole segment. The UI shows many readable areas
   that never reach the model or template matcher.

2. **Template hints are capped too aggressively.**
   `src/decipher/template_hints.py` caps each strip at `max_records=15`.
   Across 6 strips that is only 90 candidate hints before the LLM even starts.
   This is below the number of visible letter-like forms.

3. **The LLM is being asked to infer too much from broad noisy images.**
   It should not be the first detector. It should rerank and explain a dense
   programmatic candidate lattice.

4. **Fallback outputs look like transcription but are not real transcription.**
   Six segments have model/API failures and are using blob-level placeholder
   guesses. These should be clearly treated as failed readings.

5. **The UI has some text encoding damage.**
   A few labels contain mojibake from prior edits. This does not break the
   algorithm, but it makes the page look less trustworthy.

## Best Next Strategy

### 1. Promote full-line extraction to the production path

Replace the active `strip_extractor` flow in `src/decipher/pipeline.py` with
`src/decipher/line_extractor.py` for the main transcription run.

Expected behavior:

- extract all detected lines, not 6 sampled strips
- tile each line into overlapping letter-scale crops
- keep full segment coordinates for every crop
- write line/crop metadata into `result.json`

Initial target:

- at least 10 lines per segment when the ink map visually contains that many
- 40 to 100 crops per segment depending on width and density

### 2. Build a programmatic candidate lattice before using the LLM

Create `candidate_lattice.json` per segment:

- line id
- crop id
- candidate bbox in full segment coordinates
- top Greek letter hypotheses
- confidence scores
- linked blob ids
- source features: template score, mean probability, area, aspect ratio,
  stroke density, topology hints

Use this as the primary object for transcription. The LLM should receive this
lattice plus reading plate images and choose among candidates instead of freely
inventing a line.

Initial target:

- 250+ candidate boxes per dense segment
- 80%+ of detected lines have candidates
- 70%+ of accepted characters link to a detected ink blob

### 3. Retune candidate generation for letter scale

Update `template_hints.py` and/or a new lattice extractor so it does not use
one fixed `min_area=1200` for every case.

Use dynamic thresholds based on `h_med` from `line_extractor`:

- multi-threshold passes: 0.35, 0.45, 0.55
- dynamic minimum area: proportional to median blob height
- dynamic maximum area: proportional to expected letter box area
- merge nearby fragments before classification
- keep candidate clusters by x-position rather than simply truncating to 15

The goal is to represent all plausible letters, not only the most confident
large blobs.

### 4. Make the LLM a reranker and translator

For each line/crop, send:

- raw line/crop image
- reading plate image
- candidate lattice records for that crop
- neighboring crop context

Prompt rule:

- choose from programmatic candidates where possible
- use `?` for unsupported letters
- return per-letter evidence id, not just free text
- provide English translation only after a line-level reading exists
- provide a probable scroll summary at segment level

This should reduce the current problem where the model predicts noise or
outputs generic "illegible" summaries.

### 5. Update the UI for extraction quality, not just text display

The Text Deciphering page should show:

- candidate lattice overlay
- accepted characters overlay
- rejected/low-confidence candidates in a faint style
- per-line crop thumbnails
- line transcription
- English translation
- probable text summary
- coverage metrics: candidates, accepted chars, linked blobs, failed crops

Also clean the mojibake labels and keep the thinner box border that was already
requested.

### 6. Rerun in two phases

Phase A: local only, no API

- generate line/crop assets
- generate candidate lattices
- mirror to web
- inspect coverage metrics

Phase B: model rerank

- rerun only after API quota is available
- skip or preserve segments if all calls fail
- do not overwrite successful older outputs with fallback guesses

## What Not To Do Next

- Do not simply increase yellow box size again. Box size helps visibility, but
  it does not create missing letters.
- Do not rerun the same 6-strip LLM pipeline and expect a major improvement.
- Do not treat fallback blob guesses as real transcription.
- Do not add another global image filter before fixing coverage and candidate
  generation.

## Immediate Implementation Order

1. Add a lattice builder around `line_extractor.extract_segment`.
2. Save `candidate_lattice.json` and crop/line assets for every segment.
3. Update the UI to display candidates separately from accepted characters.
4. Rerun the local lattice pass on all segments and measure coverage.
5. Integrate candidate lattice records into the LLM prompt.
6. Rerun the model phase on failed/low-coverage segments once API quota is
   available.

