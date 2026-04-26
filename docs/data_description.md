# Data Description — Vesuvius Scroll Decipherment Pipeline

This document is written for anyone who wants to understand the data: what it is physically, where it came from, how it was labelled, and how the pipeline uses it. No prior knowledge of the Vesuvius project is assumed.

---

## Part 1 — The Physical Story: What Are These Scrolls?

In 79 AD, the eruption of Mount Vesuvius buried the Roman town of Herculaneum under a torrent of volcanic ash and gas. One villa — believed to be owned by Julius Caesar's father-in-law — contained a library of hundreds of papyrus scrolls. The heat and gas carbonized them: they turned into fragile black cylinders of charcoal, indistinguishable from lumps of coal. For almost 2,000 years, nobody could read them.

Modern CT (computed tomography) scanners can image the interior of the scrolls without physically unrolling them. The scanner fires X-rays through the scroll from thousands of angles and reconstructs a 3D density map — the same technology used in hospital scans, just at much finer resolution (~7.91 micrometres per voxel, roughly 1/10th the width of a human hair).

The problem: the ink the ancient scribes used is **nearly invisible in X-ray**. Papyrus and carbon-based ink have almost the same X-ray density, so the letters don't stand out the way bones stand out from tissue. This is the core challenge the pipeline is trying to solve.

---

## Part 2 — How CT Data Becomes an Image (Voxels, Segments, Z-layers)

### The raw scan: a 3D voxel volume

Think of the CT scan as a 3D grid of tiny cubes, each one storing a brightness value (how dense that point in space is). A bright voxel = dense material. A dark voxel = air or low-density material. The entire rolled scroll is captured as one enormous 3D grid.

```
Brightness scale:
  bright (high value) = dense material (e.g., minerals in ancient ink traces)
  dark  (low value)   = air, or carbon papyrus/ink with low contrast
```

### The problem with a rolled scroll

The papyrus is rolled into a cylinder, so the text spirals around the inside. You can't just take a flat 2D photograph — you'd be looking through dozens of overlapping layers of papyrus at once. You first need to computationally "unroll" the scroll.

### Segmentation: cutting out a flat patch

Researchers at EduceLab wrote software that traces the surface of one layer of papyrus through the 3D volume and flattens it into a 2D image. Each traced and flattened patch is called a **segment**. One scroll produces hundreds of such segments (400+ for PHercParis4).

```
Full 3D CT volume of rolled scroll
        │
        ▼
Virtual unrolling: trace one papyrus layer
        │
        ▼
Segment: flat 2D patch of papyrus surface   ←── this is what we work with
```

### Surface volumes and Z-layers

A segment isn't just one 2D image — it's a stack of images taken at different depths through the papyrus sheet. Imagine slicing the flattened papyrus horizontally at 33 or 65 different depths, producing 33 or 65 greyscale images stacked on top of each other.

```
Z-layer 00  ──  surface of the papyrus (outermost)
Z-layer 01  ──  slightly deeper
Z-layer 02  ──  deeper still
   ...
Z-layer 32  ──  inner surface (for 33-layer data)
Z-layer 64  ──  inner surface (for 65-layer data)
```

This stack is called the **surface volume**. Each Z-layer is stored as one TIFF image file (`00.tif`, `01.tif`, …). The model takes all Z-layers together as a 3D input and learns which depth patterns indicate ink.

### Why multiple Z-layers?

Ink sits on the surface of the papyrus, but the CT voxel values around an ink trace are subtly different across several depth layers — not just on one slice. The model sees the full depth stack and learns to pick up this 3D signature.

---

## Part 3 — How Ink Was Detected (The Crackle Pattern Discovery)

For a long time, researchers assumed carbon-based ink would be undetectable in CT. In 2023, a researcher named Casey Handmer discovered that ancient ink leaves a faint **crackle pattern** — a texture of tiny cracks and ridges — in the surface of the papyrus. This pattern is visible in the CT voxels if you look at the right depth layers and at the right scale.

The crackle is not obvious to the naked eye. It appears as subtle brightness variations across neighboring voxels in the surface volume layers. Machine learning models — especially 3D CNNs that look across all Z-layers simultaneously — can learn to recognize this pattern and output a per-pixel probability of ink being present.

```
CT surface volume (3D)
     │
     ▼
3D CNN processes all Z-layers together
     │
     ▼
2D ink probability map  (bright pixel = likely ink)
     │
     ▼
Threshold + post-process → binary ink mask → letter candidates
```

---

## Part 4 — How the Labels Were Created

Labels tell the model: "at this pixel, there is (or isn't) ink." Without labels, supervised training is impossible. There are two completely different label creation methods used in this project.

---

### The Bootstrap Problem — How Labels Existed at All

Before any model could be trained, someone had to answer a hard question: **how do you label ink that is invisible in CT, inside a sealed scroll that nobody has opened in 2,000 years?**

The answer is that you cannot — not directly. The bootstrap relied on a lucky physical accident and a series of steps, each one making the next possible:

```
Step 1 — Physical accident: some papyrus detached
   │
   │  Over centuries, some pieces broke off the scrolls
   │  and landed face-up, exposing their ink to air.
   │  These are the Kaggle "fragments."
   ▼

Step 2 — IR photography on fragments
   │
   │  Exposed fragments can be photographed with infrared light.
   │  Carbon-based ink absorbs IR differently from carbonized papyrus
   │  → letters become clearly visible in IR images even though
   │    they are invisible in ordinary light and nearly invisible in CT.
   ▼

Step 3 — Manual annotation (ground-truth labels)
   │
   │  The Vesuvius Challenge released these fragments on Kaggle
   │  with a task: train a model to find ink in the CT Z-stack
   │  using the IR image as the label source.
   │
   │  Each fragment was CT-scanned (65 layers) AND IR-photographed.
   │  The IR photo was pixel-aligned to the CT volume.
   │  Human experts traced the visible ink → binary mask (inklabels.png).
   │
   │  These are the only human-verified ink labels in existence.
   ▼

Step 4 — Train a model on fragment labels (Kaggle competition)
   │
   │  Teams worldwide trained ink-detection models on the 4 Kaggle fragments.
   │  The best models learned to find the CT crackle signature of ink
   │  by correlating it with the IR-derived ground-truth masks.
   ▼

Step 5 — Grand Prize: run the trained model on sealed scroll segments
   │
   │  The GP-winning team (Nader / Farritor / Schilliger, 2023)
   │  used a TimeSformer model trained on fragment labels and ran
   │  sliding-window inference on PHercParis4 scroll segments.
   │
   │  For the first time, ink was detected inside a sealed scroll —
   │  producing a readable ancient Greek transcription.
   ▼

Step 6 — Release model outputs as pseudo-labels
   │
   │  The GP-winner model's ink probability maps for 11 segments
   │  were released publicly as ink_labels.tif files.
   │
   │  These are what this project trains on.
   ▼

Step 7 — Our segment track: train on pseudo-labels
   │
   │  SegmentInkNet (and eventually TimeSformer) is trained
   │  on the GP-winner's outputs. We are learning to reproduce
   │  and improve on what the first readable model predicted.
   ▼

Target: surpass the pseudo-label ceiling via iterative refinement
        and by retraining the same TimeSformer architecture
        end-to-end on all 11 segments.
```

**The key insight**: the entire label chain starts from physical fragments that happened to land face-up. Without those few detached pieces, there would be no IR photos, no ground-truth masks, no Kaggle competition, no GP-winner model, and no pseudo-labels for scroll segments. Every label in this project traces back to that physical accident.

---

### Method A — Manual labels (Kaggle fragments)

**What are fragments?**
Some papyrus pieces detached from the scrolls over the centuries and landed face-up, so their ink-covered surfaces were exposed to air. Unlike the sealed scroll interior, these pieces can be photographed directly using **infrared (IR) photography**. Ancient carbon-based ink absorbs infrared light differently from carbonized papyrus, making the letters clearly visible in IR images — even when they are invisible in ordinary light.

**The labelling process:**
1. A fragment is CT-scanned to get its 3D surface volume (same 65-layer format as scroll segments).
2. The same fragment is photographed under infrared light — this produces a high-contrast image where ink is bright and papyrus is dark.
3. The IR photograph is manually aligned (registered) to the CT surface volume so they match pixel-for-pixel.
4. A human expert traces the visible ink in the IR image to create a **binary mask**: white pixel = ink present, black pixel = no ink.
5. This binary mask becomes `inklabels.png` — the ground truth label.

```
Fragment CT scan              Fragment IR photo
(3D surface volume)           (ink clearly visible)
        │                             │
        └─────── align to match ──────┘
                        │
                        ▼
              Manual annotation by human
                        │
                        ▼
              inklabels.png  (binary: 0 = no ink, 255 = ink)
```

**Quality:** These are the highest-quality labels in the project. A human looked at actual visible ink and drew the mask. These are called **ground truth** labels.

**Limitation:** Only a handful of fragments exist with exposed ink. You cannot do this for the sealed scroll — its ink is buried inside and cannot be photographed.

---

### Method B — Pseudo-labels (labelled segments from PHercParis4)

**The problem:** For the actual scroll segments, there is no IR photo — the papyrus is still sealed inside the rolled scroll, ink face down, invisible. Nobody has physically seen this text. You can't hand-draw labels.

**The solution:** Use a previously trained model's output as labels.

In 2023, the Vesuvius Challenge Grand Prize was won by a team (Youssef Nader, Luke Farritor, Julian Schilliger) who built a TimeSformer-based model that successfully detected ink in scroll segments. Their model — trained on the Kaggle fragment ground-truth labels — was run on 11 PHercParis4 scroll segments and produced ink probability maps for each one.

These probability maps were released publicly. Instead of a binary "ink / no ink" mask, they are **continuous probability images**: each pixel has a value from 0.0 to 1.0 (stored as 0–255 in uint8) representing how confident the GP-winner model was that ink was present there.

```
GP-winner model (trained on fragment ground truth)
        │
        ▼
Run inference on 11 PHercParis4 scroll segments
        │
        ▼
ink probability maps (0.0–1.0 per pixel)
        │
        ▼
Released as ink_labels.tif  ←── these are the "pseudo-labels"
```

**Why "pseudo"?** Because they are not verified by a human. They are a model's best guess. If the GP-winner model was wrong somewhere, those errors are baked into the labels. Our model will try to reproduce what the GP-winner model predicted, not what the actual ink looks like. The labels are a ceiling: our segment-track model cannot outperform the model that generated its labels.

**The ignore-band:** Because the GP-winner model was uncertain in some regions (probability ≈ 0.5), the training loss deliberately ignores pixels where the label value falls between 0.4 and 0.6. Training only on confident predictions (very likely ink or very likely not-ink) prevents the model from fitting noise.

**Implication for model choice:** Because our pseudo-labels were generated by a TimeSformer, the most direct path to surpassing them is to train our own TimeSformer — the same architecture class, retrained end-to-end on all 11 segments with a refined decoder and iteratively improved labels. See [plan/model_improvement.md](../plan/model_improvement.md) for the full roadmap.

```
Pseudo-label pixel value → training treatment:
  > 0.6  →  supervise as INK
  < 0.4  →  supervise as NO INK
  0.4–0.6  →  IGNORED (uncertain region, excluded from loss)
```

---

## Part 5 — Three Data Categories

All data in this project comes from CT scans of PHercParis4 (Herculaneum Scroll 1), with one exception: the Kaggle fragments, which are physically separate detached pieces. All use the same ~7.91 µm scanning resolution and the same TIFF stack format.

| Category | Location | Labels | Z-layers | Use |
|---|---|---|---|---|
| Kaggle fragments | `data/train/` | Manual ground truth | 65 | Train fragment-track model |
| Scroll segments | `data/scroll/` | None | 65 | Inference only |
| Labelled segments | `data/labelled_segments/` | Pseudo (GP-winner output) | 33 | Train segment-track model |

---

## Part 6 — Category Details

### Category 1 — Kaggle Fragments (`data/train/`)

Small detached papyrus pieces with exposed ink surfaces, CT-scanned and manually labelled via IR photography alignment. The only ground-truth labels in the project.

**Source:** [Vesuvius Challenge Kaggle competition](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection)

**Disk layout:**
```
data/train/
  fragment1/
    surface_volume/
      00.tif … 64.tif     # 65 Z-layers, uint16 TIFF, ~88 MB each
    mask.png               # valid papyrus region (white = usable area)
    inklabels.png          # binary ink mask — 0 = no ink, 255 = ink
  fragment2/
  fragment3/
  fragment4/
```

**Key numbers:**
- 65 Z-layers per fragment, ~88 MB per layer
- `inklabels.png` is binary (0 or 255), pixel-aligned with all Z-layers
- `mask.png` marks which pixels are actually on papyrus vs background/air

**Role:** Train `VesuviusNet` (fragment track). One fragment is held out as the validation set. The trained model is then run on scroll segments to find ink there.

---

### Category 2 — Scroll Segments (`data/scroll/`)

Full PHercParis4 scroll segments — real unread text, no ink labels. Used only for inference (running a trained model to find where ink is).

**Source:** `https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/`
400+ segments available; 7 are downloaded by this project.

**What the server stores per segment:**
```
paths/{segment_id}/
  layers/
    00.tif … 64.tif        # 65 Z-layers, ~88 MB each
  {segment_id}_mask.png    # papyrus region mask
  {segment_id}.obj         # 3D mesh of the papyrus surface geometry
  {segment_id}.ppm         # UV map connecting 3D mesh to 2D image
  meta.json                # scan metadata (resolution, date, etc.)
```

Note: the server calls the folder `layers/` and prefixes the mask with the segment ID. The download script renames these to match the local convention (`surface_volume/` and `mask.png`).

**Disk layout after download:**
```
data/scroll/
  {segment_id}/
    surface_volume/
      00.tif … 64.tif      # renamed from server's 'layers/'
    mask.png               # renamed from '{id}_mask.png'
    meta.json
```

**The 7 downloaded segments:**

| Segment ID | Size | Note |
|---|---|---|
| 20230827161847 | ~5.73 GB | Grand Prize — first confirmed readable text |
| 20230826170124 | ~2.35 GB | Adjacent region |
| 20230820174948 | ~1.87 GB | Adjacent region |
| 20230826135043 | ~1.55 GB | Adjacent region |
| 20230828154913 | ~1.40 GB | Day after Grand Prize scan |
| 20230819093803 | ~1.15 GB | Adjacent region |
| 20230504093154 | ~1.03 GB | Smaller early segment |
| **Total** | **~15.1 GB** | |

Download: `python download_segments.py`

**Role:** Inference target only. A trained model runs over these in a sliding window and outputs `predictions/{id}_prob.npy` — a probability map of where ink is located.

---

### Category 3 — Labelled Segments (`data/labelled_segments/`)

11 PHercParis4 scroll segments with pseudo-labels from the Grand Prize winning model. Same underlying CT scan as Category 2, but packaged with a shallower Z-stack (33 layers instead of 65) and with labels attached.

**Source:**
- Surface volumes: `https://data.aws.ash2txt.org/samples/PHercParis4/segments/`
- Volume ID: `7.91um-54keV-volume-20230205180739` (same CT acquisition as scroll segments)
- The AWS host uses a self-signed TLS certificate — the download script disables SSL verification for this host only.

**Labels are already downloaded** into `data/labelled_segments/` and committed to the repo. Only the surface volumes need to be downloaded.

**Server structure per segment:**
```
segments/{segment_id}/
  surface-volumes/
    7.91um-54keV-volume-20230205180739.tifs/
      00.tif … 32.tif      # 33 Z-layers
```

**Disk layout after download:**
```
data/labelled_segments/
  {segment_id}/
    ink_labels.tif          # pseudo-label probability map (uint8, 0–255) — PRE-DOWNLOADED
    surface_volume/
      00.tif … 32.tif       # 33 Z-layers — download separately
```

**`ink_labels.tif` explained:**
- Single-channel TIFF, same spatial dimensions as one Z-layer
- Pixel values 0–255, representing GP-winner model's ink probability (0 = no ink, 255 = definite ink)
- Not binary — it is a continuous probability map
- Pixels with values 102–153 (≈ 0.4–0.6 probability) are **ignored during training** because the GP-winner model was not confident there

**The 11 labelled segments:**

| # | Segment ID | Surface Vol | Label | Recommended |
|---|---|---|---|---|
| 1 | 20231221180251 | 3.5 GB | 25 MB | Start here — smallest, default val |
| 2 | 20231031143852 | 3.7 GB | 26 MB | Start here |
| 3 | 20231016151002 | 4.0 GB | 28 MB | Start here |
| 4 | 20231106155351 | 4.5 GB | 35 MB | |
| 5 | 20230702185753 | 4.6 GB | 36 MB | |
| 6 | 20231210121321 | 5.1 GB | 37 MB | |
| 7 | 20230929220926 | 6.1 GB | 50 MB | |
| 8 | 20231022170901 | 6.3 GB | 46 MB | |
| 9 | 20231005123336 | 8.6 GB | 70 MB | |
| 10 | 20231012184424 | 9.8 GB | 72 MB | |
| 11 | 20231007101619 | 14.2 GB | 111 MB | Largest |
| **Total** | | **~72 GB** | **~540 MB** | |

Download commands:
```bash
# See all segments and which labels are already present:
python download_labelled_segment.py

# Smallest 3 (~11 GB, good starting point):
python download_labelled_segment.py --seg 20231221180251 20231031143852 20231016151002

# All 11 (~72 GB):
python download_labelled_segment.py --all
```

**Disk budget guide:**

| Available disk | Recommended segments | Volume total |
|---|---|---|
| 10–12 GB | first 3 | ~11 GB |
| 15–20 GB | first 5 | ~17 GB |
| 30+ GB | first 8 | ~30 GB |
| 70+ GB | all 11 | ~72 GB |

**Role:** Train `SegmentInkNet` (segment track). `20231221180251` is held out as the default validation segment. After training, inference runs on any segment — labelled or unlabelled — to produce new probability maps.

---

## Part 7 — Comparing Category 2 vs Category 3

These two categories are both PHercParis4 scroll segments from the same CT machine. They differ in packaging and purpose:

| Dimension | Scroll segments (cat. 2) | Labelled segments (cat. 3) |
|---|---|---|
| Z-layers | **65** | **33** |
| Ink labels | **No** | **Yes** (pseudo, from GP-winner) |
| Label quality | — | Noisy — bounded by GP-winner accuracy |
| Primary use | Inference | Training |
| # on server | 400+ | 11 (curated) |
| Downloaded | 7 | up to 11 |
| Server | `dl.ash2txt.org` (public CDN) | `data.aws.ash2txt.org` (AWS, self-signed cert) |
| Extra files on server | mask, meta.json, OBJ mesh, PPM | none |
| Download script | `download_segments.py` | `download_labelled_segment.py` |
| Model that uses it | `VesuviusNet` (fragment track, inference) | `SegmentInkNet` (segment track, training) |

**The 33 vs 65 Z-layer difference matters for models.** `SegmentInkNet` uses `AdaptiveAvgPool3d` in its Z encoder so it accepts any Z depth. `VesuviusNet` expects exactly 65 layers. If you want to run `VesuviusNet` on 33-layer data, you must either pad the Z dimension or retrain.

---

## Part 8 — What Does a Probability Map Look Like?

After inference, the model outputs a file like `predictions/20230827161847_prob.npy`. Conceptually:

```
Each pixel in the original segment surface image
        │
        ▼
Has a probability value 0.0 → 1.0
  0.0 = model is confident there is NO ink here
  1.0 = model is confident there IS ink here
  0.5 = model is uncertain
```

This float32 probability map is then post-processed to produce a usable binary ink mask:
1. **Threshold** at 0.45 (fragment track) or 0.40 (segment track) → pixels above = ink
2. **Morphological closing** with a small disk kernel → closes tiny gaps within letters
3. **Remove small components** below 50 pixels → eliminates isolated noise specks

The result is a clean binary mask where white blobs correspond to individual ink strokes. These blobs are then fed to the letter isolation stage.

---

## Part 9 — Full Disk Layout

```
data/
  train/                              # Kaggle fragments — manual labels, 65-layer
    fragment1/
      surface_volume/00–64.tif        # input to model
      mask.png                        # valid region
      inklabels.png                   # ground-truth label
    fragment2/ fragment3/ fragment4/

  scroll/                             # 7 PHercParis4 segments — no labels, 65-layer
    20230827161847/                   # Grand Prize segment (first readable text)
      surface_volume/00–64.tif
      mask.png
      meta.json
    20230826170124/ … 20230504093154/

  labelled_segments/                  # 11 PHercParis4 segments — pseudo-labels, 33-layer
    20231221180251/                   # smallest — default validation segment
      ink_labels.tif                  # LABEL — pre-downloaded, uint8 probability map
      surface_volume/00–32.tif        # INPUT — download separately
    20231031143852/ … 20231007101619/

models/
  best_model.pth                      # fragment-track best weights (saved at best val F0.5)
  best_segment_model.pth              # segment-track best weights

predictions/
  {segment_id}_prob.npy               # float32 probability map from inference
  {segment_id}_prob.tif               # same as uint8 TIFF (for viewing in image tools)
  {segment_id}_vis.png                # side-by-side: CT slice / prediction / label / overlay

filter_experiments/
  candidates.csv                      # per-component top-5 Greek letter guesses + scores
  candidates_grid.png                 # visual grid of all extracted letter candidates
  templates.png                       # 24 Greek uppercase letter templates used for matching
```

---

## Part 10 — What Needs to Be Downloaded

The repository does **not** include the raw CT data (too large for git). The following must be downloaded before training or inference:

| Data | Size | How to get it |
|---|---|---|
| `data/train/` Kaggle fragments | ~2 GB | Kaggle competition page (manual download) |
| `data/scroll/` 7 scroll segments | ~15 GB | `python download_segments.py` |
| `data/labelled_segments/*/surface_volume/` | ~72 GB total | `python download_labelled_segment.py` |

**Already in the repo** (no download needed):
- `data/labelled_segments/*/ink_labels.tif` — all 11 pseudo-label files (~540 MB total)
- All model code, notebooks, scripts

---

## External References

- [Vesuvius Challenge](https://scrollprize.org/) — competition home, background on the scrolls
- [Grand Prize announcement](https://scrollprize.org/grandprize) — winning team and method
- [Kaggle Ink Detection Competition](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection) — source of fragment ground-truth labels
- [EduceLab Scrolls Dataset](https://dl.ash2txt.org/) — full scroll segment CDN (400+ segments of PHercParis4)
