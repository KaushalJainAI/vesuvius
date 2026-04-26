# Labelled Segment Map

Source: `https://data.aws.ash2txt.org/samples/PHercParis4/segments/`
Volume ID: `7.91um-54keV-volume-20230205180739` (same CT scan as your existing scroll data)
Z-layers: **33** (00–32) — note: different from your existing segments which are 65 layers

## Input / Label mapping

```
data/labelled_segments/{segment_id}/
  ink_labels.tif        ← LABEL  (downloaded, ink probability map)
  surface_volume/       ← INPUT  (to download: 33× .tif Z-stack)
    00.tif … 32.tif
```

## Segment sizes (sorted smallest → largest)

| # | Segment ID     | Surface Volume | Label (downloaded) | Total   | Recommended |
|---|---------------|---------------|-------------------|---------|-------------|
| 1 | 20231221180251 | 3.5 GB        | 25 MB ✓           | ~3.5 GB | ★ Start here |
| 2 | 20231031143852 | 3.7 GB        | 26 MB ✓           | ~3.7 GB | ★ Start here |
| 3 | 20231016151002 | 4.0 GB        | 28 MB ✓           | ~4.0 GB | ★ Start here |
| 4 | 20230702185753 | 4.6 GB        | 36 MB ✓           | ~4.6 GB |             |
| 5 | 20231106155351 | 4.5 GB        | 35 MB ✓           | ~4.5 GB |             |
| 6 | 20231210121321 | 5.1 GB        | 37 MB ✓           | ~5.1 GB |             |
| 7 | 20231022170901 | 6.3 GB        | 46 MB ✓           | ~6.3 GB |             |
| 8 | 20230929220926 | 6.1 GB        | 50 MB ✓           | ~6.1 GB |             |
| 9 | 20231005123336 | 8.6 GB        | 70 MB ✓           | ~8.6 GB |             |
|10 | 20231012184424 | 9.8 GB        | 72 MB ✓           | ~9.8 GB |             |
|11 | 20231007101619 | 14.2 GB       | 111 MB ✓          | ~14.2 GB| Largest     |

**Total if all downloaded: ~72 GB surface volumes + ~540 MB labels**

## Suggested training sets by system capability

| Available disk | Suggested segments | Surface vol total |
|---------------|-------------------|------------------|
| 10–12 GB      | 20231221180251, 20231031143852, 20231016151002 | ~11 GB |
| 15–20 GB      | above + 20230702185753, 20231106155351        | ~17 GB |
| 30+ GB        | above + 20231210121321, 20231022170901, 20230929220926 | ~30 GB |
| 70+ GB        | all 11                                        | ~72 GB |

## Model note: Z-depth mismatch

Your existing `VesuviusNet` expects 65-layer input.
These segments have **33 layers**. Before training, update `CFG['z_end'] = 33` and
adjust the first Conv3d in `VesuviusNet` to match (or pad Z dim to 65).

## Surface volume download command

To download a segment's surface volumes, run:

```bash
python download_labelled_segment.py --seg 20231221180251
```

(script to be created — see next steps)
