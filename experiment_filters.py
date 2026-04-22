"""
Filter / letter-candidate experimentation on an ink-probability map.

Pipeline:
  prediction (H,W) prob map
    → threshold + morphology cleanup
    → connected components (letter candidates)
    → for each component, crop bbox, binarize, normalize size
    → match against 24 Greek uppercase letter templates (rendered from a font)
    → rank top-K letters by normalized cross-correlation
    → output CSV + a visualization grid

Usage:
    python experiment_filters.py --pred predictions/20231221180251_prob.npy
    python experiment_filters.py --pred predictions/20231221180251_prob.tif --threshold 0.45
    python experiment_filters.py --pred ... --crop y0,y1,x0,x1   # inspect a region

Important caveat: confidences are *relative* scores from template matching
against a single font. They are NOT calibrated probabilities and will be
biased by font choice, scroll-script style, and binarization threshold.
Treat them as a ranked guess to hand to a human or an LLM, not as truth.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import scipy.ndimage as ndi
from skimage import morphology, measure

# Greek uppercase — scroll letterforms are majuscule
GREEK_UPPER = list("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")
GREEK_NAMES = ["Alpha","Beta","Gamma","Delta","Epsilon","Zeta","Eta","Theta",
               "Iota","Kappa","Lambda","Mu","Nu","Xi","Omicron","Pi","Rho",
               "Sigma","Tau","Upsilon","Phi","Chi","Psi","Omega"]
TEMPLATE_SIZE = 64  # normalized template resolution


# ---------- Template generation ---------- #

def render_letter(ch: str, size: int = TEMPLATE_SIZE) -> np.ndarray:
    """Render a Greek uppercase letter as a binary (size, size) array using matplotlib."""
    fig = Figure(figsize=(1, 1), dpi=size)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, ch, ha='center', va='center',
            fontsize=size * 0.7, family='serif', color='white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_facecolor('black')
    ax.axis('off')
    fig.patch.set_facecolor('black')
    fig.subplots_adjust(0, 0, 1, 1)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[..., :3].mean(-1)
    img = (buf > 64).astype(np.float32)
    if img.shape[0] != size:  # DPI can produce off-by-one
        img = _resize_bin(img, size, size)
    return img


def build_templates(size: int = TEMPLATE_SIZE) -> dict[str, np.ndarray]:
    return {ch: render_letter(ch, size) for ch in GREEK_UPPER}


# ---------- Helpers ---------- #

def _resize_bin(a: np.ndarray, H: int, W: int) -> np.ndarray:
    zy = H / a.shape[0]; zx = W / a.shape[1]
    r = ndi.zoom(a, (zy, zx), order=1)
    return (r > 0.5).astype(np.float32)


def load_prediction(path: Path) -> np.ndarray:
    if path.suffix == '.npy':
        p = np.load(path)
    else:
        p = tf.imread(str(path))
    p = p.astype(np.float32)
    if p.max() > 1.5:
        p = p / 255.0
    return p


# ---------- Letter-candidate extraction ---------- #

def extract_candidates(prob: np.ndarray, threshold: float, min_size: int, max_size: int) -> List[dict]:
    """Threshold + closing + remove small objects → connected components.
    Returns list of {id, bbox, centroid, area, patch (binary cropped)}."""
    binary = prob > threshold
    binary = morphology.binary_closing(binary, morphology.disk(2))
    binary = morphology.remove_small_objects(binary, min_size=min_size)

    labelled, n = ndi.label(binary)
    out = []
    for r in measure.regionprops(labelled):
        if r.area > max_size:  # likely blob of merged letters or non-letter artifact
            continue
        y0, x0, y1, x1 = r.bbox
        patch = labelled[y0:y1, x0:x1] == r.label
        out.append({
            'id': r.label,
            'bbox': (y0, x0, y1, x1),
            'centroid': r.centroid,
            'area': int(r.area),
            'patch': patch.astype(np.float32),
        })
    return out


# ---------- Template matching ---------- #

def normalize_candidate(patch: np.ndarray, size: int = TEMPLATE_SIZE) -> np.ndarray:
    """Resize candidate binary to template size while preserving aspect ratio
    (centered in a square canvas)."""
    H, W = patch.shape
    if H == 0 or W == 0:
        return np.zeros((size, size), dtype=np.float32)
    s = size - 6
    if H >= W:
        nh = s; nw = max(1, int(W * s / H))
    else:
        nw = s; nh = max(1, int(H * s / W))
    resized = _resize_bin(patch, nh, nw)
    canvas = np.zeros((size, size), dtype=np.float32)
    y0 = (size - nh) // 2; x0 = (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def score_against_templates(cand: np.ndarray, templates: dict[str, np.ndarray]) -> List[Tuple[str, float]]:
    """Return (letter, score) pairs sorted desc. Score = normalized IoU × NCC."""
    scores = []
    c = cand.flatten()
    c_norm = c - c.mean()
    c_den = np.sqrt((c_norm ** 2).sum()) + 1e-6
    for ch, tpl in templates.items():
        t = tpl.flatten()
        t_norm = t - t.mean()
        t_den = np.sqrt((t_norm ** 2).sum()) + 1e-6
        ncc = float((c_norm * t_norm).sum() / (c_den * t_den))
        inter = (cand * tpl).sum()
        union = cand.sum() + tpl.sum() - inter + 1e-6
        iou = float(inter / union)
        combined = 0.5 * ncc + 0.5 * iou
        scores.append((ch, combined))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def softmax_top(scores: List[Tuple[str, float]], k: int = 5, temp: float = 10.0) -> List[Tuple[str, float]]:
    """Turn raw scores into a pseudo-probability distribution over the top K."""
    top = scores[:k]
    xs = np.array([s for _, s in top])
    xs = xs * temp
    xs = xs - xs.max()
    p = np.exp(xs); p = p / (p.sum() + 1e-9)
    return [(ch, float(pi)) for (ch, _), pi in zip(top, p)]


# ---------- Main ---------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='.npy or .tif probability map')
    ap.add_argument('--threshold', type=float, default=0.45)
    ap.add_argument('--min-size', type=int, default=80,  help='min connected-component area')
    ap.add_argument('--max-size', type=int, default=6000, help='max cc area (to exclude merged blobs)')
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--max-candidates', type=int, default=60,
                    help='cap visualization + CSV at this many candidates (largest first)')
    ap.add_argument('--crop', type=str, default=None, help='y0,y1,x0,x1 to restrict analysis')
    ap.add_argument('--out-dir', type=Path, default=Path('filter_experiments'))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prob = load_prediction(Path(args.pred))
    if args.crop:
        y0, y1, x0, x1 = map(int, args.crop.split(','))
        prob = prob[y0:y1, x0:x1]
    print(f'Loaded prob map {prob.shape}  min={prob.min():.3f} max={prob.max():.3f}')

    print('Building Greek uppercase templates…')
    templates = build_templates()

    print('Extracting letter candidates…')
    cands = extract_candidates(prob, args.threshold, args.min_size, args.max_size)
    cands.sort(key=lambda c: c['area'], reverse=True)
    cands = cands[:args.max_candidates]
    print(f'  {len(cands)} candidates kept')

    # Score each candidate
    rows = []
    for c in cands:
        norm = normalize_candidate(c['patch'])
        raw = score_against_templates(norm, templates)
        topk = softmax_top(raw, k=args.top_k)
        rows.append({
            'id': c['id'], 'area': c['area'],
            'bbox': c['bbox'], 'topk': topk,
            'centroid': c['centroid'],
            'norm_patch': norm,
        })

    # CSV
    csv_path = args.out_dir / 'candidates.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'area', 'y0', 'x0', 'y1', 'x1',
                    *[f'top{i+1}_letter' for i in range(args.top_k)],
                    *[f'top{i+1}_conf'   for i in range(args.top_k)]])
        for r in rows:
            letters = [t[0] for t in r['topk']] + [''] * (args.top_k - len(r['topk']))
            confs   = [f"{t[1]:.3f}" for t in r['topk']] + [''] * (args.top_k - len(r['topk']))
            w.writerow([r['id'], r['area'], *r['bbox'], *letters, *confs])
    print(f'  wrote {csv_path}')

    # Visualization: grid of candidates with top-3 guesses
    if rows:
        n = len(rows)
        cols = 8; row_count = (n + cols - 1) // cols
        fig, axes = plt.subplots(row_count, cols, figsize=(cols * 1.6, row_count * 2))
        axes = np.atleast_2d(axes)
        for i, r in enumerate(rows):
            ax = axes[i // cols, i % cols]
            ax.imshow(r['norm_patch'], cmap='gray')
            top3 = r['topk'][:3]
            title = '\n'.join(f'{ch} {p:.2f}' for ch, p in top3)
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        for j in range(n, row_count * cols):
            axes[j // cols, j % cols].axis('off')
        fig.suptitle(f'{len(rows)} candidates (top-3 Greek letter guesses)', fontsize=10)
        fig.tight_layout()
        out = args.out_dir / 'candidates_grid.png'
        fig.savefig(out, dpi=120, bbox_inches='tight')
        print(f'  wrote {out}')

    # Template reference plot
    fig, ax = plt.subplots(3, 8, figsize=(12, 5))
    for i, (ch, t) in enumerate(templates.items()):
        a = ax[i // 8, i % 8]
        a.imshow(t, cmap='gray')
        a.set_title(f'{ch} ({GREEK_NAMES[i]})', fontsize=8)
        a.axis('off')
    fig.tight_layout()
    fig.savefig(args.out_dir / 'templates.png', dpi=120, bbox_inches='tight')
    print(f'  wrote {args.out_dir / "templates.png"}')
    print('Done.')


if __name__ == '__main__':
    main()
