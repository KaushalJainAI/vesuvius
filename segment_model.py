"""
Shared model + dataset for the labelled-segment ink detection pipeline.

Used by segment_ink_detection.ipynb and experiment_filters.py.
Memory-safe: loads one segment's volume into RAM at a time.
"""
from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import tifffile as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset


# ---------- Paths: auto-detect Kaggle vs local ---------- #

def find_data_root() -> Path:
    for cand in [
        Path("/kaggle/input/vesuvius-labelled-segments"),
        Path("/kaggle/working/data/labelled_segments"),
        Path(__file__).parent / "data" / "labelled_segments",
    ]:
        if cand.exists():
            return cand
    raise FileNotFoundError("Could not locate labelled_segments data root")


# ---------- Segment registry (smallest to largest) ---------- #

ALL_SEGMENTS = [
    "20231221180251", "20231031143852", "20231016151002", "20231106155351",
    "20230702185753", "20231210121321", "20230929220926", "20231022170901",
    "20231005123336", "20231012184424", "20231007101619",
]
NUM_LAYERS = 33


# ---------- Default config ---------- #

CFG = dict(
    patch_size      = 256,
    patches_per_seg = 400,          # patches sampled per segment per epoch
    batch_size      = 4,            # keep low for cross-system compatibility
    grad_accum      = 4,            # effective batch = batch_size * grad_accum
    num_epochs      = 8,
    lr              = 1e-4,
    weight_decay    = 1e-4,
    z_layers        = NUM_LAYERS,
    ink_pos_thresh  = 0.6,          # patches w/ mean(label) above this = "ink patch"
    ink_neg_thresh  = 0.05,
    ignore_band     = (0.4, 0.6),   # don't supervise in ambiguous band
    val_segment     = "20231221180251",  # held-out; smallest → fast val
    threshold       = 0.4,
    device          = "cuda" if torch.cuda.is_available() else "cpu",
)


# ---------- Data loading ---------- #

def load_segment_volume(seg_dir: Path) -> np.ndarray:
    """Load all 33 layers of one segment into a (Z,H,W) uint8 array."""
    layers = []
    for i in range(NUM_LAYERS):
        p = seg_dir / "surface_volume" / f"{i:02d}.tif"
        layers.append(tf.imread(str(p)))
    return np.stack(layers, axis=0)


def load_segment_label(seg_dir: Path) -> np.ndarray:
    """Load the uint8 ink label (H,W). Returns float32 in [0,1]."""
    lbl = tf.imread(str(seg_dir / "ink_labels.tif"))
    return lbl.astype(np.float32) / 255.0


def derive_mask(volume: np.ndarray) -> np.ndarray:
    """Boolean mask where the papyrus surface has valid (non-zero) data."""
    return (volume.max(axis=0) > 0)


# ---------- Dataset: streams patches, one segment in RAM at a time ---------- #

class SegmentStreamDataset(IterableDataset):
    """Iterates through a list of segments; for each, loads the volume once
    and yields `patches_per_seg` random patches before moving to the next."""

    def __init__(self, data_root: Path, segments: Sequence[str],
                 patch_size: int, patches_per_seg: int,
                 ink_pos_thresh: float, ink_neg_thresh: float,
                 augment: bool = True, shuffle_segments: bool = True):
        super().__init__()
        self.data_root = Path(data_root)
        self.segments = list(segments)
        self.patch_size = patch_size
        self.patches_per_seg = patches_per_seg
        self.ink_pos_thresh = ink_pos_thresh
        self.ink_neg_thresh = ink_neg_thresh
        self.augment = augment
        self.shuffle_segments = shuffle_segments

    def _sample_patch(self, vol: np.ndarray, lbl: np.ndarray, mask: np.ndarray):
        H, W = mask.shape
        ps = self.patch_size
        for _ in range(50):
            y = random.randint(0, H - ps)
            x = random.randint(0, W - ps)
            m = mask[y:y+ps, x:x+ps]
            if m.mean() < 0.6:  # mostly background
                continue
            patch_img = vol[:, y:y+ps, x:x+ps].astype(np.float32) / 255.0
            patch_lbl = lbl[y:y+ps, x:x+ps]
            # ~50/50 sampling of ink vs non-ink patches
            if random.random() < 0.5:
                if patch_lbl.mean() < self.ink_pos_thresh * 0.25:
                    continue
            return patch_img, patch_lbl, m.astype(np.float32)
        # fallback: accept whatever we got
        return patch_img, patch_lbl, m.astype(np.float32)

    def _augment(self, img: np.ndarray, lbl: np.ndarray, mask: np.ndarray):
        if random.random() < 0.5:
            img, lbl, mask = img[:, :, ::-1].copy(), lbl[:, ::-1].copy(), mask[:, ::-1].copy()
        if random.random() < 0.5:
            img, lbl, mask = img[:, ::-1, :].copy(), lbl[::-1, :].copy(), mask[::-1, :].copy()
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k, axes=(1, 2)).copy()
            lbl = np.rot90(lbl, k).copy()
            mask = np.rot90(mask, k).copy()
        return img, lbl, mask

    def __iter__(self) -> Iterator:
        segs = list(self.segments)
        if self.shuffle_segments:
            random.shuffle(segs)
        for seg_id in segs:
            seg_dir = self.data_root / seg_id
            vol = load_segment_volume(seg_dir)
            lbl = load_segment_label(seg_dir)
            mask = derive_mask(vol)
            for _ in range(self.patches_per_seg):
                img, y, m = self._sample_patch(vol, lbl, mask)
                if self.augment:
                    img, y, m = self._augment(img, y, m)
                yield (
                    torch.from_numpy(img),
                    torch.from_numpy(y),
                    torch.from_numpy(m),
                )
            del vol, lbl, mask  # free memory before next segment


# ---------- Model: 33-layer 3D→2D UNet-lite ---------- #

def conv3d_block(ci, co):
    return nn.Sequential(
        nn.Conv3d(ci, co, 3, padding=1, bias=False),
        nn.BatchNorm3d(co), nn.ReLU(inplace=True),
        nn.Conv3d(co, co, 3, padding=1, bias=False),
        nn.BatchNorm3d(co), nn.ReLU(inplace=True),
    )


def conv2d_block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1, bias=False),
        nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )


class SegmentInkNet(nn.Module):
    """Input (B, 1, Z=33, H, W) → Output (B, 1, H, W) logits.
    Z is collapsed by adaptive pooling so the same model works on 33 or 65 layers."""

    def __init__(self):
        super().__init__()
        self.enc1 = conv3d_block(1, 16)
        self.pool1 = nn.MaxPool3d((2, 1, 1))   # Z: 33 → 16
        self.enc2 = conv3d_block(16, 32)
        self.pool2 = nn.MaxPool3d((2, 1, 1))   # Z: 16 → 8
        self.enc3 = conv3d_block(32, 64)
        self.zpool = nn.AdaptiveAvgPool3d((1, None, None))  # Z → 1, H/W preserved

        self.dec1 = conv2d_block(64, 32)
        self.dec2 = conv2d_block(32, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # x: (B, Z, H, W) → add channel dim
        if x.dim() == 4:
            x = x.unsqueeze(1)
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.enc3(x)
        x = self.zpool(x).squeeze(2)   # (B, 64, H, W)
        x = self.dec1(x)
        x = self.dec2(x)
        return self.head(x)            # logits (B, 1, H, W)


# ---------- Loss with ignore-band on ambiguous soft labels ---------- #

def masked_soft_bce_dice(logits, target, mask, ignore_low=0.4, ignore_high=0.6):
    """BCE + Dice, only on pixels where label is confidently ink or non-ink."""
    logits = logits.squeeze(1)
    supervise = mask.bool() & ((target < ignore_low) | (target > ignore_high))
    if supervise.sum() == 0:
        return logits.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(
        logits[supervise], target[supervise], reduction="mean"
    )
    p = torch.sigmoid(logits)
    t = target
    m = supervise.float()
    inter = (p * t * m).sum()
    denom = (p * m).sum() + (t * m).sum() + 1e-6
    dice = 1.0 - (2 * inter + 1e-6) / denom
    return 0.5 * bce + 0.5 * dice


@torch.no_grad()
def fbeta(logits, target, mask, beta=0.5, thresh=0.4):
    p = (torch.sigmoid(logits).squeeze(1) > thresh) & mask.bool()
    t = (target > 0.5) & mask.bool()
    tp = (p & t).sum().float()
    fp = (p & ~t).sum().float()
    fn = (~p & t).sum().float()
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    return ((1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec + 1e-6)).item()


# ---------- Inference (tiled, memory-safe) ---------- #

@torch.no_grad()
def predict_segment(model: nn.Module, seg_dir: Path, patch_size: int = 256,
                    stride: int = 128, device: str = "cuda",
                    amp: bool = True) -> np.ndarray:
    """Sliding-window inference. Returns (H, W) float32 probability map in [0,1]."""
    model.eval()
    vol = load_segment_volume(seg_dir)
    mask = derive_mask(vol)
    H, W = mask.shape
    prob = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    vol_t = torch.from_numpy(vol.astype(np.float32) / 255.0)
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            if not mask[y:y+patch_size, x:x+patch_size].any():
                continue
            patch = vol_t[:, y:y+patch_size, x:x+patch_size].unsqueeze(0).to(device)
            with torch.amp.autocast(device_type="cuda", enabled=amp and device == "cuda"):
                out = torch.sigmoid(model(patch)).squeeze().float().cpu().numpy()
            prob[y:y+patch_size, x:x+patch_size] += out
            wsum[y:y+patch_size, x:x+patch_size] += 1.0
    prob = np.where(wsum > 0, prob / np.maximum(wsum, 1e-6), 0.0)
    prob *= mask
    del vol, vol_t
    return prob
