"""Data loading and patch sampling for labelled segment experiments."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import tifffile as tf
import torch
from torch.utils.data import IterableDataset

from research_config import NUM_LAYERS

try:
    from scipy.ndimage import gaussian_filter, map_coordinates
    HAS_ELASTIC = True
except ImportError:
    HAS_ELASTIC = False


def load_volume(seg_dir: Path, num_layers: int = NUM_LAYERS) -> np.ndarray:
    """Load a segment surface volume as uint8/uint16 array shaped (Z,H,W)."""
    layers = []
    for z in range(num_layers):
        layers.append(tf.imread(str(seg_dir / "surface_volume" / f"{z:02d}.tif")))
    return np.stack(layers, axis=0)


def load_label(seg_dir: Path, label_root: Optional[Path] = None) -> np.ndarray:
    """Load a float32 label map in [0,1].

    If label_root is provided, this first checks for either
    {seg}_labels_v1.tif or {seg}_labels_v1.npy. This lets the same trainer use
    refined labels without copying files into data/labelled_segments.
    """
    seg_id = seg_dir.name
    path: Path
    if label_root is not None:
        label_root = Path(label_root)
        candidates = [
            label_root / f"{seg_id}_labels_v1.npy",
            label_root / f"{seg_id}_labels_v1.tif",
            label_root / f"{seg_id}_labels.tif",
        ]
        path = next((p for p in candidates if p.exists()), seg_dir / "ink_labels.tif")
    else:
        path = seg_dir / "ink_labels.tif"

    arr = np.load(path) if path.suffix == ".npy" else tf.imread(str(path))
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def derive_mask(volume: np.ndarray) -> np.ndarray:
    return volume.max(axis=0) > 0


def elastic_deform(img: np.ndarray, lbl: np.ndarray, mask: np.ndarray,
                   alpha: float = 50, sigma: float = 5):
    H, W = lbl.shape
    dx = gaussian_filter(np.random.randn(H, W) * alpha, sigma)
    dy = gaussian_filter(np.random.randn(H, W) * alpha, sigma)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cy = np.clip(yy + dy, 0, H - 1).ravel()
    cx = np.clip(xx + dx, 0, W - 1).ravel()
    img_out = np.stack([
        map_coordinates(img[z], [cy, cx], order=1, mode="reflect").reshape(H, W)
        for z in range(img.shape[0])
    ], axis=0)
    lbl_out = map_coordinates(lbl, [cy, cx], order=1, mode="reflect").reshape(H, W)
    mask_out = map_coordinates(mask.astype("f4"), [cy, cx], order=0, mode="reflect").reshape(H, W).astype(bool)
    return img_out, lbl_out, mask_out


class SegmentPatchDataset(IterableDataset):
    """Stream random patches while keeping only one segment in RAM."""

    def __init__(
        self,
        data_root: Path,
        segments: Sequence[str],
        patch_size: int,
        patches_per_seg: int,
        ink_pos: float,
        ink_neg: float,
        *,
        label_root: Optional[Path] = None,
        augment: bool = True,
        strong_aug: bool = False,
        shuffle_segments: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.segments = list(segments)
        self.patch_size = patch_size
        self.patches_per_seg = patches_per_seg
        self.ink_pos = ink_pos
        self.ink_neg = ink_neg
        self.label_root = Path(label_root) if label_root else None
        self.augment = augment
        self.strong_aug = strong_aug
        self.shuffle_segments = shuffle_segments

    def _sample(self, vol: np.ndarray, lbl: np.ndarray, mask: np.ndarray):
        H, W = mask.shape
        ps = self.patch_size
        last = None
        for _ in range(50):
            y = random.randint(0, H - ps)
            x = random.randint(0, W - ps)
            m = mask[y:y + ps, x:x + ps]
            img = vol[:, y:y + ps, x:x + ps].astype(np.float32) / 255.0
            lab = lbl[y:y + ps, x:x + ps]
            last = (img, lab, m.astype(np.float32))
            if m.mean() < 0.6:
                continue
            if random.random() < 0.5 and lab.mean() < self.ink_pos * 0.25:
                continue
            return last
        return last

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

        if self.strong_aug:
            if random.random() < 0.5:
                img = np.clip(img + np.random.randn(*img.shape).astype(np.float32) * 0.03, 0, 1)
            if random.random() < 0.5:
                img = np.clip(img * random.uniform(0.85, 1.15), 0, 1)
            if random.random() < 0.4:
                idx = random.sample(range(img.shape[0]), random.randint(1, 3))
                img[idx] = 0.0
            if random.random() < 0.3:
                shift = random.randint(-2, 2)
                if shift:
                    img = np.roll(img, shift, axis=0)
                    if shift > 0:
                        img[:shift] = 0.0
                    else:
                        img[shift:] = 0.0
            if HAS_ELASTIC and random.random() < 0.3:
                img, lbl, mask = elastic_deform(img, lbl, mask.astype(bool))
                mask = mask.astype(np.float32)
        return img, lbl, mask

    def __iter__(self) -> Iterator:
        segs = list(self.segments)
        if self.shuffle_segments:
            random.shuffle(segs)
        for seg_id in segs:
            seg_dir = self.data_root / seg_id
            vol = load_volume(seg_dir)
            lbl = load_label(seg_dir, self.label_root)
            mask = derive_mask(vol)
            for _ in range(self.patches_per_seg):
                sample = self._sample(vol, lbl, mask)
                if sample is None:
                    continue
                img, y, m = sample
                if self.augment:
                    img, y, m = self._augment(img, y, m)
                yield torch.from_numpy(img.copy()), torch.from_numpy(y.copy()), torch.from_numpy(m.copy())
            del vol, lbl, mask

