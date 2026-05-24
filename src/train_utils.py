"""Reusable training, evaluation, and inference utilities."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable

import numpy as np
import tifffile as tf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from research_config import TrainConfig
from vesuvius_data import SegmentPatchDataset, derive_mask, load_volume


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def focal_loss(logits, target, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return (a_t * (1 - p_t) ** gamma * bce).mean()


def masked_loss(logits, target, mask, cfg: TrainConfig):
    logits = logits.squeeze(1)
    supervise = mask.bool() & ((target < cfg.ignore_low) | (target > cfg.ignore_high))
    if supervise.sum() == 0:
        return logits.sum() * 0.0

    if cfg.loss == "bce_dice":
        primary = F.binary_cross_entropy_with_logits(logits[supervise], target[supervise])
    else:
        primary = focal_loss(logits[supervise], target[supervise], cfg.focal_alpha, cfg.focal_gamma)

    p = torch.sigmoid(logits)
    m = supervise.float()
    inter = (p * target * m).sum()
    denom = (p * m).sum() + (target * m).sum() + 1e-6
    dice = 1.0 - (2.0 * inter + 1e-6) / denom
    return 0.5 * primary + 0.5 * dice


@torch.no_grad()
def fbeta_stats(logits, target, mask, beta=0.5, threshold=0.4) -> dict:
    pred = (torch.sigmoid(logits).squeeze(1) > threshold) & mask.bool()
    truth = (target > 0.5) & mask.bool()
    tp = (pred & truth).sum().float()
    fp = (pred & ~truth).sum().float()
    fn = (~pred & truth).sum().float()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    fbeta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-6)
    return {
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "f0.5": float(fbeta.item()),
    }


def warmup_cosine(optimizer, warmup_epochs: int, total_epochs: int):
    def _lr(epoch: int):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    strong_aug = cfg.model.lower().replace("-", "_") != "baseline"
    train_ds = SegmentPatchDataset(
        cfg.data_root,
        cfg.train_segments(),
        cfg.patch_train,
        cfg.patches_per_seg,
        cfg.ink_pos,
        cfg.ink_neg,
        label_root=cfg.label_root,
        augment=True,
        strong_aug=strong_aug,
        shuffle_segments=True,
    )
    val_ds = SegmentPatchDataset(
        cfg.data_root,
        [cfg.val_segment],
        cfg.patch_val,
        cfg.val_patches,
        cfg.ink_pos,
        cfg.ink_neg,
        label_root=cfg.label_root,
        augment=False,
        strong_aug=False,
        shuffle_segments=False,
    )
    pin = cfg.device == "cuda"
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=0, pin_memory=pin),
        DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=0, pin_memory=pin),
    )


def train_model(model, cfg: TrainConfig, run_dir: Path,
                on_epoch: Callable[[dict], None] | None = None) -> list[dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    train_loader, val_loader = build_loaders(cfg)

    device = torch.device(cfg.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine(optimizer, cfg.warmup_epochs, cfg.num_epochs)
    scaler = torch.amp.GradScaler(enabled=cfg.amp and cfg.device == "cuda")
    history: list[dict] = []
    best_f05 = -1.0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        pending = False

        for i, (img, y, m) in enumerate(train_loader):
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=cfg.amp and cfg.device == "cuda"):
                logits = model(img)
                loss = masked_loss(logits, y, m, cfg) / cfg.grad_accum
            scaler.scale(loss).backward()
            pending = True
            if (i + 1) % cfg.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending = False
            total += loss.item() * cfg.grad_accum
            steps += 1

        if pending:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        metric_sum = {"precision": 0.0, "recall": 0.0, "f0.5": 0.0}
        val_steps = 0
        with torch.no_grad():
            for img, y, m in val_loader:
                img = img.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", enabled=cfg.amp and cfg.device == "cuda"):
                    logits = model(img)
                    val_loss += masked_loss(logits, y, m, cfg).item()
                metrics = fbeta_stats(logits, y, m, threshold=cfg.threshold)
                for key in metric_sum:
                    metric_sum[key] += metrics[key]
                val_steps += 1

        entry = {
            "epoch": epoch,
            "train_loss": total / max(steps, 1),
            "val_loss": val_loss / max(val_steps, 1),
            "precision": metric_sum["precision"] / max(val_steps, 1),
            "recall": metric_sum["recall"] / max(val_steps, 1),
            "f0.5": metric_sum["f0.5"] / max(val_steps, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(entry)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"model_state_dict": model.state_dict(), "cfg": cfg.as_jsonable(), **entry}, run_dir / "last.pth")
        if entry["f0.5"] > best_f05:
            best_f05 = entry["f0.5"]
            torch.save({"model_state_dict": model.state_dict(), "cfg": cfg.as_jsonable(), **entry}, run_dir / "best.pth")
        if on_epoch is not None:
            on_epoch(entry)
    return history


@torch.no_grad()
def predict_segment(model, seg_dir: Path, *, patch_size: int, stride: int,
                    device: str, amp: bool = True) -> np.ndarray:
    model.eval()
    vol = load_volume(Path(seg_dir))
    mask = derive_mask(vol)
    H, W = mask.shape
    prob = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    vt = torch.from_numpy(vol.astype(np.float32) / 255.0)
    dev = torch.device(device)
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            if not mask[y:y + patch_size, x:x + patch_size].any():
                continue
            patch = vt[:, y:y + patch_size, x:x + patch_size].unsqueeze(0).to(dev)
            with torch.amp.autocast(device_type="cuda", enabled=amp and device == "cuda"):
                out = torch.sigmoid(model(patch)).squeeze().float().cpu().numpy()
            prob[y:y + patch_size, x:x + patch_size] += out
            wsum[y:y + patch_size, x:x + patch_size] += 1.0
    prob = np.where(wsum > 0, prob / np.maximum(wsum, 1e-6), 0.0)
    prob *= mask
    return prob.astype(np.float32)


def save_prediction(prob: np.ndarray, out_stem: Path) -> tuple[Path, Path]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    npy_path = out_stem.with_suffix(".npy")
    tif_path = out_stem.with_suffix(".tif")
    np.save(npy_path, prob.astype(np.float32))
    tf.imwrite(str(tif_path), (np.clip(prob, 0, 1) * 255).astype(np.uint8))
    return npy_path, tif_path

