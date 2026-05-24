"""Train a comparable segment ink model run.

Examples:
    python scripts/train_segment_model.py --model baseline --epochs 2 --max-segments 3
    python scripts/train_segment_model.py --model unet_v2 --label-root predictions/improved_labels_visual
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ink_models import build_model
from research_config import ALL_SEGMENTS, TrainConfig
from train_utils import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["baseline", "unet_v2"], default="unet_v2")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "labelled_segments")
    parser.add_argument("--label-root", type=Path, default=None,
                        help="Optional refined-label directory, e.g. predictions/improved_labels_visual.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "models" / "runs")
    parser.add_argument("--val-segment", default=ALL_SEGMENTS[0])
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patch-train", type=int, default=None)
    parser.add_argument("--patches-per-seg", type=int, default=None)
    parser.add_argument("--val-patches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--loss", choices=["focal_dice", "bce_dice"], default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig(
        model=args.model,
        run_name=args.run_name,
        data_root=args.data_root,
        label_root=args.label_root,
        output_root=args.output_root,
        val_segment=args.val_segment,
        max_segments=args.max_segments,
    )
    if args.model == "unet_v2":
        cfg.patch_train = 320
        cfg.batch_size = 2
        cfg.grad_accum = 8
        cfg.num_epochs = 80
        cfg.warmup_epochs = 5
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.patch_train is not None:
        cfg.patch_train = args.patch_train
    if args.patches_per_seg is not None:
        cfg.patches_per_seg = args.patches_per_seg
    if args.val_patches is not None:
        cfg.val_patches = args.val_patches
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.grad_accum is not None:
        cfg.grad_accum = args.grad_accum
    if args.lr is not None:
        cfg.lr = args.lr
    if args.device is not None:
        cfg.device = args.device
        cfg.amp = args.device == "cuda" and torch.cuda.is_available()
    if args.loss is not None:
        cfg.loss = args.loss
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    label_tag = "refined" if cfg.label_root else "original"
    run_name = cfg.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg.model}_{label_tag}"
    run_dir = cfg.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.as_jsonable(), indent=2))

    print(f"[train] run:       {run_name}")
    print(f"[train] model:     {cfg.model}")
    print(f"[train] labels:    {cfg.label_root or 'original ink_labels.tif'}")
    print(f"[train] train segs:{cfg.train_segments()}")
    print(f"[train] val seg:   {cfg.val_segment}")
    print(f"[train] output:    {run_dir}")

    model = build_model(cfg.model)
    print(f"[train] params:    {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    def log_epoch(entry: dict) -> None:
        print(
            f"[epoch {entry['epoch']:03d}] "
            f"train={entry['train_loss']:.4f} val={entry['val_loss']:.4f} "
            f"F0.5={entry['f0.5']:.4f} P={entry['precision']:.4f} R={entry['recall']:.4f} "
            f"lr={entry['lr']:.2e}"
        )

    history = train_model(model, cfg, run_dir, on_epoch=log_epoch)
    print(f"[train] done. best F0.5={max(e['f0.5'] for e in history):.4f}")
    print(f"[train] best checkpoint: {run_dir / 'best.pth'}")


if __name__ == "__main__":
    main()

