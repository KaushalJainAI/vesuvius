"""Run standardized sliding-window inference from a trained run checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ink_models import build_model
from research_config import ALL_SEGMENTS, TrainConfig
from train_utils import predict_segment, save_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seg", default=ALL_SEGMENTS[0])
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "labelled_segments")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "predictions" / "runs")
    parser.add_argument("--model", default=None, choices=["baseline", "unet_v2"])
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("cfg", {})
    cfg = TrainConfig()
    for key, value in raw_cfg.items():
        if hasattr(cfg, key):
            setattr(cfg, key, Path(value) if key.endswith("_root") and value else value)

    model_name = args.model or cfg.model
    patch_size = args.patch_size or cfg.patch_infer
    stride = args.stride or cfg.stride_infer
    device = args.device or cfg.device

    model = build_model(model_name)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    print(f"[predict] model={model_name} checkpoint={args.checkpoint}")
    print(f"[predict] segment={args.seg} patch={patch_size} stride={stride} device={device}")
    prob = predict_segment(
        model,
        args.data_root / args.seg,
        patch_size=patch_size,
        stride=stride,
        device=device,
        amp=(device == "cuda"),
    )

    run_name = args.checkpoint.parent.name
    out_stem = args.out_dir / run_name / f"{args.seg}_{model_name}_prob"
    npy_path, tif_path = save_prediction(prob, out_stem)
    meta = {
        "checkpoint": str(args.checkpoint),
        "segment": args.seg,
        "model": model_name,
        "patch_size": patch_size,
        "stride": stride,
        "npy": str(npy_path),
        "tif": str(tif_path),
        "mean_prob": float(prob.mean()),
    }
    (out_stem.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    print(f"[predict] saved {npy_path}")
    print(f"[predict] saved {tif_path}")


if __name__ == "__main__":
    main()

