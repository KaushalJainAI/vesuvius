"""Shared experiment configuration for the segment ink pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch


ALL_SEGMENTS = [
    "20231221180251", "20231031143852", "20231016151002", "20231106155351",
    "20230702185753", "20231210121321", "20230929220926", "20231022170901",
    "20231005123336", "20231012184424", "20231007101619",
]

NUM_LAYERS = 33


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    root = project_root()
    for candidate in [
        Path("/kaggle/working/data/labelled_segments"),
        Path("/kaggle/input/vesuvius-labelled-segments"),
        root / "data" / "labelled_segments",
    ]:
        if candidate.exists():
            return candidate
    return root / "data" / "labelled_segments"


@dataclass
class TrainConfig:
    model: str = "unet_v2"
    run_name: Optional[str] = None
    data_root: Path = default_data_root()
    label_root: Optional[Path] = None
    output_root: Path = project_root() / "models" / "runs"
    prediction_root: Path = project_root() / "predictions" / "runs"

    val_segment: str = ALL_SEGMENTS[0]
    max_segments: int = 0
    seed: int = 42

    patch_train: int = 256
    patch_val: int = 256
    patch_infer: int = 256
    stride_infer: int = 128
    patches_per_seg: int = 400
    val_patches: int = 80
    batch_size: int = 4
    grad_accum: int = 4
    num_epochs: int = 8
    warmup_epochs: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4

    loss: str = "focal_dice"
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    ignore_low: float = 0.4
    ignore_high: float = 0.6
    threshold: float = 0.4
    ink_pos: float = 0.6
    ink_neg: float = 0.05

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = torch.cuda.is_available()

    def selected_segments(self) -> list[str]:
        segments = list(ALL_SEGMENTS)
        if self.max_segments and self.max_segments > 0:
            segments = segments[: self.max_segments]
        return segments

    def train_segments(self) -> list[str]:
        return [s for s in self.selected_segments() if s != self.val_segment]

    def as_jsonable(self) -> dict:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, Path):
                out[key] = str(value)
        return out

