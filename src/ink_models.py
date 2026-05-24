"""Shared ink detection model definitions."""
from __future__ import annotations

import torch
import torch.nn as nn


def conv3d_block(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(ci, co, 3, padding=1, bias=False),
        nn.BatchNorm3d(co),
        nn.ReLU(inplace=True),
        nn.Conv3d(co, co, 3, padding=1, bias=False),
        nn.BatchNorm3d(co),
        nn.ReLU(inplace=True),
    )


def conv2d_block(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1, bias=False),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1, bias=False),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
    )


class SegmentInkNet(nn.Module):
    """Baseline 3D encoder to 2D decoder."""

    def __init__(self):
        super().__init__()
        self.enc1 = conv3d_block(1, 16)
        self.pool1 = nn.MaxPool3d((2, 1, 1))
        self.enc2 = conv3d_block(16, 32)
        self.pool2 = nn.MaxPool3d((2, 1, 1))
        self.enc3 = conv3d_block(32, 64)
        self.zpool = nn.AdaptiveAvgPool3d((1, None, None))
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.enc3(x)
        x = self.zpool(x).squeeze(2)
        x = self.dec1(x)
        x = self.dec2(x)
        return self.head(x)


class SegmentInkNetV2(nn.Module):
    """4-stage 3D encoder with 2D U-Net skip decoder."""

    def __init__(self):
        super().__init__()
        self.enc1 = conv3d_block(1, 32)
        self.pool1 = nn.MaxPool3d((2, 1, 1))
        self.enc2 = conv3d_block(32, 64)
        self.pool2 = nn.MaxPool3d((2, 1, 1))
        self.enc3 = conv3d_block(64, 128)
        self.pool3 = nn.MaxPool3d((2, 1, 1))
        self.enc4 = conv3d_block(128, 256)

        self.dec3 = conv2d_block(256 + 128, 128)
        self.dec2 = conv2d_block(128 + 64, 64)
        self.dec1 = conv2d_block(64 + 32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _zpool(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        bot = self._zpool(e4)
        s3 = self._zpool(e3)
        s2 = self._zpool(e2)
        s1 = self._zpool(e1)

        d = self.dec3(torch.cat([bot, s3], dim=1))
        d = self.dec2(torch.cat([d, s2], dim=1))
        d = self.dec1(torch.cat([d, s1], dim=1))
        return self.head(d)


def build_model(name: str) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key in {"baseline", "segment_ink_net", "segmentinknet"}:
        return SegmentInkNet()
    if key in {"unet_v2", "segment_ink_net_v2", "segmentinknetv2"}:
        return SegmentInkNetV2()
    raise ValueError(f"Unknown model: {name}")

