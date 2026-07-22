from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_groups(output_dim), output_dim),
            nn.GELU(),
            nn.Conv2d(
                output_dim,
                output_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_groups(output_dim), output_dim),
            nn.GELU(),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.block(values)


class TinyBackbone(nn.Module):
    """Small learned CNN with an FPN output at 1/4, 1/8 and 1/16 scale."""

    def __init__(
        self,
        backbone_dim: int,
        hidden_dim: int,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        stem_dim = max(16, backbone_dim // 2)
        self.stem = ConvBlock(input_channels, stem_dim, stride=2)
        self.stage4 = ConvBlock(stem_dim, backbone_dim, stride=2)
        self.stage8 = ConvBlock(backbone_dim, backbone_dim * 2, stride=2)
        self.stage16 = ConvBlock(backbone_dim * 2, backbone_dim * 4, stride=2)
        self.lateral4 = nn.Conv2d(backbone_dim, hidden_dim, kernel_size=1)
        self.lateral8 = nn.Conv2d(backbone_dim * 2, hidden_dim, kernel_size=1)
        self.lateral16 = nn.Conv2d(backbone_dim * 4, hidden_dim, kernel_size=1)
        self.refine4 = ConvBlock(hidden_dim, hidden_dim)
        self.refine8 = ConvBlock(hidden_dim, hidden_dim)
        self.refine16 = ConvBlock(hidden_dim, hidden_dim)

    def forward(self, frames: Tensor) -> dict[str, Tensor]:
        stage2 = self.stem(frames)
        stage4 = self.stage4(stage2)
        stage8 = self.stage8(stage4)
        stage16 = self.stage16(stage8)

        p16 = self.lateral16(stage16)
        p8 = self.lateral8(stage8) + F.interpolate(
            p16,
            size=stage8.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p4 = self.lateral4(stage4) + F.interpolate(
            p8,
            size=stage4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return {
            "p4": self.refine4(p4),
            "p8": self.refine8(p8),
            "p16": self.refine16(p16),
        }
