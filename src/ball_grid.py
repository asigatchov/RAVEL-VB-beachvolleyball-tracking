from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            input_dim,
            input_dim,
            kernel_size=3,
            padding=1,
            groups=input_dim,
            bias=False,
        )
        self.pointwise = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)

    def forward(self, values: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(values))


class BallGridBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(input_dim, output_dim),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.block(values)


class BallGridHead(nn.Module):
    """VballNetGridV1b-style temporal grid detector.

    A grayscale clip is presented as T input channels. For every frame the
    output contains a confidence grid and x/y offsets inside the winning cell.
    """

    def __init__(
        self,
        clip_length: int,
        confidence_prior: float = 0.01,
        output_stride: int = 8,
    ) -> None:
        super().__init__()
        if output_stride not in (8, 16):
            raise ValueError("ball grid output_stride must be 8 or 16")
        self.clip_length = clip_length
        self.output_stride = output_stride
        self.features = nn.Sequential(
            BallGridBlock(clip_length, 64),
            BallGridBlock(64, 64),
            nn.MaxPool2d(2, 2),
            BallGridBlock(64, 128),
            BallGridBlock(128, 128),
            nn.MaxPool2d(2, 2),
            BallGridBlock(128, 256),
            BallGridBlock(256, 256),
            nn.MaxPool2d(2, 2),
            *([nn.MaxPool2d(2, 2)] if output_stride == 16 else []),
            BallGridBlock(256, 256),
            BallGridBlock(256, 256),
            BallGridBlock(256, 256),
            BallGridBlock(256, 512),
            BallGridBlock(512, 512),
            BallGridBlock(512, 512),
        )
        self.output = nn.Sequential(
            nn.Conv2d(512, clip_length * 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            output_conv = self.output[0]
            assert isinstance(output_conv, nn.Conv2d)
            nn.init.zeros_(output_conv.weight)
            nn.init.zeros_(output_conv.bias)
            confidence_bias = math.log(
                confidence_prior / (1.0 - confidence_prior)
            )
            output_conv.bias[0::3] = confidence_bias

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 5 or frames.shape[1] != self.clip_length:
            raise ValueError(
                f"ball grid expects [B, {self.clip_length}, C, H, W]"
            )
        if frames.shape[2] == 1:
            grayscale = frames[:, :, 0]
        else:
            rgb = frames[:, :, :3]
            weights = rgb.new_tensor((0.299, 0.587, 0.114))[None, None, :, None, None]
            grayscale = (rgb * weights).sum(2)
        values = self.output(self.features(grayscale))
        return values.view(
            values.shape[0],
            self.clip_length,
            3,
            values.shape[-2],
            values.shape[-1],
        )
