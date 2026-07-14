"""Multi-exit ConvMixer-256/8 for CIFAR-100 / small image classification.

Source: Trockman & Kolter, "Patches Are All You Need?", ICLR 2022.
Architecture: patch embedding + N blocks of (depthwise conv + pointwise conv)
with residual, followed by AdaptiveAvgPool + Linear. CIFAR-100 canonical
~73.9% top-1 accuracy with depth=8, dim=256, kernel=9, patch=1.

Three early exits + final classifier. Chunks evenly split N=8 blocks
across 4 chunks (2 blocks each). ~0.7M params at num_classes=100.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel


class _ConvMixerBlock2D(nn.Module):
    def __init__(self, d: int, kernel: int = 9):
        super().__init__()
        self.dw_conv = nn.Conv2d(d, d, kernel_size=kernel, groups=d, padding=kernel // 2)
        self.dw_bn = nn.BatchNorm2d(d)
        self.pw_conv = nn.Conv2d(d, d, kernel_size=1)
        self.pw_bn = nn.BatchNorm2d(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.dw_bn(self.dw_conv(x)))
        x = x + h
        x = F.gelu(self.pw_bn(self.pw_conv(x)))
        return x


class _MixedPoolHead2d(nn.Module):
    def __init__(self, c: int, num_classes: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * c, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1))


class _GateHead2d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * c, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)).squeeze(-1)


class _ConfidenceHead2d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * c, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)).squeeze(-1)


class ConvMixer256_8Exit(ExitModel):
    """ConvMixer-256/8 (depth=8, d=256, kernel=9, patch=1) with 3 internal exits + final."""

    num_exits = 3

    def __init__(
        self,
        num_classes: int = 100,
        in_channels: int = 3,
        d: int = 256,
        depth: int = 8,
        kernel: int = 9,
        patch_size: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Patch embedding: 2D strided conv
        self.patch = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(d),
        )

        # 4 chunks of equal block count (8/4 = 2 each)
        per_chunk = max(1, depth // 4)
        def make_chunk(n):
            return nn.Sequential(*[_ConvMixerBlock2D(d, kernel) for _ in range(n)])

        self.chunk_0_blocks = make_chunk(per_chunk)
        self.chunk_1 = make_chunk(per_chunk)
        self.chunk_2 = make_chunk(per_chunk)
        remainder = max(1, depth - 3 * per_chunk)
        self.chunk_3 = make_chunk(remainder)

        self.exit_head_0 = _MixedPoolHead2d(d, num_classes)
        self.exit_head_1 = _MixedPoolHead2d(d, num_classes)
        self.exit_head_2 = _MixedPoolHead2d(d, num_classes)
        self.exit_head_final = _MixedPoolHead2d(d, num_classes)

        self.gate_head_0 = _GateHead2d(d)
        self.confidence_head_0 = _ConfidenceHead2d(d)
        self.gate_head_1 = _GateHead2d(d)
        self.confidence_head_1 = _ConfidenceHead2d(d)
        self.gate_head_2 = _GateHead2d(d)
        self.confidence_head_2 = _ConfidenceHead2d(d)
        self.confidence_head_final = _ConfidenceHead2d(d)

    def chunk_0(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.patch(x))
        return self.chunk_0_blocks(x)

    def forward(self, x: torch.Tensor, exit_layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if exit_layer_idx is None:
            exit_layer_idx = 0
        if exit_layer_idx == 0:
            h = self.chunk_0(x); return h, self.exit_head_0(h)
        if exit_layer_idx == 1:
            h = self.chunk_1(x); return h, self.exit_head_1(h)
        if exit_layer_idx == 2:
            h = self.chunk_2(x); return h, self.exit_head_2(h)
        if exit_layer_idx == 3:
            h = self.chunk_3(x); return h, self.exit_head_final(h)
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
        )

    def forward_with_gates(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        h = self.chunk_0(x); logits.append(self.exit_head_0(h)); gates.append(self.gate_head_0(h))
        h = self.chunk_1(h); logits.append(self.exit_head_1(h)); gates.append(self.gate_head_1(h))
        h = self.chunk_2(h); logits.append(self.exit_head_2(h)); gates.append(self.gate_head_2(h))
        h = self.chunk_3(h); logits.append(self.exit_head_final(h))
        return logits, gates

    def forward_with_confidences(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        confs: List[torch.Tensor] = []
        h = self.chunk_0(x); logits.append(self.exit_head_0(h)); confs.append(self.confidence_head_0(h))
        h = self.chunk_1(h); logits.append(self.exit_head_1(h)); confs.append(self.confidence_head_1(h))
        h = self.chunk_2(h); logits.append(self.exit_head_2(h)); confs.append(self.confidence_head_2(h))
        h = self.chunk_3(h); logits.append(self.exit_head_final(h)); confs.append(self.confidence_head_final(h))
        return logits, confs


def convmixer_256_8_exit(num_classes: int = 100, in_channels: int = 3) -> ConvMixer256_8Exit:
    return ConvMixer256_8Exit(num_classes=num_classes, in_channels=in_channels)
