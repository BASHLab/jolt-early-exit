"""Multi-exit ConvMixer-1D for HAR / time series.

1D adaptation of ConvMixer (Trockman & Kolter, ICLR 2022): patch embedding
followed by N blocks of (depthwise conv + pointwise conv) with residual
connections. Multi-exit splits: 4 chunks of equal block count.

Input convention: ``(B, in_channels, time)``.
~0.55M params at d=64, depth=8, kernel=9, patch=4, num_classes=6.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel


class _ConvMixerBlock1D(nn.Module):
    def __init__(self, d: int, kernel: int = 9):
        super().__init__()
        self.dw_conv = nn.Conv1d(d, d, kernel_size=kernel, groups=d, padding=kernel // 2)
        self.dw_bn = nn.BatchNorm1d(d)
        self.pw_conv = nn.Conv1d(d, d, kernel_size=1)
        self.pw_bn = nn.BatchNorm1d(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise + residual
        h = F.gelu(self.dw_bn(self.dw_conv(x)))
        x = x + h
        # Pointwise
        x = F.gelu(self.pw_bn(self.pw_conv(x)))
        return x


class _Head1d(nn.Module):
    def __init__(self, c_in: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c_in, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


class _GateHead1d(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c_in, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


class _ConfidenceHead1d(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c_in, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


class ConvMixer1DExit(ExitModel):
    """ConvMixer-1D with three early exits + final classifier."""

    num_exits = 3

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 9,
        d: int = 64,
        depth: int = 8,
        kernel: int = 9,
        patch_size: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Patch embedding via strided conv
        self.patch = nn.Sequential(
            nn.Conv1d(in_channels, d, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm1d(d),
        )

        per_chunk = max(1, depth // 4)
        def make_chunk(n):
            return nn.Sequential(*[_ConvMixerBlock1D(d, kernel) for _ in range(n)])

        self.chunk_0_blocks = make_chunk(per_chunk)
        self.chunk_1 = make_chunk(per_chunk)
        self.chunk_2 = make_chunk(per_chunk)
        remainder = max(1, depth - 3 * per_chunk)
        self.chunk_3 = make_chunk(remainder)

        self.exit_head_0 = _Head1d(d, num_classes)
        self.exit_head_1 = _Head1d(d, num_classes)
        self.exit_head_2 = _Head1d(d, num_classes)
        self.exit_head_final = _Head1d(d, num_classes)

        self.gate_head_0 = _GateHead1d(d)
        self.confidence_head_0 = _ConfidenceHead1d(d)
        self.gate_head_1 = _GateHead1d(d)
        self.confidence_head_1 = _ConfidenceHead1d(d)
        self.gate_head_2 = _GateHead1d(d)
        self.confidence_head_2 = _ConfidenceHead1d(d)
        self.confidence_head_final = _ConfidenceHead1d(d)

    def chunk_0(self, x: torch.Tensor) -> torch.Tensor:
        # Embed first, then run first block group
        x = F.gelu(self.patch(x))
        return self.chunk_0_blocks(x)

    def forward(
        self, x: torch.Tensor, exit_layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def forward_with_gates(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        h = self.chunk_0(x); logits.append(self.exit_head_0(h)); gates.append(self.gate_head_0(h))
        h = self.chunk_1(h); logits.append(self.exit_head_1(h)); gates.append(self.gate_head_1(h))
        h = self.chunk_2(h); logits.append(self.exit_head_2(h)); gates.append(self.gate_head_2(h))
        h = self.chunk_3(h); logits.append(self.exit_head_final(h))
        return logits, gates

    def forward_with_confidences(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        confs: List[torch.Tensor] = []
        h = self.chunk_0(x); logits.append(self.exit_head_0(h)); confs.append(self.confidence_head_0(h))
        h = self.chunk_1(h); logits.append(self.exit_head_1(h)); confs.append(self.confidence_head_1(h))
        h = self.chunk_2(h); logits.append(self.exit_head_2(h)); confs.append(self.confidence_head_2(h))
        h = self.chunk_3(h); logits.append(self.exit_head_final(h)); confs.append(self.confidence_head_final(h))
        return logits, confs


def convmixer1d_exit(num_classes: int = 6, in_channels: int = 9) -> ConvMixer1DExit:
    return ConvMixer1DExit(num_classes=num_classes, in_channels=in_channels)
