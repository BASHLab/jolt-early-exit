"""Multi-exit InceptionTime for time-series classification.

Source: Fawaz et al., "InceptionTime: Finding AlexNet for time series classification",
Data Mining and Knowledge Discovery 34(6), 2020. The original architecture stacks
six Inception modules with residual connections every three modules; we keep the
six-module depth, partition into three residual blocks of two Inception modules
each, and attach an early exit head after each block. The final classifier reads
the deepest block's pooled features.

Input convention: ``(B, in_channels, time)``.

Three early exits + one final classifier (``num_exits = 3``). At PAMAP2 default
config (n_filters=32, bottleneck=32, depth=2 per block), the network has roughly
0.35-0.5M parameters at num_classes=12.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel


class _InceptionModule(nn.Module):
    """One Inception module: bottleneck + 3 parallel convs + max-pool branch."""

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        bottleneck: int = 32,
        kernel_sizes: Tuple[int, int, int] = (10, 20, 40),
    ):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False)
            conv_in = bottleneck
        else:
            self.bottleneck = nn.Identity()
            conv_in = in_channels

        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            self.convs.append(
                nn.Conv1d(conv_in, n_filters, kernel_size=k, padding=k // 2, bias=False)
            )

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        outs = [conv(b) for conv in self.convs]
        outs.append(self.pool_conv(self.maxpool(x)))
        # Align temporal length: even kernels add an extra timestep on one side.
        T = min(o.shape[-1] for o in outs)
        outs = [o[..., :T] for o in outs]
        return F.relu(self.bn(torch.cat(outs, dim=1)), inplace=True)


class _ResidualInceptionBlock(nn.Module):
    """Two Inception modules with a 1x1 residual shortcut."""

    def __init__(self, in_channels: int, n_filters: int = 32, depth: int = 2):
        super().__init__()
        modules: List[nn.Module] = []
        c = in_channels
        out = n_filters * 4  # 4 branches per module
        for _ in range(depth):
            modules.append(_InceptionModule(c, n_filters=n_filters))
            c = out
        self.layers = nn.Sequential(*modules)
        if in_channels != out:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out, kernel_size=1, bias=False),
                nn.BatchNorm1d(out),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.layers(x) + self.shortcut(x), inplace=True)


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


class InceptionTimeExit(ExitModel):
    """InceptionTime with three early exits + final classifier."""

    num_exits = 3

    def __init__(
        self,
        num_classes: int = 12,
        in_channels: int = 27,
        n_filters: int = 32,
        depth_per_block: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        out_c = n_filters * 4  # 4 parallel branches per Inception module

        self.chunk_0 = _ResidualInceptionBlock(in_channels, n_filters, depth=depth_per_block)
        self.chunk_1 = _ResidualInceptionBlock(out_c,       n_filters, depth=depth_per_block)
        self.chunk_2 = _ResidualInceptionBlock(out_c,       n_filters, depth=depth_per_block)
        self.chunk_3 = _ResidualInceptionBlock(out_c,       n_filters, depth=depth_per_block)

        self.exit_head_0 = _Head1d(out_c, num_classes)
        self.exit_head_1 = _Head1d(out_c, num_classes)
        self.exit_head_2 = _Head1d(out_c, num_classes)
        self.exit_head_final = _Head1d(out_c, num_classes)

        # JEI-DNN gate heads + SCAR confidence heads (used only when those baselines run).
        self.gate_head_0 = _GateHead1d(out_c)
        self.confidence_head_0 = _ConfidenceHead1d(out_c)
        self.gate_head_1 = _GateHead1d(out_c)
        self.confidence_head_1 = _ConfidenceHead1d(out_c)
        self.gate_head_2 = _GateHead1d(out_c)
        self.confidence_head_2 = _ConfidenceHead1d(out_c)
        self.confidence_head_final = _ConfidenceHead1d(out_c)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self, x: torch.Tensor, exit_layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exit_layer_idx is None:
            exit_layer_idx = 0
        if exit_layer_idx == 0:
            x = self.chunk_0(x); return x, self.exit_head_0(x)
        if exit_layer_idx == 1:
            x = self.chunk_1(x); return x, self.exit_head_1(x)
        if exit_layer_idx == 2:
            x = self.chunk_2(x); return x, self.exit_head_2(x)
        if exit_layer_idx == 3:
            x = self.chunk_3(x); return x, self.exit_head_final(x)
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
        )

    def forward_with_gates(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        x = self.chunk_0(x); logits.append(self.exit_head_0(x)); gates.append(self.gate_head_0(x))
        x = self.chunk_1(x); logits.append(self.exit_head_1(x)); gates.append(self.gate_head_1(x))
        x = self.chunk_2(x); logits.append(self.exit_head_2(x)); gates.append(self.gate_head_2(x))
        x = self.chunk_3(x); logits.append(self.exit_head_final(x))
        return logits, gates

    def forward_with_confidences(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        confs: List[torch.Tensor] = []
        x = self.chunk_0(x); logits.append(self.exit_head_0(x)); confs.append(self.confidence_head_0(x))
        x = self.chunk_1(x); logits.append(self.exit_head_1(x)); confs.append(self.confidence_head_1(x))
        x = self.chunk_2(x); logits.append(self.exit_head_2(x)); confs.append(self.confidence_head_2(x))
        x = self.chunk_3(x); logits.append(self.exit_head_final(x)); confs.append(self.confidence_head_final(x))
        return logits, confs


def inceptiontime_exit(num_classes: int = 12, in_channels: int = 27) -> InceptionTimeExit:
    return InceptionTimeExit(num_classes=num_classes, in_channels=in_channels)
