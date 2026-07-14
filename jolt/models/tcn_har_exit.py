"""Multi-exit Temporal Convolutional Network (TCN) for HAR / time series.

Source: Bai, Kolter, Koltun, "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling" (arXiv 2018). Four dilated
residual blocks at dilations {1, 2, 4, 8} give a receptive field that covers
the full 171-step PAMAP2 window with a single hidden width.

We use SAME (non-causal) padding here since classification is offline; the
original TCN paper uses causal padding for autoregressive sequence modeling,
which is unnecessary for whole-window HAR classification (this is the same
relaxation used by TCN-Attention-HAR, Sci Rep 2024).

Three early exits + one final classifier (``num_exits = 3``). At hidden 64
the network has ~0.4M parameters at num_classes=12.

Input convention: ``(B, in_channels, time)``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel


class _TCNBlock(nn.Module):
    """Two dilated 1D convs + residual. SAME (non-causal) padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=pad, dilation=dilation, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=pad, dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop1(out)
        out = self.bn2(self.conv2(out))
        out = self.drop2(out)
        # Trim if even-padding produced a mismatch
        s = self.shortcut(x)
        T = min(out.shape[-1], s.shape[-1])
        return F.relu(out[..., :T] + s[..., :T], inplace=True)


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


class TCNHarExit(ExitModel):
    """TCN with three early exits + final classifier."""

    num_exits = 3

    def __init__(
        self,
        num_classes: int = 12,
        in_channels: int = 27,
        hidden: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Dilations grow geometrically; receptive field at hidden=64 / kernel=3 / 4 blocks
        # spans ~120 steps -- covers most of PAMAP2's 171-step window.
        self.chunk_0 = _TCNBlock(in_channels, hidden, dilation=1, kernel_size=kernel_size, dropout=dropout)
        self.chunk_1 = _TCNBlock(hidden,      hidden, dilation=2, kernel_size=kernel_size, dropout=dropout)
        self.chunk_2 = _TCNBlock(hidden,      hidden, dilation=4, kernel_size=kernel_size, dropout=dropout)
        self.chunk_3 = _TCNBlock(hidden,      hidden, dilation=8, kernel_size=kernel_size, dropout=dropout)

        self.exit_head_0 = _Head1d(hidden, num_classes)
        self.exit_head_1 = _Head1d(hidden, num_classes)
        self.exit_head_2 = _Head1d(hidden, num_classes)
        self.exit_head_final = _Head1d(hidden, num_classes)

        self.gate_head_0 = _GateHead1d(hidden)
        self.confidence_head_0 = _ConfidenceHead1d(hidden)
        self.gate_head_1 = _GateHead1d(hidden)
        self.confidence_head_1 = _ConfidenceHead1d(hidden)
        self.gate_head_2 = _GateHead1d(hidden)
        self.confidence_head_2 = _ConfidenceHead1d(hidden)
        self.confidence_head_final = _ConfidenceHead1d(hidden)

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


def tcn_har_exit(num_classes: int = 12, in_channels: int = 27) -> TCNHarExit:
    return TCNHarExit(num_classes=num_classes, in_channels=in_channels)
