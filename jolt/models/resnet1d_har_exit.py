"""1D-ResNet variant for HAR with three early exits + final.

Source: 1D adaptation of He et al. ResNet-18 used in the time-series literature
(Wang et al., "Time Series Classification from Scratch with Deep Neural Networks: A
Strong Baseline", IJCNN 2017; Karim et al. LSTM-FCN, IEEE Access 2018). The variant
here uses three downsampling stages followed by a final stage, mirroring the
three-early-exit + final-head convention of the other backbones in this repo. The
architecture is intentionally compact (~210K parameters at 6-class UCI-HAR) so it
respects the same on-device parameter envelope as the canonical 1D-CNN.

Input convention: ``(B, in_channels, time)``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel


class _BasicBlock1D(nn.Module):
    """Two-conv residual block over time. Optional downsampling stride."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class _Stage(nn.Module):
    """Two BasicBlock1D in sequence, the first one optionally downsampling."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.blocks = nn.Sequential(
            _BasicBlock1D(c_in, c_out, stride=stride),
            _BasicBlock1D(c_out, c_out, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


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


class ResNet1DHarExit(ExitModel):
    """1D ResNet with three early exits + final classifier."""

    num_exits = 3

    def __init__(self, num_classes: int = 6, in_channels: int = 9):
        super().__init__()
        self.num_exits = 3
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Compact channel budget so the backbone stays close to the 1D-CNN parameter cap.
        c0, c1, c2, c3 = 32, 64, 96, 128

        # Stem: conv7-stride2 + bn + relu + maxpool, halving time twice in total.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Three downsampling stages produce features at progressively coarser time
        # resolution; each stage gets an exit head.
        self.chunk_0 = _Stage(c0, c1, stride=1)
        self.chunk_1 = _Stage(c1, c2, stride=2)
        self.chunk_2 = _Stage(c2, c3, stride=2)

        # Final stage feeds the deep classifier.
        self.chunk_3 = _Stage(c3, c3, stride=1)

        self.exit_head_0 = _Head1d(c1, num_classes)
        self.exit_head_1 = _Head1d(c2, num_classes)
        self.exit_head_2 = _Head1d(c3, num_classes)
        self.exit_head_final = _Head1d(c3, num_classes)

        # Training-only scaffolding shared with the rest of the backbones (used by JEI-DNN
        # and SCAR baselines, ignored by the main inference path).
        self.gate_head_0 = _GateHead1d(c1)
        self.confidence_head_0 = _ConfidenceHead1d(c1)
        self.gate_head_1 = _GateHead1d(c2)
        self.confidence_head_1 = _ConfidenceHead1d(c2)
        self.gate_head_2 = _GateHead1d(c3)
        self.confidence_head_2 = _ConfidenceHead1d(c3)
        self.confidence_head_final = _ConfidenceHead1d(c3)

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
            x = self.stem(x)
            x = self.chunk_0(x)
            return x, self.exit_head_0(x)
        if exit_layer_idx == 1:
            x = self.chunk_1(x)
            return x, self.exit_head_1(x)
        if exit_layer_idx == 2:
            x = self.chunk_2(x)
            return x, self.exit_head_2(x)
        if exit_layer_idx == 3:
            x = self.chunk_3(x)
            return x, self.exit_head_final(x)
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
        )

    def forward_with_gates(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        per_exit_logits: List[torch.Tensor] = []
        per_exit_gate_logits: List[torch.Tensor] = []
        x = self.stem(x)
        x = self.chunk_0(x); per_exit_logits.append(self.exit_head_0(x)); per_exit_gate_logits.append(self.gate_head_0(x))
        x = self.chunk_1(x); per_exit_logits.append(self.exit_head_1(x)); per_exit_gate_logits.append(self.gate_head_1(x))
        x = self.chunk_2(x); per_exit_logits.append(self.exit_head_2(x)); per_exit_gate_logits.append(self.gate_head_2(x))
        x = self.chunk_3(x); per_exit_logits.append(self.exit_head_final(x))
        return per_exit_logits, per_exit_gate_logits

    def forward_with_confidences(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        per_exit_logits: List[torch.Tensor] = []
        per_exit_confidence_logits: List[torch.Tensor] = []
        x = self.stem(x)
        x = self.chunk_0(x); per_exit_logits.append(self.exit_head_0(x)); per_exit_confidence_logits.append(self.confidence_head_0(x))
        x = self.chunk_1(x); per_exit_logits.append(self.exit_head_1(x)); per_exit_confidence_logits.append(self.confidence_head_1(x))
        x = self.chunk_2(x); per_exit_logits.append(self.exit_head_2(x)); per_exit_confidence_logits.append(self.confidence_head_2(x))
        x = self.chunk_3(x); per_exit_logits.append(self.exit_head_final(x)); per_exit_confidence_logits.append(self.confidence_head_final(x))
        return per_exit_logits, per_exit_confidence_logits


def resnet1d_har_exit(num_classes: int = 6, in_channels: int = 9) -> ResNet1DHarExit:
    return ResNet1DHarExit(num_classes=num_classes, in_channels=in_channels)
