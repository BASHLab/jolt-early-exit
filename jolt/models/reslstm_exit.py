"""Multi-exit ResLSTM for HAR / time series.

Residual conv blocks (1D) followed by a bidirectional LSTM. Variant of
DeepConvLSTM (Ordóñez & Roggen 2016) with residual connections in the conv
trunk, reported on UCI-HAR at 96.34% in the published HAR literature.

Three early exits + final classifier (``num_exits = 3``).
Input convention: ``(B, in_channels, time)``.
~0.55M params at num_classes=6, in_channels=9.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel


class _ResConvBlock1D(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=k, padding=k // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        if c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1, bias=False),
                nn.BatchNorm1d(c_out),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


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


class _LinearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _LinearConfidence(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ResLSTMExit(ExitModel):
    """ResLSTM with three early exits + final classifier (LSTM)."""

    num_exits = 3

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 9,
        conv_channels: Tuple[int, int, int] = (64, 96, 128),
        lstm_hidden: int = 128,
        lstm_dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        c0, c1, c2 = conv_channels
        self.chunk_0 = _ResConvBlock1D(in_channels, c0)
        self.chunk_1 = _ResConvBlock1D(c0, c1)
        self.chunk_2 = _ResConvBlock1D(c1, c2)

        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        lstm_out_dim = (2 if bidirectional else 1) * lstm_hidden
        self.fc = nn.Linear(lstm_out_dim, num_classes)

        self.exit_head_0 = _Head1d(c0, num_classes)
        self.exit_head_1 = _Head1d(c1, num_classes)
        self.exit_head_2 = _Head1d(c2, num_classes)

        self.gate_head_0 = _GateHead1d(c0)
        self.confidence_head_0 = _ConfidenceHead1d(c0)
        self.gate_head_1 = _GateHead1d(c1)
        self.confidence_head_1 = _ConfidenceHead1d(c1)
        self.gate_head_2 = _GateHead1d(c2)
        self.confidence_head_2 = _ConfidenceHead1d(c2)
        self.gate_head_final = _LinearGate(lstm_out_dim)
        self.confidence_head_final = _LinearConfidence(lstm_out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _final_classify(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = x.permute(0, 2, 1).contiguous()
        output, _ = self.lstm(seq)
        last = output[:, -1, :]
        return last, self.fc(last)

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
            last, logits = self._final_classify(x); return last, logits
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
        last, lg = self._final_classify(x)
        logits.append(lg); gates.append(self.gate_head_final(last))
        return logits, gates

    def forward_with_confidences(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        confs: List[torch.Tensor] = []
        x = self.chunk_0(x); logits.append(self.exit_head_0(x)); confs.append(self.confidence_head_0(x))
        x = self.chunk_1(x); logits.append(self.exit_head_1(x)); confs.append(self.confidence_head_1(x))
        x = self.chunk_2(x); logits.append(self.exit_head_2(x)); confs.append(self.confidence_head_2(x))
        last, lg = self._final_classify(x)
        logits.append(lg); confs.append(self.confidence_head_final(last))
        return logits, confs


def reslstm_exit(num_classes: int = 6, in_channels: int = 9) -> ResLSTMExit:
    return ResLSTMExit(num_classes=num_classes, in_channels=in_channels)
