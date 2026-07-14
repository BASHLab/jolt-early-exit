"""Multi-exit DeepConvLSTM (Ordonez & Roggen, Sensors 2016).

Four 1D convolutional blocks then two bidirectional LSTM layers, with three exit
classifiers placed after conv2, conv4, and lstm2 (the final exit). All three exits
share the same trunk in a single forward pass; the ``ExitModel`` interface threads
the intermediate activation between calls.

Input: (B, in_channels, time). For PAMAP2 the canonical layout is in_channels=18
(ankle + chest + hand at 6 channels each) and time=120 (the pre-windowed NPY shape
on disk).

``num_exits = 2`` (two early exits at conv2 / conv4; the final exit at lstm2 is
index 2).

Notes on the architecture vs the v1 in this repo:
  - The LSTM is now bidirectional, matching the canonical published variant; the
    classifier sees the concatenated forward+backward final hidden state.
  - Conv channels grow with depth (64, 64, 128, 128) so the conv4 exit sees richer
    features than the conv2 exit; the prior all-64 layout left conv4 with the same
    discriminative capacity as conv2, which contributed to the middle exit being
    routed-around at every operating point.
  - LSTM dropout reduced from 0.5 to 0.3.
  - Per-exit gate heads (JEI-DNN) and confidence heads (SCAR) added so those
    baselines run on this cell.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .base import ExitModel


def _conv_block(in_c: int, out_c: int, k: int = 5) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=k, padding=k // 2),
        nn.BatchNorm1d(out_c),
        nn.ReLU(inplace=True),
    )


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


class _LinearGateHead(nn.Module):
    """Gate head operating on a flat feature vector (used at the LSTM exit)."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _LinearConfidenceHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DeepConvLSTMExit(ExitModel):
    num_exits = 2

    def __init__(
        self,
        num_classes: int = 12,
        in_channels: int = 18,
        conv_channels: Tuple[int, int, int, int] = (64, 64, 128, 128),
        lstm_hidden: int = 128,
        lstm_dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        c0, c1, c2, c3 = conv_channels
        self.conv1 = _conv_block(in_channels, c0)
        self.conv2 = _conv_block(c0, c1)
        self.exit1_head = nn.Linear(c1, num_classes)

        self.conv3 = _conv_block(c1, c2)
        self.conv4 = _conv_block(c2, c3)
        self.exit2_head = nn.Linear(c3, num_classes)

        self.lstm = nn.LSTM(
            input_size=c3,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        lstm_out_dim = (2 if bidirectional else 1) * lstm_hidden
        self.fc = nn.Linear(lstm_out_dim, num_classes)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Training-time scaffolding for JEI-DNN gate heads and SCAR confidence heads.
        # The inference forward path does not invoke these; they only get used when
        # the corresponding baseline is the active loss method.
        self.gate_head_0 = _GateHead1d(c1)
        self.confidence_head_0 = _ConfidenceHead1d(c1)
        self.gate_head_1 = _GateHead1d(c3)
        self.confidence_head_1 = _ConfidenceHead1d(c3)
        self.gate_head_final = _LinearGateHead(lstm_out_dim)
        self.confidence_head_final = _LinearConfidenceHead(lstm_out_dim)

    def _conv_head(self, features: torch.Tensor, head: nn.Module) -> torch.Tensor:
        y = self.pool(features).squeeze(-1)
        return head(y)

    def _final_classify(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # features carries the conv4 feature map: (B, C, T). LSTM wants (B, T, C).
        seq = features.permute(0, 2, 1).contiguous()
        output, _ = self.lstm(seq)
        last = output[:, -1, :]
        return last, self.fc(last)

    def forward(
        self, x: torch.Tensor, exit_layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exit_layer_idx is None:
            exit_layer_idx = 0
        if exit_layer_idx == 0:
            x = self.conv1(x)
            x = self.conv2(x)
            return x, self._conv_head(x, self.exit1_head)
        if exit_layer_idx == 1:
            x = self.conv3(x)
            x = self.conv4(x)
            return x, self._conv_head(x, self.exit2_head)
        if exit_layer_idx == 2:
            last, logits = self._final_classify(x)
            return last, logits
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
        )

    def forward_with_gates(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        per_exit_logits: List[torch.Tensor] = []
        per_exit_gate_logits: List[torch.Tensor] = []
        x = self.conv1(x); x = self.conv2(x)
        per_exit_logits.append(self._conv_head(x, self.exit1_head))
        per_exit_gate_logits.append(self.gate_head_0(x))
        x = self.conv3(x); x = self.conv4(x)
        per_exit_logits.append(self._conv_head(x, self.exit2_head))
        per_exit_gate_logits.append(self.gate_head_1(x))
        last, logits = self._final_classify(x)
        per_exit_logits.append(logits)
        return per_exit_logits, per_exit_gate_logits

    def forward_with_confidences(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        per_exit_logits: List[torch.Tensor] = []
        per_exit_confidence_logits: List[torch.Tensor] = []
        x = self.conv1(x); x = self.conv2(x)
        per_exit_logits.append(self._conv_head(x, self.exit1_head))
        per_exit_confidence_logits.append(self.confidence_head_0(x))
        x = self.conv3(x); x = self.conv4(x)
        per_exit_logits.append(self._conv_head(x, self.exit2_head))
        per_exit_confidence_logits.append(self.confidence_head_1(x))
        last, logits = self._final_classify(x)
        per_exit_logits.append(logits)
        per_exit_confidence_logits.append(self.confidence_head_final(last))
        return per_exit_logits, per_exit_confidence_logits


def deepconvlstm_exit(num_classes: int = 12, in_channels: int = 18) -> DeepConvLSTMExit:
    return DeepConvLSTMExit(num_classes=num_classes, in_channels=in_channels)
