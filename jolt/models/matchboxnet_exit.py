"""MatchboxNet-3x1x64 with three early exits + final.

Source: Majumdar & Ginsburg, "MatchboxNet: 1D Time-Channel Separable Convolutional Neural
Network Architecture for Speech Commands Recognition" (Interspeech 2020). MatchboxNet uses
1D time-channel separable convolutions over a mel-spectrogram, with each "block" repeating
K depthwise + pointwise sub-blocks. The default 3x1x64 variant is three blocks (B1, B2, B3)
with one sub-block each at base channel width 64.

Input convention: ``(B, 1, mel_bands, time)`` for compatibility with the existing GSC
mel-spectrogram dataloader (40 mels x 101 frames at 16 kHz with 10 ms hop). The mel axis is
squeezed and treated as the channel axis of the 1D convolutions.

~ 90K params at 35-class GSC v2 output.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel


class _SepConv1d(nn.Module):
    """Time-channel separable 1D conv: depthwise time-conv then pointwise channel mix."""

    def __init__(self, c_in: int, c_out: int, kernel: int):
        super().__init__()
        pad = kernel // 2
        self.depthwise = nn.Conv1d(c_in, c_in, kernel, padding=pad, groups=c_in, bias=False)
        self.pointwise = nn.Conv1d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


class _MBBlock(nn.Module):
    """MatchboxNet block: K sub-blocks plus a residual pointwise + BN over the block input."""

    def __init__(self, c_in: int, c_out: int, kernel: int, k_sub: int = 1):
        super().__init__()
        layers = []
        for i in range(k_sub):
            layers.append(_SepConv1d(c_in if i == 0 else c_out, c_out, kernel))
        self.body = nn.Sequential(*layers)
        self.residual = nn.Sequential(
            nn.Conv1d(c_in, c_out, 1, bias=False),
            nn.BatchNorm1d(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.residual(x))


class _Head1d(nn.Module):
    """Per-exit classifier head: AdaptiveAvgPool1d -> Linear."""

    def __init__(self, c_in: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class _GateHead1d(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).squeeze(-1))


class _ConfidenceHead1d(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).squeeze(-1))


class MatchboxNetExit(ExitModel):
    """MatchboxNet-3x1x64 with exits after each of the three MB blocks."""

    num_exits = 3

    def __init__(self, num_classes: int = 35, in_channels: int = 1, mel_bands: int = 40):
        super().__init__()
        self.num_exits = 3
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.mel_bands = mel_bands

        # Prologue: 1D conv that maps the mel-band axis into 128 channels over time.
        c0 = 128
        self.prologue = _SepConv1d(mel_bands, c0, kernel=11)

        # Three MB blocks at growing channel widths.
        c1, c2, c3 = 64, 64, 64
        self.chunk_0 = _MBBlock(c0, c1, kernel=13)
        self.chunk_1 = _MBBlock(c1, c2, kernel=15)
        self.chunk_2 = _MBBlock(c2, c3, kernel=17)

        # Epilogue 1D conv before the final classifier.
        self.epilogue = _SepConv1d(c3, 128, kernel=29)
        c_final = 128

        # Per-exit classifier heads.
        self.exit_head_0 = _Head1d(c1, num_classes)
        self.exit_head_1 = _Head1d(c2, num_classes)
        self.exit_head_2 = _Head1d(c3, num_classes)
        self.exit_head_final = _Head1d(c_final, num_classes)

        # Training-only scaffolding: per-exit gate heads (JEI-DNN) and confidence heads (SCAR).
        # Inference forward path does not use these.
        self.gate_head_0 = _GateHead1d(c1)
        self.confidence_head_0 = _ConfidenceHead1d(c1)
        self.gate_head_1 = _GateHead1d(c2)
        self.confidence_head_1 = _ConfidenceHead1d(c2)
        self.gate_head_2 = _GateHead1d(c3)
        self.confidence_head_2 = _ConfidenceHead1d(c3)
        self.confidence_head_final = _ConfidenceHead1d(c_final)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze the channel-axis "1" of (B, 1, mel, time) -> (B, mel, time)
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        return self.prologue(x)

    def _final_classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.exit_head_final(self.epilogue(x))

    def forward(
        self, x: torch.Tensor, exit_layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exit_layer_idx is None:
            exit_layer_idx = 0
        if exit_layer_idx == 0:
            x = self._prep(x)
            h = self.chunk_0(x)
            return h, self.exit_head_0(h)
        if exit_layer_idx == 1:
            h = self.chunk_1(x)
            return h, self.exit_head_1(h)
        if exit_layer_idx == 2:
            h = self.chunk_2(x)
            return h, self.exit_head_2(h)
        if exit_layer_idx == 3:
            return x, self._final_classify(x)
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
        )

    def forward_with_gates(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x = self._prep(x)
        per_exit_logits: List[torch.Tensor] = []
        per_exit_gate_logits: List[torch.Tensor] = []
        x = self.chunk_0(x); per_exit_logits.append(self.exit_head_0(x)); per_exit_gate_logits.append(self.gate_head_0(x))
        x = self.chunk_1(x); per_exit_logits.append(self.exit_head_1(x)); per_exit_gate_logits.append(self.gate_head_1(x))
        x = self.chunk_2(x); per_exit_logits.append(self.exit_head_2(x)); per_exit_gate_logits.append(self.gate_head_2(x))
        per_exit_logits.append(self._final_classify(x))
        return per_exit_logits, per_exit_gate_logits

    def forward_with_confidences(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x = self._prep(x)
        per_exit_logits: List[torch.Tensor] = []
        per_exit_confidence_logits: List[torch.Tensor] = []
        x = self.chunk_0(x); per_exit_logits.append(self.exit_head_0(x)); per_exit_confidence_logits.append(self.confidence_head_0(x))
        x = self.chunk_1(x); per_exit_logits.append(self.exit_head_1(x)); per_exit_confidence_logits.append(self.confidence_head_1(x))
        x = self.chunk_2(x); per_exit_logits.append(self.exit_head_2(x)); per_exit_confidence_logits.append(self.confidence_head_2(x))
        x = self.epilogue(x)
        per_exit_logits.append(self.exit_head_final(x))
        per_exit_confidence_logits.append(self.confidence_head_final(x))
        return per_exit_logits, per_exit_confidence_logits


def matchboxnet_exit(num_classes: int = 35, in_channels: int = 1) -> MatchboxNetExit:
    return MatchboxNetExit(num_classes=num_classes, in_channels=in_channels)
