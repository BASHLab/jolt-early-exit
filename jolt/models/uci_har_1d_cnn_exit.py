"""UCI-HAR 1D-CNN with three early exits + final (4 total classifiers).

Canonical 1D-CNN multi-exit backbone for EE-HAR. The architecture follows the standard HAR
1D-CNN convention (Jordao et al. 2020 survey + Tang et al. 2022 baselines): three pooling
stages with channel doubling, exit placement at each pooling boundary.

This is the v4-canonical UCI-HAR cell; the v3 2D-ResNet-18 reshape (data layout
``(n_steps=4, n_length=32, channels=9)``) is non-canonical for EE-HAR and is now legacy.

Input: raw 9-channel inertial signal (body acc xyz + body gyro xyz + total acc xyz), 128
timesteps, channels-first ``(B, 9, 128)``.

Architecture (3 early exits + 1 final = 4 classifiers):

    stem:    Conv1d(9 -> 32, k=3, p=1) -> BN -> ReLU
    block_1: Conv1d(32 -> 64, k=3, p=1) -> BN -> ReLU -> MaxPool1d(2)   [64 ts] EXIT 0
    block_2: Conv1d(64 -> 128, k=3, p=1) -> BN -> ReLU -> MaxPool1d(2)  [32 ts] EXIT 1
    block_3: Conv1d(128 -> 128, k=3, p=1) -> BN -> ReLU -> MaxPool1d(2) [16 ts] EXIT 2
    block_4: Conv1d(128 -> 128, k=3, p=1) -> BN -> ReLU -> AdaptiveAvgPool1d(1) -> FC [FINAL]

Each early exit gets a mixed-pool 1D head (AdaptiveAvgPool1d + AdaptiveMaxPool1d to length 1)
followed by Linear(2C, num_classes), matching the mixed-pool head pattern used in our CIFAR
cells but in 1D. ``num_exits = 3`` (early-exit count); total classifiers = 4.

Param count at num_classes=6: ~140K, well under the 5M on-device HAR cap.

SCAR confidence heads (``confidence_head_k``) and JEI-DNN gate heads (``gate_head_k``)
mirror the per-exit shape.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel


class _Conv1dBlock(nn.Module):
    """Conv1d -> BN -> ReLU, followed by MaxPool1d(2) when ``downsample`` is True."""

    def __init__(self, in_channels: int, out_channels: int, *, downsample: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        return self.pool(x)


class _MixedPoolHead1d(nn.Module):
    """Mixed-pool 1D internal classifier: AdaptiveAvgPool1d || AdaptiveMaxPool1d -> Linear(2C, num_classes)."""

    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(2 * channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_pool(x).flatten(1)
        peak = self.max_pool(x).flatten(1)
        return self.fc(torch.cat([avg, peak], dim=1))


class _GateHead1d(nn.Module):
    """JEI-DNN routing gate (1D variant): mixed-pool -> Linear(2C, 1)."""

    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(2 * channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_pool(x).flatten(1)
        peak = self.max_pool(x).flatten(1)
        return self.fc(torch.cat([avg, peak], dim=1)).squeeze(-1)


class _ConfidenceHead1d(nn.Module):
    """SCAR confidence score (1D variant): mixed-pool -> Linear(2C, 1)."""

    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(2 * channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_pool(x).flatten(1)
        peak = self.max_pool(x).flatten(1)
        return self.fc(torch.cat([avg, peak], dim=1)).squeeze(-1)


class UCIHAR1DCNNExit(ExitModel):
    """UCI-HAR 1D-CNN with configurable early-exit count.

    The network always has four backbone chunks (chunk_0..chunk_3) for parity. The
    ``num_early_exits`` parameter controls how many of those chunks carry a
    classification head, taken from the shallowest end. For example,
    ``num_early_exits=1`` keeps exit_head_0 only (so a single shallow early exit);
    ``num_early_exits=3`` is the default and matches the paper-table cell.
    """

    def __init__(self, num_classes: int = 6, in_channels: int = 9, num_early_exits: int = 3):
        super().__init__()
        if num_early_exits not in (1, 2, 3):
            raise ValueError(f"num_early_exits must be 1, 2, or 3 (got {num_early_exits})")
        self.num_exits = num_early_exits
        self.num_classes = num_classes

        c0, c1, c2 = 32, 64, 128

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c0, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c0),
            nn.ReLU(inplace=True),
        )

        # Three pool-driven downsampling stages (128 -> 64 -> 32 -> 16 timesteps).
        self.chunk_0 = _Conv1dBlock(c0, c1, downsample=True)
        if num_early_exits >= 1:
            self.exit_head_0 = _MixedPoolHead1d(c1, num_classes)
            self.gate_head_0 = _GateHead1d(c1)
            self.confidence_head_0 = _ConfidenceHead1d(c1)

        self.chunk_1 = _Conv1dBlock(c1, c2, downsample=True)
        if num_early_exits >= 2:
            self.exit_head_1 = _MixedPoolHead1d(c2, num_classes)
            self.gate_head_1 = _GateHead1d(c2)
            self.confidence_head_1 = _ConfidenceHead1d(c2)

        self.chunk_2 = _Conv1dBlock(c2, c2, downsample=True)
        if num_early_exits >= 3:
            self.exit_head_2 = _MixedPoolHead1d(c2, num_classes)
            self.gate_head_2 = _GateHead1d(c2)
            self.confidence_head_2 = _ConfidenceHead1d(c2)

        # Final block: standard final-classifier path (no extra pool; the head's
        # AdaptiveAvgPool1d reduces whatever sequence length remains).
        self.chunk_3 = _Conv1dBlock(c2, c2, downsample=False)
        self.final_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, num_classes)
        self.confidence_head_final = _ConfidenceHead1d(c2)

        # Track the backbone chunks that need to run between exit i and the final
        # classifier. With num_early_exits=K we still have 4 chunks total; chunks
        # K..3 form the "remaining" backbone that the final call runs after the
        # K-th forward call.
        self._all_chunks = (self.chunk_0, self.chunk_1, self.chunk_2, self.chunk_3)

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

    def _final_classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.final_avg_pool(x).flatten(1))

    def _exit_head(self, idx: int) -> nn.Module:
        return [self.exit_head_0 if hasattr(self, "exit_head_0") else None,
                self.exit_head_1 if hasattr(self, "exit_head_1") else None,
                self.exit_head_2 if hasattr(self, "exit_head_2") else None][idx]

    def _gate_head(self, idx: int) -> nn.Module:
        return [self.gate_head_0 if hasattr(self, "gate_head_0") else None,
                self.gate_head_1 if hasattr(self, "gate_head_1") else None,
                self.gate_head_2 if hasattr(self, "gate_head_2") else None][idx]

    def _confidence_head(self, idx: int) -> nn.Module:
        return [self.confidence_head_0 if hasattr(self, "confidence_head_0") else None,
                self.confidence_head_1 if hasattr(self, "confidence_head_1") else None,
                self.confidence_head_2 if hasattr(self, "confidence_head_2") else None][idx]

    def forward(
        self, x: torch.Tensor, exit_layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exit_layer_idx is None:
            exit_layer_idx = 0
        K = self.num_exits
        # FINAL classifier call. Caller hands us x = output of chunk_{K-1}
        # (after stem). We run the remaining backbone chunks and apply the
        # final fc head.
        if exit_layer_idx == K:
            if K <= 1:
                x = self.chunk_1(x)
            if K <= 2:
                x = self.chunk_2(x)
            x = self.chunk_3(x)
            return x, self._final_classify(x)
        # Early-exit call.
        if exit_layer_idx == 0:
            x = self.stem(x)
            x = self.chunk_0(x)
            return x, self.exit_head_0(x)
        if exit_layer_idx == 1 and K >= 2:
            x = self.chunk_1(x)
            return x, self.exit_head_1(x)
        if exit_layer_idx == 2 and K >= 3:
            x = self.chunk_2(x)
            return x, self.exit_head_2(x)
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for K={K} early exits"
        )

    def forward_with_gates(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Single-pass forward returning K+1 classifier logits + K gate logits."""
        per_exit_logits: List[torch.Tensor] = []
        per_exit_gate_logits: List[torch.Tensor] = []
        x = self.stem(x)
        x = self.chunk_0(x)
        if self.num_exits >= 1:
            per_exit_logits.append(self.exit_head_0(x))
            per_exit_gate_logits.append(self.gate_head_0(x))
        x = self.chunk_1(x)
        if self.num_exits >= 2:
            per_exit_logits.append(self.exit_head_1(x))
            per_exit_gate_logits.append(self.gate_head_1(x))
        x = self.chunk_2(x)
        if self.num_exits >= 3:
            per_exit_logits.append(self.exit_head_2(x))
            per_exit_gate_logits.append(self.gate_head_2(x))
        x = self.chunk_3(x)
        per_exit_logits.append(self._final_classify(x))
        return per_exit_logits, per_exit_gate_logits

    def forward_with_confidences(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Single-pass forward returning K+1 classifier logits + K+1 confidence logits."""
        per_exit_logits: List[torch.Tensor] = []
        per_exit_confidence_logits: List[torch.Tensor] = []
        x = self.stem(x)
        x = self.chunk_0(x)
        if self.num_exits >= 1:
            per_exit_logits.append(self.exit_head_0(x))
            per_exit_confidence_logits.append(self.confidence_head_0(x))
        x = self.chunk_1(x)
        if self.num_exits >= 2:
            per_exit_logits.append(self.exit_head_1(x))
            per_exit_confidence_logits.append(self.confidence_head_1(x))
        x = self.chunk_2(x)
        if self.num_exits >= 3:
            per_exit_logits.append(self.exit_head_2(x))
            per_exit_confidence_logits.append(self.confidence_head_2(x))
        x = self.chunk_3(x)
        per_exit_logits.append(self._final_classify(x))
        per_exit_confidence_logits.append(self.confidence_head_final(x))
        return per_exit_logits, per_exit_confidence_logits


def uci_har_1d_cnn_exit(num_classes: int = 6, in_channels: int = 9,
                        num_early_exits: int = 3) -> UCIHAR1DCNNExit:
    return UCIHAR1DCNNExit(num_classes=num_classes, in_channels=in_channels,
                           num_early_exits=num_early_exits)
