"""Multi-exit WideResNet-28-2 for CIFAR.

WideResNet (Zagoruyko & Komodakis, BMVC 2016) with depth N=28, widening factor k=2.
Pre-activation BasicBlock without dropout. Three groups of 4 blocks each at channel
widths 16k=32 / 32k=64 / 64k=128; downsampling at the first block of groups 2 and 3
via stride=2. ~1.5M params at num_classes=100. Reported canonical CIFAR-100 ~73%.

Three early exits at the canonical SDN depth ratios 0.33 / 0.66 / 1.0:
    chunk_0: stem + group_1 (32 ch, 32x32) -> exit 0
    chunk_1: group_2          (64 ch, 16x16) -> exit 1
    chunk_2: group_3          (128 ch, 8x8) -> exit 2
    chunk_3: final BN-ReLU + avg pool + linear FC

``num_exits = 3`` early exits (3 internal + 1 final = 4 classifiers).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel


class _PreActBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x), inplace=True)
        s = self.shortcut(out if isinstance(self.shortcut, nn.Conv2d) else x)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + s


def _make_group(n_blocks: int, in_c: int, out_c: int, stride: int) -> nn.Sequential:
    layers: List[nn.Module] = [_PreActBasicBlock(in_c, out_c, stride=stride)]
    for _ in range(n_blocks - 1):
        layers.append(_PreActBasicBlock(out_c, out_c, stride=1))
    return nn.Sequential(*layers)


class _MixedPoolHead(nn.Module):
    """SDN-style internal classifier: AvgPool || MaxPool -> Linear(2*C, num_classes)."""
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1))


class _GateHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * channels, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)).squeeze(-1)


class _ConfidenceHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * channels, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)).squeeze(-1)


class WideResNet28_2Exit(ExitModel):
    """WRN-28-2 with three internal exits + final classifier."""

    num_exits = 3

    def __init__(self, num_classes: int = 100, in_channels: int = 3,
                 depth: int = 28, widen: int = 2):
        super().__init__()
        self.num_classes = num_classes

        # Depth 28 = 4 (stem+final) + 6 * n_blocks_per_group  -> n_per_group = 4
        assert (depth - 4) % 6 == 0, "depth must be 6n+4"
        n_per_group = (depth - 4) // 6  # =4 for depth=28

        c = [16, 16 * widen, 32 * widen, 64 * widen]  # [16, 32, 64, 128]

        self.stem = nn.Conv2d(in_channels, c[0], kernel_size=3, padding=1, bias=False)

        self.chunk_0 = nn.Sequential(_make_group(n_per_group, c[0], c[1], stride=1))
        self.exit_head_0 = _MixedPoolHead(c[1], num_classes)
        self.gate_head_0 = _GateHead(c[1])
        self.confidence_head_0 = _ConfidenceHead(c[1])

        self.chunk_1 = _make_group(n_per_group, c[1], c[2], stride=2)
        self.exit_head_1 = _MixedPoolHead(c[2], num_classes)
        self.gate_head_1 = _GateHead(c[2])
        self.confidence_head_1 = _ConfidenceHead(c[2])

        self.chunk_2 = _make_group(n_per_group, c[2], c[3], stride=2)
        self.exit_head_2 = _MixedPoolHead(c[3], num_classes)
        self.gate_head_2 = _GateHead(c[3])
        self.confidence_head_2 = _ConfidenceHead(c[3])

        # Final BN-ReLU + AvgPool + FC (canonical WRN finisher)
        self.final_bn = nn.BatchNorm2d(c[3])
        self.final_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.final_classifier = nn.Linear(c[3], num_classes)
        self.confidence_head_final = _ConfidenceHead(c[3])

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def _final_classify(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.final_bn(x), inplace=True)
        y = self.final_avg_pool(x).flatten(1)
        return self.final_classifier(y)

    def forward(self, x: torch.Tensor, exit_layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if exit_layer_idx is None:
            exit_layer_idx = 0
        if exit_layer_idx == 0:
            h = self.chunk_0(self.stem(x)); return h, self.exit_head_0(h)
        if exit_layer_idx == 1:
            h = self.chunk_1(x); return h, self.exit_head_1(h)
        if exit_layer_idx == 2:
            h = self.chunk_2(x); return h, self.exit_head_2(h)
        if exit_layer_idx == 3:
            return x, self._final_classify(x)
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
        )

    def forward_with_gates(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        h = self.chunk_0(self.stem(x)); logits.append(self.exit_head_0(h)); gates.append(self.gate_head_0(h))
        h = self.chunk_1(h); logits.append(self.exit_head_1(h)); gates.append(self.gate_head_1(h))
        h = self.chunk_2(h); logits.append(self.exit_head_2(h)); gates.append(self.gate_head_2(h))
        logits.append(self._final_classify(h))
        return logits, gates

    def forward_with_confidences(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        confs: List[torch.Tensor] = []
        h = self.chunk_0(self.stem(x)); logits.append(self.exit_head_0(h)); confs.append(self.confidence_head_0(h))
        h = self.chunk_1(h); logits.append(self.exit_head_1(h)); confs.append(self.confidence_head_1(h))
        h = self.chunk_2(h); logits.append(self.exit_head_2(h)); confs.append(self.confidence_head_2(h))
        logits.append(self._final_classify(h)); confs.append(self.confidence_head_final(h))
        return logits, confs


def wideresnet28_2_exit(num_classes: int = 100, in_channels: int = 3) -> WideResNet28_2Exit:
    return WideResNet28_2Exit(num_classes=num_classes, in_channels=in_channels)
