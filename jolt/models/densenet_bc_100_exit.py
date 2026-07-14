"""Multi-exit DenseNet-BC-100 (k=12) for CIFAR-100 / 32x32 image classification.

Source: Huang et al., "Densely Connected Convolutional Networks", CVPR 2017.
The BC ("bottleneck + compression") variant uses 1x1 bottlenecks before each
3x3 conv (BC) and halves channels at transitions (compression 0.5).
Configuration here matches the canonical L=100, k=12 setting from the paper:

    initial conv: 24ch (2*k) -> 32x32
    dense block 1: 16 layers, 24 -> 24 + 16*12 = 216ch -> 32x32
    transition 1: BN-ReLU-1x1conv-avgpool -> 108ch -> 16x16
    dense block 2: 16 layers, 108 -> 108 + 16*12 = 300ch -> 16x16
    transition 2: BN-ReLU-1x1conv-avgpool -> 150ch -> 8x8
    dense block 3: 16 layers, 150 -> 150 + 16*12 = 342ch -> 8x8
    final BN-ReLU-avgpool-FC

~0.8M params at num_classes=100 with k=12 / depth=100 / compression=0.5. Reported
CIFAR-100 top-1 accuracy: 77.93% (Huang 2017, Table 2, DenseNet-BC depth=100 k=12).

Multi-exit splits (3 internal + 1 final = 4 classifiers):
    chunk_0 = stem + dense_block_1 + transition_1   (108ch, 16x16)
    chunk_1 = dense_block_2 + transition_2          (150ch, 8x8)
    chunk_2 = dense_block_3                         (342ch, 8x8)
    chunk_3 = final BN-ReLU + final classifier

Heads use SDN mixed-pool (AvgPool || MaxPool -> Linear(2C, num_classes)).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel


class _BottleneckLayer(nn.Module):
    """BN-ReLU-1x1conv(4k)-BN-ReLU-3x3conv(k) bottleneck dense layer."""

    def __init__(self, in_channels: int, growth: int):
        super().__init__()
        inter = 4 * growth
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter)
        self.conv2 = nn.Conv2d(inter, growth, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, growth: int):
        super().__init__()
        layers: List[nn.Module] = []
        c = in_channels
        for _ in range(num_layers):
            layers.append(_BottleneckLayer(c, growth))
            c += growth
        self.layers = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _Transition(nn.Module):
    """BN-ReLU-1x1conv(compression)-avgpool(2)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(F.relu(self.bn(x), inplace=True))
        return F.avg_pool2d(out, kernel_size=2, stride=2)


class _MixedPoolHead2d(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1))


class _GateHead2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)).squeeze(-1)


class _ConfidenceHead2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)).squeeze(-1)


class DenseNetBC100Exit(ExitModel):
    """DenseNet-BC-100 (depth=100, growth=12, compression=0.5) with 3 internal exits + final."""

    num_exits = 3

    def __init__(
        self,
        num_classes: int = 100,
        in_channels: int = 3,
        growth: int = 12,
        compression: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Depth 100, BC variant: each dense layer counts as 2 (1x1 + 3x3),
        # so the per-block depth is (100 - 4) / 6 = 16 layers per block.
        block_depth = 16

        # Stem: 3x3 conv to 2*growth channels (CIFAR convention).
        c = 2 * growth
        self.stem = nn.Conv2d(in_channels, c, kernel_size=3, padding=1, bias=False)

        # Block 1
        self.dense1 = _DenseBlock(block_depth, c, growth)
        c1_in = self.dense1.out_channels                 # 24 + 16*12 = 216
        c1_out = max(1, int(c1_in * compression))        # 108
        self.trans1 = _Transition(c1_in, c1_out)
        self.chunk_0 = nn.Sequential(self.stem, self.dense1, self.trans1)
        self.exit_head_0 = _MixedPoolHead2d(c1_out, num_classes)
        self.gate_head_0 = _GateHead2d(c1_out)
        self.confidence_head_0 = _ConfidenceHead2d(c1_out)

        # Block 2
        self.dense2 = _DenseBlock(block_depth, c1_out, growth)
        c2_in = self.dense2.out_channels                 # 108 + 16*12 = 300
        c2_out = max(1, int(c2_in * compression))        # 150
        self.trans2 = _Transition(c2_in, c2_out)
        self.chunk_1 = nn.Sequential(self.dense2, self.trans2)
        self.exit_head_1 = _MixedPoolHead2d(c2_out, num_classes)
        self.gate_head_1 = _GateHead2d(c2_out)
        self.confidence_head_1 = _ConfidenceHead2d(c2_out)

        # Block 3 (no transition after final dense block per Huang 2017)
        self.dense3 = _DenseBlock(block_depth, c2_out, growth)
        c3_out = self.dense3.out_channels                # 150 + 16*12 = 342
        self.chunk_2 = self.dense3
        self.exit_head_2 = _MixedPoolHead2d(c3_out, num_classes)
        self.gate_head_2 = _GateHead2d(c3_out)
        self.confidence_head_2 = _ConfidenceHead2d(c3_out)

        # Final classifier: BN-ReLU-AvgPool-FC (canonical DenseNet finisher)
        self.final_bn = nn.BatchNorm2d(c3_out)
        self.final_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.final_classifier = nn.Linear(c3_out, num_classes)
        self.chunk_3 = nn.Identity()  # final compute is in _final_classify
        self.confidence_head_final = _ConfidenceHead2d(c3_out)

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
            return x, self._final_classify(x)
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
        logits.append(self._final_classify(x))
        return logits, gates

    def forward_with_confidences(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        confs: List[torch.Tensor] = []
        x = self.chunk_0(x); logits.append(self.exit_head_0(x)); confs.append(self.confidence_head_0(x))
        x = self.chunk_1(x); logits.append(self.exit_head_1(x)); confs.append(self.confidence_head_1(x))
        x = self.chunk_2(x); logits.append(self.exit_head_2(x)); confs.append(self.confidence_head_2(x))
        logits.append(self._final_classify(x)); confs.append(self.confidence_head_final(x))
        return logits, confs


def densenet_bc_100_exit(num_classes: int = 100, in_channels: int = 3) -> DenseNetBC100Exit:
    return DenseNetBC100Exit(num_classes=num_classes, in_channels=in_channels)
