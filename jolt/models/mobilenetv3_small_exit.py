"""Multi-exit MobileNetV3-Small for CIFAR-100 / small image classification.

Source: Howard et al., "Searching for MobileNetV3" (ICCV 2019). torchvision's
canonical mobilenet_v3_small wrapped with three internal exit heads + final.

Block layout (with stem_stride=1, for 32x32 input):
    block  0 (stem)            -> 16ch
    block  1                   -> 16ch
    block  2-3                 -> 24ch
    block  4-6                 -> 40ch
    block  7-8                 -> 48ch
    block  9-11                -> 96ch
    block 12 (Conv1x1)         -> 576ch
    -> classifier (576 -> 1024 -> num_classes)

Multi-exit splits:
    chunk_0 = blocks 0-3  (exit at 24ch)
    chunk_1 = blocks 4-6  (40ch)
    chunk_2 = blocks 7-11 (96ch)
    chunk_3 = block 12 + canonical classifier (576ch -> ...).

3 early exits + 1 final = 4 classifiers. Heads use SDN mixed-pool (AvgPool || MaxPool ->
Linear(2C, num_classes)) matching the convention in mobilenetv3_large_exit.

~2.5M params at num_classes=100. Different routing surface than MV2 / MN3-Large
because the early exits see lower-channel features earlier in the network.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .base import ExitModel


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


class MobileNetV3SmallExit(ExitModel):
    """MobileNetV3-Small with 3 internal mixed-pool exits + final classifier."""

    num_exits = 3

    def __init__(
        self,
        num_classes: int = 100,
        in_channels: int = 3,
        stem_stride: int = 1,
    ):
        from torchvision.models import mobilenet_v3_small
        super().__init__()
        self.num_classes = num_classes

        base = mobilenet_v3_small(num_classes=num_classes)
        features = list(base.features)

        if stem_stride != 2:
            stem_conv = features[0][0]
            stem_conv.stride = (stem_stride, stem_stride)

        # Per the block inventory: 13 features blocks (0..12), output channels
        # 16, 16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96, 576.
        self.chunk_0 = nn.Sequential(*features[0:4])    # ends 24ch
        self.exit_head_0 = _MixedPoolHead2d(24, num_classes)
        self.gate_head_0 = _GateHead2d(24)
        self.confidence_head_0 = _ConfidenceHead2d(24)

        self.chunk_1 = nn.Sequential(*features[4:7])    # ends 40ch
        self.exit_head_1 = _MixedPoolHead2d(40, num_classes)
        self.gate_head_1 = _GateHead2d(40)
        self.confidence_head_1 = _ConfidenceHead2d(40)

        self.chunk_2 = nn.Sequential(*features[7:12])   # ends 96ch
        self.exit_head_2 = _MixedPoolHead2d(96, num_classes)
        self.gate_head_2 = _GateHead2d(96)
        self.confidence_head_2 = _ConfidenceHead2d(96)

        # Final chunk: block 12 (Conv1x1 -> 576ch) plus the canonical MN3-Small
        # classifier (576 -> 1024 -> num_classes).
        self.chunk_3 = nn.Sequential(*features[12:])
        self.final_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.final_classifier = base.classifier
        self.confidence_head_final = _ConfidenceHead2d(576)

    def _final_classify(self, x: torch.Tensor) -> torch.Tensor:
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
            x = self.chunk_3(x); return x, self._final_classify(x)
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
        x = self.chunk_3(x); logits.append(self._final_classify(x))
        return logits, gates

    def forward_with_confidences(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        confs: List[torch.Tensor] = []
        x = self.chunk_0(x); logits.append(self.exit_head_0(x)); confs.append(self.confidence_head_0(x))
        x = self.chunk_1(x); logits.append(self.exit_head_1(x)); confs.append(self.confidence_head_1(x))
        x = self.chunk_2(x); logits.append(self.exit_head_2(x)); confs.append(self.confidence_head_2(x))
        x = self.chunk_3(x); logits.append(self._final_classify(x)); confs.append(self.confidence_head_final(x))
        return logits, confs


def mobilenetv3_small_exit(
    num_classes: int = 100, in_channels: int = 3, stem_stride: int = 1
) -> MobileNetV3SmallExit:
    return MobileNetV3SmallExit(num_classes=num_classes, in_channels=in_channels, stem_stride=stem_stride)
