"""Multi-exit MobileNetV3-Large for Tiny-ImageNet (parallel fallback to CCT-7/3x2).

Source: Howard et al., "Searching for MobileNetV3" (ICCV 2019). torchvision provides the
canonical implementation; we wrap it with multi-exit hooks at the SDN-style placements.

Tiny-ImageNet (64x64 input) versus the model's native ImageNet (224x224) recipe: we set the
stem stride to 1 so spatial resolution stays large enough to be useful at the early exits.
Block layout (with stem_stride=1, input 64x64):

    block  0 (stem)            -> 64x64
    block  1                   -> 64x64 (16ch)
    block  2-3                 -> 32x32 (24ch)
    block  4-6                 -> 16x16 (40ch)
    block  7-10                ->  8x8 (80ch)
    block 11-12                ->  8x8 (112ch)
    block 13-15                ->  4x4 (160ch)
    block 16 (Conv1x1)         ->  4x4 (960ch)
    -> classifier (960 -> 1280 -> num_classes)

Multi-exit splits: chunk_0 = blocks 0-6 (exit at 16x16, 40ch), chunk_1 = blocks 7-10 (8x8,
80ch), chunk_2 = blocks 11-14 (4x4, 160ch), chunk_3 = blocks 15-16 + canonical classifier.
3 internal exits + 1 final = 4 classifiers. Heads use SDN mixed-pool (AvgPool || MaxPool ->
Linear(2C, num_classes)) matching the convention in resnet56_sdn_exit / bcresnet_exit.

~4.5M params at num_classes=200. Published Tiny-ImageNet top-1 ~63.68% (third-party
benchmark, recipe-dependent). Not the same as Hassani CCT-7's 66.9% but a cleaner
non-transformer fallback if CCT-7 doesn't pan out.
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


class MobileNetV3LargeExit(ExitModel):
    """MobileNetV3-Large with 3 internal mixed-pool exits + final classifier."""

    num_exits = 3

    def __init__(
        self,
        num_classes: int = 200,
        in_channels: int = 3,
        stem_stride: int = 1,
        dropout: float = 0.2,
    ):
        from torchvision.models import mobilenet_v3_large
        super().__init__()
        self.num_classes = num_classes

        base = mobilenet_v3_large(num_classes=num_classes)
        features = list(base.features)

        if stem_stride != 2:
            # Override stem stride. The stem is features[0] = Conv2dNormActivation with a
            # Conv2d(3, 16, kernel=3, stride=(2,2), padding=(1,1)).
            stem_conv = features[0][0]
            stem_conv.stride = (stem_stride, stem_stride)

        # SDN-style chunk split:
        #   chunk_0: blocks 0-6  -> ends at 40ch
        #   chunk_1: blocks 7-10 -> ends at 80ch
        #   chunk_2: blocks 11-14 -> ends at 160ch
        #   chunk_3: blocks 15-16 + classifier
        self.chunk_0 = nn.Sequential(*features[0:7])
        self.exit_head_0 = _MixedPoolHead2d(40, num_classes)
        self.gate_head_0 = _GateHead2d(40)
        self.confidence_head_0 = _ConfidenceHead2d(40)

        self.chunk_1 = nn.Sequential(*features[7:11])
        self.exit_head_1 = _MixedPoolHead2d(80, num_classes)
        self.gate_head_1 = _GateHead2d(80)
        self.confidence_head_1 = _ConfidenceHead2d(80)

        self.chunk_2 = nn.Sequential(*features[11:15])
        self.exit_head_2 = _MixedPoolHead2d(160, num_classes)
        self.gate_head_2 = _GateHead2d(160)
        self.confidence_head_2 = _ConfidenceHead2d(160)

        # Final chunk: blocks 15-16 (last InvertedResidual + Conv2dNormActivation -> 960ch)
        # plus the canonical MobileNetV3 classifier (960 -> 1280 -> num_classes).
        self.chunk_3 = nn.Sequential(*features[15:])
        self.final_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.final_classifier = base.classifier
        self.confidence_head_final = _ConfidenceHead2d(960)

    def _final_classify(self, x: torch.Tensor) -> torch.Tensor:
        y = self.final_avg_pool(x).flatten(1)
        return self.final_classifier(y)

    def forward(
        self, x: torch.Tensor, exit_layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exit_layer_idx is None:
            exit_layer_idx = 0
        if exit_layer_idx == 0:
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
            return x, self._final_classify(x)
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
        )

    def forward_with_gates(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        per_exit_logits: List[torch.Tensor] = []
        per_exit_gate_logits: List[torch.Tensor] = []
        x = self.chunk_0(x)
        per_exit_logits.append(self.exit_head_0(x))
        per_exit_gate_logits.append(self.gate_head_0(x))
        x = self.chunk_1(x)
        per_exit_logits.append(self.exit_head_1(x))
        per_exit_gate_logits.append(self.gate_head_1(x))
        x = self.chunk_2(x)
        per_exit_logits.append(self.exit_head_2(x))
        per_exit_gate_logits.append(self.gate_head_2(x))
        x = self.chunk_3(x)
        per_exit_logits.append(self._final_classify(x))
        return per_exit_logits, per_exit_gate_logits

    def forward_with_confidences(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        per_exit_logits: List[torch.Tensor] = []
        per_exit_confidence_logits: List[torch.Tensor] = []
        x = self.chunk_0(x)
        per_exit_logits.append(self.exit_head_0(x))
        per_exit_confidence_logits.append(self.confidence_head_0(x))
        x = self.chunk_1(x)
        per_exit_logits.append(self.exit_head_1(x))
        per_exit_confidence_logits.append(self.confidence_head_1(x))
        x = self.chunk_2(x)
        per_exit_logits.append(self.exit_head_2(x))
        per_exit_confidence_logits.append(self.confidence_head_2(x))
        x = self.chunk_3(x)
        per_exit_logits.append(self._final_classify(x))
        per_exit_confidence_logits.append(self.confidence_head_final(x))
        return per_exit_logits, per_exit_confidence_logits


def mobilenetv3_large_exit(
    num_classes: int = 200, in_channels: int = 3, stem_stride: int = 1
) -> MobileNetV3LargeExit:
    return MobileNetV3LargeExit(num_classes=num_classes, in_channels=in_channels, stem_stride=stem_stride)
