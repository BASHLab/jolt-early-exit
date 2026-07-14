"""EfficientNet-B0 with three early exits + final.

Wraps the torchvision ``efficientnet_b0`` backbone with classifier heads inserted after
its 2nd, 4th, and 6th MBConv stages, mirroring the three-early-exit + final-head
convention used by the other backbones in this repo (BC-ResNet-8, CCT-7/3x2, MobileNetV2).
Stage output channels at the cut points are 24 / 80 / 192 / 1280 (last is after the final
1x1 conv that produces the 1280-channel feature map).

Input convention: ``(B, in_channels, H, W)``. The default constructor accepts
``in_channels=1`` for log-mel-spectrogram inputs and replicates the channel three times so
the standard 3-channel EfficientNet stem can be re-used without weight surgery.

~ 4.1M params at 50-class ESC-50 output, well under the 5M on-device cap.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights

from .base import ExitModel


# Cuts inside the torchvision ``features`` Sequential: indices 0..8 where 0 is the stem
# Conv2dNormActivation, 1..7 are the MBConv stages with growing channel counts, and 8 is
# the final 1x1 conv to 1280 channels. We bundle the stem + first two MBConvs into chunk_0,
# the next two into chunk_1, the next two into chunk_2, and the rest (last MBConv + final
# 1x1) into the final exit path.
CUTS = [(0, 3), (3, 5), (5, 7), (7, 9)]  # python-slice [start, end) into features
CHANNELS_AT_CUT = [24, 80, 192, 1280]


class _Head2d(nn.Module):
    def __init__(self, c_in: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _GateHead2d(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


class _ConfidenceHead2d(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


class EfficientNetB0Exit(ExitModel):
    num_exits = 3

    def __init__(self, num_classes: int, in_channels: int = 1, pretrained: bool = True):
        super().__init__()
        self.num_exits = 3
        self.num_classes = num_classes
        self.in_channels = in_channels

        # ImageNet-pretrained weights for the feature extractor. The classifier
        # head is replaced by our per-exit heads, so loading with the default
        # 1000-class classifier is fine: we only use ``base.features`` below.
        # The treat-spectrogram-as-image transfer recipe is the standard ESC-50
        # baseline at ~85-90% accuracy (Park et al. 2020).
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b0(weights=weights)
        feats = base.features
        # Bundle features into our four chunks.
        self.chunk_0 = nn.Sequential(*[feats[i] for i in range(CUTS[0][0], CUTS[0][1])])
        self.chunk_1 = nn.Sequential(*[feats[i] for i in range(CUTS[1][0], CUTS[1][1])])
        self.chunk_2 = nn.Sequential(*[feats[i] for i in range(CUTS[2][0], CUTS[2][1])])
        self.chunk_final = nn.Sequential(*[feats[i] for i in range(CUTS[3][0], CUTS[3][1])])

        c0, c1, c2, cF = CHANNELS_AT_CUT

        self.exit_head_0 = _Head2d(c0, num_classes)
        self.exit_head_1 = _Head2d(c1, num_classes)
        self.exit_head_2 = _Head2d(c2, num_classes)
        self.exit_head_final = _Head2d(cF, num_classes)

        # Training-only scaffolding for JEI-DNN and SCAR baselines.
        self.gate_head_0 = _GateHead2d(c0)
        self.confidence_head_0 = _ConfidenceHead2d(c0)
        self.gate_head_1 = _GateHead2d(c1)
        self.confidence_head_1 = _ConfidenceHead2d(c1)
        self.gate_head_2 = _GateHead2d(c2)
        self.confidence_head_2 = _ConfidenceHead2d(c2)
        self.confidence_head_final = _ConfidenceHead2d(cF)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate 1-channel input to 3 channels for the torchvision stem.
        if x.size(1) == 1 and self.in_channels == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

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
            h = self.chunk_final(x)
            return h, self.exit_head_final(h)
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
        x = self.chunk_final(x); per_exit_logits.append(self.exit_head_final(x))
        return per_exit_logits, per_exit_gate_logits

    def forward_with_confidences(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x = self._prep(x)
        per_exit_logits: List[torch.Tensor] = []
        per_exit_confidence_logits: List[torch.Tensor] = []
        x = self.chunk_0(x); per_exit_logits.append(self.exit_head_0(x)); per_exit_confidence_logits.append(self.confidence_head_0(x))
        x = self.chunk_1(x); per_exit_logits.append(self.exit_head_1(x)); per_exit_confidence_logits.append(self.confidence_head_1(x))
        x = self.chunk_2(x); per_exit_logits.append(self.exit_head_2(x)); per_exit_confidence_logits.append(self.confidence_head_2(x))
        x = self.chunk_final(x); per_exit_logits.append(self.exit_head_final(x)); per_exit_confidence_logits.append(self.confidence_head_final(x))
        return per_exit_logits, per_exit_confidence_logits


def efficientnet_b0_exit(num_classes: int, in_channels: int = 1) -> EfficientNetB0Exit:
    return EfficientNetB0Exit(num_classes=num_classes, in_channels=in_channels)
