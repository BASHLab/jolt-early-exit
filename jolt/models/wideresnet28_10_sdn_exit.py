"""WideResNet-28-10 with seven exits at SDN compute-fraction placements.

WideResNet (Zagoruyko & Komodakis, BMVC 2016): N=28 depth, k=10 widening factor. Pre-activation
BasicBlock with dropout=0.3 between convs. Three groups of 4 blocks each at channel widths
16k=160 / 32k=320 / 64k=640; downsampling at the first block of groups 2 and 3 via stride=2.
Total ~36M params -- over the v4 on-device cap (~5M), within the relaxed image-cell cap (~40M).
Standard 200-epoch SGD-cosine recipe reaches ~82-83% on CIFAR-100 from scratch.

Multi-exit layout matches the canonical SDN compute-fraction placements (~15/30/45/60/75/90%
+ final). With 12 blocks of roughly equal MAC cost (since each group doubles channels but
halves spatial resolution), the placement maps cleanly to "after block k of 12" with k in
{2, 4, 5, 7, 9, 11} plus the final at block 12.

Chunks for the 7 exits:
    chunk_0: blocks  1- 2   (group 1, 160 ch, 32x32)
    chunk_1: blocks  3- 4   (group 1, 160 ch, 32x32)
    chunk_2: block   5      (group 1 -> group 2 transition; ends 320 ch, 16x16)
    chunk_3: blocks  6- 7   (group 2, 320 ch, 16x16)
    chunk_4: blocks  8- 9   (group 2 -> group 3 transition; ends 640 ch, 8x8)
    chunk_5: blocks 10-11   (group 3, 640 ch, 8x8)
    chunk_6: block  12      (group 3, 640 ch, 8x8) + final BN-ReLU + avg pool + linear FC

``num_exits = 6`` per the ExitModel contract (6 early exits + 1 final).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel


class _MixedPoolHead(nn.Module):
    """SDN-style internal classifier: AvgPool || MaxPool -> Linear(2*C, num_classes)."""

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_pool(x).flatten(1)
        peak = self.max_pool(x).flatten(1)
        return self.fc(torch.cat([avg, peak], dim=1))


class _GateHead(nn.Module):
    """JEI-DNN gate (Regol et al. ICLR 2024): AvgPool || MaxPool -> Linear(2*C, 1).

    Returns one logit per sample; sigmoid(logit) is the Bernoulli probability of routing
    a sample to this exit. Default init puts initial sigmoid near 0.5.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_pool(x).flatten(1)
        peak = self.max_pool(x).flatten(1)
        return self.fc(torch.cat([avg, peak], dim=1)).squeeze(-1)


class _ConfidenceHead(nn.Module):
    """SCAR confidence head (v5): AvgPool || MaxPool -> Linear(2*C, 1).

    Returns one logit per sample; sigmoid(logit) is the SCAR selection score s_j(x) used
    both for the structure-aware rank surrogate at train and for exit-time routing at eval.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_pool(x).flatten(1)
        peak = self.max_pool(x).flatten(1)
        return self.fc(torch.cat([avg, peak], dim=1)).squeeze(-1)


class _WRNBlock(nn.Module):
    """Pre-activation WideResNet BasicBlock with dropout (Zagoruyko-Komodakis 2016).

    Structure:
        x -> BN -> ReLU -> Conv3x3 -> Dropout -> BN -> ReLU -> Conv3x3 -> (+) -> out
                       \\-> [Conv1x1 shortcut if dims differ, applied to BN-ReLU'd x] -->/
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.3):
        super().__init__()
        self.equal_dim = (in_channels == out_channels and stride == 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if not self.equal_dim:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False,
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.equal_dim:
            out = F.relu(self.bn1(x), inplace=True)
        else:
            x = F.relu(self.bn1(x), inplace=True)
            out = x
        out = self.conv1(out)
        out = self.dropout(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        if self.shortcut is not None:
            return out + self.shortcut(x)
        return out + x


def _make_blocks(specs: List[Tuple[int, int, int]], dropout: float = 0.3) -> nn.Sequential:
    return nn.Sequential(*[_WRNBlock(c_in, c_out, s, dropout) for c_in, c_out, s in specs])


class WideResNet2810SDNExit(ExitModel):
    """WideResNet-28-10 with six internal SDN classifiers + final classifier (7 total)."""

    num_exits = 6

    def __init__(
        self,
        num_classes: int = 100,
        in_channels: int = 3,
        widening_factor: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        k = widening_factor
        c0, c1, c2, c3 = 16, 16 * k, 32 * k, 64 * k  # 16, 160, 320, 640 for k=10

        # Stem: conv3x3 only; the BN-ReLU happens at the start of the first block (pre-act).
        self.stem = nn.Conv2d(in_channels, c0, kernel_size=3, stride=1, padding=1, bias=False)

        # 12 blocks, 4 per group. First block of groups 2 and 3 is stride 2.
        # chunk_0: blocks 1-2 (16->160 stride 1 + 160->160 stride 1)
        self.chunk_0 = _make_blocks([(c0, c1, 1), (c1, c1, 1)], dropout)
        self.exit_head_0 = _MixedPoolHead(c1, num_classes)
        self.gate_head_0 = _GateHead(c1)
        self.confidence_head_0 = _ConfidenceHead(c1)

        # chunk_1: blocks 3-4 (both 160->160)
        self.chunk_1 = _make_blocks([(c1, c1, 1), (c1, c1, 1)], dropout)
        self.exit_head_1 = _MixedPoolHead(c1, num_classes)
        self.gate_head_1 = _GateHead(c1)
        self.confidence_head_1 = _ConfidenceHead(c1)

        # chunk_2: block 5 (160->320 stride 2, start of group 2)
        self.chunk_2 = _make_blocks([(c1, c2, 2)], dropout)
        self.exit_head_2 = _MixedPoolHead(c2, num_classes)
        self.gate_head_2 = _GateHead(c2)
        self.confidence_head_2 = _ConfidenceHead(c2)

        # chunk_3: blocks 6-7 (both 320->320)
        self.chunk_3 = _make_blocks([(c2, c2, 1), (c2, c2, 1)], dropout)
        self.exit_head_3 = _MixedPoolHead(c2, num_classes)
        self.gate_head_3 = _GateHead(c2)
        self.confidence_head_3 = _ConfidenceHead(c2)

        # chunk_4: blocks 8-9 (320->320 + 320->640 stride 2 at block 9)
        self.chunk_4 = _make_blocks([(c2, c2, 1), (c2, c3, 2)], dropout)
        self.exit_head_4 = _MixedPoolHead(c3, num_classes)
        self.gate_head_4 = _GateHead(c3)
        self.confidence_head_4 = _ConfidenceHead(c3)

        # chunk_5: blocks 10-11 (both 640->640)
        self.chunk_5 = _make_blocks([(c3, c3, 1), (c3, c3, 1)], dropout)
        self.exit_head_5 = _MixedPoolHead(c3, num_classes)
        self.gate_head_5 = _GateHead(c3)
        self.confidence_head_5 = _ConfidenceHead(c3)

        # chunk_6: block 12 + final BN-ReLU + avg pool + linear FC. No gate on the final
        # exit (residual probability). SCAR DOES use a confidence head at every exit
        # including the final one.
        self.chunk_6 = _make_blocks([(c3, c3, 1)], dropout)
        self.final_bn = nn.BatchNorm2d(c3)
        self.final_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, num_classes)
        self.confidence_head_final = _ConfidenceHead(c3)

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
        y = F.relu(self.final_bn(x), inplace=True)
        y = self.final_avg_pool(y).flatten(1)
        return self.fc(y)

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
            return x, self.exit_head_3(x)
        if exit_layer_idx == 4:
            x = self.chunk_4(x)
            return x, self.exit_head_4(x)
        if exit_layer_idx == 5:
            x = self.chunk_5(x)
            return x, self.exit_head_5(x)
        if exit_layer_idx == 6:
            x = self.chunk_6(x)
            return x, self._final_classify(x)
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
        )

    def forward_with_gates(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Single-pass forward returning per-exit classifier logits AND per-exit gate logits.

        Used by the JEI-DNN training path. Returns 7 classifier logits and 6 gate logits
        (one per non-final exit; the final exit has no gate). Gate logits have shape [B];
        classifier logits have shape [B, num_classes].
        """
        per_exit_logits: List[torch.Tensor] = []
        per_exit_gate_logits: List[torch.Tensor] = []
        x = self.stem(x)
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
        per_exit_logits.append(self.exit_head_3(x))
        per_exit_gate_logits.append(self.gate_head_3(x))
        x = self.chunk_4(x)
        per_exit_logits.append(self.exit_head_4(x))
        per_exit_gate_logits.append(self.gate_head_4(x))
        x = self.chunk_5(x)
        per_exit_logits.append(self.exit_head_5(x))
        per_exit_gate_logits.append(self.gate_head_5(x))
        x = self.chunk_6(x)
        per_exit_logits.append(self._final_classify(x))
        return per_exit_logits, per_exit_gate_logits

    def forward_with_confidences(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Single-pass forward returning per-exit classifier logits AND per-exit confidence
        logits s_j (one per exit, including the final).

        Used by the SCAR training path (v5). Returns 7 classifier logits and 7 confidence
        logits. Confidence logits have shape [B]; sigmoid(logit) is the SCAR selection
        score s_j(x) ∈ [0, 1]. Classifier logits have shape [B, num_classes]. The final
        confidence head reads the pre-BN chunk_6 feature map for symmetry with the early
        exits (the classifier itself reads BN+ReLU'd features via _final_classify).
        """
        per_exit_logits: List[torch.Tensor] = []
        per_exit_confidence_logits: List[torch.Tensor] = []
        x = self.stem(x)
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
        per_exit_logits.append(self.exit_head_3(x))
        per_exit_confidence_logits.append(self.confidence_head_3(x))
        x = self.chunk_4(x)
        per_exit_logits.append(self.exit_head_4(x))
        per_exit_confidence_logits.append(self.confidence_head_4(x))
        x = self.chunk_5(x)
        per_exit_logits.append(self.exit_head_5(x))
        per_exit_confidence_logits.append(self.confidence_head_5(x))
        x = self.chunk_6(x)
        per_exit_logits.append(self._final_classify(x))
        per_exit_confidence_logits.append(self.confidence_head_final(x))
        return per_exit_logits, per_exit_confidence_logits


def wideresnet28_10_sdn_exit(num_classes: int = 100, in_channels: int = 3) -> WideResNet2810SDNExit:
    return WideResNet2810SDNExit(num_classes=num_classes, in_channels=in_channels)
