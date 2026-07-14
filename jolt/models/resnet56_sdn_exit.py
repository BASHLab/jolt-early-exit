"""ResNet-56 with seven exits at SDN compute-fraction placements.

Canonical SDN construction (Kaya, Hong & Dumitras, "Shallow-Deep Networks", ICML 2019):
internal classifiers placed at 15 / 30 / 45 / 60 / 75 / 90 percent of the network's inference
cost, plus the final classifier (7 outputs total). Each internal classifier (IC) is a
mixed-pool (avg + max) head followed by a fully connected layer, per the SDN paper.

For ResNet-56 (1 stem + 27 BasicBlocks + 1 FC, three stages of 9 blocks at widths 16 / 32 /
64; the spatial-halving / channel-doubling pattern keeps each block at roughly equal MACs),
the per-cent-of-compute placements map cleanly to "after block k of 27" with k in {4, 8, 12,
16, 20, 24}, and the final exit sits at the end of block 27 plus the avg-pool + linear FC.

Block partition into 7 chunks:
    chunk_0: blocks  1- 4      (stage 1, 16 channels, 32x32)
    chunk_1: blocks  5- 8      (stage 1, 16 channels, 32x32)
    chunk_2: blocks  9-12      (stage 1 -> stage 2 boundary; ends 32 channels, 16x16)
    chunk_3: blocks 13-16      (stage 2, 32 channels, 16x16)
    chunk_4: blocks 17-20      (stage 2 -> stage 3 boundary; ends 64 channels, 8x8)
    chunk_5: blocks 21-24      (stage 3, 64 channels, 8x8)
    chunk_6: blocks 25-27      (stage 3, 64 channels, 8x8) + standard ResNet final classifier

``num_exits = 6`` (the count of EARLY exits) per the ExitModel contract; total classifiers
are 7. Parameter count at num_classes=100 is ~0.90 M, well within the on-device band.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .base import ExitModel
from .resnet56_exit import BasicBlock


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
    a sample to this exit. Sized to match the SDN exit head so the gate sees the same
    feature representation the classifier sees. Default Linear init (std=0.01, bias=0)
    puts initial sigmoid(logit) near 0.5, an uninformative prior the joint loss anneals.
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
    """SCAR confidence head: AvgPool || MaxPool -> Linear(2*C, 1).

    Per the v5 SCAR design, returns one logit per sample at each exit. sigmoid(logit) is
    used as the structure-aware selection score s_j at eval; at training the logit feeds
    into the rank surrogate L_rank and a sigmoid-transformed value feeds into the TCP
    regression anchor L_tcp. Sized to match the exit classifier head so the rank score
    sees the same feature representation. Architecturally identical to _GateHead but kept
    as a separate class for readability and to avoid confusing the JEI-DNN gates with the
    SCAR confidence scores at use sites.
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


def _make_blocks(specs: List[Tuple[int, int, int]]) -> nn.Sequential:
    """Build a Sequential of BasicBlocks from (in_channels, out_channels, stride) tuples."""
    return nn.Sequential(*[BasicBlock(c_in, c_out, s) for c_in, c_out, s in specs])


class ResNet56SDNExit(ExitModel):
    """ResNet-56 SDN with 6 internal classifiers (mixed-pool head) plus the final classifier.

    Forward pass threads features through the 7 chunks per the ``ExitModel`` contract.
    Exit indices 0..5 emit the mixed-pool internal logits at the chunk boundaries; index 6
    runs the final 3 blocks then standard ResNet (AvgPool + Linear) classification.
    """

    num_exits = 6

    def __init__(self, num_classes: int = 100, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Stage 1 = blocks 1-9 at 16ch / 32x32, all stride 1.
        # Stage 2 = blocks 10-18 at 32ch / 16x16, block 10 stride 2.
        # Stage 3 = blocks 19-27 at 64ch / 8x8, block 19 stride 2.
        # We split into 7 chunks at block boundaries 4 / 8 / 12 / 16 / 20 / 24 / 27.

        # chunk_0: blocks 1-4 (all 16->16 stride 1).
        self.chunk_0 = _make_blocks([(16, 16, 1)] * 4)
        self.exit_head_0 = _MixedPoolHead(16, num_classes)
        self.gate_head_0 = _GateHead(16)
        self.confidence_head_0 = _ConfidenceHead(16)

        # chunk_1: blocks 5-8 (all 16->16 stride 1).
        self.chunk_1 = _make_blocks([(16, 16, 1)] * 4)
        self.exit_head_1 = _MixedPoolHead(16, num_classes)
        self.gate_head_1 = _GateHead(16)
        self.confidence_head_1 = _ConfidenceHead(16)

        # chunk_2: block 9 (16->16) + blocks 10-12 (10 is 16->32 stride 2, 11/12 are 32->32).
        self.chunk_2 = _make_blocks([
            (16, 16, 1),
            (16, 32, 2),
            (32, 32, 1),
            (32, 32, 1),
        ])
        self.exit_head_2 = _MixedPoolHead(32, num_classes)
        self.gate_head_2 = _GateHead(32)
        self.confidence_head_2 = _ConfidenceHead(32)

        # chunk_3: blocks 13-16 (all 32->32).
        self.chunk_3 = _make_blocks([(32, 32, 1)] * 4)
        self.exit_head_3 = _MixedPoolHead(32, num_classes)
        self.gate_head_3 = _GateHead(32)
        self.confidence_head_3 = _ConfidenceHead(32)

        # chunk_4: blocks 17-18 (32->32) + 19 (32->64 stride 2) + 20 (64->64).
        self.chunk_4 = _make_blocks([
            (32, 32, 1),
            (32, 32, 1),
            (32, 64, 2),
            (64, 64, 1),
        ])
        self.exit_head_4 = _MixedPoolHead(64, num_classes)
        self.gate_head_4 = _GateHead(64)
        self.confidence_head_4 = _ConfidenceHead(64)

        # chunk_5: blocks 21-24 (all 64->64).
        self.chunk_5 = _make_blocks([(64, 64, 1)] * 4)
        self.exit_head_5 = _MixedPoolHead(64, num_classes)
        self.gate_head_5 = _GateHead(64)
        self.confidence_head_5 = _ConfidenceHead(64)

        # chunk_6: blocks 25-27 + canonical ResNet final classifier. No gate on the final exit
        # (its routing probability is the residual: 1 - sum_{i<6} pi_i). SCAR DOES use a
        # confidence head at every exit including the final one because s_j is the selection
        # score along the whole curve.
        self.chunk_6 = _make_blocks([(64, 64, 1)] * 3)
        self.final_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.confidence_head_final = _ConfidenceHead(64)

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
        y = self.final_avg_pool(x).flatten(1)
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

        Used by the JEI-DNN training path (Regol et al. ICLR 2024). Returns 7 classifier
        logits (one per exit) and 6 gate logits (one per non-final exit; the final exit has
        no gate, since its routing probability is the residual). Gate logits have shape [B]
        (one Bernoulli per sample); classifier logits have shape [B, num_classes].
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
        logits. Confidence logits have shape [B] -- sigmoid(logit) is the SCAR selection
        score s_j(x) ∈ [0, 1] used both for the rank surrogate at train and for exit-time
        routing at eval. Classifier logits have shape [B, num_classes].
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


def resnet56_sdn_exit(num_classes: int = 100, in_channels: int = 3) -> ResNet56SDNExit:
    return ResNet56SDNExit(num_classes=num_classes, in_channels=in_channels)
