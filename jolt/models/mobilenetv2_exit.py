"""Multi-exit MobileNetV2.

Ported from the canonical copy (identical across standalone CIFAR-100 and the Bravery
monorepo). Reconciliation fix: the original hardcoded 100 output classes because
``MobileNetV2.__init__`` built each exit without forwarding ``class_num``; that forced a
separate forked file for CIFAR-10. Here ``num_classes`` and ``in_channels`` are threaded
through, so one file serves CIFAR-10/100 and Imagenette.

Reference: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks",
https://arxiv.org/abs/1801.04381
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        return residual


def _make_stage(repeat, in_channels, out_channels, stride, t):
    layers = [LinearBottleNeck(in_channels, out_channels, stride, t)]
    while repeat - 1:
        layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
        repeat -= 1
    return nn.Sequential(*layers)


class _Pre(nn.Module):
    def __init__(self, in_channels=3, stem_stride=1):
        super().__init__()
        if stem_stride == 1:
            # Original CIFAR stem (32x32 inputs); kept byte-for-byte so prior results are unaffected.
            stem = nn.Conv2d(in_channels, 32, 1, padding=1)
        else:
            # Native-resolution stem: a strided 3x3 conv downsamples large inputs (e.g. 160/192px
            # Imagenette) before the inverted-residual stages, keeping feature maps tractable.
            # RESEARCH GAP: matching torchvision MobileNetV2's full stride schedule for native res
            # is a further refinement; here a single strided stem plus adaptive pooling suffices.
            stem = nn.Conv2d(in_channels, 32, 3, stride=stem_stride, padding=1)
        self.pre = nn.Sequential(stem, nn.BatchNorm2d(32), nn.ReLU6(inplace=True))
        self.stage1 = LinearBottleNeck(32, 16, 1, 1)

    def forward(self, x):
        return self.stage1(self.pre(x))


class _ExitBlock(nn.Module):
    """One early/final exit: a feature stage, a widening conv, pool, and a 1x1 classifier.

    Carries small auxiliary heads to match the multi-method comparison convention used in
    our other backbones: a JEI-DNN gate head and a SCAR confidence head, each is an
    AvgPool + Linear(C, 1) on the post-widening feature (matching the channel count of the
    classifier input). out_features=1 so ``convert_last_layers_to_laplace`` skips them.
    """

    def __init__(self, stage: nn.Module, neck_in: int, neck_out: int, head_mult: int, num_classes: int):
        super().__init__()
        self.stage = stage
        self.stage7 = LinearBottleNeck(neck_in, neck_out, 1, 6)
        head_channels = neck_out * head_mult
        self.conv1 = nn.Sequential(
            nn.Conv2d(neck_out, head_channels, 1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU6(inplace=True),
        )
        self.conv2 = nn.Conv2d(head_channels, num_classes, 1)
        # JEI-DNN gate (Regol et al. ICLR 2024) and SCAR confidence head, both
        # AvgPool -> Linear(head_channels, 1). Defined on the post-widening feature so
        # they see the same representation as the classifier.
        self.gate_head = nn.Linear(head_channels, 1)
        self.confidence_head = nn.Linear(head_channels, 1)

    def _pooled_feat(self, y: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(y, 1).flatten(1)

    def forward(self, x):
        x = self.stage(x)
        y = self.stage7(x)
        y = self.conv1(y)
        y = F.adaptive_avg_pool2d(y, 1)
        y = self.conv2(y)
        return x, y.view(y.size(0), -1)

    def forward_with_gate(self, x):
        """Return (next_x, logits, gate_logit) for JEI-DNN training."""
        x = self.stage(x)
        y = self.stage7(x)
        y = self.conv1(y)
        pooled = self._pooled_feat(y)
        cls_y = self.conv2(F.adaptive_avg_pool2d(y, 1))
        return x, cls_y.view(cls_y.size(0), -1), self.gate_head(pooled).squeeze(-1)

    def forward_with_confidence(self, x):
        """Return (next_x, logits, confidence_logit) for SCAR training."""
        x = self.stage(x)
        y = self.stage7(x)
        y = self.conv1(y)
        pooled = self._pooled_feat(y)
        cls_y = self.conv2(F.adaptive_avg_pool2d(y, 1))
        return x, cls_y.view(cls_y.size(0), -1), self.confidence_head(pooled).squeeze(-1)


class MobileNetV2(ExitModel):
    def __init__(self, num_classes: int = 100, in_channels: int = 3, stem_stride: int = 1):
        super().__init__()
        self.num_exits = 4
        self.pre = _Pre(in_channels, stem_stride=stem_stride)
        # (stage, neck_in, neck_out, head_mult)
        # original widening conv is always neck_out * 4 (written as 48*4, 32*8, 64*8, ...)
        self.exit1 = _ExitBlock(_make_stage(2, 16, 24, 2, 6), 24, 48, 4, num_classes)
        self.exit2 = _ExitBlock(_make_stage(3, 24, 32, 2, 6), 32, 64, 4, num_classes)
        self.exit3 = _ExitBlock(_make_stage(4, 32, 64, 2, 6), 64, 128, 4, num_classes)
        self.exit4 = _ExitBlock(_make_stage(3, 64, 96, 1, 6), 96, 192, 4, num_classes)
        self.exit5 = _ExitBlock(_make_stage(3, 96, 160, 1, 6), 160, 320, 4, num_classes)
        self._exits = [self.exit1, self.exit2, self.exit3, self.exit4, self.exit5]

    def forward(
        self, x: torch.Tensor, exit_layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exit_layer_idx is None:
            exit_layer_idx = 0
        if not 0 <= exit_layer_idx <= self.num_exits:
            raise ValueError(
                f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
            )
        if exit_layer_idx == 0:
            x = self.pre(x)
        return self._exits[exit_layer_idx](x)

    def forward_with_gates(self, x):
        """Single-pass forward returning per-exit logits AND per-exit gate logits.

        Convention matches the other backbones: 5 classifier logits and 4 gate logits (the
        final exit has no gate; its routing probability is the residual 1 - sum_i pi_i).
        """
        per_exit_logits = []
        per_exit_gate_logits = []
        h = self.pre(x)
        for i, exit_block in enumerate(self._exits):
            if i < self.num_exits:  # early exit: emit gate
                h, logits, gate = exit_block.forward_with_gate(h)
                per_exit_logits.append(logits)
                per_exit_gate_logits.append(gate)
            else:  # final exit: no gate
                h, logits = exit_block(h)
                per_exit_logits.append(logits)
        return per_exit_logits, per_exit_gate_logits

    def forward_with_confidences(self, x):
        """Single-pass forward returning per-exit logits AND per-exit confidence logits.

        Used by the SCAR training path: 5 classifier logits and 5 confidence logits (all
        exits including the final get a confidence head, since the selection score s_j is
        defined along the full curve).
        """
        per_exit_logits = []
        per_exit_confidence_logits = []
        h = self.pre(x)
        for exit_block in self._exits:
            h, logits, conf = exit_block.forward_with_confidence(h)
            per_exit_logits.append(logits)
            per_exit_confidence_logits.append(conf)
        return per_exit_logits, per_exit_confidence_logits


def mobilenetv2_exit(num_classes: int = 100, in_channels: int = 3, stem_stride: int = 1) -> MobileNetV2:
    return MobileNetV2(num_classes=num_classes, in_channels=in_channels, stem_stride=stem_stride)
