"""Multi-exit ResNet-56 for CIFAR.

CIFAR-style ResNet-56 (He et al. 2015, Sec. 4.2 ImageNet variant adapted to CIFAR): a 3x3
stride-1 stem, then three stages of 9 BasicBlocks each at channel widths 16 / 32 / 64
(stages 2 and 3 begin with a stride-2 downsampling block), followed by adaptive average
pooling and a linear classifier. The depth count `56 = 1 (stem) + 2 * 27 (blocks) + 1 (fc)`.

Multi-exit layout matches the canonical SDN / ZTW arrangement at depth ratios 0.33 / 0.66 / 1.0:
- Exit 0 at end of stage 1 (16 channels, depth 9 of 27 blocks)
- Exit 1 at end of stage 2 (32 channels, depth 18 of 27 blocks)
- Exit 2 at end of stage 3 (64 channels, depth 27 of 27 blocks; this is the FINAL exit)

`num_exits = 2` (the count of EARLY exits) per the `ExitModel` contract; total classifiers
are 3 (early exits 0, 1 plus final exit 2).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import ExitModel


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.residual_function(x) + self.shortcut(x))


class ResNet56Exit(ExitModel):
    """ResNet-56 with two internal classifiers and a final classifier.

    The intermediate-activation threading follows the ``ExitModel`` contract: each call
    advances the trunk one stage and returns ``(features_for_next_exit, logits_at_this_exit)``.
    """

    num_exits = 2  # early exits at end of stages 1 and 2; final exit at end of stage 3.

    def __init__(self, num_classes: int = 100, in_channels: int = 3, blocks_per_stage: int = 9):
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(16, 16, blocks_per_stage, stride=1)
        self.exit1_head = nn.Linear(16, num_classes)

        self.stage2 = self._make_stage(16, 32, blocks_per_stage, stride=2)
        self.exit2_head = nn.Linear(32, num_classes)

        self.stage3 = self._make_stage(32, 64, blocks_per_stage, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        c_in = in_channels
        for s in strides:
            blocks.append(BasicBlock(c_in, out_channels, s))
            c_in = out_channels
        return nn.Sequential(*blocks)

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

    def _classify(self, features: torch.Tensor, head: nn.Module) -> torch.Tensor:
        y = self.avg_pool(features)
        y = y.view(y.size(0), -1)
        return head(y)

    def forward(
        self, x: torch.Tensor, exit_layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exit_layer_idx is None:
            exit_layer_idx = 0
        if exit_layer_idx == 0:
            x = self.stem(x)
            x = self.stage1(x)
            return x, self._classify(x, self.exit1_head)
        if exit_layer_idx == 1:
            x = self.stage2(x)
            return x, self._classify(x, self.exit2_head)
        if exit_layer_idx == 2:
            x = self.stage3(x)
            return x, self._classify(x, self.fc)
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
        )


def resnet56_exit(num_classes: int = 100, in_channels: int = 3) -> ResNet56Exit:
    return ResNet56Exit(num_classes=num_classes, in_channels=in_channels)


def resnet110_exit(num_classes: int = 100, in_channels: int = 3) -> ResNet56Exit:
    """Multi-exit ResNet-110 (depth=110 = 6n+2 with n=18 blocks/stage). ~1.7M params.

    Shares the ResNet56Exit class because the CIFAR-style architecture differs only
    in `blocks_per_stage` (9 -> 18). Three internal classifiers at depth ratios
    0.33 / 0.66 / 1.0 (the canonical SDN / ZTW placement).
    """
    return ResNet56Exit(num_classes=num_classes, in_channels=in_channels, blocks_per_stage=18)
