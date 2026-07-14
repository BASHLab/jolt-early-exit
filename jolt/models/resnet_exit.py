"""Multi-exit ResNet.

Ported from the canonical copy (identical across standalone UCI-HAR and the Bravery
monorepo). Used for UCI-HAR IMU data with 4 input channels; ``in_channels`` is now a
parameter rather than hardcoded. Attribute names are preserved so existing ResNet
checkpoints remain loadable.

Reference: He et al., "Deep Residual Learning for Image Recognition",
https://arxiv.org/abs/1512.03385
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import ExitModel


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(ExitModel):
    def __init__(self, block, num_block, num_classes: int = 6, in_channels: int = 4):
        super().__init__()
        self.num_exits = 3
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # different input size than the original paper, so conv2_x stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.fc2 = nn.Linear(64 * block.expansion, num_classes)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.fc3 = nn.Linear(128 * block.expansion, num_classes)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.fc4 = nn.Linear(256 * block.expansion, num_classes)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _head(self, fc: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.view(y.size(0), -1)
        return fc(y)

    def forward(
        self, x: torch.Tensor, exit_layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exit_layer_idx is None:
            exit_layer_idx = 0
        if exit_layer_idx == 0:
            x = self.conv1(x)
            x = self.conv2_x(x)
            return x, self._head(self.fc2, x)
        if exit_layer_idx == 1:
            x = self.conv3_x(x)
            return x, self._head(self.fc3, x)
        if exit_layer_idx == 2:
            x = self.conv4_x(x)
            return x, self._head(self.fc4, x)
        if exit_layer_idx == 3:
            x = self.conv5_x(x)
            return x, self._head(self.fc, x)
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
        )


def resnet18_exit(num_classes: int = 6, in_channels: int = 4) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)


def resnet34_exit(num_classes: int = 6, in_channels: int = 4) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)


def resnet50_exit(num_classes: int = 6, in_channels: int = 4) -> ResNet:
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)
