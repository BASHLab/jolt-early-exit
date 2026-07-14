"""Multi-exit MobileViT-XXS for CIFAR-100 / small image classification.

Source: Mehta & Rastegari, "MobileViT: Light-weight, General-purpose, and
Mobile-friendly Vision Transformer", ICLR 2022. Wrapped from timm
``mobilevit_xxs``. Hybrid CNN + transformer architecture: MV2-style stem
+ depthwise-separable blocks + MobileViT-attention blocks at deeper stages.

At img_size=32 the model has ~1M params (timm default). Stages 0-4 form
the trunk: 0+1 are early conv stem (lightweight), 2-4 are progressively
deeper MobileViT-attention blocks. Stage outputs at 32x32: chunk_0 captures
the conv stem features, chunk_1/2 the early attention layers, chunk_3 the
final attention layer + classifier head.
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


def _channels_after(module: nn.Module, x: torch.Tensor) -> int:
    """Helper: forward a sample tensor and return output channel count."""
    with torch.no_grad():
        return module(x).shape[1]


class MobileViTXXSExit(ExitModel):
    """MobileViT-XXS with three internal mixed-pool exits + final classifier."""

    num_exits = 3

    def __init__(
        self,
        num_classes: int = 100,
        in_channels: int = 3,
        img_size: int = 32,
    ):
        import timm
        super().__init__()
        self.num_classes = num_classes
        # Build the base MobileViT-XXS from timm with the right input size.
        base = timm.create_model(
            "mobilevit_xxs", num_classes=num_classes, img_size=img_size, in_chans=in_channels,
        )
        stages = list(base.stages)
        # Split: chunk_0 = stem + stages[0,1] (early); chunk_1 = stages[2]; chunk_2 = stages[3];
        # chunk_3 = stages[4] + final_conv + head
        self.stem = base.stem
        self.chunk_0_stages = nn.Sequential(*stages[0:2])
        self.chunk_1 = stages[2]
        self.chunk_2 = stages[3]
        self.chunk_3_stage = stages[4]
        self.final_conv = base.final_conv
        self.final_head = base.head

        # Probe channels at each exit point with a sample tensor on CPU
        with torch.no_grad():
            x = torch.zeros(1, in_channels, img_size, img_size)
            x = self.stem(x)
            x = self.chunk_0_stages(x); c0 = x.shape[1]
            x = self.chunk_1(x); c1 = x.shape[1]
            x = self.chunk_2(x); c2 = x.shape[1]

        self.exit_head_0 = _MixedPoolHead2d(c0, num_classes)
        self.gate_head_0 = _GateHead2d(c0)
        self.confidence_head_0 = _ConfidenceHead2d(c0)
        self.exit_head_1 = _MixedPoolHead2d(c1, num_classes)
        self.gate_head_1 = _GateHead2d(c1)
        self.confidence_head_1 = _ConfidenceHead2d(c1)
        self.exit_head_2 = _MixedPoolHead2d(c2, num_classes)
        self.gate_head_2 = _GateHead2d(c2)
        self.confidence_head_2 = _ConfidenceHead2d(c2)

        with torch.no_grad():
            x = torch.zeros(1, c2, img_size // 8, img_size // 8)  # rough; just for channel probe
            try:
                x = self.chunk_3_stage(x); x = self.final_conv(x); c_final = x.shape[1]
            except Exception:
                # Fallback: just use a plausible final channel count
                c_final = 320  # MobileViT-XXS canonical last_channel
        self.confidence_head_final = _ConfidenceHead2d(c_final)

    def chunk_0(self, x: torch.Tensor) -> torch.Tensor:
        return self.chunk_0_stages(self.stem(x))

    def chunk_3(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_conv(self.chunk_3_stage(x))

    def _final_classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_head(x)

    def forward(self, x: torch.Tensor, exit_layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if exit_layer_idx is None:
            exit_layer_idx = 0
        if exit_layer_idx == 0:
            h = self.chunk_0(x); return h, self.exit_head_0(h)
        if exit_layer_idx == 1:
            h = self.chunk_1(x); return h, self.exit_head_1(h)
        if exit_layer_idx == 2:
            h = self.chunk_2(x); return h, self.exit_head_2(h)
        if exit_layer_idx == 3:
            h = self.chunk_3(x); return h, self._final_classify(h)
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
        )

    def forward_with_gates(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        h = self.chunk_0(x); logits.append(self.exit_head_0(h)); gates.append(self.gate_head_0(h))
        h = self.chunk_1(h); logits.append(self.exit_head_1(h)); gates.append(self.gate_head_1(h))
        h = self.chunk_2(h); logits.append(self.exit_head_2(h)); gates.append(self.gate_head_2(h))
        h = self.chunk_3(h); logits.append(self._final_classify(h))
        return logits, gates

    def forward_with_confidences(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        confs: List[torch.Tensor] = []
        h = self.chunk_0(x); logits.append(self.exit_head_0(h)); confs.append(self.confidence_head_0(h))
        h = self.chunk_1(h); logits.append(self.exit_head_1(h)); confs.append(self.confidence_head_1(h))
        h = self.chunk_2(h); logits.append(self.exit_head_2(h)); confs.append(self.confidence_head_2(h))
        h = self.chunk_3(h); logits.append(self._final_classify(h)); confs.append(self.confidence_head_final(h))
        return logits, confs


def mobilevit_xxs_exit(num_classes: int = 100, in_channels: int = 3, img_size: int = 32) -> MobileViTXXSExit:
    return MobileViTXXSExit(num_classes=num_classes, in_channels=in_channels, img_size=img_size)
