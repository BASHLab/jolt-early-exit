"""BC-ResNet-8 (Broadcasted Residual Network) with three early exits + final.

Source: Kim, Chang, Lee, Sung & Han, "Broadcasted Residual Learning for Efficient Keyword
Spotting" (INTERSPEECH 2021). BC-ResNet decomposes 2D conv on (mel, time) into a 2D conv
over the frequency dimension plus a 1D depthwise conv over the time dimension whose output
is broadcast over the frequency axis. This separation gives strong KWS accuracy at much
smaller parameter counts than vanilla 2D ResNets.

BC-ResNet-8 (the version we use here) has 6 BC-ResBlocks in 3 stages (1 + 2 + 3 per stage)
with channel widths 8 -> 16 -> 24 -> 32 (depth multiplier 8). Each stage starts with a
"transition" block (3x3 conv on the stem, then BC blocks). Exits are placed AFTER each of
the three stages (so 3 early exits) plus the final classifier = 4 classifiers total.

Input: mel-spectrogram (1, 40, 101) at 16 kHz with 10 ms hop. The width axis (40 mel-bands)
gets pooled by the global avg pool at the final classifier; intermediate exits use
mixed-pool 2D heads (AdaptiveAvgPool2d + AdaptiveMaxPool2d to (1, 1)) matching the
convention in our other backbones.

~325K params at 35-class GSC v2 output, well under the 5M on-device cap.

SCAR confidence heads and JEI-DNN gate heads follow the per-exit shape.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel


class _SubSpectralNorm(nn.Module):
    """Sub-spectral normalisation (Chang et al. INTERSPEECH 2021).

    Splits the frequency axis into ``num_sub`` groups and applies a separate BatchNorm per
    group. Helps when local frequency statistics differ (the original BC-ResNet motivation
    for going beyond plain BN over the full mel axis). Falls back to plain BN if the mel
    dimension isn't divisible by ``num_sub``.
    """

    def __init__(self, channels: int, num_sub: int = 5):
        super().__init__()
        self.num_sub = num_sub
        self.bn = nn.BatchNorm2d(channels * num_sub)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        if f % self.num_sub != 0:
            return F.batch_norm(
                x, self.bn.running_mean[: c], self.bn.running_var[: c],
                self.bn.weight[: c], self.bn.bias[: c],
                self.bn.training, self.bn.momentum, self.bn.eps,
            )
        x = x.view(b, c * self.num_sub, f // self.num_sub, t)
        x = self.bn(x)
        return x.view(b, c, f, t)


class _BCResBlock(nn.Module):
    """A BC-ResNet building block.

    f2: 3x3 depthwise conv on the frequency axis
    SubSpectralNorm: per-group BN over the mel axis
    f1: 1x1 conv (pointwise mix)
    + AvgPool over frequency: collapse to (B, C, 1, T) for the time path
    f3: 1x9 depthwise temporal conv on the time axis (after avg-pool over freq)
    BN
    f4: 1x1 conv
    SwishActivation
    Output: time_path broadcast over the frequency axis + the (original f2/f1) freq path
    Residual: input added back if shapes match.

    ``transition`` blocks (used at the start of each stage when the channel count changes
    or stride downsamples) skip the residual since input/output shapes don't match.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride_freq: int = 1,
        stride_time: int = 1,
        dilation: int = 1,
        is_transition: bool = False,
        num_sub: int = 5,
    ):
        super().__init__()
        self.is_transition = is_transition
        self.same_shape = (in_channels == out_channels and stride_freq == 1 and stride_time == 1)

        # Frequency-domain path
        self.f2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 1), stride=(stride_freq, 1),
            padding=(1, 0), groups=in_channels, bias=False,
        )
        self.ssn = _SubSpectralNorm(in_channels, num_sub=num_sub)
        self.f1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Time-domain path (after pooling over the frequency axis)
        self.f3 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(1, 9), stride=(1, stride_time),
            padding=(0, 4 * dilation), dilation=(1, dilation),
            groups=out_channels, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.f4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frequency path
        identity = x
        z = self.f2(x)
        z = self.ssn(z)
        z = self.f1(z)
        freq_path = self.bn1(z)
        # Time path on the freq-averaged feature, then broadcast back
        time_in = freq_path.mean(dim=2, keepdim=True)  # (B, C, 1, T)
        t = self.f3(time_in)
        t = F.silu(self.bn2(t), inplace=True)
        t = self.f4(t)
        t = self.bn3(t)
        # Broadcast time path back along the frequency axis
        out = freq_path + t
        out = F.silu(out, inplace=True)
        if self.same_shape and not self.is_transition:
            out = out + identity
        return out


def _make_stage(in_channels: int, out_channels: int, num_blocks: int, *, stride_freq: int, num_sub: int) -> nn.Sequential:
    blocks = []
    blocks.append(
        _BCResBlock(in_channels, out_channels, stride_freq=stride_freq, is_transition=True, num_sub=num_sub)
    )
    for _ in range(num_blocks - 1):
        blocks.append(_BCResBlock(out_channels, out_channels, num_sub=num_sub))
    return nn.Sequential(*blocks)


class _MixedPoolHead2d(nn.Module):
    """Mixed-pool 2D internal classifier: AdaptiveAvgPool2d || AdaptiveMaxPool2d -> Linear(2C, num_classes)."""

    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_pool(x).flatten(1)
        peak = self.max_pool(x).flatten(1)
        return self.fc(torch.cat([avg, peak], dim=1))


class _GateHead2d(nn.Module):
    """JEI-DNN routing gate (2D variant)."""

    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_pool(x).flatten(1)
        peak = self.max_pool(x).flatten(1)
        return self.fc(torch.cat([avg, peak], dim=1)).squeeze(-1)


class _ConfidenceHead2d(nn.Module):
    """SCAR confidence score (2D variant)."""

    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2 * channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_pool(x).flatten(1)
        peak = self.max_pool(x).flatten(1)
        return self.fc(torch.cat([avg, peak], dim=1)).squeeze(-1)


class BCResNet8Exit(ExitModel):
    """BC-ResNet-8 with three internal mixed-pool classifiers + final classifier."""

    num_exits = 3

    def __init__(
        self,
        num_classes: int = 35,
        in_channels: int = 1,
        base_channels: int = 32,
        num_sub: int = 5,
    ):
        super().__init__()
        self.num_classes = num_classes

        c1 = base_channels * 1   # 8
        c2 = base_channels * 2   # 16
        c3 = base_channels * 3   # 24
        c4 = base_channels * 4   # 32

        # Stem: a 5x5 conv with stride 2 over both axes (per the BC-ResNet paper).
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # Stage 1: 1 transition block (no downsample on freq -- transition handles channel change)
        # The stem already downsamples spatially. We use stride_freq=2 in the stage-2 transition
        # to keep the spatial reductions sensible.
        self.chunk_0 = _make_stage(c1, c2, num_blocks=2, stride_freq=1, num_sub=num_sub)
        self.exit_head_0 = _MixedPoolHead2d(c2, num_classes)
        self.gate_head_0 = _GateHead2d(c2)
        self.confidence_head_0 = _ConfidenceHead2d(c2)

        # Stage 2
        self.chunk_1 = _make_stage(c2, c3, num_blocks=2, stride_freq=2, num_sub=num_sub)
        self.exit_head_1 = _MixedPoolHead2d(c3, num_classes)
        self.gate_head_1 = _GateHead2d(c3)
        self.confidence_head_1 = _ConfidenceHead2d(c3)

        # Stage 3
        self.chunk_2 = _make_stage(c3, c4, num_blocks=2, stride_freq=2, num_sub=num_sub)
        self.exit_head_2 = _MixedPoolHead2d(c4, num_classes)
        self.gate_head_2 = _GateHead2d(c4)
        self.confidence_head_2 = _ConfidenceHead2d(c4)

        # Final classifier
        self.final_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c4, num_classes)
        self.confidence_head_final = _ConfidenceHead2d(c4)

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
        return self.fc(self.final_avg_pool(x).flatten(1))

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
            return x, self._final_classify(x)
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
        )

    def forward_with_gates(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        per_exit_logits: List[torch.Tensor] = []
        per_exit_gate_logits: List[torch.Tensor] = []
        x = self.stem(x)
        x = self.chunk_0(x); per_exit_logits.append(self.exit_head_0(x)); per_exit_gate_logits.append(self.gate_head_0(x))
        x = self.chunk_1(x); per_exit_logits.append(self.exit_head_1(x)); per_exit_gate_logits.append(self.gate_head_1(x))
        x = self.chunk_2(x); per_exit_logits.append(self.exit_head_2(x)); per_exit_gate_logits.append(self.gate_head_2(x))
        per_exit_logits.append(self._final_classify(x))
        return per_exit_logits, per_exit_gate_logits

    def forward_with_confidences(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        per_exit_logits: List[torch.Tensor] = []
        per_exit_confidence_logits: List[torch.Tensor] = []
        x = self.stem(x)
        x = self.chunk_0(x); per_exit_logits.append(self.exit_head_0(x)); per_exit_confidence_logits.append(self.confidence_head_0(x))
        x = self.chunk_1(x); per_exit_logits.append(self.exit_head_1(x)); per_exit_confidence_logits.append(self.confidence_head_1(x))
        x = self.chunk_2(x); per_exit_logits.append(self.exit_head_2(x)); per_exit_confidence_logits.append(self.confidence_head_2(x))
        per_exit_logits.append(self._final_classify(x)); per_exit_confidence_logits.append(self.confidence_head_final(x))
        return per_exit_logits, per_exit_confidence_logits


def bcresnet8_exit(num_classes: int = 35, in_channels: int = 1) -> BCResNet8Exit:
    return BCResNet8Exit(num_classes=num_classes, in_channels=in_channels)
