"""ResNet-56-SDN with variational (LiteLaplace) mixed-pool heads.

C4 LiteLaplace-EE per the v4 sweep return: each exit's mixed-pool head carries a diagonal
Gaussian posterior over its weights with a variance floor, sampled at train time and
expected-out at eval time via MacKay's probit approximation. The KL-to-unit-Gaussian
regulariser is computed by the model and summed across heads via
``litelaplace_kl_sum()``; the trainer adds beta * KL to the per-exit CE loss.

What's IN this first pass:
- Variational mixed-pool heads with per-weight diagonal sigma (variance floor sigma_min^2 = 1e-4).
- MacKay probit approximation at eval (Gibbs 1997; SNGP-style closed-form predictive).
- Closed-form KL to unit Gaussian (per-head, summed across heads).

Not included in this implementation:
- Spectral-norm constraint on penultimate conv features (SNGP distance-awareness). The
  variational head story is testable without the spectral norm; SNGP's marginal gain on top
  of the variational head is the "distance-awareness" effect, which we test only if the
  variational head alone shows traction on cell 1.

Architecture: identical to ``ResNet56SDNExit`` (7 exits at 15/30/45/60/75/90% + final, mixed
pool heads). Only the heads change.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel
from .resnet56_exit import BasicBlock


class _VariationalMixedPoolHead(nn.Module):
    """SDN-style internal classifier with variational (diagonal Gaussian) weight posterior.

    At training time, samples W and bias from q(W) = N(mu, diag(sigma^2)) and computes logits.
    At eval time, returns MacKay-probit-approximated predictive logits (mean logits scaled
    down by sqrt(1 + pi/8 * variance(logits))).
    """

    def __init__(self, channels: int, num_classes: int, sigma_min: float = 1e-2):
        super().__init__()
        self.in_features = 2 * channels
        self.out_features = num_classes
        self.sigma_min = sigma_min
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Mean parameters: standard Linear layer init.
        self.mu_W = nn.Parameter(torch.empty(num_classes, 2 * channels))
        self.mu_b = nn.Parameter(torch.zeros(num_classes))
        nn.init.normal_(self.mu_W, std=0.01)
        # Per-weight log-sigma. Init at log(sigma_min) so initial posterior is tight.
        init_log_sigma = math.log(sigma_min)
        self.log_sigma_W = nn.Parameter(torch.full((num_classes, 2 * channels), init_log_sigma))
        self.log_sigma_b = nn.Parameter(torch.full((num_classes,), init_log_sigma))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)

    def _sigma_W(self) -> torch.Tensor:
        return torch.clamp_min(self.log_sigma_W.exp(), self.sigma_min)

    def _sigma_b(self) -> torch.Tensor:
        return torch.clamp_min(self.log_sigma_b.exp(), self.sigma_min)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._features(x)
        if self.training:
            sigma_W = self._sigma_W()
            sigma_b = self._sigma_b()
            eps_W = torch.randn_like(sigma_W)
            eps_b = torch.randn_like(sigma_b)
            W = self.mu_W + eps_W * sigma_W
            b = self.mu_b + eps_b * sigma_b
            return F.linear(h, W, b)
        # MacKay probit approximation: divide mean logits by sqrt(1 + pi/8 * variance).
        mean_logits = F.linear(h, self.mu_W, self.mu_b)
        sigma_W = self._sigma_W()
        sigma_b = self._sigma_b()
        h_sq = h ** 2  # (B, 2C)
        var = h_sq @ (sigma_W ** 2).t() + (sigma_b ** 2)
        scale = (1.0 + (math.pi / 8.0) * var).clamp_min(1e-12).sqrt()
        return mean_logits / scale

    def kl_to_unit_normal(self) -> torch.Tensor:
        """KL(q(W) || N(0, I)) summed over all weight + bias dimensions."""
        sigma_W = self._sigma_W()
        sigma_b = self._sigma_b()
        kl_W = 0.5 * (self.mu_W ** 2 + sigma_W ** 2 - 1.0 - 2.0 * sigma_W.log()).sum()
        kl_b = 0.5 * (self.mu_b ** 2 + sigma_b ** 2 - 1.0 - 2.0 * sigma_b.log()).sum()
        return kl_W + kl_b


def _make_blocks(specs: List[Tuple[int, int, int]]) -> nn.Sequential:
    return nn.Sequential(*[BasicBlock(c_in, c_out, s) for c_in, c_out, s in specs])


class ResNet56SDNLiteLaplaceExit(ExitModel):
    """ResNet-56-SDN with variational (LiteLaplace) mixed-pool heads. Trunk identical to
    ``ResNet56SDNExit``; only the heads differ. The final classifier is also variational so
    that the KL-to-unit-Gaussian regularisation is consistent across all 7 exits."""

    num_exits = 6

    def __init__(self, num_classes: int = 100, in_channels: int = 3, sigma_min: float = 1e-2):
        super().__init__()
        self.num_classes = num_classes
        self.sigma_min = sigma_min

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        # Same chunk layout as the deterministic ResNet56SDNExit.
        self.chunk_0 = _make_blocks([(16, 16, 1)] * 4)
        self.chunk_1 = _make_blocks([(16, 16, 1)] * 4)
        self.chunk_2 = _make_blocks([(16, 16, 1), (16, 32, 2), (32, 32, 1), (32, 32, 1)])
        self.chunk_3 = _make_blocks([(32, 32, 1)] * 4)
        self.chunk_4 = _make_blocks([(32, 32, 1), (32, 32, 1), (32, 64, 2), (64, 64, 1)])
        self.chunk_5 = _make_blocks([(64, 64, 1)] * 4)
        self.chunk_6 = _make_blocks([(64, 64, 1)] * 3)

        self.exit_head_0 = _VariationalMixedPoolHead(16, num_classes, sigma_min)
        self.exit_head_1 = _VariationalMixedPoolHead(16, num_classes, sigma_min)
        self.exit_head_2 = _VariationalMixedPoolHead(32, num_classes, sigma_min)
        self.exit_head_3 = _VariationalMixedPoolHead(32, num_classes, sigma_min)
        self.exit_head_4 = _VariationalMixedPoolHead(64, num_classes, sigma_min)
        self.exit_head_5 = _VariationalMixedPoolHead(64, num_classes, sigma_min)
        self.exit_head_6 = _VariationalMixedPoolHead(64, num_classes, sigma_min)

        # Trunk + stem conv/BN init (mean parameters of heads are init in the head class).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self, x: torch.Tensor, exit_layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exit_layer_idx is None:
            exit_layer_idx = 0
        chunks = [self.chunk_0, self.chunk_1, self.chunk_2, self.chunk_3,
                  self.chunk_4, self.chunk_5, self.chunk_6]
        heads = [self.exit_head_0, self.exit_head_1, self.exit_head_2, self.exit_head_3,
                 self.exit_head_4, self.exit_head_5, self.exit_head_6]
        if exit_layer_idx < 0 or exit_layer_idx > self.num_exits:
            raise ValueError(
                f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
            )
        if exit_layer_idx == 0:
            x = self.stem(x)
        x = chunks[exit_layer_idx](x)
        return x, heads[exit_layer_idx](x)

    def litelaplace_kl_sum(self) -> torch.Tensor:
        """Sum of KL(q(W_e) || N(0,I)) across all 7 variational heads."""
        return sum(h.kl_to_unit_normal() for h in [
            self.exit_head_0, self.exit_head_1, self.exit_head_2, self.exit_head_3,
            self.exit_head_4, self.exit_head_5, self.exit_head_6,
        ])


def resnet56_sdn_litelaplace_exit(num_classes: int = 100, in_channels: int = 3) -> ResNet56SDNLiteLaplaceExit:
    return ResNet56SDNLiteLaplaceExit(num_classes=num_classes, in_channels=in_channels)
