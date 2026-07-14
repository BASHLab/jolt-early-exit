"""JOLT training losses: variance-based scaling and adaptive uncertainty weighting.

Unified from the Bravery monorepo (Nov 2025), which standardized every dataset onto the
softmax-variance running-average formulation (the older standalone CIFAR-100/UCI-HAR copies
used a divergent logit-variance masking variant; that has been superseded).

Two independent switches control the method:

* ``use_b`` (variance-based scaling): divide each exit's cross-entropy by ``b``, the running
  mean softmax variance across exits seen so far. ``b`` is detached, so it scales the loss
  without contributing gradient.
* :class:`MultiTaskLoss` (adaptive uncertainty weighting): learnable per-exit weights.

The "variance-scaling-only" configuration (``use_b=True`` with MultiTaskLoss disabled) is the
one that collapses to chance accuracy on CIFAR-100; it is kept reproducible on purpose.

The loss runs the forward pass itself because it needs every exit's logits; it relies on the
:class:`jolt.models.base.ExitModel` threading contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Adaptive uncertainty weighting over per-exit losses (Kendall et al., 2018)."""

    def __init__(self, eta: List[float]):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor(list(eta), dtype=torch.float32))

    def forward(self, loss_list):
        loss_sum = 0.0
        for i, loss in enumerate(loss_list):
            loss_sum = loss_sum + 0.5 / (self.eta[i] ** 2) * loss + torch.log(1 + self.eta[i] ** 2)
        return self.eta, loss_sum


class PoEStateModule(nn.Module):
    """Learnable per-exit alpha for the v5 PoE-Anneal candidate.

    Holds J scalar parameters initialised to 1.0. The PoE prediction at exit j is
    ptilde_j(x) ∝ prod_{l≤j} p_l(x) ** alpha_l, computed in log space and renormalised
    by log_softmax. The alphas are trained jointly with the model via standard SGD on
    the PoE-Anneal loss; ``forward()`` returns the alphas tensor for use in the loss.
    """

    def __init__(self, num_exits: int):
        super().__init__()
        if num_exits < 1:
            raise ValueError(f"PoEStateModule needs num_exits >= 1; got {num_exits}.")
        self.alphas = nn.Parameter(torch.ones(num_exits, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.alphas


@dataclass
class CascadingLossOutput:
    convex_loss: torch.Tensor        # mean scaled loss across exits (used when MultiTaskLoss is off)
    per_exit_losses: List[torch.Tensor]  # scaled per-exit losses (feed to MultiTaskLoss)
    per_exit_logits: List[torch.Tensor]
    b: torch.Tensor                  # running-average softmax variance ("true_b")


def forward_all_exits(model: nn.Module, images: torch.Tensor) -> List[torch.Tensor]:
    """Run every exit, threading the intermediate activation, and return per-exit logits.

    This is the single forward used by every training method (JOLT and the baselines), so
    each method differs only in how it combines the per-exit logits.
    """
    carry = images
    per_exit_logits: List[torch.Tensor] = []
    for layer_idx in range(model.num_exits + 1):
        carry, logits = model(carry, exit_layer_idx=layer_idx)
        per_exit_logits.append(logits)
    return per_exit_logits


def variance_scaled_losses(
    per_exit_logits: List[torch.Tensor],
    labels: torch.Tensor,
    criterion: nn.Module,
    *,
    use_b: bool = True,
):
    """Per-exit losses scaled by the running mean softmax variance ``b`` (when ``use_b``).

    Returns ``(per_exit_losses, b)``. ``b`` is detached, so it scales the loss without
    contributing gradient.
    """
    total_b = per_exit_logits[0].new_zeros(())
    true_b = per_exit_logits[0].new_ones(())
    softmax = nn.Softmax(dim=1)
    scaled: List[torch.Tensor] = []
    for idx, logits in enumerate(per_exit_logits):
        loss = criterion(logits, labels)
        b_cascading = torch.mean(torch.var(softmax(logits.detach()), dim=1), dim=0)
        total_b = total_b + b_cascading
        true_b = total_b / (idx + 1)
        b_mean = true_b if use_b else per_exit_logits[0].new_ones(())
        scaled.append(loss / b_mean)
    return scaled, true_b


def cascading_convex_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    num_exits: int,
    *,
    use_b: bool = True,
    divide_b: float = 10.0,
) -> CascadingLossOutput:
    """JOLT's variance-scaled multi-exit loss: one forward, then variance-scaled per-exit CE.

    The convex loss is the mean of the scaled per-exit losses (feed ``per_exit_losses`` to
    :class:`MultiTaskLoss` for the full JOLT objective). ``num_exits`` is kept for API
    compatibility; the exit count is taken from the model. ``divide_b`` is retained for parity
    with the source (it scales the monitoring uncertainty, not this loss).
    """
    per_exit_logits = forward_all_exits(model, images)
    per_exit_losses, true_b = variance_scaled_losses(per_exit_logits, labels, criterion, use_b=use_b)
    convex_loss = torch.stack(per_exit_losses).mean()
    return CascadingLossOutput(convex_loss, per_exit_losses, per_exit_logits, true_b)


def exit_uncertainty(
    per_exit_logits: List[torch.Tensor],
    num_classes: int,
    *,
    divide_b: float = 10.0,
) -> torch.Tensor:
    """Monitoring-only uncertainty: per-exit prediction disagreement minus scaled variance.

    Ported from ``calculate_uncertainty``. Not part of the training gradient; used for
    logging and the collapse diagnostics.
    """
    num_layers = len(per_exit_logits)
    if num_layers == 0:
        return per_exit_logits[0].new_zeros(())
    batch_size = per_exit_logits[0].size(0)
    device = per_exit_logits[0].device
    softmax = nn.Softmax(dim=1)

    y_counts = torch.zeros((batch_size, num_classes), device=device)
    b_mean_in_layer = torch.zeros((num_layers, batch_size), device=device)
    for choice, outputs in enumerate(per_exit_logits):
        preds = torch.argmax(outputs.detach(), dim=1)
        y_counts += F.one_hot(preds.to(torch.int64), num_classes=num_classes).float()
        b_mean_in_layer[choice] = torch.var(softmax(outputs.detach()), dim=1)

    b_mean_all_layers = num_classes * torch.mean(b_mean_in_layer, dim=0) / divide_b
    y_pairwise = torch.sum(y_counts * (num_layers - y_counts), dim=1)
    if num_layers > 1:
        y_pairwise = y_pairwise / (num_layers * (num_layers - 1))
    return (y_pairwise - b_mean_all_layers).mean()
