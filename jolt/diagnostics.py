"""Collapse diagnostics (config-gated).

The variance-scaling-only configuration collapses to chance accuracy on CIFAR-100. To make
that failure mode observable, these helpers log, per exit and per epoch: the predicted
class-probability distribution (a collapsed exit predicts one class, so its max class
fraction approaches 1), the gradient norms of each exit block, and the first exit's entropy
distribution over epochs. Everything is gated by ``config.diagnostics.enabled`` so normal
runs pay nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy


def predicted_class_fractions(logits: torch.Tensor, num_classes: int) -> np.ndarray:
    preds = torch.argmax(logits.detach(), dim=1).cpu().numpy()
    counts = np.bincount(preds, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


def entropy_distribution(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
    return entropy(probs, axis=1)


def module_grad_norms(model: nn.Module) -> Dict[str, float]:
    """L2 gradient norm grouped by the model's immediate child modules (per-exit blocks)."""
    norms: Dict[str, float] = {}
    for name, child in model.named_children():
        total_sq = 0.0
        for param in child.parameters():
            if param.grad is not None:
                total_sq += float(param.grad.detach().norm(2).item()) ** 2
        norms[name] = total_sq ** 0.5
    return norms


@dataclass
class CollapseLogger:
    num_classes: int
    log_class_distribution: bool = True
    log_gradient_norms: bool = True
    log_exit1_entropy: bool = True
    records: List[dict] = field(default_factory=list)

    def record_epoch(self, epoch: int, model: nn.Module, per_exit_logits: List[torch.Tensor]) -> None:
        """Record one snapshot. Call after backward so gradients are populated."""
        rec: dict = {"epoch": epoch}
        if self.log_class_distribution:
            rec["max_class_fraction"] = {
                i: float(predicted_class_fractions(lg, self.num_classes).max())
                for i, lg in enumerate(per_exit_logits)
            }
        if self.log_gradient_norms:
            rec["grad_norms"] = module_grad_norms(model)
        if self.log_exit1_entropy and per_exit_logits:
            ent = entropy_distribution(per_exit_logits[0])
            rec["exit1_entropy_mean"] = float(np.mean(ent))
            rec["exit1_entropy_std"] = float(np.std(ent))
        self.records.append(rec)
