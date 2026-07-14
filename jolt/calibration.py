"""Offline, validation-guided per-exit threshold calibration.

Lifted from ``bravery/common/exit_var_utils.py::estimate_thresholds_for_accuracy``. Given a
target accuracy, it picks the strictest per-exit entropy threshold whose cumulative accuracy
on the validation set meets the target. The model must expose
``forward(x, exit_layer_idx=i) -> (intermediate, logits)`` (see ``jolt.models.base``).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy


def _confidence_head_for(model: nn.Module, layer_idx: int) -> nn.Module:
    """Look up the SCAR confidence head for a given early-exit index on the model.

    SDN backbones register them as ``confidence_head_0`` ... ``confidence_head_{num_layers-1}``
    plus ``confidence_head_final``. ``layer_idx`` should be < num_layers (the final head is
    not used for routing because the final exit accepts all remaining samples).
    """
    name = f"confidence_head_{layer_idx}"
    head = getattr(model, name, None)
    if head is None:
        raise AttributeError(
            f"Model has no '{name}' module; cannot run learned_confidence cutoff. "
            "Confidence-aware methods require a backbone with per-exit confidence heads "
            "(e.g. resnet56_sdn_exit, wideresnet28_10_sdn_exit)."
        )
    return head


def estimate_thresholds_for_accuracy(
    target_accuracy: float,
    *,
    model: nn.Module,
    val_loader,
    num_layers: int,
    device: torch.device,
    cutoff_type: str = "entropy",
) -> List[float]:
    """Estimate per-exit thresholds that meet ``target_accuracy`` on ``val_loader``.

    Supports two routing scores: per-exit softmax entropy (default; lower = more confident
    = exit) and the SCAR learned confidence score s_j(x) = sigmoid(confidence_head_j(carry))
    (higher = more confident = exit). The threshold-selection logic is identical: sort
    samples by the routing score in the direction that puts confident samples first, take
    the loosest threshold whose exited population still meets ``target_accuracy``.

    ``num_layers`` is the number of early exits (excludes the final head). Returns one
    threshold per early exit; a placeholder of ``-10000.0`` (entropy) or ``10000.0``
    (learned_confidence) is used when an exit sees no samples.
    """
    if cutoff_type not in ("entropy", "learned_confidence", "poe_entropy", "poe_scar_entropy", "router_argmax"):
        raise ValueError(f"Unsupported cutoff_type '{cutoff_type}'.")
    if cutoff_type == "router_argmax":
        # router_argmax does NOT need per-exit thresholds (the router argmax over s_j is
        # deterministic); calibration is skipped. Return placeholder thresholds.
        return [0.0] * num_layers

    if cutoff_type in ("poe_entropy", "poe_scar_entropy"):
        poe_alphas = getattr(model, "poe_alphas", None)
        if poe_alphas is None:
            raise AttributeError(
                "Model has no 'poe_alphas' buffer; poe_entropy / poe_scar_entropy cutoffs "
                "require training a PoE-equipped candidate that registers per-exit alphas."
            )

    softmax = nn.Softmax(dim=1)
    log_softmax_fn = nn.LogSoftmax(dim=1)
    cutoff_data: List[list] = [[] for _ in range(num_layers)]

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            tensor_after_layer = images.to(device)
            labels = labels.to(device)
            running_sum = None  # cumulative log-PoE sum across exits (poe_entropy variants)
            for layer_idx in range(num_layers):
                tensor_after_layer, logits = model(tensor_after_layer, exit_layer_idx=layer_idx)
                if cutoff_type == "entropy":
                    probs = softmax(logits)
                    preds = torch.argmax(logits, dim=1)
                    scores = entropy(probs.cpu(), axis=1).astype(np.float32)
                elif cutoff_type == "learned_confidence":
                    preds = torch.argmax(logits, dim=1)
                    conf_head = _confidence_head_for(model, layer_idx)
                    s_logit = conf_head(tensor_after_layer)
                    scores = torch.sigmoid(s_logit).cpu().numpy().astype(np.float32)
                elif cutoff_type == "poe_entropy":
                    log_p = log_softmax_fn(logits)
                    alpha = poe_alphas[layer_idx]
                    running_sum = log_p * alpha if running_sum is None else running_sum + alpha * log_p
                    log_ptilde = log_softmax_fn(running_sum)
                    ptilde = log_ptilde.exp()
                    preds = torch.argmax(log_ptilde, dim=1)
                    scores = entropy(ptilde.cpu(), axis=1).astype(np.float32)
                else:  # poe_scar_entropy: combine cumulative PoE entropy with SCAR s_j
                    log_p = log_softmax_fn(logits)
                    alpha = poe_alphas[layer_idx]
                    running_sum = log_p * alpha if running_sum is None else running_sum + alpha * log_p
                    log_ptilde = log_softmax_fn(running_sum)
                    ptilde = log_ptilde.exp()
                    preds = torch.argmax(log_ptilde, dim=1)
                    poe_ent = entropy(ptilde.cpu(), axis=1).astype(np.float32)
                    # Normalise PoE entropy to [0, 1] (divide by ln(C) -- log of class count).
                    nclass = log_ptilde.size(-1)
                    poe_ent_norm = poe_ent / float(np.log(max(nclass, 2)))
                    # SCAR confidence s_j in [0, 1]; higher = more confident.
                    conf_head = _confidence_head_for(model, layer_idx)
                    s = torch.sigmoid(conf_head(tensor_after_layer)).cpu().numpy().astype(np.float32)
                    # Hybrid "uncertainty" score: weighted geometric of PoE-entropy + (1-s).
                    # Lower score = more confident = exit. alpha=0.5 by default.
                    hybrid_alpha = 0.5
                    scores = hybrid_alpha * poe_ent_norm + (1.0 - hybrid_alpha) * (1.0 - s)
                correct = (preds == labels).cpu().numpy().astype(np.float32)
                cutoff_data[layer_idx].extend(zip(scores.tolist(), correct.tolist()))

    thresholds: List[float] = []
    for layer_idx in range(num_layers):
        if not cutoff_data[layer_idx]:
            if cutoff_type in ("entropy", "poe_entropy", "poe_scar_entropy"):
                thresholds.append(-10000.0)
            else:
                thresholds.append(10000.0)
            continue
        data = np.array(cutoff_data[layer_idx], dtype=np.float32)
        # Sort so the most-confident samples (low entropy / low hybrid-uncertainty / high
        # learned_confidence) come first; threshold-selection logic is identical across types.
        if cutoff_type in ("entropy", "poe_entropy", "poe_scar_entropy"):
            order = np.argsort(data[:, 0])
        else:
            order = np.argsort(-data[:, 0])
        scores_sorted = data[order, 0]
        correct_sorted = data[order, 1]
        cumulative_accuracy = np.cumsum(correct_sorted) / (np.arange(len(correct_sorted)) + 1)
        valid = np.where(cumulative_accuracy >= target_accuracy)[0]
        if valid.size > 0:
            selected_idx = int(valid[-1])
        else:
            selected_idx = int(np.argmax(cumulative_accuracy))
        thresholds.append(float(scores_sorted[selected_idx]))
    return thresholds
