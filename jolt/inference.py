"""Early-exit inference and evaluation.

Lifted from ``bravery/common/exit_var_utils.py::evaluate_dynamic_exit`` with three changes:
EMAR is computed via :func:`jolt.metrics.compute_emar` (configurable population exponent),
ECE/NLL are reported alongside it, and a split-inference seam serializes the forwarded
activation at a configurable exit index (see :mod:`jolt.deploy.split_inference`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy
from sklearn.metrics import classification_report

from .deploy.split_inference import Transport, handoff
from .metrics import (
    aurc,
    brier_score,
    compute_emar,
    expected_calibration_error,
    negative_log_likelihood,
)


@dataclass
class ExitEvaluation:
    accuracy: float
    emar: float
    mean_variance: float
    ece: float
    nll: float
    brier: float = 0.0
    aurc: float = 0.0
    exit_counts: List[int] = field(default_factory=list)
    per_exit_accuracy: List[float] = field(default_factory=list)
    per_exit_support: List[float] = field(default_factory=list)


def _normalize_thresholds(thresholds, num_layers: int) -> List[float]:
    if thresholds is None:
        return [-10000.0] * num_layers
    if not isinstance(thresholds, (list, tuple)):
        return [float(thresholds)] * num_layers
    if len(thresholds) != num_layers:
        raise ValueError(f"Expected {num_layers} thresholds, got {len(thresholds)}.")
    return [float(t) for t in thresholds]


def build_exit_evaluation(
    num_positions: int,
    pred_y: List[list],
    true_y: List[list],
    macs: Sequence[float],
    total_samples: int,
    probs_all: List[np.ndarray],
    labels_all: List[np.ndarray],
    *,
    mean_variance: float = 0.0,
) -> ExitEvaluation:
    """Aggregate per-exit predictions into accuracy/EMAR/ECE/NLL. Shared by the vision and
    text dynamic-exit loops (they collect the same per-exit prediction lists)."""
    overall_accuracy = 0.0
    per_exit_accuracy: List[float] = []
    per_exit_support: List[float] = []
    exit_counts: List[int] = []
    for pos in range(num_positions):
        if not true_y[pos]:
            per_exit_accuracy.append(0.0)
            per_exit_support.append(0.0)
            exit_counts.append(0)
            continue
        report = classification_report(true_y[pos], pred_y[pos], output_dict=True, zero_division=0)
        support = report["weighted avg"]["support"]
        accuracy = report.get("accuracy", 0.0)
        overall_accuracy += (support * accuracy) / total_samples
        per_exit_accuracy.append(float(accuracy))
        per_exit_support.append(float(support))
        exit_counts.append(int(support))

    probs_concat = np.concatenate(probs_all, axis=0)
    labels_concat = np.concatenate(labels_all, axis=0)
    # Dynamic-policy confidence/correctness: each sample is scored at the exit it left from.
    confidences = probs_concat.max(axis=1)
    correctness = (probs_concat.argmax(axis=1) == labels_concat).astype(np.float64)
    return ExitEvaluation(
        accuracy=float(overall_accuracy),
        emar=compute_emar(per_exit_accuracy, per_exit_support, macs, total_samples),
        mean_variance=float(mean_variance),
        ece=expected_calibration_error(probs_concat, labels_concat),
        nll=negative_log_likelihood(probs_concat, labels_concat),
        brier=brier_score(probs_concat, labels_concat),
        aurc=aurc(confidences, correctness),
        exit_counts=exit_counts,
        per_exit_accuracy=per_exit_accuracy,
        per_exit_support=per_exit_support,
    )


def _evaluate_router_argmax(
    model: nn.Module,
    test_loader,
    *,
    macs: Sequence[float],
    num_layers: int,
    device: torch.device,
) -> "ExitEvaluation":
    """Inference for the moe_router method: joint argmax routing over per-exit confidence heads.

    For each batch: forward through every exit (running the entire backbone), collect the
    per-exit confidence-head logits s_j(x), softmax across j per sample, and route each
    sample to the argmax exit. Used as the eval cutoff for ``moe_router``.

    Returns the same ExitEvaluation schema as evaluate_dynamic_exit so the downstream
    metric collection (build_exit_evaluation) treats it identically.
    """
    softmax = nn.Softmax(dim=1)
    pred_y: List[list] = [[] for _ in range(num_layers + 1)]
    true_y: List[list] = [[] for _ in range(num_layers + 1)]
    var_by_layer: List[list] = [[] for _ in range(num_layers + 1)]
    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            # Forward all exits in one pass to obtain (logits, conf_logits) per exit.
            per_exit_logits, per_exit_conf_logits = model.forward_with_confidences(images)
            s_logits = torch.stack(per_exit_conf_logits, dim=0)  # [J, B]
            router_argmax = torch.argmax(s_logits, dim=0)  # [B] in [0, J-1]
            for layer_idx in range(num_layers + 1):
                mask = router_argmax == layer_idx
                if not mask.any():
                    continue
                exit_logits = per_exit_logits[layer_idx][mask]
                exit_probs = softmax(exit_logits)
                exit_labels = labels[mask]
                preds = exit_logits.argmax(dim=1)
                pred_y[layer_idx].extend(preds.cpu().numpy().tolist())
                true_y[layer_idx].extend(exit_labels.cpu().numpy().tolist())
                var_by_layer[layer_idx].extend(torch.var(exit_logits, dim=1).cpu().numpy().tolist())
                probs_all.append(exit_probs.cpu().numpy())
                labels_all.append(exit_labels.cpu().numpy())
                total_samples += int(mask.sum().item())

    if total_samples == 0:
        return ExitEvaluation(0.0, 0.0, 0.0, 0.0, 0.0)

    all_variances = [v for layer in var_by_layer for v in layer]
    mean_variance = float(np.mean(all_variances)) if all_variances else 0.0
    return build_exit_evaluation(
        num_layers + 1, pred_y, true_y, macs, total_samples, probs_all, labels_all,
        mean_variance=mean_variance,
    )


def evaluate_dynamic_exit(
    model: nn.Module,
    thresholds: Sequence[float],
    test_loader,
    *,
    macs: Sequence[float],
    num_layers: int,
    device: torch.device,
    cutoff_type: str = "entropy",
    split_at: Optional[int] = None,
    transport: Optional[Transport] = None,
) -> ExitEvaluation:
    """Run the dynamic early-exit policy and report accuracy, EMAR, variance, ECE, and NLL.

    ``num_layers`` is the number of early exits (the final head is exit index ``num_layers``).
    ``split_at`` optionally names an early-exit index at which the activation forwarded to the
    next exit is serialized and pushed through ``transport`` (default in-process identity, so
    the result is numerically identical to a non-split run).
    """
    thresholds = _normalize_thresholds(thresholds, num_layers)
    softmax = nn.Softmax(dim=1)
    log_softmax_fn = nn.LogSoftmax(dim=1)

    # router_argmax cutoff (moe_router method) routes via joint softmax over per-exit
    # confidence-head logits; each sample picks its argmax-exit. Calibration thresholds
    # are unused. Implemented as a separate code path because the decision is per-sample
    # over ALL exits collectively, not the sequential per-exit threshold rule.
    if cutoff_type == "router_argmax":
        return _evaluate_router_argmax(
            model, test_loader, macs=macs, num_layers=num_layers, device=device,
        )

    pred_y: List[list] = [[] for _ in range(num_layers + 1)]
    true_y: List[list] = [[] for _ in range(num_layers + 1)]
    var_by_layer: List[list] = [[] for _ in range(num_layers + 1)]
    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    total_samples = 0

    poe_alphas = None
    if cutoff_type in ("poe_entropy", "poe_scar_entropy"):
        poe_alphas = getattr(model, "poe_alphas", None)
        if poe_alphas is None:
            raise AttributeError(
                "Model has no 'poe_alphas' buffer; poe_entropy / poe_scar_entropy cutoffs "
                "require the PoE-equipped training path to register per-exit alphas."
            )

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            layer_idx = 0
            input_tensors = images.to(device)
            labels = labels.to(device)
            poe_running_sum = None  # cumulative alpha-weighted log_p across remaining samples
            while layer_idx <= num_layers and labels.size(0) > 0:
                if layer_idx < num_layers:
                    intermediate, logits = model(input_tensors, exit_layer_idx=layer_idx)
                    probs = softmax(logits)
                    if cutoff_type == "confidence":
                        exit_mask = torch.max(probs, dim=1).values > thresholds[layer_idx]
                    elif cutoff_type == "entropy":
                        exit_mask = torch.as_tensor(
                            entropy(probs.cpu(), axis=1) < thresholds[layer_idx],
                            device=device,
                        )
                    elif cutoff_type == "learned_confidence":
                        # SCAR routing: read the per-exit confidence head s_j on the
                        # intermediate carry and exit when s_j > threshold.
                        head_name = f"confidence_head_{layer_idx}"
                        conf_head = getattr(model, head_name, None)
                        if conf_head is None:
                            raise AttributeError(
                                f"Model has no '{head_name}' module; learned_confidence "
                                "cutoff requires per-exit confidence heads."
                            )
                        s_logit = conf_head(intermediate)
                        s = torch.sigmoid(s_logit)
                        exit_mask = s > thresholds[layer_idx]
                    elif cutoff_type == "poe_entropy":
                        # PoE-Anneal routing: cumulative log-PoE across exits; route on
                        # entropy of the cumulative prediction.
                        log_p = log_softmax_fn(logits)
                        alpha_j = poe_alphas[layer_idx]
                        if poe_running_sum is None:
                            poe_running_sum = alpha_j * log_p
                        else:
                            poe_running_sum = poe_running_sum + alpha_j * log_p
                        log_ptilde = log_softmax_fn(poe_running_sum)
                        ptilde = log_ptilde.exp()
                        # Override per-exit prediction / probs with the cumulative PoE
                        probs = ptilde
                        logits = log_ptilde  # use the log-PoE as the "logits" for argmax / var
                        exit_mask = torch.as_tensor(
                            entropy(ptilde.cpu(), axis=1) < thresholds[layer_idx],
                            device=device,
                        )
                    elif cutoff_type == "poe_scar_entropy":
                        # Hybrid routing: weighted sum of normalised PoE entropy and (1 - s_j).
                        log_p = log_softmax_fn(logits)
                        alpha_j = poe_alphas[layer_idx]
                        if poe_running_sum is None:
                            poe_running_sum = alpha_j * log_p
                        else:
                            poe_running_sum = poe_running_sum + alpha_j * log_p
                        log_ptilde = log_softmax_fn(poe_running_sum)
                        ptilde = log_ptilde.exp()
                        # Use cumulative PoE for the prediction (better accuracy than raw exit).
                        probs = ptilde
                        logits = log_ptilde
                        poe_ent = entropy(ptilde.cpu(), axis=1).astype(np.float32)
                        nclass = log_ptilde.size(-1)
                        poe_ent_norm = poe_ent / float(np.log(max(nclass, 2)))
                        head_name = f"confidence_head_{layer_idx}"
                        conf_head = getattr(model, head_name, None)
                        if conf_head is None:
                            raise AttributeError(
                                f"Model has no '{head_name}'; poe_scar_entropy cutoff requires per-exit confidence heads."
                            )
                        s = torch.sigmoid(conf_head(intermediate)).cpu().numpy().astype(np.float32)
                        hybrid_alpha = 0.5
                        hybrid_uncertainty = hybrid_alpha * poe_ent_norm + (1.0 - hybrid_alpha) * (1.0 - s)
                        exit_mask = torch.as_tensor(
                            hybrid_uncertainty < thresholds[layer_idx], device=device,
                        )
                    else:
                        raise ValueError(f"Unsupported cutoff_type '{cutoff_type}'.")
                    accepted_logits = logits[exit_mask]
                    accepted_probs = probs[exit_mask]
                    accepted_labels = labels[exit_mask]
                    labels = labels[~exit_mask]
                    input_tensors = intermediate[~exit_mask]
                    if poe_running_sum is not None:
                        poe_running_sum = poe_running_sum[~exit_mask]
                    if split_at is not None and layer_idx == split_at and input_tensors.size(0) > 0:
                        input_tensors = handoff(
                            input_tensors, exit_index=layer_idx, transport=transport
                        )
                else:
                    _, final_logits = model(input_tensors, exit_layer_idx=layer_idx)
                    if cutoff_type in ("poe_entropy", "poe_scar_entropy"):
                        # Fold the final exit into the running PoE prediction
                        log_p = log_softmax_fn(final_logits)
                        alpha_j = poe_alphas[layer_idx]
                        if poe_running_sum is None:
                            poe_running_sum = alpha_j * log_p
                        else:
                            poe_running_sum = poe_running_sum + alpha_j * log_p
                        log_ptilde = log_softmax_fn(poe_running_sum)
                        accepted_logits = log_ptilde
                        accepted_probs = log_ptilde.exp()
                    else:
                        accepted_logits = final_logits
                        accepted_probs = softmax(accepted_logits)
                    accepted_labels = labels
                    labels = labels[:0]

                if accepted_labels.numel() == 0:
                    layer_idx += 1
                    continue

                _, preds = torch.max(accepted_logits, dim=1)
                var_by_layer[layer_idx].extend(torch.var(accepted_logits, dim=1).cpu().numpy().tolist())
                pred_y[layer_idx].extend(preds.cpu().numpy().tolist())
                true_y[layer_idx].extend(accepted_labels.cpu().numpy().tolist())
                probs_all.append(accepted_probs.cpu().numpy())
                labels_all.append(accepted_labels.cpu().numpy())
                total_samples += accepted_labels.size(0)
                layer_idx += 1

    if total_samples == 0:
        return ExitEvaluation(0.0, 0.0, 0.0, 0.0, 0.0)

    all_variances = [v for layer in var_by_layer for v in layer]
    mean_variance = float(np.mean(all_variances)) if all_variances else 0.0
    return build_exit_evaluation(
        num_layers + 1, pred_y, true_y, macs, total_samples, probs_all, labels_all,
        mean_variance=mean_variance,
    )
