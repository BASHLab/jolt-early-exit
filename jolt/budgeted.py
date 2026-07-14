"""Budgeted-batch evaluation (MSDNet-style) from saved checkpoints.

MSDNet (Huang et al., ICLR 2018) evaluates early-exit networks under "budgeted
batch classification": per-exit thresholds are solved on validation so that a
target fraction q of the samples reaching each exit terminates there, which
populates every exit by construction and traces an accuracy-vs-compute curve
as q sweeps. This module implements that protocol on top of the existing exit
models.

The expensive part (one full forward pass through all exits per split) is done
once per model and cached as score/correctness matrices; threshold solving and
routing simulation are pure NumPy on those matrices, so any number of budget
points, population caps, or alternative operating-point rules can be evaluated
offline without touching the GPU again.

Score convention: lower score = more confident = exit (entropy-like). The
per-method prediction rule matches jolt.inference / jolt.text_inference: PoE
methods predict from the cumulative product-of-experts distribution, all other
methods from the raw per-exit softmax.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# Matrix collection (one forward pass through all exits, no early stopping)
# --------------------------------------------------------------------------------------

def collect_exit_matrix(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    cutoff_type: str = "entropy",
    score_type: str = "entropy",
) -> Dict[str, np.ndarray]:
    """Forward every sample through every exit; return per-exit routing scores and
    per-exit prediction correctness.

    ``score_type`` selects the routing score computed from the (method-appropriate)
    per-exit probability vector: "entropy" (Shannon entropy) or "margin"
    (1 - (top1 - top2)). Both use the lower-is-more-confident convention, so the
    downstream quantile/threshold machinery is unchanged.

    Returns dict with:
      scores  [N, num_exits]      routing score at each early exit (lower = exit)
      correct [N, num_exits + 1]  prediction correctness at each exit incl. final,
                                  under the method's prediction rule
      labels  [N]
    """
    if cutoff_type not in ("entropy", "poe_entropy"):
        raise ValueError(f"collect_exit_matrix supports entropy/poe_entropy, got '{cutoff_type}'.")
    if score_type not in ("entropy", "margin"):
        raise ValueError(f"score_type must be entropy|margin, got '{score_type}'.")
    poe_alphas = None
    if cutoff_type == "poe_entropy":
        poe_alphas = getattr(model, "poe_alphas", None)
        if poe_alphas is None:
            raise AttributeError("poe_entropy cutoff requires a 'poe_alphas' buffer on the model.")

    softmax = nn.Softmax(dim=1)
    log_softmax = nn.LogSoftmax(dim=1)
    num_early = model.num_exits

    scores_rows: List[np.ndarray] = []
    correct_rows: List[np.ndarray] = []
    labels_rows: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            x = images.to(device)
            labels = labels.to(device)
            b = labels.size(0)
            batch_scores = np.zeros((b, num_early), dtype=np.float32)
            batch_correct = np.zeros((b, num_early + 1), dtype=np.float32)
            running_sum = None
            for pos in range(num_early + 1):
                x, logits = model(x, exit_layer_idx=pos)
                if cutoff_type == "poe_entropy":
                    log_p = log_softmax(logits)
                    alpha = poe_alphas[pos]
                    running_sum = alpha * log_p if running_sum is None else running_sum + alpha * log_p
                    log_ptilde = log_softmax(running_sum)
                    probs = log_ptilde.exp()
                else:
                    probs = softmax(logits)
                preds = probs.argmax(dim=1)
                batch_correct[:, pos] = (preds == labels).float().cpu().numpy()
                if pos < num_early:
                    if score_type == "margin":
                        top2 = probs.topk(2, dim=1).values
                        score = 1.0 - (top2[:, 0] - top2[:, 1])
                    else:
                        score = -(probs.clamp_min(1e-12).log() * probs).sum(dim=1)
                    batch_scores[:, pos] = score.cpu().numpy()
            scores_rows.append(batch_scores)
            correct_rows.append(batch_correct)
            labels_rows.append(labels.cpu().numpy())

    return {
        "scores": np.concatenate(scores_rows, axis=0),
        "correct": np.concatenate(correct_rows, axis=0),
        "labels": np.concatenate(labels_rows, axis=0),
    }


def collect_exit_matrix_text(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    cutoff_type: str = "entropy",
) -> Dict[str, np.ndarray]:
    """Text-path (BertEarlyExit) counterpart of collect_exit_matrix."""
    from .text_inference import _move, _run_exit

    if cutoff_type not in ("entropy", "poe_entropy"):
        raise ValueError(f"collect_exit_matrix_text supports entropy/poe_entropy, got '{cutoff_type}'.")
    poe_alphas = None
    if cutoff_type == "poe_entropy":
        poe_alphas = getattr(model, "poe_alphas", None)
        if poe_alphas is None:
            raise AttributeError("poe_entropy cutoff requires a 'poe_alphas' buffer on the model.")

    softmax = nn.Softmax(dim=1)
    log_softmax = nn.LogSoftmax(dim=1)
    num_pos = len(model.exit_layers)
    num_early = model.num_exits

    scores_rows: List[np.ndarray] = []
    correct_rows: List[np.ndarray] = []
    labels_rows: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            labels = batch["labels"]
            b = labels.size(0)
            batch_scores = np.zeros((b, num_early), dtype=np.float32)
            batch_correct = np.zeros((b, num_pos), dtype=np.float32)
            hidden_state, cache = None, None
            running_sum = None
            for pos in range(num_pos):
                out = _run_exit(model, first=(pos == 0), batch=batch, pos=pos,
                                hidden_state=hidden_state, cache=cache)
                hidden_state, cache = out["hidden_state"], out["cache"]
                logits = out["logits"]
                if cutoff_type == "poe_entropy":
                    log_p = log_softmax(logits)
                    alpha = poe_alphas[pos]
                    running_sum = alpha * log_p if running_sum is None else running_sum + alpha * log_p
                    log_ptilde = log_softmax(running_sum)
                    probs = log_ptilde.exp()
                else:
                    probs = softmax(logits)
                preds = probs.argmax(dim=-1)
                batch_correct[:, pos] = (preds == labels).float().cpu().numpy()
                if pos < num_early:
                    ent = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1)
                    batch_scores[:, pos] = ent.cpu().numpy()
            scores_rows.append(batch_scores)
            correct_rows.append(batch_correct)
            labels_rows.append(labels.cpu().numpy())

    return {
        "scores": np.concatenate(scores_rows, axis=0),
        "correct": np.concatenate(correct_rows, axis=0),
        "labels": np.concatenate(labels_rows, axis=0),
    }


# --------------------------------------------------------------------------------------
# Threshold solving and routing simulation (pure NumPy)
# --------------------------------------------------------------------------------------

def thresholds_for_population(scores_val: np.ndarray, q: float) -> List[float]:
    """MSDNet-style sequential population thresholds.

    At each early exit, the q-quantile of the routing scores of the samples that
    REACH that exit becomes the threshold, so a fraction ~q of the remaining
    samples terminates there. Sequential: samples below the threshold are
    removed before the next exit's quantile is taken.
    """
    n, num_early = scores_val.shape
    remaining = np.ones(n, dtype=bool)
    thresholds: List[float] = []
    for j in range(num_early):
        s = scores_val[remaining, j]
        if s.size == 0:
            thresholds.append(float("-inf"))
            continue
        t = float(np.quantile(s, q))
        thresholds.append(t)
        exited = remaining.copy()
        exited[remaining] = scores_val[remaining, j] < t
        remaining &= ~exited
    return thresholds


def simulate_routing(
    scores: np.ndarray,
    correct: np.ndarray,
    thresholds: Sequence[float],
    per_exit_macs: Sequence[float],
) -> Dict[str, object]:
    """Route every sample sequentially through the exits at the given thresholds
    and aggregate accuracy / populations / compute. Mirrors evaluate_dynamic_exit
    semantics (exit when score < threshold; final exit accepts all)."""
    n, num_early = scores.shape
    exit_idx = np.full(n, num_early, dtype=np.int64)
    remaining = np.ones(n, dtype=bool)
    for j in range(num_early):
        take = remaining & (scores[:, j] < thresholds[j])
        exit_idx[take] = j
        remaining &= ~take

    cum_macs = np.cumsum(np.asarray(per_exit_macs, dtype=np.float64))
    deep = float(cum_macs[-1])
    exit_counts = [int((exit_idx == j).sum()) for j in range(num_early + 1)]
    per_exit_acc = []
    for j in range(num_early + 1):
        mask = exit_idx == j
        per_exit_acc.append(float(correct[mask, j].mean()) if mask.any() else 0.0)
    sample_correct = correct[np.arange(n), exit_idx]
    avg_macs = float(cum_macs[exit_idx].mean())

    # EMAR (early exits only): compute-fraction x population x accuracy
    emar = 0.0
    for j in range(num_early):
        if exit_counts[j] == 0:
            continue
        emar += (float(per_exit_macs[0]) / float(cum_macs[j])) * (exit_counts[j] / n) * per_exit_acc[j]

    return {
        "accuracy": float(sample_correct.mean()),
        "exit_counts": exit_counts,
        "per_exit_accuracy": per_exit_acc,
        "avg_macs": avg_macs,
        "macs_frac": avg_macs / deep,
        "mac_saved_pct": (1.0 - avg_macs / deep) * 100.0,
        "emar": emar * 100.0,
    }


DEFAULT_Q_GRID = [round(q, 3) for q in np.linspace(0.02, 0.98, 33)]


def budget_curve(
    mats_val: Dict[str, np.ndarray],
    mats_test: Dict[str, np.ndarray],
    per_exit_macs: Sequence[float],
    q_grid: Optional[Sequence[float]] = None,
) -> List[dict]:
    """Sweep exit-population fraction q; solve thresholds on validation; report
    validation and test routing statistics per q."""
    rows = []
    for q in (q_grid if q_grid is not None else DEFAULT_Q_GRID):
        thr = thresholds_for_population(mats_val["scores"], q)
        val = simulate_routing(mats_val["scores"], mats_val["correct"], thr, per_exit_macs)
        test = simulate_routing(mats_test["scores"], mats_test["correct"], thr, per_exit_macs)
        rows.append({
            "q": float(q),
            "thresholds": [float(t) for t in thr],
            "val": val,
            "test": test,
        })
    return rows
