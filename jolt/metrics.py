"""Early-exit evaluation metrics: EMAR, ECE, NLL.

EMAR (Exit-population and MAC-scaled Anti-Risk) sums per-exit contributions of the form
``(M_1 / cumulative_MAC_i) * (S_i / N) * A_i`` over the early exits.

Hypervolume on (accuracy, MACs) and per-sample MACs helpers live here too. They are used
to confirm that the EMAR ranking is consistent with the joint accuracy--compute Pareto
frontier at the deployment operating point.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def compute_emar(
    per_exit_accuracy: Sequence[float],
    per_exit_support: Sequence[float],
    macs: Sequence[float],
    total_samples: float,
) -> float:
    """EMAR over the early exits.

    For each early exit i (every exit except the final one), the contribution is::

        (macs[0] / sum(macs[:i+1])) * (support_i / total_samples) * accuracy_i

    ``macs`` holds per-exit compute (length = number of exits including the final head);
    ``macs[0]`` is the first-exit cost and the cumulative sum is the cost of reaching
    exit ``i``.
    """
    macs = [float(m) for m in macs]
    n_exits = len(macs)
    if total_samples <= 0:
        return 0.0

    emar = 0.0
    for i in range(n_exits - 1):  # early exits only; final head is not an "early" saving
        support = float(per_exit_support[i])
        if support <= 0:
            continue
        mac_sum = sum(macs[: i + 1]) or 1.0
        efficiency = macs[0] / mac_sum
        population = support / total_samples
        emar += efficiency * population * float(per_exit_accuracy[i])
    return float(emar)


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 15
) -> float:
    """Expected Calibration Error over equal-width confidence bins."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array of shape (n_samples, n_classes).")
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    n = labels.shape[0]
    if n == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_confidence = confidences[mask].mean()
        bin_accuracy = accuracies[mask].mean()
        ece += (count / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def negative_log_likelihood(
    probs: np.ndarray, labels: np.ndarray, *, eps: float = 1e-12
) -> float:
    """Mean negative log-likelihood of the true class under predicted probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array of shape (n_samples, n_classes).")
    n = labels.shape[0]
    if n == 0:
        return 0.0
    true_class = probs[np.arange(n), labels]
    return float(-np.mean(np.log(np.clip(true_class, eps, 1.0))))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean over samples of sum_c (p_c - 1{y=c})^2. Lower is better."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array of shape (n_samples, n_classes).")
    n = labels.shape[0]
    if n == 0:
        return 0.0
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def risk_coverage_curve(confidences: np.ndarray, correctness: np.ndarray):
    """Selective-prediction risk-coverage curve: sort by confidence (high first), and at each
    coverage k/n report the error rate among the k most-confident samples.

    Returns ``(coverage, risk)`` arrays of length n.
    """
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    correctness = np.asarray(correctness, dtype=np.float64).reshape(-1)
    n = confidences.shape[0]
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(-confidences)  # most confident first
    correct_sorted = correctness[order]
    ranks = np.arange(1, n + 1)
    coverage = ranks / n
    risk = 1.0 - np.cumsum(correct_sorted) / ranks
    return coverage, risk


def aurc(confidences: np.ndarray, correctness: np.ndarray) -> float:
    """Area Under the Risk-Coverage curve (Geifman & El-Yaniv, 2018). Lower is better: a model
    whose confidence ranks correct predictions above wrong ones keeps risk low at low coverage.

    Reported as a co-primary metric with EMAR because ECE alone is a poor proxy for early-exit
    networks (Kubaty et al., 2025).

    # RESEARCH GAP: for the early-exit *dynamic policy* we score each sample at the confidence of
    # the exit it actually left from (policy-level AURC over the accepted-sample set). Whether to
    # instead report per-exit risk-coverage, and the exact Kubaty et al. 2025 methodology, is to be
    # confirmed in follow-up research.
    """
    coverage, risk = risk_coverage_curve(confidences, correctness)
    if coverage.size == 0:
        return 0.0
    trapezoid = getattr(np, "trapezoid", np.trapz)  # np.trapz deprecated in NumPy 2.x
    return float(trapezoid(risk, coverage))


def per_sample_macs(exit_counts: Sequence[int], per_exit_macs: Sequence[float]) -> float:
    """Average per-sample compute for one operating point of a dynamic-exit policy.

    A sample that left at exit i incurred cost ``sum(per_exit_macs[:i+1])`` (the cumulative
    cost of reaching that exit). The average over all accepted samples is the cost axis for
    the (accuracy, MACs) Pareto check.
    """
    total = int(sum(int(c) for c in exit_counts))
    if total <= 0:
        return 0.0
    n_exits = min(len(per_exit_macs), len(exit_counts))
    cumulative = [float(sum(float(m) for m in per_exit_macs[: i + 1])) for i in range(n_exits)]
    weighted = sum(cumulative[i] * int(exit_counts[i]) for i in range(n_exits))
    return float(weighted / total)


def hypervolume_2d(
    points: Sequence[Tuple[float, float]],
    reference: Tuple[float, float],
) -> float:
    """2D hypervolume over (accuracy, MACs) points: MAXIMIZE accuracy, MINIMIZE MACs.

    Reference is the worst corner (worst-accuracy, worst-MACs). HV is the area of the union of
    (per-frontier-point) rectangles bounded between the point and the reference. Higher HV is
    better. Within-cell rankings are invariant under shifts of the reference because every
    method's HV scales by the same constant.

    Algorithm: filter to points dominating the reference, extract the Pareto frontier (sorted
    by accuracy descending these have MACs descending too), then sum strip areas
    (a_i - a_{i+1}) * (ref_macs - m_i) with a_{K+1} = ref_acc. Canonical 2D-HV sweep,
    O(n log n). Returns 0.0 if no point dominates the reference (e.g., a collapsed method).
    """
    ref_acc, ref_macs = float(reference[0]), float(reference[1])
    useful = [(float(a), float(m)) for (a, m) in points if a > ref_acc and m < ref_macs]
    if not useful:
        return 0.0
    # Sort by accuracy descending; tie-break by MACs ascending for determinism.
    useful.sort(key=lambda p: (-p[0], p[1]))
    # Pareto frontier extraction: keep points with strictly lower MACs than any earlier-seen
    # (earlier-seen are higher accuracy, so dominating MACs must drop strictly).
    frontier: List[Tuple[float, float]] = []
    min_macs_seen = float("inf")
    for a, m in useful:
        if m < min_macs_seen:
            frontier.append((a, m))
            min_macs_seen = m
    # Sweep: strip i spans accuracy [a_{i+1}, a_i] with a_{K+1} := ref_acc; width = ref_macs - m_i.
    hv = 0.0
    for i, (a, m) in enumerate(frontier):
        next_a = frontier[i + 1][0] if i + 1 < len(frontier) else ref_acc
        hv += (a - next_a) * (ref_macs - m)
    return float(hv)
