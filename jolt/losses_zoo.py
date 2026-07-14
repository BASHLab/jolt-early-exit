"""Composable loss components for early-exit training (the candidate-loss zoo).

Each component operates on per-exit logits and is combined by :func:`composite_loss` according
to a :class:`LossComponents` spec, so every candidate and every ablation row is just a config.

Provenance: JOLT's novelty is the *integration* (Candidate A: focal + soft-binned ECE + BYOT
self-distillation + a monotonicity hinge on a depth-increasing schedule). Every individual
component here is prior art used as a comparator. Several exact formulations are not fully
specified in the source literature; those use documented defaults and are marked
``# RESEARCH GAP:`` for the follow-up research pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Per-exit classification terms
# --------------------------------------------------------------------------------------

def focal_loss(logits: torch.Tensor, labels: torch.Tensor, *, gamma: float = 3.0,
               sample_dependent: bool = False) -> torch.Tensor:
    """Focal loss (Lin et al. 2017); calibration analysis Mukhoti et al. 2020.

    FL = -(1 - p_y)^gamma * log p_y. ``sample_dependent`` uses FLSD-53 (gamma=5 if p_y<0.2 else 3).
    """
    log_probs = F.log_softmax(logits, dim=1)
    log_py = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    py = log_py.exp()
    if sample_dependent:
        gamma_t = torch.where(py < 0.2, py.new_full((), 5.0), py.new_full((), 3.0))
    else:
        gamma_t = py.new_full((), float(gamma))
    return (-((1.0 - py) ** gamma_t) * log_py).mean()


def ce_loss(logits: torch.Tensor, labels: torch.Tensor, *, label_smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def brier_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Differentiable multiclass Brier: sum_c (p_c - 1{y=c})^2, mean over the batch."""
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(labels, num_classes=logits.size(1)).to(probs.dtype)
    return ((probs - onehot) ** 2).sum(dim=1).mean()


def brier_ce_blend(logits: torch.Tensor, labels: torch.Tensor, *, beta: float = 0.5) -> torch.Tensor:
    return (1.0 - beta) * F.cross_entropy(logits, labels) + beta * brier_loss(logits, labels)


# --------------------------------------------------------------------------------------
# Calibration regularizers
# --------------------------------------------------------------------------------------

def soft_binned_ece(logits: torch.Tensor, labels: torch.Tensor, *, n_bins: int = 15,
                    temperature: float = 0.01) -> torch.Tensor:
    """Differentiable soft-binned ECE surrogate (Karandikar et al. 2021).

    # RESEARCH GAP: the exact soft-binning kernel/temperature and reduction in Karandikar et al.
    # 2021 are not fully reproduced here. This implementation soft-assigns each sample's
    # confidence to bin centers with a Gaussian/softmax kernel, then sums |conf - acc| per bin
    # weighted by soft membership. Gradient flows through confidence and membership (the hard
    # accuracy indicator is constant w.r.t. logits). temperature is a guessed default.
    """
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).to(probs.dtype)
    centers = torch.linspace(0.0, 1.0, n_bins, device=logits.device, dtype=probs.dtype)
    dist = -((conf.unsqueeze(1) - centers.unsqueeze(0)) ** 2) / temperature
    membership = torch.softmax(dist, dim=1)            # (N, n_bins)
    bin_weight = membership.sum(dim=0).clamp_min(1e-12)
    bin_conf = (membership * conf.unsqueeze(1)).sum(dim=0) / bin_weight
    bin_acc = (membership * correct.unsqueeze(1)).sum(dim=0) / bin_weight
    frac = bin_weight / conf.size(0)
    return (frac * (bin_conf - bin_acc).abs()).sum()


def soft_avuc(logits: torch.Tensor, labels: torch.Tensor, *, kappa: float = 1.0) -> torch.Tensor:
    """Soft accuracy-vs-uncertainty calibration (Krishnan & Tickoo 2020; soft variant Karandikar
    et al. 2021).

    # RESEARCH GAP: the exact S-AvUC formulation (sigmoid surrogates and entropy thresholds) is
    # not fully reproduced. This surrogate penalizes accurate-but-uncertain and
    # inaccurate-but-certain samples using normalized entropy as the (soft) certainty; gradient
    # flows through the entropy term.
    """
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    correct = (pred == labels).to(probs.dtype)
    num_classes = probs.size(1)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=1)
    norm_entropy = entropy / torch.log(torch.tensor(float(num_classes), device=logits.device))
    certain = 1.0 - norm_entropy
    accurate_uncertain = correct * (1.0 - certain)
    inaccurate_certain = (1.0 - correct) * certain
    return kappa * (accurate_uncertain.mean() + inaccurate_certain.mean())


# --------------------------------------------------------------------------------------
# Distillation from the final exit (Be Your Own Teacher; Zhang et al. 2019)
# --------------------------------------------------------------------------------------

def byot_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, *, T: float = 3.0) -> torch.Tensor:
    """Temperature-T KL from the (detached) final-exit teacher to an earlier exit.

    # RESEARCH GAP: the original BYOT also distills feature maps via a bottleneck; here we use
    # logit-KL only. Whether feature distillation is needed to match the published gains is TBD.
    """
    teacher = F.log_softmax(teacher_logits.detach() / T, dim=1)
    student = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(student, teacher.exp(), reduction="batchmean") * (T * T)


# --------------------------------------------------------------------------------------
# Cross-exit monotonicity hinge (training-time analog of Jazbec et al. 2023)
# --------------------------------------------------------------------------------------

def monotonicity_hinge(per_exit_logits: List[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """Penalize a later exit having higher per-batch NLL than the previous exit.

    The target (earlier exit) NLL is detached; the penalty is differentiable through the later
    exit. The pair ending at the deepest exit is excluded so exit K is never optimized against
    monotonicity.

    # RESEARCH GAP: exact training-time monotonicity formulation (vs Jazbec et al.'s post-hoc PoE).
    """
    n = len(per_exit_logits)
    nll = [F.cross_entropy(logits, labels) for logits in per_exit_logits]
    penalty = per_exit_logits[0].new_zeros(())
    for e in range(n - 2):  # pairs (e, e+1) with e+1 <= n-2 -> exclude deepest exit (index n-1)
        penalty = penalty + F.relu(nll[e + 1] - nll[e].detach())
    return penalty


# --------------------------------------------------------------------------------------
# Energy-based confidence margin (Candidate F; Liu et al. 2020)
# --------------------------------------------------------------------------------------

def energy_margin(logits: torch.Tensor, labels: torch.Tensor, *, T: float = 1.0,
                  margin: float = 1.0) -> torch.Tensor:
    """Push correct samples to lower free energy than wrong ones by a margin.

    Free energy E(z) = -T * logsumexp(z / T).

    # RESEARCH GAP: energy scoring was designed for OOD detection; its value for in-distribution
    # early-exit gating is uncertain. This margin form is a reasonable default, not from a paper.
    """
    energy = -T * torch.logsumexp(logits / T, dim=1)
    correct = logits.argmax(dim=1) == labels
    if correct.any() and (~correct).any():
        return F.relu(energy[correct].mean() - energy[~correct].mean() + margin)
    return logits.new_zeros(())


# --------------------------------------------------------------------------------------
# RACS: Routing-Aware Coverage Surrogate (Zhou et al. arXiv:2505.23463, 2025)
# --------------------------------------------------------------------------------------

def aurc_soft_cdf(confidence: torch.Tensor, *, smoothing: float = 0.05) -> torch.Tensor:
    """Differentiable kernel CDF estimate of confidence over a batch.

    For each sample i, returns G_hat(s_i) = (1/n) sum_j sigmoid((s_i - s_j) / nu), where
    s = confidence values and nu = smoothing. As nu -> 0 this approaches the empirical CDF
    (the rank of s_i divided by n). For batch_size B, this is O(B^2); for B = 128 that is
    ~16K sigmoid evaluations per exit, negligible relative to the forward pass.

    The kernel form is the smoothed-empirical-CDF basis of Zhou et al. (2025)'s differentiable
    AURC; the SELE rank form (Franc, Prusa, Voracek, JMLR 2023) is the cheaper O(n log n)
    drop-in if the soft-CDF pass becomes a bottleneck.
    """
    s_i = confidence.unsqueeze(1)
    s_j = confidence.unsqueeze(0)
    return torch.sigmoid((s_i - s_j) / smoothing).mean(dim=1)


def racs_beta(confidence: torch.Tensor, *, smoothing: float = 0.05,
              delta_cap: float = 1e-3) -> torch.Tensor:
    """Per-sample AURC-coverage weight beta_i = -log(1 - clamp(G_hat(s_i), 0, 1-delta_cap)).

    Mathematically equivalent to Zhou et al. (2025) Definition 3.3: under sort-by-confidence,
    minimising sum_i beta_i * CE_i is a smooth lower bound of AURC because samples with high
    confidence rank get more loss weight, pushing the network to make confident samples correct
    (correct rank-order being the proxy that actually drives EE routing per Kubaty et al.,
    NeurIPS 2025). The bound delta_cap keeps beta_i <= -log(delta_cap) ~ 6.9, gradient-safe.
    """
    g = aurc_soft_cdf(confidence, smoothing=smoothing)
    return -torch.log1p(-g.clamp(max=1.0 - delta_cap))


def ce_racs(logits: torch.Tensor, labels: torch.Tensor, *, racs_lambda: float = 0.5,
            smoothing: float = 0.05, delta_cap: float = 1e-3) -> torch.Tensor:
    """RACS per-exit term: CE plus an AURC-targeted coverage reweighting.

    L_e = (1/n) sum_i CE(z_i, y_i) + racs_lambda * (1/n) sum_i beta_i * CE(z_i, y_i),
    with beta_i computed from the within-batch CDF of max-softmax confidence at exit e.
    racs_lambda = 0 recovers plain CE (identity test).
    """
    per_sample_ce = F.cross_entropy(logits, labels, reduction="none")
    base = per_sample_ce.mean()
    if racs_lambda == 0.0:
        return base
    confidence = F.softmax(logits, dim=1).max(dim=1).values.detach()
    beta = racs_beta(confidence, smoothing=smoothing, delta_cap=delta_cap)
    shaped = (beta * per_sample_ce).mean()
    return base + racs_lambda * shaped


# --------------------------------------------------------------------------------------
# PFW: smooth-Tchebycheff Pareto-frontier-weighted aggregation
# (Lin et al., "Smooth Tchebycheff Scalarization for Multi-Objective Optimization",
# ICML 2024, PMLR 235:30479-30509; arXiv:2402.19078)
# --------------------------------------------------------------------------------------

def smooth_tchebycheff(per_exit_terms: torch.Tensor, weights: torch.Tensor,
                       *, mu: float = 0.1, reference: float = 0.0) -> torch.Tensor:
    """L_PFW = mu * log( sum_e exp( w_e * (term_e - r_e) / mu ) ).

    As mu -> 0 this recovers exact (nonsmooth) Tchebycheff max_e w_e * (term_e - r_e); mu > 0
    gives bounded gradients via a softmax over exits and lets every exit move (whichever exit
    is currently furthest from its frontier-ideal gets the largest gradient). Linear
    depth-weighted sums cannot reach non-convex parts of the (accuracy, MACs) Pareto front
    (Das & Dennis 1997); Tchebycheff can.

    The reference r_e is a per-exit "ideal" offset; using r_e = 0 (the simplest, sweep-default
    choice) keeps the aggregation positive and gradient-safe.
    """
    scaled = weights * (per_exit_terms - reference) / mu
    return mu * torch.logsumexp(scaled, dim=0)


# --------------------------------------------------------------------------------------
# MUTUAL-EE: bidirectional symmetric KL between adjacent exits
# (Deep Mutual Learning; Zhang, Xiang, Hospedales, Lu, CVPR 2018)
# --------------------------------------------------------------------------------------

def mutual_ee_kl(per_exit_logits: List[torch.Tensor], *, T: float = 1.0) -> torch.Tensor:
    """Symmetric KL across adjacent (e, e+1) exit pairs, no detach.

    For each adjacent pair return 0.5 * (KL(p_{e+1} || p_e) + KL(p_e || p_{e+1})) scaled by T*T;
    average over (n - 1) pairs so the magnitude is invariant to depth. Mutual learning lets
    adjacent exits regularize each other symmetrically, smoothing the confidence trajectory
    across depth without privileging the final exit (the way candidate_b's BYOT-from-final does).
    """
    n = len(per_exit_logits)
    if n < 2:
        return per_exit_logits[0].new_zeros(())
    penalty = per_exit_logits[0].new_zeros(())
    for e in range(n - 1):
        log_p_e = F.log_softmax(per_exit_logits[e] / T, dim=1)
        log_p_next = F.log_softmax(per_exit_logits[e + 1] / T, dim=1)
        kl_next_to_e = F.kl_div(log_p_e, log_p_next.exp(), reduction="batchmean")
        kl_e_to_next = F.kl_div(log_p_next, log_p_e.exp(), reduction="batchmean")
        penalty = penalty + 0.5 * (kl_next_to_e + kl_e_to_next) * (T * T)
    return penalty / (n - 1)


# --------------------------------------------------------------------------------------
# EDL-EE: evidential / Dirichlet per-exit loss
# (Sensoy, Kandemir, Kaplan, "Evidential Deep Learning to Quantify Classification
# Uncertainty", NeurIPS 2018)
# --------------------------------------------------------------------------------------

def _dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(Dir(alpha) || Dir(1, 1, ..., 1)) computed per row.

        KL = log Gamma(sum_i alpha_i) - sum_i log Gamma(alpha_i) - log Gamma(K)
             + sum_i (alpha_i - 1) * (digamma(alpha_i) - digamma(sum_i alpha_i))
    """
    K = alpha.size(-1)
    S = alpha.sum(dim=-1, keepdim=True)
    lgamma_S = torch.lgamma(S.squeeze(-1))
    sum_lgamma_alpha = torch.lgamma(alpha).sum(dim=-1)
    lgamma_K = torch.lgamma(alpha.new_tensor(float(K)))
    digamma_S = torch.digamma(S)
    digamma_alpha = torch.digamma(alpha)
    sum_term = ((alpha - 1.0) * (digamma_alpha - digamma_S)).sum(dim=-1)
    return lgamma_S - sum_lgamma_alpha - lgamma_K + sum_term


def edl_loss(logits: torch.Tensor, labels: torch.Tensor, *, lambda_kl: float = 0.1) -> torch.Tensor:
    """Evidential Dirichlet loss with Brier-decomposed risk plus KL-to-uniform regularizer.

    evidence  e_k = softplus(z_k);  alpha_k = e_k + 1;  S = sum_k alpha_k;  p_hat = alpha / S.
    L = mean( sum_k [(y_k - p_hat_k)^2 + p_hat_k (1 - p_hat_k) / (S + 1)] )
        + lambda_kl * mean( KL(Dir(alpha_tilde) || Dir(1)) )
    where alpha_tilde = y + (1 - y) * alpha removes evidence for the true class from the KL.
    """
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    onehot = F.one_hot(labels, num_classes=logits.size(1)).to(alpha.dtype)
    p_hat = alpha / S
    sq_err = (onehot - p_hat).pow(2).sum(dim=1)
    var = (p_hat * (1.0 - p_hat) / (S + 1.0)).sum(dim=1)
    brier = (sq_err + var).mean()
    if lambda_kl == 0.0:
        return brier
    alpha_tilde = onehot + (1.0 - onehot) * alpha
    kl_reg = _dirichlet_kl_to_uniform(alpha_tilde).mean()
    return brier + lambda_kl * kl_reg


# --------------------------------------------------------------------------------------
# Exit-weight schedules
# --------------------------------------------------------------------------------------

def exit_weights(num_exits_total: int, schedule: str = "uniform", *, alpha: float = 0.7,
                 device: Optional[torch.device] = None) -> torch.Tensor:
    """Normalized per-exit weights (sum to 1). ``num_exits_total`` includes the final exit.

    ``cost_discount`` is the MAC-cost-discount aggregation used by RACS: g_e = m_0 / sum_{j<=e} m_j.
    With uniformly-spaced exits (the canonical SDN / ResNet-56 / DeepConvLSTM layout used in
    Stage 2), incremental MACs per exit are roughly equal so g_e ~ 1/(e+1); this is the
    normalised default. The exact MAC-weighted form is a refinement to layer in once per-exit
    MAC measurements land.
    """
    if schedule == "uniform":
        w = torch.ones(num_exits_total)
    elif schedule == "increasing":
        w = torch.arange(1, num_exits_total + 1, dtype=torch.float)
    elif schedule == "decreasing":
        w = torch.tensor([alpha ** i for i in range(num_exits_total)], dtype=torch.float)
    elif schedule == "cost_discount":
        w = torch.tensor([1.0 / (e + 1) for e in range(num_exits_total)], dtype=torch.float)
    else:
        raise ValueError(f"Unknown weight schedule '{schedule}'.")
    w = w / w.sum()
    return w.to(device) if device is not None else w


# --------------------------------------------------------------------------------------
# Composite loss
# --------------------------------------------------------------------------------------

@dataclass
class LossComponents:
    per_exit_term: str = "ce"            # ce | focal | ce_label_smoothing | brier_blend | ce_racs | edl
    focal_gamma: float = 3.0
    focal_sample_dependent: bool = False
    label_smoothing: float = 0.1
    brier_beta: float = 0.5
    calib_reg: str = "none"              # none | sb_ece | s_avuc
    lambda_cal: float = 0.5
    distill: str = "none"                # none | byot
    distill_T: float = 3.0
    lambda_aux: float = 1.0
    monotonic: bool = False
    lambda_mono: float = 0.1
    energy: bool = False
    lambda_energy: float = 0.1
    weight_schedule: str = "uniform"     # uniform | increasing | decreasing | cost_discount
    branchynet_alpha: float = 0.7
    # RACS knobs (used when per_exit_term == "ce_racs").
    racs_lambda: float = 0.5
    racs_smoothing: float = 0.05
    racs_delta_cap: float = 1e-3
    # PFW knobs (used when aggregation == "pfw").
    aggregation: str = "weighted_sum"    # weighted_sum | pfw
    pfw_mu: float = 0.1
    # MUTUAL-EE knobs (additive top-level term).
    mutual_kl: bool = False
    lambda_mutual: float = 1.0
    mutual_T: float = 1.0
    # EDL-EE knobs (used when per_exit_term == "edl").
    edl_lambda_kl: float = 0.1


def _per_exit_term(logits, labels, c: LossComponents, regression: bool) -> torch.Tensor:
    if regression:
        return F.mse_loss(logits.squeeze(-1), labels.float())
    if c.per_exit_term == "ce":
        return ce_loss(logits, labels)
    if c.per_exit_term == "ce_label_smoothing":
        return ce_loss(logits, labels, label_smoothing=c.label_smoothing)
    if c.per_exit_term == "focal":
        return focal_loss(logits, labels, gamma=c.focal_gamma, sample_dependent=c.focal_sample_dependent)
    if c.per_exit_term == "brier_blend":
        return brier_ce_blend(logits, labels, beta=c.brier_beta)
    if c.per_exit_term == "ce_racs":
        return ce_racs(logits, labels, racs_lambda=c.racs_lambda,
                       smoothing=c.racs_smoothing, delta_cap=c.racs_delta_cap)
    if c.per_exit_term == "edl":
        return edl_loss(logits, labels, lambda_kl=c.edl_lambda_kl)
    raise ValueError(f"Unknown per_exit_term '{c.per_exit_term}'.")


def _calibration_term(logits, labels, c: LossComponents) -> Optional[torch.Tensor]:
    if c.calib_reg == "none":
        return None
    if c.calib_reg == "sb_ece":
        return soft_binned_ece(logits, labels)
    if c.calib_reg == "s_avuc":
        return soft_avuc(logits, labels)
    raise ValueError(f"Unknown calib_reg '{c.calib_reg}'.")


def composite_loss(per_exit_logits: Sequence[torch.Tensor], labels: torch.Tensor,
                   components: LossComponents, *, regression: bool = False) -> torch.Tensor:
    """Combine per-exit terms, calibration, distillation, monotonicity, and energy per the spec.

    L = sum_e w_e * [ term_e + lambda_cal*calib_e + lambda_aux*byot_e(<final) ]
        + lambda_mono * monotonicity_hinge + lambda_energy * mean_e energy_margin_e
    """
    per_exit_logits = list(per_exit_logits)
    n = len(per_exit_logits)
    final = per_exit_logits[-1]
    weights = exit_weights(n, components.weight_schedule, alpha=components.branchynet_alpha,
                           device=labels.device)
    per_exit_terms: List[torch.Tensor] = []
    for e, logits in enumerate(per_exit_logits):
        term = _per_exit_term(logits, labels, components, regression)
        if not regression:
            calib = _calibration_term(logits, labels, components)
            if calib is not None:
                term = term + components.lambda_cal * calib
            if components.distill == "byot" and e < n - 1:
                term = term + components.lambda_aux * byot_kl(logits, final, T=components.distill_T)
        per_exit_terms.append(term)

    if components.aggregation == "weighted_sum":
        total = sum((weights[e] * per_exit_terms[e] for e in range(n)),
                    per_exit_logits[0].new_zeros(()))
    elif components.aggregation == "pfw":
        stacked = torch.stack(per_exit_terms)
        total = smooth_tchebycheff(stacked, weights, mu=components.pfw_mu)
    else:
        raise ValueError(f"Unknown aggregation '{components.aggregation}'.")

    if not regression and components.monotonic and n >= 3:
        total = total + components.lambda_mono * monotonicity_hinge(per_exit_logits, labels)
    if not regression and components.mutual_kl and n >= 2:
        total = total + components.lambda_mutual * mutual_ee_kl(per_exit_logits, T=components.mutual_T)
    if not regression and components.energy:
        energy = torch.stack([energy_margin(logits, labels) for logits in per_exit_logits]).mean()
        total = total + components.lambda_energy * energy
    return total


def selectivenet_loss(
    per_exit_logits: Sequence[torch.Tensor],
    per_exit_selection: Sequence[torch.Tensor],
    per_exit_aux: Sequence[torch.Tensor],
    labels: torch.Tensor,
    *,
    coverage_targets: Sequence[float],
    lam_coverage: float = 32.0,
    alpha: float = 0.5,
    weight_schedule: str = "increasing",
) -> torch.Tensor:
    """SelectiveNet objective (Geifman & El-Yaniv 2019) per exit, summed with a weight schedule.

    Per exit e with selection score g (sigmoid) and auxiliary head h:
        selective_risk = mean(g * CE) / mean(g)
        coverage_penalty = lam_coverage * relu(c_e - mean(g))^2
        total_e = alpha * (selective_risk + coverage_penalty) + (1 - alpha) * CE(h, y)

    # RESEARCH GAP: the per-exit coverage targets c_e and their tie-in to the calibrated exit
    # thresholds are unspecified in the source; ``coverage_targets`` is supplied by config.
    """
    n = len(per_exit_logits)
    weights = exit_weights(n, weight_schedule, device=labels.device)
    total = per_exit_logits[0].new_zeros(())
    for e in range(n):
        logits, selection, aux = per_exit_logits[e], per_exit_selection[e], per_exit_aux[e]
        ce = F.cross_entropy(logits, labels, reduction="none")
        g = selection.squeeze(-1)
        coverage = g.mean().clamp_min(1e-6)
        selective_risk = (g * ce).mean() / coverage
        coverage_penalty = lam_coverage * F.relu(coverage.new_tensor(coverage_targets[e]) - coverage) ** 2
        aux_ce = F.cross_entropy(aux, labels)
        total = total + weights[e] * (alpha * (selective_risk + coverage_penalty) + (1.0 - alpha) * aux_ce)
    return total
