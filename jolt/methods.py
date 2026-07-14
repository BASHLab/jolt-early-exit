"""Multi-exit training methods: JOLT plus the comparison baselines.

JOLT (variance-based scaling + adaptive uncertainty weighting via :class:`MultiTaskLoss`) is
handled in the runners because it owns learnable parameters. The baselines below are pure
functions over the per-exit logits, so they work unchanged for vision, IMU, and text. They
are reproduced from the published methods (formulations follow the source's GLUE refactor).

Each baseline takes ``(per_exit_logits, labels)`` and returns a scalar training loss.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F

from .losses_zoo import LossComponents, composite_loss

# Diagnostic: composed-loss component sums written here by each loss call. Read + reset
# from train_one_epoch to log per-epoch component magnitudes (NaN detection, dead-component
# detection). Keys: "L_poe", "L_mono", "L_distill", "L_brier", "L_mac", and "n" (count).
_LOSS_COMPONENTS: Dict[str, float] = {"L_poe": 0.0, "L_mono": 0.0, "L_distill": 0.0,
                                       "L_brier": 0.0, "L_mac": 0.0, "n": 0}


def reset_loss_components() -> None:
    for k in _LOSS_COMPONENTS:
        _LOSS_COMPONENTS[k] = 0.0


def pop_loss_components() -> Dict[str, float]:
    """Return per-call average of each component since the last reset, then reset."""
    n = max(_LOSS_COMPONENTS["n"], 1)
    out = {k: (v / n if k != "n" else int(v)) for k, v in _LOSS_COMPONENTS.items()}
    reset_loss_components()
    return out


def _stash(key: str, value) -> None:
    if torch.is_tensor(value):
        value = float(value.detach().item())
    _LOSS_COMPONENTS[key] = _LOSS_COMPONENTS.get(key, 0.0) + float(value)

BASELINE_METHODS = [
    "adaloss", "branchynet", "eenet", "td", "meronen", "meronen_laplace",
    "deep_only",
    "boostnet", "boostnet_lin", "ztw_cascade", "jei_dnn",
    "cwet", "route_consensus", "sat_consensus", "anytime_stable",
    "anytime_stable_cold", "anytime_stable_cheap", "asym_select",
    "litelaplace", "anytime_stable_distill",
    "scar",                        # v5 candidate C1 (AURC axis)
    "budget_boost",                # v5 candidate C2 (MAC-saved axis)
    "poe_anneal",                  # v5 candidate C3 (EMAR axis; cumulative PoE + monotonicity)
    "asym_select_mac",             # v5 enhancement of v4 C5 (AsymSelect + MAC penalty)
    "anytime_stable_distill_mono", # v5 enhancement of v4 C7 (distill + monotonicity)
    "tri_axis",                    # v6 kitchen-sink (SCAR + BudgetBoost + PoE-Anneal)
    "poe_distill",                 # v6 PoE-Anneal + BEEM-style self-distillation
    "poe_multitask",               # v6 PoE-Anneal + Kendall MultiTaskLoss adaptive eta
    "poe_anytime",                 # v6 PoE-Anneal + AnytimeStable cost-aware alpha
    "poe_asym",                    # v6 PoE-Anneal + AsymSelect kappa-weighted NLL
    "scar_distill",                # v6 SCAR + BEEM-style self-distillation
    "budget_boost_distill",        # v6 BudgetBoost + KL teacher (anti-collapse anchor)
    "budget_boost_multitask",      # v6 BudgetBoost + Kendall MTL on CE floor
    "tri_axis_distill",            # v6 tri_axis + BEEM-style self-distillation
    "poe_brier",                   # v6 PoE-Anneal + Brier-score calibration anchor
    "scar_brier",                  # v6 SCAR + Brier-score calibration anchor
    "tri_axis_brier",              # v6 tri_axis + Brier-score calibration anchor
    "scar_poe",                    # v6 hybrid: trains SCAR + PoE, eval combines both signals
    "moe_router",                  # v6 new mechanism class: joint softmax router over SCAR confidence heads
    "poe_jazbec",                  # v6 baseline: Jazbec et al. NeurIPS 2023 post-hoc PoE (uniform alphas, no training change)
    "poe_distill_mtl",             # v6 poe_distill + Kendall MTL adaptive eta
    "poe_distill_brier",           # v6 poe_distill + Brier-score anchor
    "poe_distill_asym",            # v6 poe_distill + asymmetric kappa-CE on cumulative ptilde
    "poe_distill_anytime",         # v6 poe_distill + AnytimeStable cost-aware alpha
    "poe_distill_mtl_brier",       # v6 poe_distill + MTL + Brier (multi-mech)
    "poe_multitask_brier",         # v6 poe_multitask + Brier (ablation: full minus distill)
    "poe_distill_mtl_asym",        # v6 poe_distill + MTL + asymmetric kappa-CE (multi-mech)
    "poe_distill_mtl_brier_asym",  # v6 poe_distill + MTL + Brier + asym (full stack)
    "budget_boost_distill_brier",  # v6 BudgetBoost + distill + Brier (multi-mech)
    "budget_boost_distill_asym",   # v6 BudgetBoost + distill + asymmetric kappa-CE (multi-mech)
    "budget_boost_distill_mtl",    # v6 BudgetBoost + distill + MTL (multi-mech)
    "budget_boost_distill_mtl_brier",  # v6 BudgetBoost + distill + MTL + Brier (full stack)
    "poe_distill_mtl_brier_mac",       # v6 poe_distill + MTL + Brier + differentiable MAC penalty (HV ceiling break)
    "distill_mtl_brier",               # MTL + KL distill + Brier WITHOUT PoE training-time machinery
]

# Candidate presets (component configs). Candidate A is our novelty (the integration); every
# other preset is a prior-art comparator. Candidate C (SelectiveNet) needs model selection heads
# and is registered separately once those exist.
CANDIDATE_PRESETS: Dict[str, LossComponents] = {
    # A (LEAD / novelty): focal + soft-binned ECE + BYOT distillation + monotonicity hinge.
    "candidate_a": LossComponents(
        per_exit_term="focal", focal_sample_dependent=True, calib_reg="sb_ece", lambda_cal=0.5,
        distill="byot", lambda_aux=1.0, monotonic=True, lambda_mono=0.1, weight_schedule="increasing",
    ),
    # B (fallback): cascaded self-distillation + label smoothing + S-AvUC.
    "candidate_b": LossComponents(
        per_exit_term="ce_label_smoothing", label_smoothing=0.1, calib_reg="s_avuc", lambda_cal=1.0,
        distill="byot", lambda_aux=1.0, weight_schedule="increasing",
    ),
    # D (proper-score control): Brier + CE blend.
    "candidate_d": LossComponents(per_exit_term="brier_blend", brier_beta=0.5, weight_schedule="increasing"),
    # E (ablation): training-time conditional monotonicity. RESEARCH GAP: true training-time PoE;
    # approximated here by the monotonicity hinge on a label-smoothed CE base.
    "candidate_e": LossComponents(
        per_exit_term="ce_label_smoothing", label_smoothing=0.1, monotonic=True, lambda_mono=0.1,
        weight_schedule="increasing",
    ),
    # F (baseline): energy-based confidence margin.
    "candidate_f": LossComponents(per_exit_term="ce", energy=True, lambda_energy=0.1, weight_schedule="increasing"),
    # Component baselines (isolate one piece each).
    "focal_only": LossComponents(per_exit_term="focal", weight_schedule="increasing"),
    "byot_only": LossComponents(per_exit_term="ce", distill="byot", weight_schedule="increasing"),
    "ls": LossComponents(per_exit_term="ce_label_smoothing", label_smoothing=0.1, weight_schedule="increasing"),
    "s_avuc": LossComponents(per_exit_term="ce", calib_reg="s_avuc", weight_schedule="increasing"),
    "brier": LossComponents(per_exit_term="brier_blend", weight_schedule="increasing"),
    # Stage-1b lite ablations of candidate_a: each removes one suspected over-regularizer to
    # isolate which term drives the underperformance on small/easy tasks.
    "candidate_a_lite_focal": LossComponents(
        per_exit_term="focal", focal_gamma=1.0, focal_sample_dependent=False,
        calib_reg="sb_ece", lambda_cal=0.5, distill="byot", lambda_aux=1.0,
        monotonic=True, lambda_mono=0.1, weight_schedule="increasing",
    ),
    "candidate_a_lite_cal": LossComponents(
        per_exit_term="focal", focal_sample_dependent=True, calib_reg="none", lambda_cal=0.0,
        distill="byot", lambda_aux=1.0, monotonic=True, lambda_mono=0.1, weight_schedule="increasing",
    ),
    "candidate_a_lite_aux": LossComponents(
        per_exit_term="focal", focal_sample_dependent=True, calib_reg="sb_ece", lambda_cal=0.5,
        distill="byot", lambda_aux=0.3, monotonic=True, lambda_mono=0.1, weight_schedule="increasing",
    ),
    "candidate_a_lite_mono": LossComponents(
        per_exit_term="focal", focal_sample_dependent=True, calib_reg="sb_ece", lambda_cal=0.5,
        distill="byot", lambda_aux=1.0, monotonic=False, lambda_mono=0.0, weight_schedule="increasing",
    ),
    # RACS (v3 sweep lead): per-exit CE reweighted by an AURC-coverage beta + MAC-cost-discount
    # aggregation. Designed to beat candidate_b on AURC + EMAR while being invariant to the
    # confidence-magnitude shift CutMix and label smoothing induce. Source math: Zhou, Gruber,
    # Popordanoska, Blaschko, arXiv:2505.23463 (2025); framing: Kubaty et al. NeurIPS 2025.
    "racs": LossComponents(
        per_exit_term="ce_racs", racs_lambda=0.5, racs_smoothing=0.05,
        weight_schedule="cost_discount",
    ),
    # RACS ablation row 1: cost-discount aggregation only, no beta reweight (isolates the
    # aggregation contribution).
    "racs_no_beta": LossComponents(
        per_exit_term="ce", weight_schedule="cost_discount",
    ),
    # RACS ablation row 2: beta reweight only, uniform aggregation (isolates the per-exit
    # routing-aware shaping).
    "racs_uniform_agg": LossComponents(
        per_exit_term="ce_racs", racs_lambda=0.5, racs_smoothing=0.05,
        weight_schedule="uniform",
    ),
    # PFW (v3 sweep fallback lead): smooth-Tchebycheff Pareto-frontier aggregation over plain
    # per-exit CE, MAC-cost-discount preference weights. Pulls up whichever exit is currently
    # furthest from its frontier-ideal; reaches non-convex parts of the (accuracy, MACs) frontier
    # that depth-weighted sums cannot. Source: Lin et al. ICML 2024 (arXiv:2402.19078).
    "pfw": LossComponents(
        per_exit_term="ce", aggregation="pfw", pfw_mu=0.1,
        weight_schedule="cost_discount",
    ),
    # MUTUAL-EE (v3 sweep secondary): bidirectional symmetric KL between adjacent exits, no
    # detach. Smooths the confidence trajectory across depth without privileging the final
    # exit. Source: Deep Mutual Learning (Zhang et al. CVPR 2018).
    "mutual_ee": LossComponents(
        per_exit_term="ce", mutual_kl=True, lambda_mutual=1.0, mutual_T=1.0,
        weight_schedule="uniform",
    ),
    # EDL-EE (v3 sweep secondary, high-upside / high-variance): evidential Dirichlet per-exit
    # term plus a KL-to-uniform regularizer that strips evidence on the true class. Provides a
    # non-calibration epistemic uncertainty signal for the accept/defer decision, orthogonal to
    # candidate_b's calibration stack. Source: Sensoy, Kandemir, Kaplan, NeurIPS 2018.
    # In-band Stage-2 validation flag: source backbones are LeNet-class; trust on ResNet-56 only
    # after verifying it does not collapse on small batches.
    "edl_ee": LossComponents(
        per_exit_term="edl", edl_lambda_kl=0.1, weight_schedule="cost_discount",
    ),
    # BEEM (Bajpai et al., ICLR 2025): linearly i-weighted CE + KL-to-final with T=1.0.
    # Decomposes exactly into composite_loss with ce + byot at distill_T=1.0 + increasing
    # weight schedule + lambda_aux=1.0. Listed as a CANDIDATE_PRESETS entry (not BASELINES)
    # because it routes through the composite dispatcher; semantically it is a published EE
    # baseline that always runs as a comparator.
    "beem": LossComponents(
        per_exit_term="ce", distill="byot", distill_T=1.0, lambda_aux=1.0,
        weight_schedule="increasing",
    ),
    # Move-A variants of the three v3 candidates that collapsed on Tiny-ImageNet
    # (pfw 8% acc, edl_ee 0.8%, boostnet 13%). Each tweaks a single hyperparameter the
    # collapse-mode analysis identified: pfw's mu=0.1 made the smooth-max too sharp, edl_ee's
    # fixed lambda_kl=0.1 over-regularised on 200 classes without epoch annealing.
    # (boostnet_lin lives in BASELINES because boostnet is a top-level baseline function.)
    "pfw_mu05": LossComponents(
        per_exit_term="ce", aggregation="pfw", pfw_mu=0.5,
        weight_schedule="cost_discount",
    ),
    "edl_ee_light": LossComponents(
        per_exit_term="edl", edl_lambda_kl=0.01, weight_schedule="cost_discount",
    ),
}

# candidate_c (SelectiveNet) is handled by a dedicated training path (needs selection heads), not
# the composite dispatcher, so it is listed here but not in CANDIDATE_PRESETS.
ALL_METHODS = ["jolt", "candidate_c"] + BASELINE_METHODS + list(CANDIDATE_PRESETS) + ["composite"]


def _ce_or_mse(logits: torch.Tensor, labels: torch.Tensor, regression: bool) -> torch.Tensor:
    if regression:
        return F.mse_loss(logits.squeeze(-1), labels.float())
    return F.cross_entropy(logits, labels)


def _per_exit_ce(per_exit_logits, labels, regression):
    return torch.stack([_ce_or_mse(logits, labels, regression) for logits in per_exit_logits])


def adaloss_loss(per_exit_logits, labels, *, regression: bool = False, **_) -> torch.Tensor:
    """Uniform-mean fallback used only when MultiTaskLoss is disabled.

    The FAITHFUL AdaLoss path (per the original train_anytime.py: per-exit CE without variance
    scaling, fed to a Kendall et al. 2018 MultiTaskLoss with learnable eta) lives in
    jolt/train.py under ``method == "adaloss"``. It is selected when ``cfg.loss.method ==
    "adaloss"`` AND ``cfg.loss.use_multitask`` is True (the default). This stateless function
    is the fallback for ablations where multitask is explicitly disabled.
    """
    return _per_exit_ce(per_exit_logits, labels, regression).mean()


def meronen_loss(per_exit_logits, labels, *, regression: bool = False, **_) -> torch.Tensor:
    """Summed cross-entropy across exits."""
    return _per_exit_ce(per_exit_logits, labels, regression).sum()


def deep_only_loss(per_exit_logits, labels, *, regression: bool = False, **_) -> torch.Tensor:
    """Cross-entropy on the final exit only — backbone-only training (no EE supervision)."""
    return _per_exit_ce([per_exit_logits[-1]], labels, regression).sum()


def branchynet_loss(per_exit_logits, labels, *, regression: bool = False, alpha: float = 0.7, **_) -> torch.Tensor:
    """Depth-decaying weights w_i ∝ alpha**i (earlier exits weighted more), normalized."""
    n = len(per_exit_logits)
    weights = torch.tensor([alpha ** i for i in range(n)], device=labels.device, dtype=torch.float)
    weights = weights / weights.sum()
    return (weights * _per_exit_ce(per_exit_logits, labels, regression)).sum()


def eenet_loss(
    per_exit_logits, labels, *,
    regression: bool = False,
    epoch_progress: float = 0.0,
    kl_T: float = 3.0,
    kl_alpha: float = 0.01,
    kl_warmup_frac: float = 0.75,
    **_,
) -> torch.Tensor:
    """EENet (Ilhan et al., github.com/git-disl/EENet).

    L = sum_j (j+1) / (K*(K+1)) * CE(z_j, y)
        + [epoch_progress > kl_warmup_frac] * sum_{j<K-1} kl_alpha * T^2 * KL(p_j^T || p_final^T)

    where p_x^T = softmax(z_x / T) and T = 3, kl_alpha = 0.01, warmup at 75% of training, all
    per the Ilhan EENet train.py. KL uses default reduction='mean' to match Ilhan (averages
    over batch * num_classes), with T^2 compensating for the temperature. No detach on the
    teacher (matches Ilhan; final-exit gradient flows through both CE and KL).
    """
    n = len(per_exit_logits)
    weights = torch.arange(1, n + 1, device=labels.device, dtype=torch.float) / (n * (n + 1))
    ce_term = (weights * _per_exit_ce(per_exit_logits, labels, regression)).sum()

    if regression or n < 2 or epoch_progress <= kl_warmup_frac:
        return ce_term

    final_logits = per_exit_logits[-1]
    teacher_probs = F.softmax(final_logits / kl_T, dim=-1)
    kl_total = ce_term.new_zeros(())
    for j in range(n - 1):
        student_log_p = F.log_softmax(per_exit_logits[j] / kl_T, dim=-1)
        kl_total = kl_total + F.kl_div(student_log_p, teacher_probs, reduction="mean") * kl_alpha * (kl_T ** 2)
    return ce_term + kl_total


def td_loss(
    per_exit_logits, labels, *, regression: bool = False, td_lambda: float = 0.0, **_
) -> torch.Tensor:
    """Temporal-difference TD(lambda) loss (Iuzzolino, Mozer & Bengio, NeurIPS 2021).

    Faithful depth-multi-exit adaptation of their cascaded-readout objective. The
    target at exit t is the lambda-weighted bootstrap over LATER exits' own
    (detached) predictions plus a true-label tail:

        y_t = (1 - lambda) * sum_{i>=1} lambda^{i-1} * yhat_{t+i}  +  lambda^{T-t} * y_true

    With lambda=0 (TD(0), the paper's canonical default) this reduces to
    y_t = detach(softmax(z_{t+1})): each exit chases the next exit's prediction, and
    the final exit trains on the ground-truth label. The stop-gradient on the target
    is load-bearing per the paper. lambda=1 recovers the all-exits-on-labels form the
    paper shows is worst.

    NOTE (2026-07-06): the previous implementation here was final-exit KD
    (alpha*CE(final) + KL(final||early)), which is NOT the cited method and gave
    early exits no ground-truth signal at all — the likely cause of the GSC v2
    collapse. Rewritten to the faithful bootstrapped-target form.
    """
    K = len(per_exit_logits)
    final = per_exit_logits[-1]
    ce_final = _ce_or_mse(final, labels, regression)
    if K < 2:
        return ce_final
    if regression:
        total = ce_final
        for t in range(K - 1):
            target = per_exit_logits[t + 1].detach().squeeze(-1)
            total = total + F.mse_loss(per_exit_logits[t].squeeze(-1), target)
        return total / K

    lam = float(td_lambda)
    with torch.no_grad():
        probs = [F.softmax(z, dim=-1) for z in per_exit_logits]
        onehot = F.one_hot(labels, num_classes=final.size(-1)).float()
    total = ce_final
    for t in range(K - 1):
        if lam == 0.0:
            target = probs[t + 1]
        else:
            target = torch.zeros_like(probs[t + 1])
            for i in range(1, K - t):
                target = target + (1.0 - lam) * (lam ** (i - 1)) * probs[t + i]
            target = target + (lam ** (K - 1 - t)) * onehot
            target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        log_student = F.log_softmax(per_exit_logits[t], dim=-1)
        total = total + (-(target * log_student).sum(dim=-1)).mean()
    return total / K


def boostnet_loss(
    per_exit_logits, labels, *, regression: bool = False, boost_t: float = 0.5, **_
) -> torch.Tensor:
    """BoostNet additive-logit ensemble training (Yu et al. AAAI 2023).

    The paper's recursion is F_n = t * F_{n-1} + f_n on PRE-softmax logits, with the
    prior ensemble F_{n-1} detached inside head n's CE so f_n is forced to fit the
    residual error of the frozen ensemble. The reweighting temperature t is the
    paper's most sensitive hyperparameter (their ablation: t=1 -> deep classifiers
    barely learn because too little high-loss data survives the ensemble shift;
    t=0 -> collapses to plain per-exit CE). t=0.5 is the paper default.

    NOTE (2026-07-06): the previous implementation summed an UNDECAYED carry
    (equivalent to t=1, the setting the authors document as degenerate) — the likely
    cause of the deep-exit under-training and seed instability on small cells.
    Rewritten to the paper's recursive t-decayed form. Their gradient rescaling
    (1/(N-n+1) into shared blocks) needs per-block hooks and is NOT implemented;
    the t-decay is the dominant stabilizer per their ablation.
    """
    t = float(boost_t)
    ensemble = None
    total = per_exit_logits[0].new_zeros(())
    for h_k in per_exit_logits:
        z_k = h_k if ensemble is None else h_k + t * ensemble.detach()
        total = total + _ce_or_mse(z_k, labels, regression)
        ensemble = z_k
    return total


def ztw_cascade_loss(
    per_exit_logits, labels, *, regression: bool = False, kd_lambda: float = 0.5, **_
) -> torch.Tensor:
    """ZTW cascade-distillation (distill_next variant; Wolczyk et al. NeurIPS 2021).

    L = sum_i w_i * CE(z_i, y) + (kd_lambda / (E-1)) * sum_{i=0..E-2} CE_soft(z_i, p_{i+1}.detach())
    where w_i are GPF-growing weights (i + 1) / sum(j + 1), CE_soft is soft-target CE with no
    temperature, and the teacher (the immediately-deeper exit) is detached. KD direction is
    reversed vs td_loss / byot — students learn from a slightly-more-mature neighbour rather
    than from the deepest head, the structurally distinct mechanism this baseline tests.
    """
    n = len(per_exit_logits)
    base_ces = _per_exit_ce(per_exit_logits, labels, regression)
    weights = torch.arange(1, n + 1, device=labels.device, dtype=torch.float)
    weights = weights / weights.sum()
    ce_total = (weights * base_ces).sum()
    if regression or n < 2:
        return ce_total
    aux_terms = []
    for i in range(n - 1):
        teacher = F.softmax(per_exit_logits[i + 1].detach(), dim=-1)
        student_logp = F.log_softmax(per_exit_logits[i], dim=-1)
        aux_terms.append(-(teacher * student_logp).sum(dim=-1).mean())
    aux = torch.stack(aux_terms).mean()
    return ce_total + kd_lambda * aux


def _gated_ensemble_teacher_logits(per_exit_logits, gate: str = "uniform") -> torch.Tensor:
    """Return the ensemble teacher's logits: g_e-weighted sum of per-exit logits.

    The v4 sweep proposed g = softmax(theta) with learnable theta for CWET / RouteConsensus.
    We use uniform gates (g_e = 1/K), the default-equivalent at initialization.
    Learnable gates are a possible extension.
    """
    stacked = torch.stack(list(per_exit_logits), dim=0)  # (K, B, C)
    if gate == "uniform":
        return stacked.mean(dim=0)
    raise ValueError(f"Unknown gate '{gate}'.")


def cwet_loss(
    per_exit_logits, labels, *,
    regression: bool = False,
    distill_T: float = 2.0,
    lambda_aux: float = 1.0,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """CWET (Consensus-Weighted Ensemble-Teacher distillation; v4 candidate C1).

    L = sum_e w_e * CE(p_e, y) + lambda_aux * sum_e T^2 * KL( stop_grad(p_ens^T) || p_e^T )

    where p_ens = softmax(sum_e g_e * z_e) is a gated ensemble teacher (uniform gates here;
    learnable g = softmax(theta) is a possible extension). Extends BEEM by replacing the
    KL-to-FINAL teacher with a KL-to-gated-ensemble teacher built from ALL exits — the cheap
    early exits often out-predict the final classifier on easy samples, so the native
    ensemble is a stronger teacher than the final logits alone. Source: Lan, Zhu & Gong
    (NeurIPS 2018, ONE / gated native-ensemble teacher); Zhang et al. CVPR 2018 (DML); BEEM
    i-weighting (Bajpai & Hanawal, ICLR 2025).
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()

    weights = exit_weights(n, weight_schedule, device=labels.device)
    per_exit_ce = _per_exit_ce(per_exit_logits, labels, regression=False)
    ce_total = (weights * per_exit_ce).sum()

    ens_logits = _gated_ensemble_teacher_logits(per_exit_logits, gate="uniform")
    teacher_log_p = F.log_softmax(ens_logits / distill_T, dim=-1).detach()
    teacher_p = teacher_log_p.exp()
    aux_terms = []
    for z_e in per_exit_logits:
        student_log_p = F.log_softmax(z_e / distill_T, dim=-1)
        aux_terms.append(
            F.kl_div(student_log_p, teacher_p, reduction="batchmean") * (distill_T ** 2)
        )
    aux = torch.stack(aux_terms).sum()
    return ce_total + lambda_aux * aux


def route_consensus_loss(
    per_exit_logits, labels, *,
    regression: bool = False,
    distill_T: float = 2.0,
    lambda_aux: float = 1.0,
    lambda_cost: float = 0.1,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """RouteConsensus (v4 candidate C6; the headline EMAR bet).

    L = sum_e w_e * CE(p_e, y)
        + lambda_aux * sum_e T^2 * KL( stop_grad(p_ens^T) || p_e^T )    [CWET / consensus term]
        + lambda_cost * sum_e c_e * (1 - s_e)                            [bounded routing-cost term]

    where p_ens is the gated ensemble teacher (uniform gates this pass), c_e is the normalized
    cumulative cost at exit e (approximated as (e+1)/K), and s_e is the batch-mean TARGET-CLASS
    probability at exit e (i.e., p_e[i, y_i] averaged over the batch). The first pass
    used max-softmax confidence for s_e, which collapsed the model to ~1 % accuracy because it
    rewards confidence on whatever class the model is currently predicting -- regardless of
    correctness -- and at random init the model's argmax is the wrong class. Switching to
    p_e[y] makes the cost gradient always push the correct class's probability up, which is
    the calibrated-confidence behaviour the sweep's "calibrated exit-confidence surrogate"
    phrase implied. The cost term remains bounded (c_e in [0,1], (1 - s_e) in [0,1]).
    Source: BEEM i-weighting (Bajpai & Hanawal, ICLR 2025); Lan et al. NeurIPS 2018 ensemble
    teacher; JEI-DNN cost-awareness contrast (Regol et al. ICLR 2024).
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()

    weights = exit_weights(n, weight_schedule, device=labels.device)
    per_exit_ce = _per_exit_ce(per_exit_logits, labels, regression=False)
    ce_total = (weights * per_exit_ce).sum()

    ens_logits = _gated_ensemble_teacher_logits(per_exit_logits, gate="uniform")
    teacher_log_p = F.log_softmax(ens_logits / distill_T, dim=-1).detach()
    teacher_p = teacher_log_p.exp()
    aux_terms = []
    for z_e in per_exit_logits:
        student_log_p = F.log_softmax(z_e / distill_T, dim=-1)
        aux_terms.append(
            F.kl_div(student_log_p, teacher_p, reduction="batchmean") * (distill_T ** 2)
        )
    aux = torch.stack(aux_terms).sum()

    # Routing-cost term: target-class probability, batch-meaned. Bounded in [0, c_e].
    cost_terms = []
    for e, z_e in enumerate(per_exit_logits):
        c_e = float(e + 1) / n
        probs = F.softmax(z_e, dim=-1)
        s_e = probs.gather(1, labels.unsqueeze(1)).squeeze(1).mean()  # mean target-class prob
        cost_terms.append(c_e * (1.0 - s_e))
    cost = torch.stack(cost_terms).sum()

    return ce_total + lambda_aux * aux + lambda_cost * cost


def sat_consensus_loss(
    per_exit_logits, labels, *,
    regression: bool = False,
    alpha: float = 0.9,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """SAT-Consensus (v4 candidate C3; the headline AURC bet).

    Faithful spec is a per-sample per-exit soft target t_e ← alpha * t_e + (1 - alpha) * p_ens
    with a 10-epoch warm-up on hard labels (Huang et al., NeurIPS 2020 Self-Adaptive Training).

    This first-pass impl is loss-only and stateless: the soft target is built from the CURRENT
    batch's ensemble teacher (alpha * one_hot(y) + (1 - alpha) * stop_grad(p_ens)). It keeps
    the consensus-distillation half (the part that matters for AURC ranking per Kubaty 2025)
    but discards the cross-epoch refurbishment half. Plumbing the per-sample EMA bank
    (~140 MB on CIFAR-100 / 7 exits) is a possible extension once evidence supports the
    direction. Per the sweep: no label smoothing; the soft target is data-driven, not a
    uniform prior, so it does NOT incur the Xia-2025 selective-classification penalty.
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()

    weights = exit_weights(n, weight_schedule, device=labels.device)
    num_classes = per_exit_logits[0].size(-1)
    onehot = F.one_hot(labels, num_classes=num_classes).float()
    ens_logits = _gated_ensemble_teacher_logits(per_exit_logits, gate="uniform")
    ens_probs = F.softmax(ens_logits, dim=-1).detach()
    soft_target = alpha * onehot + (1.0 - alpha) * ens_probs

    terms = []
    for z_e in per_exit_logits:
        log_p = F.log_softmax(z_e, dim=-1)
        # CE against soft target: -sum_c t_c * log p_c.
        terms.append(-(soft_target * log_p).sum(dim=-1).mean())
    return (weights * torch.stack(terms)).sum()


def anytime_stable_loss(
    per_exit_logits, labels, *,
    regression: bool = False,
    tau: float = 1.0,
    eta: float = 0.5,
    alpha_min: float = 0.05,
    **_,
) -> torch.Tensor:
    """AnytimeStable (v4 candidate C2; PFW-collapse-immune probe).

    alpha_e = softmax_e((acc_e - eta * c_e) / tau), then floored at alpha_min and renormalised.
    L = sum_e alpha_e * CE(p_e, y).

    Faithful spec uses acc_e^EMA across batches; this first-pass impl uses the CURRENT batch's
    per-exit accuracy (a simplification, documented). c_e is the uniform-proxy cumulative cost
    (e+1)/K, same as RouteConsensus. The alpha_min floor + renormalisation is the bounded,
    non-degenerate replacement for PFW's max/Tchebycheff surrogate (no exit's gradient can
    vanish).
    """
    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()

    # Per-batch per-exit accuracy (stateless proxy for the spec's EMA accuracy).
    accs = []
    for z_e in per_exit_logits:
        with torch.no_grad():
            preds = z_e.argmax(dim=-1)
            accs.append((preds == labels).float().mean())
    acc_vec = torch.stack(accs)  # (n,)
    cost = torch.tensor(
        [(e + 1) / n for e in range(n)], device=labels.device, dtype=acc_vec.dtype,
    )
    raw = (acc_vec - eta * cost) / tau
    alpha = F.softmax(raw, dim=0)
    alpha = torch.clamp(alpha, min=alpha_min)
    alpha = alpha / alpha.sum()
    # Detach alpha so gradient flows only through the per-exit CE terms (the weights are derived
    # from non-differentiable argmax-accuracy anyway).
    alpha = alpha.detach()

    per_exit_ce = _per_exit_ce(per_exit_logits, labels, regression=False)
    return (alpha * per_exit_ce).sum()


def anytime_stable_distill_loss(
    per_exit_logits, labels, *,
    regression: bool = False,
    tau: float = 1.0,
    eta: float = 0.5,
    alpha_min: float = 0.05,
    gamma: float = 0.5,
    distill_T: float = 1.0,
    **_,
) -> torch.Tensor:
    """AnytimeStable-Distill: cost-aware adaptive weights + alpha-modulated KL to final exit.

    Per exit e (for e < K-1):
        L_e = CE(z_e, y) + gamma * T^2 * KL(softmax(z_e/T) || softmax(z_final/T).detach())
    Final exit:
        L_{K-1} = CE(z_{K-1}, y)
    Total:
        L = sum_e alpha_e * L_e

    alpha_e is computed exactly as in anytime_stable_loss: softmax((acc_e - eta * c_e) / tau)
    with a renormalised alpha_min floor. The distillation term is novel: BEEM uses a uniform
    KL-to-final pattern with linearly-increasing weights, AdaLoss has no KL, AnytimeStable
    has no distillation. Combining alpha-modulated KL with the cost-aware floor gives the
    AURC win (from the KL teacher) without sacrificing the EMAR / MAC-saved wins (from the
    cost-aware alpha).
    """
    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()

    accs = []
    for z_e in per_exit_logits:
        with torch.no_grad():
            preds = z_e.argmax(dim=-1)
            accs.append((preds == labels).float().mean())
    acc_vec = torch.stack(accs)
    cost = torch.tensor(
        [(e + 1) / n for e in range(n)], device=labels.device, dtype=acc_vec.dtype,
    )
    raw = (acc_vec - eta * cost) / tau
    alpha = F.softmax(raw, dim=0)
    alpha = torch.clamp(alpha, min=alpha_min)
    alpha = (alpha / alpha.sum()).detach()

    final_logits = per_exit_logits[-1]
    teacher_log_probs = F.log_softmax(final_logits.detach() / distill_T, dim=-1)
    teacher_probs = teacher_log_probs.exp()

    per_exit_loss = []
    for e, z_e in enumerate(per_exit_logits):
        ce = F.cross_entropy(z_e, labels)
        if e == n - 1:
            per_exit_loss.append(ce)
            continue
        student_log_probs = F.log_softmax(z_e / distill_T, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        per_exit_loss.append(ce + gamma * (distill_T ** 2) * kl)
    stacked = torch.stack(per_exit_loss)
    return (alpha * stacked).sum()


def jei_dnn_loss(
    per_exit_logits, per_exit_gate_logits, labels, *,
    cost_lambda: float = 0.1,
    per_exit_costs=None,
    regression: bool = False,
    epoch_progress: float = 1.0,
    warmup_frac: float = 0.2,
    **_,
) -> torch.Tensor:
    """JEI-DNN joint loss (Regol et al., ICLR 2024).

    Warm-up phase (paper's Phase 1): for the first ``warmup_frac`` of training the
    IMs are trained on plain summed CE with the gates untouched. The paper searches
    the warm-up length and describes it as necessary because gate targets computed
    from unconverged IMs destabilize training; our previous implementation trained
    gates jointly from epoch 0 (the configuration the authors explicitly call
    unstable), which is the likely cause of the seed collapses on small cells.
    After warm-up, the joint routing-weighted objective below applies.

    L = sum_i pi_i * CE(z_i, y) + cost_lambda * sum_i pi_i * c_i

    pi_0 = sigmoid(g_0)
    pi_i = sigmoid(g_i) * prod_{j<i} (1 - sigmoid(g_j))   for 0 < i < K-1
    pi_{K-1} = prod_{j<K-1} (1 - sigmoid(g_j))            (residual: final exit has no gate)

    Per-sample routing probabilities sum to 1. CE_i is per-sample cross-entropy at exit i;
    c_i is the cumulative compute cost at exit i in [0, 1]; cost_lambda controls the
    accuracy-cost trade-off (smaller -> deeper exits dominate, larger -> earlier exits favored).
    """
    K = len(per_exit_logits)
    if K < 2 or regression:
        return F.cross_entropy(per_exit_logits[0], labels) if not regression else F.mse_loss(per_exit_logits[0].squeeze(-1), labels.float())
    if len(per_exit_gate_logits) != K - 1:
        raise ValueError(f"Expected {K-1} gate logits for {K} exits, got {len(per_exit_gate_logits)}.")
    if per_exit_costs is None:
        per_exit_costs = [float(i + 1) / K for i in range(K)]

    if epoch_progress < warmup_frac:
        # Phase 1 warm-up: IMs only, uniform CE; gates receive no gradient.
        return torch.stack([F.cross_entropy(z, labels) for z in per_exit_logits]).mean()

    gates = [torch.sigmoid(g) for g in per_exit_gate_logits]
    pi_list = []
    not_taken = torch.ones_like(gates[0])
    for i in range(K - 1):
        pi_list.append(gates[i] * not_taken)
        not_taken = not_taken * (1.0 - gates[i])
    pi_list.append(not_taken)

    classification = per_exit_logits[0].new_zeros(())
    cost = per_exit_logits[0].new_zeros(())
    for i in range(K):
        per_sample_ce = F.cross_entropy(per_exit_logits[i], labels, reduction="none")
        classification = classification + (pi_list[i] * per_sample_ce).mean()
        cost = cost + (pi_list[i] * float(per_exit_costs[i])).mean()
    return classification + cost_lambda * cost


def poe_anneal_loss(
    per_exit_logits, labels, *,
    alphas,
    epoch_progress: float = 0.0,
    regression: bool = False,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """PoE-Anneal loss (v5 candidate C3; EMAR axis, exit-policy regularization).

    Cumulative product-of-experts prediction with an entropy-monotonicity hinge, per the
    v5 brief (Jazbec et al. NeurIPS 2023 + ZTW geometric ensembling + Sensoy-style
    annealing). For J exits with logits z_j, define the cumulative PoE in log space:

      log_ptilde_j(x) = log_softmax( sum_{l≤j} alpha_l * log p_l(x) )

    where alpha_l are the per-exit learnable scalars (PoEStateModule, initialised to 1).
    The training objective is

      L_poe   = sum_j w_j NLL(log_ptilde_j, y)
      L_mono  = sum_{j>=1} max(0, H(ptilde_j) - H(ptilde_{j-1}))
      L_PoE   = L_poe + rho(t) L_mono

    with rho(t) = rho_max * min(epoch_progress / anneal_fraction, 1). Default rho_max=0.5,
    anneal_fraction=0.4 -- rho ramps from 0 to 0.5 over the first 40 % of epochs, after
    which the monotonicity penalty is at full strength. The CE-only warm-up of the first
    epochs lets the classifier converge before the geometric ensemble is constrained.

    The inference-time prediction is argmax(ptilde_j) (cumulative PoE), implemented by
    the "poe_entropy" cutoff in jolt/calibration.py and jolt/inference.py.

    Citations: Jazbec, Allingham, Zhang & Nalisnick NeurIPS 2023 (conditional monotonicity);
    Wolczyk et al. NeurIPS 2021 (ZTW geometric ensembling); Sensoy et al. NeurIPS 2018
    (annealing schedule form).
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if alphas is None or alphas.numel() != n:
        raise ValueError(
            f"PoE-Anneal needs alphas of length {n}; got "
            f"{None if alphas is None else alphas.numel()}."
        )

    device = labels.device
    # Per-exit log_softmax (with grad)
    log_probs = [F.log_softmax(z, dim=-1) for z in per_exit_logits]  # list of [B, C]

    # Cumulative log_ptilde_j = log_softmax(sum_{l<=j} alpha_l * log_probs_l)
    running_sum = torch.zeros_like(log_probs[0])
    log_ptilde_list = []
    entropy_list = []
    for j in range(n):
        running_sum = running_sum + alphas[j] * log_probs[j]
        log_ptilde_j = F.log_softmax(running_sum, dim=-1)  # renormalize
        log_ptilde_list.append(log_ptilde_j)
        # Entropy H(ptilde_j) = -sum_c p_c log p_c
        ptilde_j = log_ptilde_j.exp()
        entropy_list.append(-(ptilde_j * log_ptilde_j).sum(dim=-1).mean())

    # L_poe = sum_j w_j NLL(log_ptilde_j, y)
    weights = exit_weights(n, weight_schedule, device=device)
    nll_per_exit = torch.stack([
        F.nll_loss(lp, labels, reduction="mean") for lp in log_ptilde_list
    ], dim=0)
    L_poe = (weights * nll_per_exit).sum()

    # L_mono: penalise entropy increases across consecutive exits
    if n > 1:
        ent = torch.stack(entropy_list, dim=0)  # [J]
        deltas = ent[1:] - ent[:-1]  # [J-1]; positive means entropy increased -> penalize
        L_mono = torch.clamp(deltas, min=0.0).sum()
    else:
        L_mono = labels.new_zeros((), dtype=torch.float32)

    rho_t = rho_max * min(epoch_progress / max(anneal_fraction, 1e-9), 1.0)
    return L_poe + rho_t * L_mono


def tri_axis_loss(
    per_exit_logits, per_exit_confidence_logits, labels, *,
    per_exit_macs,
    regression: bool = False,
    epoch_progress: float = 0.0,
    beta_rank: float = 1.0,
    gamma_tcp: float = 0.5,
    T_rank: float = 0.1,
    lambda_mac: float = 0.15,
    T_gate: float = 0.05,
    tau: float = 0.5,
    rho_max: float = 0.3,
    anneal_fraction: float = 0.4,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """Tri-Axis loss (v6 candidate; SCAR + BudgetBoost + PoE-Anneal mechanisms composed).

    Built in response to the v5 finding that no single candidate is top-3 across all four
    Pareto axes (accuracy, EMAR, AURC, MAC saved) on every cell. Tri-Axis applies pressure
    on three axes simultaneously through a single scalar objective:

      L = sum_j w_j CE(z_j, y)                                             [accuracy/AdaLoss anchor]
          + beta_rank * sum_j w_j L_rank_j  (SCAR structure-aware rank)    [AURC axis]
          + gamma_tcp * sum_j w_j (s_j - p_true_j)^2                       [AURC anchor / TCP regression]
          + lambda_mac * E[r_j * (m_j / m_J)]   (differentiable r through s_j)  [MAC axis]
          + rho(t) * sum_{j>=1} max(0, H(p_j) - H(p_{j-1}))                [EMAR monotonicity]

    Distinguishing design choices:

    * The gating responsibility r_j uses ``s_j = sigmoid(confidence_head_j)`` rather than
      max-softmax, so the SCAR confidence head simultaneously serves AURC ranking AND
      MAC-aware routing. Gradient flows through r_j, making the MAC term an active lever
      on s_j (not the inert measurement BudgetBoost's detached version produces).
    * No PoEStateModule -- the monotonicity penalty is computed on raw per-exit softmax,
      reducing infrastructure overhead. The penalty is annealed via the same rho(t)
      schedule used by PoE-Anneal.
    * Inference routing uses ``learned_confidence`` (auto-overridden in evaluate_curve when
      method == 'tri_axis'); s_j is the routing signal.

    Defaults are softer than the standalone candidates to reflect the joint pressure:
    lambda_mac=0.15 (vs BudgetBoost's 0.3), rho_max=0.3 (vs PoE-Anneal's 0.5).

    Per-component citations: Kubaty et al. arXiv:2508.21495 (rank-AURC link); Franc et al.
    JMLR 2023 (SELE rank surrogate); Corbiere et al. NeurIPS 2019 (TCP target); Huang et al.
    ICLR 2018 (MSDNet budget); Jazbec et al. NeurIPS 2023 (monotonicity rationale); Sensoy
    et al. NeurIPS 2018 (annealing schedule form).
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if len(per_exit_confidence_logits) != n:
        raise ValueError(
            f"tri_axis needs {n} confidence logit tensors; got {len(per_exit_confidence_logits)}."
        )
    if per_exit_macs is None or len(per_exit_macs) != n:
        raise ValueError(
            f"tri_axis needs per_exit_macs of length {n}; got "
            f"{None if per_exit_macs is None else len(per_exit_macs)}."
        )

    device = labels.device
    weights = exit_weights(n, weight_schedule, device=device)
    mac_tensor = torch.as_tensor(per_exit_macs, dtype=torch.float32, device=device)
    mac_norm = mac_tensor / mac_tensor[-1].clamp_min(1.0)

    # Stack everything for vectorised computation.
    s_logits = torch.stack(per_exit_confidence_logits, dim=0)  # [J, B]
    s = torch.sigmoid(s_logits)  # [J, B]
    log_probs_stack = torch.stack(
        [F.log_softmax(z, dim=-1) for z in per_exit_logits], dim=0
    )  # [J, B, C]
    probs_stack = log_probs_stack.exp()
    per_sample_ce = -log_probs_stack.gather(2, labels.view(1, -1, 1).expand(n, -1, 1)).squeeze(-1)  # [J, B]

    # Hard correctness for structure-aware mask (detached).
    with torch.no_grad():
        preds = probs_stack.argmax(dim=-1)  # [J, B]
        correct = (preds == labels.unsqueeze(0)).float()  # [J, B]
        if n > 1:
            deeper_sum = correct.flip(0).cumsum(dim=0).flip(0) - correct
        else:
            deeper_sum = torch.zeros_like(correct)
        no_deeper_correct = (deeper_sum == 0).float()
        ybar = torch.clamp(correct + (1.0 - correct) * no_deeper_correct, max=1.0)  # [J, B]
        # TCP target (true-class probability per exit).
        tcp_target = probs_stack.gather(2, labels.view(1, -1, 1).expand(n, -1, 1)).squeeze(-1)  # [J, B]

    # SCAR rank surrogate per exit (batch-local pairwise sigmoid kernel).
    L_rank_per_exit = []
    for j in range(n):
        pos = ybar[j]
        neg = 1.0 - pos
        n_pos = pos.sum().clamp_min(1.0)
        n_neg = neg.sum().clamp_min(1.0)
        diff = s_logits[j].unsqueeze(0) - s_logits[j].unsqueeze(1)  # [B, B]
        kernel = torch.sigmoid(diff / T_rank)
        mask = pos.unsqueeze(1) * neg.unsqueeze(0)
        L_rank_per_exit.append((kernel * mask).sum() / (n_pos * n_neg))
    L_rank = torch.stack(L_rank_per_exit, dim=0)  # [J]
    L_tcp = ((s - tcp_target) ** 2).mean(dim=1)  # [J]
    per_exit_ce_mean = per_sample_ce.mean(dim=1)  # [J]

    accuracy_term = (weights * (per_exit_ce_mean + beta_rank * L_rank + gamma_tcp * L_tcp)).sum()

    # BudgetBoost-style differentiable responsibility using SCAR's s as the gate signal.
    # The standard chain identity: r_j = sigmoid_gate_j * prod_{l<j} (1 - sigmoid_gate_l);
    # the final exit's responsibility is the residual prod_{l<J-1} (1 - g_l).
    gates = torch.sigmoid((s - tau) / T_gate)  # [J, B]
    early_gates = gates[:-1]
    one_minus = 1.0 - early_gates + 1e-9
    cum_one_minus = torch.cumprod(one_minus, dim=0)
    prev_survival = torch.cat([
        torch.ones_like(early_gates[:1]),
        cum_one_minus[:-1],
    ], dim=0)  # [J-1, B]
    r_early = early_gates * prev_survival  # [J-1, B]
    r_final = cum_one_minus[-1:].clone()
    responsibility = torch.cat([r_early, r_final], dim=0)  # [J, B], sums to ~1
    responsibility = responsibility / responsibility.sum(dim=0, keepdim=True).clamp_min(1e-6)
    L_mac = (responsibility * mac_norm.view(-1, 1)).sum(dim=0).mean()

    # PoE-Anneal-style entropy-monotonicity penalty on raw per-exit softmax (no PoE state).
    entropies = -(probs_stack * log_probs_stack).sum(dim=-1).mean(dim=1)  # [J]
    if n > 1:
        deltas = entropies[1:] - entropies[:-1]
        L_mono = torch.clamp(deltas, min=0.0).sum()
    else:
        L_mono = labels.new_zeros((), dtype=torch.float32)

    rho_t = rho_max * min(epoch_progress / max(anneal_fraction, 1e-9), 1.0)
    return accuracy_term + lambda_mac * L_mac + rho_t * L_mono


def _poe_distill_core(per_exit_logits, labels, alphas, gamma_distill, distill_T):
    """Shared core for poe_distill family: build log_ptilde_list and the KL teacher term."""
    n = len(per_exit_logits)
    log_probs = [F.log_softmax(z, dim=-1) for z in per_exit_logits]
    running_sum = torch.zeros_like(log_probs[0])
    log_ptilde_list = []
    entropy_list = []
    for j in range(n):
        running_sum = running_sum + alphas[j] * log_probs[j]
        log_ptilde_j = F.log_softmax(running_sum, dim=-1)
        log_ptilde_list.append(log_ptilde_j)
        ptilde_j = log_ptilde_j.exp()
        entropy_list.append(-(ptilde_j * log_ptilde_j).sum(dim=-1).mean())
    # KL teacher (final exit -> early exits)
    final_logits = per_exit_logits[-1]
    teacher_log_probs = F.log_softmax(final_logits.detach() / distill_T, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    distill_per_exit = []
    for j in range(n - 1):
        student_log_probs = F.log_softmax(log_ptilde_list[j] / distill_T, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        distill_per_exit.append(kl)
    return log_ptilde_list, entropy_list, distill_per_exit


def poe_distill_mtl_loss(
    per_exit_logits, labels, *,
    alphas, multitask,
    epoch_progress: float = 0.0,
    regression: bool = False,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    lambda_ce_final: float = 0.0,
    **_,
) -> torch.Tensor:
    """poe_distill + Kendall MultiTaskLoss adaptive eta (replaces fixed exit_weights)."""
    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if multitask is None:
        raise ValueError("poe_distill_mtl needs a MultiTaskLoss instance.")
    log_ptilde_list, entropy_list, distill_per_exit = _poe_distill_core(
        per_exit_logits, labels, alphas, gamma_distill, distill_T,
    )
    # Per-exit NLL with adaptive MTL weighting (instead of fixed exit_weights)
    per_exit_nll = [F.nll_loss(lp, labels, reduction="mean") for lp in log_ptilde_list]
    _, L_poe = multitask(per_exit_nll)
    # KL distill weighted by the same MTL etas (drop the final-exit weight; J-1 distills)
    if distill_per_exit:
        # The MTL output is a scalar sum; we just use the same weighted-sum convention with
        # uniform weighting over the J-1 distill terms.
        L_distill = torch.stack(distill_per_exit).mean() * (distill_T ** 2)
    else:
        L_distill = labels.new_zeros((), dtype=torch.float32)
    if n > 1:
        ent = torch.stack(entropy_list, dim=0)
        L_mono = torch.clamp(ent[1:] - ent[:-1], min=0.0).sum()
    else:
        L_mono = labels.new_zeros((), dtype=torch.float32)
    rho_t = rho_max * min(epoch_progress / max(anneal_fraction, 1e-9), 1.0)
    if lambda_ce_final > 0.0:
        L_ce_final = F.cross_entropy(per_exit_logits[-1], labels, reduction="mean")
        return L_poe + rho_t * L_mono + gamma_distill * L_distill + lambda_ce_final * L_ce_final
    return L_poe + rho_t * L_mono + gamma_distill * L_distill


def poe_distill_brier_loss(
    per_exit_logits, labels, *,
    alphas,
    epoch_progress: float = 0.0,
    regression: bool = False,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    lambda_brier: float = 0.2,
    weight_schedule: str = "increasing",
    lambda_ce_final: float = 0.0,
    **_,
) -> torch.Tensor:
    """poe_distill + Brier-score anchor on per-exit raw softmax (calibration axis)."""
    from .losses_zoo import exit_weights
    base = poe_distill_loss(
        per_exit_logits, labels, alphas=alphas, epoch_progress=epoch_progress,
        regression=regression, rho_max=rho_max, anneal_fraction=anneal_fraction,
        gamma_distill=gamma_distill, distill_T=distill_T, weight_schedule=weight_schedule,
    )
    if regression or len(per_exit_logits) < 2:
        return base
    weights = exit_weights(len(per_exit_logits), weight_schedule, device=labels.device)
    if lambda_ce_final > 0.0:
        L_ce_final = F.cross_entropy(per_exit_logits[-1], labels, reduction="mean")
        return base + lambda_brier * _brier_anchor(per_exit_logits, labels, weights) + lambda_ce_final * L_ce_final
    return base + lambda_brier * _brier_anchor(per_exit_logits, labels, weights)


def poe_distill_asym_loss(
    per_exit_logits, labels, *,
    alphas,
    epoch_progress: float = 0.0,
    regression: bool = False,
    kappa: float = 2.0,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """poe_distill + asymmetric κ-CE on cumulative ptilde (errors penalised κ× more)."""
    from .losses_zoo import exit_weights
    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    log_ptilde_list, entropy_list, distill_per_exit = _poe_distill_core(
        per_exit_logits, labels, alphas, gamma_distill, distill_T,
    )
    weights = exit_weights(n, weight_schedule, device=labels.device)
    # Asymmetric per-exit NLL: weight errors κ× more than corrects (sample-level mask).
    asym_nll_per_exit = []
    for j, log_p in enumerate(log_ptilde_list):
        per_sample_nll = F.nll_loss(log_p, labels, reduction="none")
        with torch.no_grad():
            preds = log_p.argmax(dim=-1)
            sample_w = (preds == labels).float() + kappa * (preds != labels).float()
        asym_nll_per_exit.append((sample_w * per_sample_nll).mean())
    L_poe = (weights * torch.stack(asym_nll_per_exit, dim=0)).sum()
    if distill_per_exit:
        L_distill = (weights[:-1] * torch.stack(distill_per_exit)).sum() * (distill_T ** 2)
    else:
        L_distill = labels.new_zeros((), dtype=torch.float32)
    if n > 1:
        ent = torch.stack(entropy_list, dim=0)
        L_mono = torch.clamp(ent[1:] - ent[:-1], min=0.0).sum()
    else:
        L_mono = labels.new_zeros((), dtype=torch.float32)
    rho_t = rho_max * min(epoch_progress / max(anneal_fraction, 1e-9), 1.0)
    return L_poe + rho_t * L_mono + gamma_distill * L_distill


def poe_distill_anytime_loss(
    per_exit_logits, labels, *,
    alphas,
    epoch_progress: float = 0.0,
    regression: bool = False,
    tau: float = 1.0,
    eta_cost: float = 0.5,
    alpha_min: float = 0.05,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    **_,
) -> torch.Tensor:
    """poe_distill + AnytimeStable cost-aware adaptive weights (replaces exit_weights)."""
    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    log_ptilde_list, entropy_list, distill_per_exit = _poe_distill_core(
        per_exit_logits, labels, alphas, gamma_distill, distill_T,
    )
    device = labels.device
    # AnytimeStable cost-aware weights, no grad on weights.
    with torch.no_grad():
        accs = []
        for log_p in log_ptilde_list:
            preds = log_p.argmax(dim=-1)
            accs.append((preds == labels).float().mean())
        acc_vec = torch.stack(accs)
        cost = torch.tensor(
            [(e + 1) / n for e in range(n)], device=device, dtype=acc_vec.dtype,
        )
        raw = (acc_vec - eta_cost * cost) / tau
        weights = F.softmax(raw, dim=0)
        weights = torch.clamp(weights, min=alpha_min)
        weights = (weights / weights.sum()).detach()
    nll_per_exit = torch.stack(
        [F.nll_loss(lp, labels, reduction="mean") for lp in log_ptilde_list], dim=0,
    )
    L_poe = (weights * nll_per_exit).sum()
    if distill_per_exit:
        L_distill = (weights[:-1] * torch.stack(distill_per_exit)).sum() * (distill_T ** 2)
    else:
        L_distill = labels.new_zeros((), dtype=torch.float32)
    if n > 1:
        ent = torch.stack(entropy_list, dim=0)
        L_mono = torch.clamp(ent[1:] - ent[:-1], min=0.0).sum()
    else:
        L_mono = labels.new_zeros((), dtype=torch.float32)
    rho_t = rho_max * min(epoch_progress / max(anneal_fraction, 1e-9), 1.0)
    return L_poe + rho_t * L_mono + gamma_distill * L_distill


def poe_distill_mtl_brier_loss(
    per_exit_logits, labels, *,
    alphas, multitask,
    epoch_progress: float = 0.0,
    regression: bool = False,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    lambda_brier: float = 0.2,
    weight_schedule: str = "increasing",
    lambda_ce_final: float = 0.0,
    **_,
) -> torch.Tensor:
    """poe_distill + MultiTaskLoss adaptive eta + Brier-score calibration anchor.

    Inlined (vs delegating to poe_distill_mtl_loss) so each component (L_poe, L_mono,
    L_distill, L_brier) is computed once and stashed in _LOSS_COMPONENTS for per-epoch
    diagnostic logging. The math is identical to the prior delegating form.
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if multitask is None:
        raise ValueError("poe_distill_mtl_brier needs a MultiTaskLoss instance.")

    log_ptilde_list, entropy_list, distill_per_exit = _poe_distill_core(
        per_exit_logits, labels, alphas, gamma_distill, distill_T,
    )
    per_exit_nll = [F.nll_loss(lp, labels, reduction="mean") for lp in log_ptilde_list]
    _, L_poe = multitask(per_exit_nll)

    if distill_per_exit:
        L_distill = torch.stack(distill_per_exit).mean() * (distill_T ** 2)
    else:
        L_distill = labels.new_zeros((), dtype=torch.float32)

    ent = torch.stack(entropy_list, dim=0)
    L_mono = torch.clamp(ent[1:] - ent[:-1], min=0.0).sum()
    rho_t = rho_max * min(epoch_progress / max(anneal_fraction, 1e-9), 1.0)

    weights = exit_weights(n, weight_schedule, device=labels.device)
    L_brier = _brier_anchor(per_exit_logits, labels, weights)

    _stash("L_poe", L_poe)
    _stash("L_mono", rho_t * L_mono)
    _stash("L_distill", gamma_distill * L_distill)
    _stash("L_brier", lambda_brier * L_brier)
    _LOSS_COMPONENTS["n"] += 1

    # Optional plain-CE anchor on the FINAL exit's own logits. The cumulative PoE
    # likelihood supervises the deep exit only through the ensemble product, which
    # under-serves it on many-class cells; a direct CE term restores label pressure
    # at full depth without touching the shallow exits.
    if lambda_ce_final > 0.0:
        L_ce_final = F.cross_entropy(per_exit_logits[-1], labels, reduction="mean")
        _stash("L_ce_final", lambda_ce_final * L_ce_final)
        return (L_poe + rho_t * L_mono + gamma_distill * L_distill
                + lambda_brier * L_brier + lambda_ce_final * L_ce_final)
    return L_poe + rho_t * L_mono + gamma_distill * L_distill + lambda_brier * L_brier


def distill_mtl_brier_loss(
    per_exit_logits, labels, *,
    multitask,
    epoch_progress: float = 0.0,
    regression: bool = False,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    lambda_brier: float = 0.2,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """Distill + MTL + Brier WITHOUT training-time PoE.

    Per-exit cross-entropy weighted by MultiTaskLoss adaptive eta, plus KL distillation
    from the final exit's softmax (BEEM/TD-style teacher, no cumulative log-product), plus
    a Brier calibration anchor. Same MTL + distill + Brier components as
    ``poe_distill_mtl_brier_loss`` but with the PoE-Anneal training-time machinery
    stripped: no learnable alphas, no cumulative log-products, no entropy-monotonicity
    hinge. Inference uses the standard entropy cutoff (NOT poe_entropy).

    Targets cells where training-time PoE-Anneal underperforms post-hoc PoE inference
    (e.g. CIFAR-100 MobileNetV2 lowwd, where bare poe_anneal trains to ~0.32 EMAR while
    summed-CE + post-hoc poe_jazbec inference hits ~0.53). Stripping the PoE training
    machinery should let the candidate match meronen-class accuracy with the extra MTL +
    distill + Brier mechanisms applied on top.
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if multitask is None:
        raise ValueError("distill_mtl_brier needs a MultiTaskLoss instance.")

    per_exit_ce = [F.cross_entropy(lg, labels, reduction="mean") for lg in per_exit_logits]
    _, L_ce = multitask(per_exit_ce)

    # KL distill: final exit (teacher, detached softmax at temperature T) -> each early
    # exit's log-softmax at temperature T. Standard TD/KD form (no cumulative ptilde).
    with torch.no_grad():
        teacher_log_prob = F.log_softmax(per_exit_logits[-1] / distill_T, dim=-1)
        teacher_prob = teacher_log_prob.exp()
    distill_per_exit = []
    for i in range(n - 1):
        student_log_prob = F.log_softmax(per_exit_logits[i] / distill_T, dim=-1)
        kl = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
        distill_per_exit.append(kl)
    if distill_per_exit:
        L_distill = torch.stack(distill_per_exit).mean() * (distill_T ** 2)
    else:
        L_distill = labels.new_zeros((), dtype=torch.float32)

    weights = exit_weights(n, weight_schedule, device=labels.device)
    L_brier = _brier_anchor(per_exit_logits, labels, weights)

    _stash("L_poe", L_ce)
    _stash("L_distill", gamma_distill * L_distill)
    _stash("L_brier", lambda_brier * L_brier)
    _LOSS_COMPONENTS["n"] += 1

    return L_ce + gamma_distill * L_distill + lambda_brier * L_brier


def poe_distill_mtl_asym_loss(
    per_exit_logits, labels, *,
    alphas, multitask,
    epoch_progress: float = 0.0,
    regression: bool = False,
    kappa: float = 2.0,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    **_,
) -> torch.Tensor:
    """poe_distill + MultiTaskLoss adaptive eta + asymmetric κ-CE on cumulative ptilde.

    Asymmetric per-exit NLL replaces the standard NLL before being fed into the MTL
    adaptive-weighting head: errors are penalised κ× more than corrects (sample-level
    mask, no gradient on the mask). The MTL etas adapt across exits as in poe_distill_mtl;
    the rest of the loss (PoE entropy monotonicity hinge, KL distill to final exit) is
    unchanged from poe_distill_mtl.
    """
    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if multitask is None:
        raise ValueError("poe_distill_mtl_asym needs a MultiTaskLoss instance.")
    log_ptilde_list, entropy_list, distill_per_exit = _poe_distill_core(
        per_exit_logits, labels, alphas, gamma_distill, distill_T,
    )
    asym_nll_per_exit = []
    for log_p in log_ptilde_list:
        per_sample_nll = F.nll_loss(log_p, labels, reduction="none")
        with torch.no_grad():
            preds = log_p.argmax(dim=-1)
            sample_w = (preds == labels).float() + kappa * (preds != labels).float()
        asym_nll_per_exit.append((sample_w * per_sample_nll).mean())
    _, L_poe = multitask(asym_nll_per_exit)
    if distill_per_exit:
        L_distill = torch.stack(distill_per_exit).mean() * (distill_T ** 2)
    else:
        L_distill = labels.new_zeros((), dtype=torch.float32)
    if n > 1:
        ent = torch.stack(entropy_list, dim=0)
        L_mono = torch.clamp(ent[1:] - ent[:-1], min=0.0).sum()
    else:
        L_mono = labels.new_zeros((), dtype=torch.float32)
    rho_t = rho_max * min(epoch_progress / max(anneal_fraction, 1e-9), 1.0)
    return L_poe + rho_t * L_mono + gamma_distill * L_distill


def poe_distill_mtl_brier_asym_loss(
    per_exit_logits, labels, *,
    alphas, multitask,
    epoch_progress: float = 0.0,
    regression: bool = False,
    kappa: float = 2.0,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    lambda_brier: float = 0.2,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """poe_distill + MTL adaptive eta + asymmetric κ-CE + Brier-score anchor (full stack)."""
    from .losses_zoo import exit_weights
    base = poe_distill_mtl_asym_loss(
        per_exit_logits, labels, alphas=alphas, multitask=multitask,
        epoch_progress=epoch_progress, regression=regression, kappa=kappa,
        rho_max=rho_max, anneal_fraction=anneal_fraction,
        gamma_distill=gamma_distill, distill_T=distill_T,
    )
    if regression or len(per_exit_logits) < 2:
        return base
    weights = exit_weights(len(per_exit_logits), weight_schedule, device=labels.device)
    return base + lambda_brier * _brier_anchor(per_exit_logits, labels, weights)


def poe_distill_mtl_brier_mac_loss(
    per_exit_logits, labels, *,
    alphas, multitask, per_exit_macs,
    epoch_progress: float = 0.0,
    regression: bool = False,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    lambda_brier: float = 0.2,
    lambda_mac: float = 0.3,
    T_gate: float = 0.05,
    tau: float = 0.5,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """poe_distill + MTL + Brier + differentiable MAC routing penalty.

    Designed to break the hypervolume ceiling on (accuracy, op_MACs). Wraps the EMAR-
    winning poe_distill_mtl_brier and adds a DIFFERENTIABLE responsibility-weighted MAC
    term. Unlike budget_boost_loss (which stopgrads confidences → L_mac contributes
    nothing to gradient), this version lets gradient flow through softmax confidences
    so L_mac actively pushes routing mass toward cheaper exits during training.

      r_{j,i}  = sigmoid((c_j(x_i) - tau)/T_gate) * prod_{l<j} (1 - sigmoid(...))
                 (c_j is differentiable softmax-max-confidence; gates are differentiable)
      L_mac    = sum_i sum_j r_{j,i} (m_j / m_J)        (responsibility-weighted MAC)
      L_total  = poe_distill_mtl_brier(...) + lambda_mac * L_mac

    The gradient through L_mac increases confidence at low-MAC exits, biasing routing.
    Combined with the PoE-distill + Brier base, this targets the (accuracy, MAC) curve
    shape directly rather than only the EMAR operating point.
    """
    base = poe_distill_mtl_brier_loss(
        per_exit_logits, labels, alphas=alphas, multitask=multitask,
        epoch_progress=epoch_progress, regression=regression,
        rho_max=rho_max, anneal_fraction=anneal_fraction,
        gamma_distill=gamma_distill, distill_T=distill_T,
        lambda_brier=lambda_brier, weight_schedule=weight_schedule,
    )
    n = len(per_exit_logits)
    if regression or n < 2:
        return base
    if per_exit_macs is None or len(per_exit_macs) != n:
        raise ValueError(
            f"poe_distill_mtl_brier_mac needs per_exit_macs of length {n}; got "
            f"{None if per_exit_macs is None else len(per_exit_macs)}."
        )
    device = labels.device
    mac_tensor = torch.as_tensor(per_exit_macs, dtype=torch.float32, device=device)
    mac_norm = mac_tensor / mac_tensor[-1].clamp_min(1.0)
    # Differentiable confidences (no no_grad wrapper): gradient flows through softmax.
    confidences = torch.stack(
        [F.softmax(z, dim=-1).max(dim=-1).values for z in per_exit_logits], dim=0,
    )
    gates = torch.sigmoid((confidences - tau) / T_gate)
    early_gates = gates[:-1]
    one_minus = 1.0 - early_gates + 1e-9
    cum_one_minus = torch.cumprod(one_minus, dim=0)
    prev_survival = torch.cat([
        torch.ones_like(early_gates[:1]), cum_one_minus[:-1],
    ], dim=0)
    r_early = early_gates * prev_survival
    r_final = cum_one_minus[-1:].clone()
    responsibility = torch.cat([r_early, r_final], dim=0)
    responsibility = responsibility / responsibility.sum(dim=0, keepdim=True).clamp_min(1e-6)
    L_mac = (responsibility * mac_norm.view(-1, 1)).sum(dim=0).mean()
    return base + lambda_mac * L_mac


def poe_jazbec_loss(
    per_exit_logits, labels, *,
    regression: bool = False,
    weight_schedule: str = "uniform",
    **_,
) -> torch.Tensor:
    """Jazbec et al. NeurIPS 2023 baseline: train as summed-CE, apply PoE post-hoc at eval.

    Per the "Anytime Predictions in Cascaded Networks via Conditional Monotonicity" paper,
    the PoE machinery is purely an INFERENCE-time transform: train the network with any
    standard anytime loss (the paper uses summed CE per exit), then at eval combine
    exits via a uniform-weighted geometric mean of softmaxes. The cumulative PoE prediction
    satisfies conditional monotonicity by construction (their main theoretical contribution).

    This loss IS just summed cross-entropy across exits -- it does NOT train against the
    cumulative ptilde_j. The "PoE" part of the method is realised at eval through the
    poe_entropy cutoff, which uses the model's poe_alphas buffer (set to uniform 1.0 for
    this baseline). All training-time PoE work (learnable alphas, monotonicity hinge,
    annealing) is omitted to faithfully reproduce the Jazbec et al. 2023 method.

    Distinct from our poe_anneal/poe_distill/poe_multitask variants which train against the
    cumulative ptilde_j with learnable alphas + explicit monotonicity penalty.
    """
    return _per_exit_ce(per_exit_logits, labels, regression).sum()


def moe_router_loss(
    per_exit_logits, per_exit_confidence_logits, labels, *,
    per_exit_macs,
    regression: bool = False,
    lambda_mac: float = 0.1,
    lambda_ent: float = 0.05,
    eta_floor: float = 0.2,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """MoE-style joint router over per-exit confidence heads (v6, new mechanism class).

    Distinct from JEI-DNN (per-exit independent sigmoid gates with chained probabilities):
    here the router probability is a JOINT softmax over the J per-exit confidence logits
    s_j, giving a single discrete-style routing distribution per sample. The classification
    loss is responsibility-weighted NLL; the router is regularised toward low entropy
    (decisive routing) and low expected MAC cost.

      r_{j,i}  = softmax_j(s_{j,i})    (joint softmax over exits per sample)
      L_route  = sum_i sum_j r_{j,i} * NLL(z_{j,i}, y_i)   (route-weighted CE)
      L_mac    = sum_i sum_j r_{j,i} * (m_j / m_J)         (route-weighted compute)
      L_ent    = mean_i H(r_{., i})                        (router entropy, minimised)
      L_floor  = sum_j w_j * mean_i CE(z_{j,i}, y_i)       (anti-collapse anchor)

      L_total  = L_route + lambda_mac * L_mac - lambda_ent * (-L_ent) + eta_floor * L_floor
               = L_route + lambda_mac * L_mac + lambda_ent * L_ent + eta_floor * L_floor

    Note: ``+ lambda_ent * L_ent`` PENALISES high entropy (drives the router to commit).
    eval routes via ``router_argmax`` cutoff in jolt/calibration.py and inference.py.

    Citations: standard MoE objective (Jacobs et al. 1991; Shazeer et al. 2017 sparse MoE);
    differentiated from JEI-DNN (Regol et al. ICLR 2024) by joint softmax routing and
    responsibility-weighted CE (not chained Bernoulli gates).
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if len(per_exit_confidence_logits) != n:
        raise ValueError(f"moe_router needs {n} confidence logits; got {len(per_exit_confidence_logits)}.")
    if per_exit_macs is None or len(per_exit_macs) != n:
        raise ValueError("moe_router needs per_exit_macs of length n.")

    device = labels.device
    mac_tensor = torch.as_tensor(per_exit_macs, dtype=torch.float32, device=device)
    mac_norm = mac_tensor / mac_tensor[-1].clamp_min(1.0)
    weights = exit_weights(n, weight_schedule, device=device)

    s_logits = torch.stack(per_exit_confidence_logits, dim=0)  # [J, B]
    router = F.softmax(s_logits, dim=0)  # [J, B], soft routing distribution per sample

    per_exit_ce = torch.stack(
        [F.cross_entropy(z, labels, reduction="none") for z in per_exit_logits], dim=0,
    )  # [J, B]

    # Route-weighted CE: each sample contributes mostly via the exit the router selects
    L_route = (router * per_exit_ce).sum(dim=0).mean()
    L_mac = (router * mac_norm.view(-1, 1)).sum(dim=0).mean()
    # Entropy of router distribution per sample (minimised -> decisive routing)
    L_ent = -(router * torch.log(router.clamp_min(1e-9))).sum(dim=0).mean()
    # Uniform CE floor: anti-collapse anchor (keeps all exits trainable)
    L_floor = (weights * per_exit_ce.mean(dim=1)).sum()

    return L_route + lambda_mac * L_mac + lambda_ent * L_ent + eta_floor * L_floor


def scar_poe_loss(
    per_exit_logits, per_exit_confidence_logits, labels, *,
    alphas,
    epoch_progress: float = 0.0,
    regression: bool = False,
    beta: float = 1.0,
    gamma_tcp: float = 0.5,
    T_rank: float = 0.1,
    rho_max: float = 0.3,
    anneal_fraction: float = 0.4,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """SCAR + PoE-Anneal joint training (v6 hybrid).

    Trains SCAR's per-exit confidence head AND PoE-Anneal's cumulative alphas in one
    objective. At eval, the routing combines the two signals (s_j and cumulative-PoE
    entropy) via the ``poe_scar_entropy`` cutoff in jolt/calibration.py.

    Per-exit objective:
      sum_j w_j ( CE(z_j, y) + beta * L_rank_j + gamma_tcp * (s_j - p_true_j)^2 )

    Cumulative-PoE objective:
      log_ptilde_j  = log_softmax(sum_{l<=j} alpha_l * log_softmax(z_l))
      L_poe         = sum_j w_j NLL(log_ptilde_j, y)
      L_mono        = sum_{j>=1} max(0, H(ptilde_j) - H(ptilde_{j-1}))

    Total: L_scar_per_exit + L_poe + rho(t) * L_mono.

    Hypothesis: each loss component shapes a different operating-point signal (s_j for
    AURC, ptilde_j for accuracy/EMAR). Training both jointly creates a richer routing
    surface for the hybrid eval cutoff.
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if len(per_exit_confidence_logits) != n:
        raise ValueError("scar_poe needs one confidence logit tensor per exit.")
    if alphas is None or alphas.numel() != n:
        raise ValueError(f"scar_poe needs alphas of length {n}.")

    device = labels.device
    weights = exit_weights(n, weight_schedule, device=device)

    # SCAR core: per-exit CE + rank + TCP. Identical to scar_loss internals.
    with torch.no_grad():
        preds = torch.stack([z.argmax(dim=-1) for z in per_exit_logits], dim=0)
        correct = (preds == labels.unsqueeze(0)).float()
        if n > 1:
            deeper_sum = correct.flip(0).cumsum(dim=0).flip(0) - correct
        else:
            deeper_sum = torch.zeros_like(correct)
        no_deeper_correct = (deeper_sum == 0).float()
        ybar = torch.clamp(correct + (1.0 - correct) * no_deeper_correct, max=1.0)

    per_sample_ce = torch.stack(
        [F.cross_entropy(z, labels, reduction="none") for z in per_exit_logits], dim=0,
    )
    s_logits = torch.stack(per_exit_confidence_logits, dim=0)
    s = torch.sigmoid(s_logits)
    with torch.no_grad():
        probs_stack = torch.stack([F.softmax(z, dim=-1) for z in per_exit_logits], dim=0)
        tcp_target = probs_stack.gather(2, labels.view(1, -1, 1).expand(n, -1, 1)).squeeze(-1)
    L_tcp = ((s - tcp_target) ** 2).mean(dim=1)

    L_rank_per_exit = []
    for j in range(n):
        pos = ybar[j]; neg = 1.0 - pos
        n_pos = pos.sum().clamp_min(1.0); n_neg = neg.sum().clamp_min(1.0)
        diff = s_logits[j].unsqueeze(0) - s_logits[j].unsqueeze(1)
        kernel = torch.sigmoid(diff / T_rank)
        mask = pos.unsqueeze(1) * neg.unsqueeze(0)
        L_rank_per_exit.append((kernel * mask).sum() / (n_pos * n_neg))
    L_rank = torch.stack(L_rank_per_exit, dim=0)
    scar_term = (weights * (per_sample_ce.mean(dim=1) + beta * L_rank + gamma_tcp * L_tcp)).sum()

    # PoE-Anneal core: cumulative log_ptilde NLL + monotonicity hinge.
    log_probs = [F.log_softmax(z, dim=-1) for z in per_exit_logits]
    running_sum = torch.zeros_like(log_probs[0])
    log_ptilde_list = []
    entropy_list = []
    for j in range(n):
        running_sum = running_sum + alphas[j] * log_probs[j]
        log_ptilde_j = F.log_softmax(running_sum, dim=-1)
        log_ptilde_list.append(log_ptilde_j)
        ptilde_j = log_ptilde_j.exp()
        entropy_list.append(-(ptilde_j * log_ptilde_j).sum(dim=-1).mean())

    nll_per_exit = torch.stack([F.nll_loss(lp, labels, reduction="mean") for lp in log_ptilde_list], dim=0)
    L_poe = (weights * nll_per_exit).sum()
    if n > 1:
        ent = torch.stack(entropy_list, dim=0)
        L_mono = torch.clamp(ent[1:] - ent[:-1], min=0.0).sum()
    else:
        L_mono = labels.new_zeros((), dtype=torch.float32)

    rho_t = rho_max * min(epoch_progress / max(anneal_fraction, 1e-9), 1.0)
    return scar_term + L_poe + rho_t * L_mono


def _brier_anchor(per_exit_logits, labels, weights):
    """Per-exit weighted Brier-score anchor: sum_j w_j * brier(softmax(z_j), y).

    Brier(p, y) = sum_c (p_c - 1[y=c])^2. A proper scoring rule (Brier 1950) that
    penalises miscalibration on top of mere ranking — orthogonal to CE because Brier
    can keep gradient pressure even when CE has saturated. Designed to be added with a
    small weight (lambda_brier ~ 0.1-0.5) as an additive regulariser to any classifier
    loss. Drops the calibration-axis pressure none of our current top candidates apply.
    """
    if not per_exit_logits:
        return labels.new_zeros((), dtype=torch.float32)
    nclass = per_exit_logits[0].shape[-1]
    y_onehot = F.one_hot(labels, nclass).float()
    per_exit_brier = torch.stack(
        [((F.softmax(z, dim=-1) - y_onehot) ** 2).sum(dim=-1).mean() for z in per_exit_logits],
        dim=0,
    )
    return (weights * per_exit_brier).sum()


def poe_brier_loss(
    per_exit_logits, labels, *,
    alphas,
    epoch_progress: float = 0.0,
    regression: bool = False,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    lambda_brier: float = 0.2,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """PoE-Anneal + Brier-score anchor (v6 composition; calibration axis).

    L = L_poe + rho(t) * L_mono + lambda_brier * sum_j w_j * brier(softmax(z_j), y)

    Adds a proper-scoring regulariser to PoE-Anneal. Brier penalises miscalibration
    explicitly (the only term in our v6 candidate set that does this beyond CE / TCP).
    Weight schedule is shared with the PoE NLL term so the anchor mirrors which exits
    matter most.
    """
    from .losses_zoo import exit_weights

    base = poe_anneal_loss(
        per_exit_logits, labels, alphas=alphas, epoch_progress=epoch_progress,
        rho_max=rho_max, anneal_fraction=anneal_fraction,
        weight_schedule=weight_schedule, regression=regression,
    )
    if regression or len(per_exit_logits) < 2:
        return base
    weights = exit_weights(len(per_exit_logits), weight_schedule, device=labels.device)
    return base + lambda_brier * _brier_anchor(per_exit_logits, labels, weights)


def poe_multitask_brier_loss(
    per_exit_logits, labels, *,
    alphas, multitask,
    epoch_progress: float = 0.0,
    regression: bool = False,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    lambda_brier: float = 0.2,
    weight_schedule: str = "increasing",
    lambda_ce_final: float = 0.0,
    **_,
) -> torch.Tensor:
    """PoE-Anneal + Kendall MultiTaskLoss + Brier anchor.

    Used in the leave-one-out ablation as ``poe_distill_mtl_brier`` minus the KL
    distillation term. The composition follows ``poe_distill_brier_loss`` and
    ``poe_multitask_loss``: the MTL loss replaces the fixed exit_weights schedule on
    the PoE NLL, and a Brier-score anchor with the same exit-weight schedule is added.
    """
    from .losses_zoo import exit_weights

    base = poe_multitask_loss(
        per_exit_logits, labels, alphas=alphas, multitask=multitask,
        epoch_progress=epoch_progress, regression=regression,
        rho_max=rho_max, anneal_fraction=anneal_fraction,
    )
    if regression or len(per_exit_logits) < 2:
        return base
    weights = exit_weights(len(per_exit_logits), weight_schedule, device=labels.device)
    if lambda_ce_final > 0.0:
        L_ce_final = F.cross_entropy(per_exit_logits[-1], labels, reduction="mean")
        return base + lambda_brier * _brier_anchor(per_exit_logits, labels, weights) + lambda_ce_final * L_ce_final
    return base + lambda_brier * _brier_anchor(per_exit_logits, labels, weights)


def scar_brier_loss(
    per_exit_logits, per_exit_confidence_logits, labels, *,
    regression: bool = False,
    beta: float = 1.0,
    gamma: float = 0.5,
    T_rank: float = 0.1,
    lambda_brier: float = 0.2,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """SCAR + Brier-score anchor (v6 composition).

    L = L_scar + lambda_brier * sum_j w_j * brier(softmax(z_j), y).

    SCAR's TCP regression already pushes s_j toward p_{true} but the classifier logits
    themselves are only shaped by CE. Adding a Brier anchor regularises the classifier's
    calibration (AURC's underlying ranking is over predicted probabilities), which should
    push AURC down further than SCAR alone.
    """
    from .losses_zoo import exit_weights

    base = scar_loss(
        per_exit_logits, per_exit_confidence_logits, labels,
        regression=regression, beta=beta, gamma=gamma, T_rank=T_rank,
        weight_schedule=weight_schedule,
    )
    if regression or len(per_exit_logits) < 2:
        return base
    weights = exit_weights(len(per_exit_logits), weight_schedule, device=labels.device)
    return base + lambda_brier * _brier_anchor(per_exit_logits, labels, weights)


def tri_axis_brier_loss(
    per_exit_logits, per_exit_confidence_logits, labels, *,
    per_exit_macs,
    regression: bool = False,
    epoch_progress: float = 0.0,
    beta_rank: float = 1.0,
    gamma_tcp: float = 0.5,
    T_rank: float = 0.1,
    lambda_mac: float = 0.15,
    T_gate: float = 0.05,
    tau: float = 0.5,
    rho_max: float = 0.3,
    anneal_fraction: float = 0.4,
    lambda_brier: float = 0.2,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """tri_axis + Brier-score anchor (v6 composition).

    Adds the calibration-axis regulariser to the three-axis kitchen sink. May rescue
    tri_axis's mid-pack rank by adding a complementary signal CE / rank / TCP / MAC /
    monotonicity don't directly enforce.
    """
    from .losses_zoo import exit_weights

    base = tri_axis_loss(
        per_exit_logits, per_exit_confidence_logits, labels,
        per_exit_macs=per_exit_macs, epoch_progress=epoch_progress,
        beta_rank=beta_rank, gamma_tcp=gamma_tcp, T_rank=T_rank,
        lambda_mac=lambda_mac, T_gate=T_gate, tau=tau,
        rho_max=rho_max, anneal_fraction=anneal_fraction,
        weight_schedule=weight_schedule, regression=regression,
    )
    if regression or len(per_exit_logits) < 2:
        return base
    weights = exit_weights(len(per_exit_logits), weight_schedule, device=labels.device)
    return base + lambda_brier * _brier_anchor(per_exit_logits, labels, weights)


def poe_anytime_loss(
    per_exit_logits, labels, *,
    alphas,
    epoch_progress: float = 0.0,
    regression: bool = False,
    tau: float = 1.0,
    eta_cost: float = 0.5,
    alpha_min: float = 0.05,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    **_,
) -> torch.Tensor:
    """PoE-Anneal + AnytimeStable cost-aware adaptive weights (v6 composition).

    Replaces poe_anneal's fixed ``exit_weights("increasing")`` with AnytimeStable's
    cost-aware adaptive alpha:

      w_e = softmax((acc_e - eta_cost * c_e) / tau)   (clamped to alpha_min, renormalised)

    where acc_e is the per-exit accuracy on the current batch (no-grad) and c_e is the
    normalised exit cost ((e+1)/J). High-accuracy early exits get more weight; deeper
    exits with marginal accuracy gains get less. The cumulative PoE prediction and the
    monotonicity hinge are unchanged.

    Hypothesis: adaptive per-exit weighting matches each exit's actual difficulty, which
    should improve EMAR + accuracy without compromising AURC. Combines our two strongest
    cross-cell mechanisms (PoE + cost-aware alpha used by anytime_stable + anytime_stable_distill).
    """
    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if alphas is None or alphas.numel() != n:
        raise ValueError(f"poe_anytime needs alphas of length {n}.")

    device = labels.device
    log_probs = [F.log_softmax(z, dim=-1) for z in per_exit_logits]
    running_sum = torch.zeros_like(log_probs[0])
    log_ptilde_list = []
    entropy_list = []
    for j in range(n):
        running_sum = running_sum + alphas[j] * log_probs[j]
        log_ptilde_j = F.log_softmax(running_sum, dim=-1)
        log_ptilde_list.append(log_ptilde_j)
        ptilde_j = log_ptilde_j.exp()
        entropy_list.append(-(ptilde_j * log_ptilde_j).sum(dim=-1).mean())

    # AnytimeStable cost-aware adaptive weights (no grad on weights, like anytime_stable).
    with torch.no_grad():
        accs = []
        for log_p in log_ptilde_list:
            preds = log_p.argmax(dim=-1)
            accs.append((preds == labels).float().mean())
        acc_vec = torch.stack(accs)
        cost = torch.tensor(
            [(e + 1) / n for e in range(n)], device=device, dtype=acc_vec.dtype,
        )
        raw = (acc_vec - eta_cost * cost) / tau
        weights = F.softmax(raw, dim=0)
        weights = torch.clamp(weights, min=alpha_min)
        weights = (weights / weights.sum()).detach()

    nll_per_exit = torch.stack([
        F.nll_loss(lp, labels, reduction="mean") for lp in log_ptilde_list
    ], dim=0)
    L_poe = (weights * nll_per_exit).sum()

    if n > 1:
        ent = torch.stack(entropy_list, dim=0)
        L_mono = torch.clamp(ent[1:] - ent[:-1], min=0.0).sum()
    else:
        L_mono = labels.new_zeros((), dtype=torch.float32)

    rho_t = rho_max * min(epoch_progress / max(anneal_fraction, 1e-9), 1.0)
    return L_poe + rho_t * L_mono


def poe_asym_loss(
    per_exit_logits, labels, *,
    alphas,
    epoch_progress: float = 0.0,
    regression: bool = False,
    kappa: float = 2.0,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """PoE-Anneal + AsymSelect κ-weighted asymmetric CE (v6 composition).

    Per-exit cumulative-PoE NLL is replaced by an asymmetric variant:

      L_asym,j = mean_i [ (correct_{j,i} + kappa * wrong_{j,i}) * NLL(log_ptilde_j[i], y_i) ]

    Errors (wrong predictions at exit j on cumulative ptilde_j) are weighted kappa× more
    than corrects. Combines PoE's monotone-ensemble training with AsymSelect's failure-
    penalising objective, which dominated AURC in our v4 results.
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if alphas is None or alphas.numel() != n:
        raise ValueError(f"poe_asym needs alphas of length {n}.")

    device = labels.device
    log_probs = [F.log_softmax(z, dim=-1) for z in per_exit_logits]
    running_sum = torch.zeros_like(log_probs[0])
    log_ptilde_list = []
    entropy_list = []
    for j in range(n):
        running_sum = running_sum + alphas[j] * log_probs[j]
        log_ptilde_j = F.log_softmax(running_sum, dim=-1)
        log_ptilde_list.append(log_ptilde_j)
        ptilde_j = log_ptilde_j.exp()
        entropy_list.append(-(ptilde_j * log_ptilde_j).sum(dim=-1).mean())

    weights = exit_weights(n, weight_schedule, device=device)
    nll_per_exit = []
    for j, log_p in enumerate(log_ptilde_list):
        per_sample_nll = F.nll_loss(log_p, labels, reduction="none")  # [B]
        with torch.no_grad():
            preds = log_p.argmax(dim=-1)
            sample_w = (preds == labels).float() + kappa * (preds != labels).float()
        nll_per_exit.append((sample_w * per_sample_nll).mean())
    nll_tensor = torch.stack(nll_per_exit, dim=0)
    L_poe = (weights * nll_tensor).sum()

    if n > 1:
        ent = torch.stack(entropy_list, dim=0)
        L_mono = torch.clamp(ent[1:] - ent[:-1], min=0.0).sum()
    else:
        L_mono = labels.new_zeros((), dtype=torch.float32)

    rho_t = rho_max * min(epoch_progress / max(anneal_fraction, 1e-9), 1.0)
    return L_poe + rho_t * L_mono


def budget_boost_multitask_loss(
    per_exit_logits, labels, *,
    per_exit_macs, multitask,
    regression: bool = False,
    lambda_mac: float = 0.3,
    eta_floor: float = 0.2,
    T_gate: float = 0.05,
    tau: float = 0.5,
    **_,
) -> torch.Tensor:
    """BudgetBoost + Kendall MultiTaskLoss adaptive CE-floor weighting (v6 composition).

    BudgetBoost's L_fit + lambda_mac * L_mac terms are unchanged. The CE-floor term
    (uniform CE across exits, the anti-collapse anchor) was previously weighted by a
    fixed exit_weights schedule; this variant uses MultiTaskLoss adaptive etas instead.
    Hypothesis: adaptive weighting on the CE floor stabilises training on backbones where
    BudgetBoost previously collapsed (notably WRN-28-10 at 1 % accuracy in v5).
    """
    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if per_exit_macs is None or len(per_exit_macs) != n:
        raise ValueError("budget_boost_multitask needs per_exit_macs.")
    if multitask is None:
        raise ValueError("budget_boost_multitask needs a MultiTaskLoss instance.")

    device = labels.device
    mac_tensor = torch.as_tensor(per_exit_macs, dtype=torch.float32, device=device)
    mac_norm = mac_tensor / mac_tensor[-1].clamp_min(1.0)
    per_exit_ce = torch.stack(
        [F.cross_entropy(z, labels, reduction="none") for z in per_exit_logits], dim=0,
    )
    with torch.no_grad():
        confidences = torch.stack(
            [F.softmax(z, dim=-1).max(dim=-1).values for z in per_exit_logits], dim=0,
        )
    gates = torch.sigmoid((confidences - tau) / T_gate)
    early_gates = gates[:-1]
    one_minus = 1.0 - early_gates + 1e-9
    cum_one_minus = torch.cumprod(one_minus, dim=0)
    prev_survival = torch.cat([
        torch.ones_like(early_gates[:1]), cum_one_minus[:-1],
    ], dim=0)
    r_early = early_gates * prev_survival
    r_final = cum_one_minus[-1:].clone()
    responsibility = torch.cat([r_early, r_final], dim=0)
    responsibility = responsibility / responsibility.sum(dim=0, keepdim=True).clamp_min(1e-6)

    L_fit = (responsibility * per_exit_ce).sum(dim=0).mean()
    L_mac = (responsibility * mac_norm.view(-1, 1)).sum(dim=0).mean()
    # CE floor with Kendall MTL adaptive weights (replaces the fixed exit_weights term)
    per_exit_ce_mean_list = [per_exit_ce[j].mean() for j in range(n)]
    _, L_floor_adaptive = multitask(per_exit_ce_mean_list)
    return L_fit + lambda_mac * L_mac + eta_floor * L_floor_adaptive


def tri_axis_distill_loss(
    per_exit_logits, per_exit_confidence_logits, labels, *,
    per_exit_macs,
    regression: bool = False,
    epoch_progress: float = 0.0,
    beta_rank: float = 1.0,
    gamma_tcp: float = 0.5,
    T_rank: float = 0.1,
    lambda_mac: float = 0.15,
    T_gate: float = 0.05,
    tau: float = 0.5,
    rho_max: float = 0.3,
    anneal_fraction: float = 0.4,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """tri_axis + BEEM-style self-distillation (v6 composition).

    tri_axis lands mid-pack in our v5 runs (rank 13/18 on UCI-HAR, 13/17 on SST-2). The
    failure mode looks like the multi-objective composition diluting the classification
    signal. Adding a KL teacher from the final exit anchors the early classifiers to a
    strong target, which should bring back accuracy without disturbing the AURC/MAC/EMAR
    machinery.

    L = L_tri_axis + gamma_distill * sum_{j<J-1} w_j * T^2 * KL(softmax(z_j/T) || softmax(z_{J-1}/T).detach())
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()

    base = tri_axis_loss(
        per_exit_logits, per_exit_confidence_logits, labels,
        per_exit_macs=per_exit_macs, epoch_progress=epoch_progress,
        beta_rank=beta_rank, gamma_tcp=gamma_tcp, T_rank=T_rank,
        lambda_mac=lambda_mac, T_gate=T_gate, tau=tau,
        rho_max=rho_max, anneal_fraction=anneal_fraction,
        weight_schedule=weight_schedule,
    )

    weights = exit_weights(n, weight_schedule, device=labels.device)
    final_logits = per_exit_logits[-1]
    teacher_log_probs = F.log_softmax(final_logits.detach() / distill_T, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    distill_per_exit = []
    for j in range(n - 1):
        student_log_probs = F.log_softmax(per_exit_logits[j] / distill_T, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        distill_per_exit.append(kl)
    if distill_per_exit:
        distill_tensor = torch.stack(distill_per_exit, dim=0)
        L_distill = (weights[:-1] * distill_tensor).sum() * (distill_T ** 2)
    else:
        L_distill = labels.new_zeros((), dtype=torch.float32)
    return base + gamma_distill * L_distill


def poe_multitask_loss(
    per_exit_logits, labels, *,
    alphas, multitask,
    epoch_progress: float = 0.0,
    regression: bool = False,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    **_,
) -> torch.Tensor:
    """PoE-Anneal with Kendall MultiTaskLoss adaptive per-exit weights (v6 composition).

    Replaces poe_anneal's fixed ``exit_weights("increasing")`` with a learnable per-exit
    weighting via Kendall et al. 2018 MultiTaskLoss (the same mechanism that powers
    AdaLoss). The per-exit NLL on cumulative PoE predictions is fed to MultiTaskLoss
    which produces an adaptively-weighted sum via the learnable eta parameters.

    The hypothesis: per-cell HP tuning of the fixed schedule is brittle; an adaptive
    eta lets the model find the right per-exit balance dynamically. AdaLoss empirically
    wins HV on multiple cells in our pack, so layering the same machinery onto PoE
    should improve cross-cell consistency.

    Citations: Kendall, Gal & Cipolla CVPR 2018 (MultiTaskLoss); Jazbec et al. NeurIPS
    2023 (PoE monotonicity).
    """
    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if alphas is None or alphas.numel() != n:
        raise ValueError(f"poe_multitask needs alphas of length {n}.")
    if multitask is None:
        raise ValueError("poe_multitask requires a MultiTaskLoss instance.")

    log_probs = [F.log_softmax(z, dim=-1) for z in per_exit_logits]
    running_sum = torch.zeros_like(log_probs[0])
    log_ptilde_list = []
    entropy_list = []
    for j in range(n):
        running_sum = running_sum + alphas[j] * log_probs[j]
        log_ptilde_j = F.log_softmax(running_sum, dim=-1)
        log_ptilde_list.append(log_ptilde_j)
        ptilde_j = log_ptilde_j.exp()
        entropy_list.append(-(ptilde_j * log_ptilde_j).sum(dim=-1).mean())

    per_exit_nll = [F.nll_loss(lp, labels, reduction="mean") for lp in log_ptilde_list]
    _, L_poe = multitask(per_exit_nll)

    if n > 1:
        ent = torch.stack(entropy_list, dim=0)
        L_mono = torch.clamp(ent[1:] - ent[:-1], min=0.0).sum()
    else:
        L_mono = labels.new_zeros((), dtype=torch.float32)

    rho_t = rho_max * min(epoch_progress / max(anneal_fraction, 1e-9), 1.0)
    return L_poe + rho_t * L_mono


def scar_distill_loss(
    per_exit_logits, per_exit_confidence_logits, labels, *,
    regression: bool = False,
    beta: float = 1.0,
    gamma_tcp: float = 0.5,
    T_rank: float = 0.1,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """SCAR + BEEM-style self-distillation (v6 composition).

    SCAR's confidence-rank machinery + a KL teacher from the final exit's softmax to
    each early exit's classifier prediction. Addresses SCAR's weakness on small backbones
    (the per-exit classifier isn't strong enough on its own); the KL teacher pulls early
    exits toward the final exit's prediction while the rank head shapes failure-prediction.

    L = sum_j w_j ( CE_j + beta * L_rank_j + gamma_tcp * L_tcp_j )
        + gamma_distill * sum_{j<J-1} w_j * T^2 * KL(softmax(z_j/T) || softmax(z_{J-1}/T).detach())

    Eval routing uses learned_confidence (SCAR's s_j); the KL term only shapes training.
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if len(per_exit_confidence_logits) != n:
        raise ValueError("scar_distill needs one confidence logit tensor per exit.")

    weights = exit_weights(n, weight_schedule, device=labels.device)

    # SCAR core (rank + TCP + CE) -- mirrors scar_loss exactly.
    with torch.no_grad():
        preds = torch.stack([z.argmax(dim=-1) for z in per_exit_logits], dim=0)
        correct = (preds == labels.unsqueeze(0)).float()
        if n > 1:
            deeper_sum = correct.flip(0).cumsum(dim=0).flip(0) - correct
        else:
            deeper_sum = torch.zeros_like(correct)
        no_deeper_correct = (deeper_sum == 0).float()
        ybar = torch.clamp(correct + (1.0 - correct) * no_deeper_correct, max=1.0)

    per_sample_ce_stack = torch.stack(
        [F.cross_entropy(z, labels, reduction="none") for z in per_exit_logits], dim=0,
    )
    s_logits = torch.stack(per_exit_confidence_logits, dim=0)
    s = torch.sigmoid(s_logits)
    with torch.no_grad():
        probs_stack = torch.stack([F.softmax(z, dim=-1) for z in per_exit_logits], dim=0)
        tcp_target = probs_stack.gather(2, labels.view(1, -1, 1).expand(n, -1, 1)).squeeze(-1)
    L_tcp = ((s - tcp_target) ** 2).mean(dim=1)

    L_rank_per_exit = []
    for j in range(n):
        pos = ybar[j]; neg = 1.0 - pos
        n_pos = pos.sum().clamp_min(1.0); n_neg = neg.sum().clamp_min(1.0)
        diff = s_logits[j].unsqueeze(0) - s_logits[j].unsqueeze(1)
        kernel = torch.sigmoid(diff / T_rank)
        mask = pos.unsqueeze(1) * neg.unsqueeze(0)
        L_rank_per_exit.append((kernel * mask).sum() / (n_pos * n_neg))
    L_rank = torch.stack(L_rank_per_exit, dim=0)

    per_exit_ce = per_sample_ce_stack.mean(dim=1)
    scar_term = (weights * (per_exit_ce + beta * L_rank + gamma_tcp * L_tcp)).sum()

    # BEEM-style self-distillation: final exit as teacher for the early exits.
    final_logits = per_exit_logits[-1]
    teacher_log_probs = F.log_softmax(final_logits.detach() / distill_T, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    distill_per_exit = []
    for j in range(n - 1):
        student_log_probs = F.log_softmax(per_exit_logits[j] / distill_T, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        distill_per_exit.append(kl)
    if distill_per_exit:
        distill_tensor = torch.stack(distill_per_exit, dim=0)
        L_distill = (weights[:-1] * distill_tensor).sum() * (distill_T ** 2)
    else:
        L_distill = labels.new_zeros((), dtype=torch.float32)
    return scar_term + gamma_distill * L_distill


def budget_boost_distill_loss(
    per_exit_logits, labels, *,
    per_exit_macs,
    regression: bool = False,
    lambda_mac: float = 0.3,
    eta: float = 0.2,
    T_gate: float = 0.05,
    tau: float = 0.5,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """BudgetBoost + BEEM-style self-distillation (v6 composition).

    BudgetBoost collapses on large backbones (1 % accuracy on WRN-28-10 in our v5 runs).
    The failure mode looks like shallow-collapse: with too few gradient signals to the
    early exits, they end up producing arbitrary predictions and the MAC penalty rewards
    routing samples there. Adding a KL teacher (final exit -> each exit's prediction)
    forces the early exits to track the strong final prediction, which should prevent
    the shallow-collapse pathology while preserving the MAC reduction.

    L = L_fit + lambda_mac * L_mac + eta * L_floor
        + gamma_distill * sum_{j<J-1} w_j * T^2 * KL(softmax(z_j/T) || softmax(z_{J-1}/T).detach())
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if per_exit_macs is None or len(per_exit_macs) != n:
        raise ValueError("budget_boost_distill needs per_exit_macs of length len(per_exit_logits).")

    device = labels.device
    mac_tensor = torch.as_tensor(per_exit_macs, dtype=torch.float32, device=device)
    mac_norm = mac_tensor / mac_tensor[-1].clamp_min(1.0)
    per_exit_ce = torch.stack(
        [F.cross_entropy(z, labels, reduction="none") for z in per_exit_logits], dim=0,
    )
    with torch.no_grad():
        confidences = torch.stack(
            [F.softmax(z, dim=-1).max(dim=-1).values for z in per_exit_logits], dim=0,
        )
    gates = torch.sigmoid((confidences - tau) / T_gate)
    early_gates = gates[:-1]
    one_minus = 1.0 - early_gates + 1e-9
    cum_one_minus = torch.cumprod(one_minus, dim=0)
    prev_survival = torch.cat([
        torch.ones_like(early_gates[:1]), cum_one_minus[:-1],
    ], dim=0)
    r_early = early_gates * prev_survival
    r_final = cum_one_minus[-1:].clone()
    responsibility = torch.cat([r_early, r_final], dim=0)
    responsibility = responsibility / responsibility.sum(dim=0, keepdim=True).clamp_min(1e-6)

    L_fit = (responsibility * per_exit_ce).sum(dim=0).mean()
    L_mac = (responsibility * mac_norm.view(-1, 1)).sum(dim=0).mean()
    weights = exit_weights(n, weight_schedule, device=device)
    L_floor = (weights * per_exit_ce.mean(dim=1)).sum()
    base = L_fit + lambda_mac * L_mac + eta * L_floor

    # Self-distillation on classifier logits (anti-collapse anchor).
    final_logits = per_exit_logits[-1]
    teacher_log_probs = F.log_softmax(final_logits.detach() / distill_T, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    distill_per_exit = []
    for j in range(n - 1):
        student_log_probs = F.log_softmax(per_exit_logits[j] / distill_T, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        distill_per_exit.append(kl)
    if distill_per_exit:
        distill_tensor = torch.stack(distill_per_exit, dim=0)
        L_distill = (weights[:-1] * distill_tensor).sum() * (distill_T ** 2)
    else:
        L_distill = labels.new_zeros((), dtype=torch.float32)
    return base + gamma_distill * L_distill


def budget_boost_distill_brier_loss(
    per_exit_logits, labels, *,
    per_exit_macs,
    regression: bool = False,
    lambda_mac: float = 0.3,
    eta: float = 0.2,
    T_gate: float = 0.05,
    tau: float = 0.5,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    lambda_brier: float = 0.2,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """budget_boost_distill + Brier-score calibration anchor (v6 multi-mech composition)."""
    from .losses_zoo import exit_weights
    base = budget_boost_distill_loss(
        per_exit_logits, labels, per_exit_macs=per_exit_macs, regression=regression,
        lambda_mac=lambda_mac, eta=eta, T_gate=T_gate, tau=tau,
        gamma_distill=gamma_distill, distill_T=distill_T, weight_schedule=weight_schedule,
    )
    if regression or len(per_exit_logits) < 2:
        return base
    weights = exit_weights(len(per_exit_logits), weight_schedule, device=labels.device)
    return base + lambda_brier * _brier_anchor(per_exit_logits, labels, weights)


def budget_boost_distill_asym_loss(
    per_exit_logits, labels, *,
    per_exit_macs,
    regression: bool = False,
    kappa: float = 2.0,
    lambda_mac: float = 0.3,
    eta: float = 0.2,
    T_gate: float = 0.05,
    tau: float = 0.5,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    lambda_asym: float = 0.5,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """budget_boost_distill + asymmetric κ-CE penalty (errors κ× more) (v6 multi-mech).

    Adds an asymmetric per-sample CE term on top of budget_boost_distill: wrong samples
    contribute κ× more than correct ones. lambda_asym controls the strength of the
    additive asym penalty so it doesn't dominate the base BudgetBoost + distill terms.
    """
    from .losses_zoo import exit_weights
    base = budget_boost_distill_loss(
        per_exit_logits, labels, per_exit_macs=per_exit_macs, regression=regression,
        lambda_mac=lambda_mac, eta=eta, T_gate=T_gate, tau=tau,
        gamma_distill=gamma_distill, distill_T=distill_T, weight_schedule=weight_schedule,
    )
    if regression or len(per_exit_logits) < 2:
        return base
    weights = exit_weights(len(per_exit_logits), weight_schedule, device=labels.device)
    asym_per_exit = []
    for z in per_exit_logits:
        per_sample = F.cross_entropy(z, labels, reduction="none")
        with torch.no_grad():
            preds = z.argmax(dim=-1)
            sample_w = (preds == labels).float() + kappa * (preds != labels).float()
        asym_per_exit.append((sample_w * per_sample).mean())
    L_asym = (weights * torch.stack(asym_per_exit, dim=0)).sum()
    return base + lambda_asym * L_asym


def budget_boost_distill_mtl_loss(
    per_exit_logits, labels, *,
    multitask,
    per_exit_macs,
    regression: bool = False,
    lambda_mac: float = 0.3,
    eta: float = 0.2,
    T_gate: float = 0.05,
    tau: float = 0.5,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """BudgetBoost + KL distill + Kendall MultiTaskLoss adaptive eta (v6 multi-mech)."""
    from .losses_zoo import exit_weights
    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if multitask is None:
        raise ValueError("budget_boost_distill_mtl needs a MultiTaskLoss instance.")
    if per_exit_macs is None or len(per_exit_macs) != n:
        raise ValueError("budget_boost_distill_mtl needs per_exit_macs of length n.")

    device = labels.device
    mac_tensor = torch.as_tensor(per_exit_macs, dtype=torch.float32, device=device)
    mac_norm = mac_tensor / mac_tensor[-1].clamp_min(1.0)
    per_exit_ce = torch.stack(
        [F.cross_entropy(z, labels, reduction="none") for z in per_exit_logits], dim=0,
    )
    with torch.no_grad():
        confidences = torch.stack(
            [F.softmax(z, dim=-1).max(dim=-1).values for z in per_exit_logits], dim=0,
        )
    gates = torch.sigmoid((confidences - tau) / T_gate)
    early_gates = gates[:-1]
    one_minus = 1.0 - early_gates + 1e-9
    cum_one_minus = torch.cumprod(one_minus, dim=0)
    prev_survival = torch.cat([
        torch.ones_like(early_gates[:1]), cum_one_minus[:-1],
    ], dim=0)
    r_early = early_gates * prev_survival
    r_final = cum_one_minus[-1:].clone()
    responsibility = torch.cat([r_early, r_final], dim=0)
    responsibility = responsibility / responsibility.sum(dim=0, keepdim=True).clamp_min(1e-6)

    L_fit = (responsibility * per_exit_ce).sum(dim=0).mean()
    L_mac = (responsibility * mac_norm.view(-1, 1)).sum(dim=0).mean()
    # MTL adaptive CE floor (replaces fixed weighted sum)
    per_exit_ce_mean = [per_exit_ce[j].mean() for j in range(n)]
    _, L_floor_mtl = multitask(per_exit_ce_mean)
    base = L_fit + lambda_mac * L_mac + eta * L_floor_mtl

    # KL distill from final exit (same fixed-weights convention as budget_boost_distill)
    weights = exit_weights(n, weight_schedule, device=device)
    final_logits = per_exit_logits[-1]
    teacher_log_probs = F.log_softmax(final_logits.detach() / distill_T, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    distill_per_exit = []
    for j in range(n - 1):
        student_log_probs = F.log_softmax(per_exit_logits[j] / distill_T, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        distill_per_exit.append(kl)
    if distill_per_exit:
        distill_tensor = torch.stack(distill_per_exit, dim=0)
        L_distill = (weights[:-1] * distill_tensor).sum() * (distill_T ** 2)
    else:
        L_distill = labels.new_zeros((), dtype=torch.float32)
    return base + gamma_distill * L_distill


def budget_boost_distill_mtl_brier_loss(
    per_exit_logits, labels, *,
    multitask,
    per_exit_macs,
    regression: bool = False,
    lambda_mac: float = 0.3,
    eta: float = 0.2,
    T_gate: float = 0.05,
    tau: float = 0.5,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    lambda_brier: float = 0.2,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """BudgetBoost + KL distill + MTL adaptive eta + Brier-score anchor (v6 full stack)."""
    from .losses_zoo import exit_weights
    base = budget_boost_distill_mtl_loss(
        per_exit_logits, labels, multitask=multitask, per_exit_macs=per_exit_macs,
        regression=regression, lambda_mac=lambda_mac, eta=eta, T_gate=T_gate, tau=tau,
        gamma_distill=gamma_distill, distill_T=distill_T, weight_schedule=weight_schedule,
    )
    if regression or len(per_exit_logits) < 2:
        return base
    weights = exit_weights(len(per_exit_logits), weight_schedule, device=labels.device)
    return base + lambda_brier * _brier_anchor(per_exit_logits, labels, weights)


def poe_distill_loss(
    per_exit_logits, labels, *,
    alphas,
    epoch_progress: float = 0.0,
    regression: bool = False,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    gamma_distill: float = 0.5,
    distill_T: float = 1.0,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """PoE-Anneal + BEEM-style self-distillation (v6 candidate).

    Composes the cumulative product-of-experts training objective (poe_anneal) with the
    BEEM/AnytimeStable-Distill convention of using the final exit's softmax as a teacher
    for the early exits:

      ptilde_j(x) = log_softmax( sum_{l<=j} alpha_l * log_softmax(z_l) )    [cumulative PoE]
      L_poe      = sum_j w_j NLL(log_ptilde_j, y)                            [PoE training]
      L_mono     = sum_{j>=1} max(0, H(ptilde_j) - H(ptilde_{j-1}))         [EMAR monotonicity]
      teacher    = softmax(z_{J-1} / T).detach()                             [final exit, frozen]
      L_distill  = sum_{j < J-1} w_j * T^2 * KL(softmax(ptilde_j / T) || teacher)
      L_total    = L_poe + rho(t) * L_mono + gamma_distill * L_distill

    The distillation term gives the cumulative PoE prediction at each early exit an explicit
    target (the final exit's prediction), which BEEM evidence suggests improves AURC + accuracy
    even when the cumulative ensemble is already monotone. Final exit has no distill term (it
    IS the teacher).

    Inference: cumulative PoE prediction via poe_entropy cutoff (same as poe_anneal).

    Per-component citations: Jazbec et al. NeurIPS 2023 (PoE monotonicity); Bajpai et al.
    ICLR 2025 (BEEM linearly-weighted CE + KL-to-final); Hinton et al. 2015 (distillation
    temperature scaling).
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if alphas is None or alphas.numel() != n:
        raise ValueError(
            f"poe_distill needs alphas of length {n}; got "
            f"{None if alphas is None else alphas.numel()}."
        )

    device = labels.device
    log_probs = [F.log_softmax(z, dim=-1) for z in per_exit_logits]
    running_sum = torch.zeros_like(log_probs[0])
    log_ptilde_list = []
    entropy_list = []
    for j in range(n):
        running_sum = running_sum + alphas[j] * log_probs[j]
        log_ptilde_j = F.log_softmax(running_sum, dim=-1)
        log_ptilde_list.append(log_ptilde_j)
        ptilde_j = log_ptilde_j.exp()
        entropy_list.append(-(ptilde_j * log_ptilde_j).sum(dim=-1).mean())

    weights = exit_weights(n, weight_schedule, device=device)
    nll_per_exit = torch.stack([
        F.nll_loss(lp, labels, reduction="mean") for lp in log_ptilde_list
    ], dim=0)
    L_poe = (weights * nll_per_exit).sum()

    if n > 1:
        ent = torch.stack(entropy_list, dim=0)
        L_mono = torch.clamp(ent[1:] - ent[:-1], min=0.0).sum()
    else:
        L_mono = labels.new_zeros((), dtype=torch.float32)

    # Self-distillation: teacher = softmax(z_final / T).detach(); student = ptilde_j scaled by T
    final_logits = per_exit_logits[-1]
    teacher_log_probs = F.log_softmax(final_logits.detach() / distill_T, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    distill_per_exit = []
    for j in range(n - 1):  # final exit (j = n-1) IS the teacher; no distill term
        student_log_probs = F.log_softmax(log_ptilde_list[j] / distill_T, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        distill_per_exit.append(kl)
    if distill_per_exit:
        # Weight each early-exit distill term by w_j (same schedule as L_poe), scale by T^2.
        distill_tensor = torch.stack(distill_per_exit, dim=0)
        L_distill = (weights[:-1] * distill_tensor).sum() * (distill_T ** 2)
    else:
        L_distill = labels.new_zeros((), dtype=torch.float32)

    rho_t = rho_max * min(epoch_progress / max(anneal_fraction, 1e-9), 1.0)
    return L_poe + rho_t * L_mono + gamma_distill * L_distill


def budget_boost_loss(
    per_exit_logits, labels, *,
    per_exit_macs,
    regression: bool = False,
    lambda_mac: float = 0.3,
    eta: float = 0.2,
    T_gate: float = 0.05,
    tau: float = 0.5,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """BudgetBoost loss (v5 candidate C2; MAC-saved axis).

    Cumulative-cost-aware reweighting per the v5 brief. At each step, compute a soft
    per-sample responsibility r_{j,i} that approximates "the probability sample i exits
    at j" via a differentiable surrogate of the standard threshold rule:

      r_{j,i} = sigmoid((c_j(x_i) - tau)/T_gate) * prod_{l<j} (1 - sigmoid((c_l(x_i) - tau)/T_gate))
      r_{J,i} = prod_{l<J} (1 - sigmoid((c_l(x_i) - tau)/T_gate))   (residual at final exit)

    where c_j(x) = max-softmax confidence at exit j. The loss reweights CE by responsibility
    (so each sample is trained mostly at the exit it actually uses) and adds an explicit
    MAC penalty plus a small uniform CE floor to keep all heads trainable:

      L_fit = sum_i sum_j r_{j,i} CE_j(x_i)
      L_mac = sum_i sum_j r_{j,i} (m_j / m_J)
      L_BudgetBoost = L_fit + lambda_mac * L_mac + eta * sum_j w_j CE_j

    The MAC penalty pulls responsibility mass toward earlier exits whenever doing so does
    not raise L_fit; the CE floor (weighted by ``exit_weights`` like other JOLT methods)
    is the load-bearing guard against the shallow-collapse failure mode (the v5 brief
    notes BoostNet-style joint optimization for this purpose).

    Defaults per the v5 brief: lambda_mac=0.3, eta=0.2, T_gate=0.05, tau=0.5. ``per_exit_macs``
    is the per-exit cumulative MAC table; normalisation uses the deepest exit (per_exit_macs[-1])
    so L_mac is bounded in [0, 1].

    Citations: Huang et al. ICLR 2018 (MSDNet) budgeted-batch objective; Zeng et al. AAAI
    2024 (ConsistentEE) one-correct-exit principle; Yu et al. AAAI 2023 (BoostNet) joint
    multi-classifier optimisation.
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if per_exit_macs is None or len(per_exit_macs) != n:
        raise ValueError(
            f"BudgetBoost needs per_exit_macs of length {n}; got "
            f"{None if per_exit_macs is None else len(per_exit_macs)}."
        )

    device = labels.device
    mac_tensor = torch.as_tensor(per_exit_macs, dtype=torch.float32, device=device)
    mac_max = mac_tensor[-1].clamp_min(1.0)
    mac_norm = mac_tensor / mac_max  # [J] in (0, 1]

    # Per-sample max-confidence per exit (no grad through the gating surrogate target so the
    # MAC term doesn't backprop through the classifier's logit-magnitude artifacts; the L_fit
    # term carries the CE gradient).
    per_exit_ce = []
    for j in range(n):
        per_exit_ce.append(F.cross_entropy(per_exit_logits[j], labels, reduction="none"))
    per_exit_ce = torch.stack(per_exit_ce, dim=0)  # [J, B]

    with torch.no_grad():
        confidences = torch.stack(
            [F.softmax(z, dim=-1).max(dim=-1).values for z in per_exit_logits], dim=0
        )  # [J, B], detached

    # Soft gate probability g_j = sigmoid((c_j - tau)/T_gate) for each EARLY exit; the
    # final exit has no gate (it always accepts the residual). With J total exits, gates
    # are defined for j = 0 .. J-2.
    #
    # r_j = g_j * prod_{l<j} (1 - g_l)   for j in 0..J-2
    # r_{J-1} = prod_{l<J-1} (1 - g_l)   (the "first-success" residual at the final exit)
    # These responsibilities sum to 1 per sample by the standard chain identity.
    gates = torch.sigmoid((confidences - tau) / T_gate)  # [J, B]
    early_gates = gates[:-1]  # [J-1, B]
    one_minus = 1.0 - early_gates + 1e-9  # [J-1, B]
    cum_one_minus = torch.cumprod(one_minus, dim=0)  # [J-1, B]
    # prev_survival[j] = prod_{l<j} (1 - g_l) for j in 0..J-2
    prev_survival = torch.cat([
        torch.ones_like(early_gates[:1]),
        cum_one_minus[:-1],
    ], dim=0)  # [J-1, B]
    r_early = early_gates * prev_survival  # [J-1, B]
    r_final = cum_one_minus[-1:].clone()  # [1, B], = prod_{l<J-1} (1 - g_l)
    responsibility = torch.cat([r_early, r_final], dim=0)  # [J, B], sums to ~1
    responsibility = responsibility / responsibility.sum(dim=0, keepdim=True).clamp_min(1e-6)

    L_fit = (responsibility * per_exit_ce).sum(dim=0).mean()  # avg over batch
    L_mac = (responsibility * mac_norm.view(-1, 1)).sum(dim=0).mean()

    weights = exit_weights(n, weight_schedule, device=device)
    L_floor = (weights * per_exit_ce.mean(dim=1)).sum()

    return L_fit + lambda_mac * L_mac + eta * L_floor


def scar_loss(
    per_exit_logits, per_exit_confidence_logits, labels, *,
    regression: bool = False,
    beta: float = 1.0,
    gamma: float = 0.5,
    T_rank: float = 0.1,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """SCAR loss (v5 candidate C1; AURC axis).

    Structure-Aware Confidence-Rank loss after the v5 brief, combining three terms per
    exit j:

      L_CE,j   = standard cross-entropy on the classifier logits z_j
      L_rank,j = pairwise rank surrogate over structure-aware positives ybar_{j,i}:
                 ybar_{j,i} = 1 iff arg-max(z_j[i]) = y_i (exit j correct), OR all deeper
                 exits l > j are also wrong (so exiting here wastes no accuracy).
                 The surrogate is a logistic kernel on rank violations:
                     L_rank,j = E_{i pos, k neg} sigma((s_j(x_k) - s_j(x_i)) / T_rank)
                 with T_rank a small temperature (default 0.1).
      L_tcp,j  = (s_j(x) - p_{j, y}(x))^2 -- a regression anchor toward ConfidNet's
                 true-class probability target. Anchors the scale of s_j and supplies
                 gradient when a batch has few rank-violating pairs.

      L_SCAR = sum_j w_j ( L_CE,j + beta L_rank,j + gamma L_tcp,j )

    Per the v5 brief defaults: beta=1.0, gamma=0.5, T_rank=0.1. w_j follows the same
    increasing/exit_weights convention as the rest of the codebase. The confidence head
    output s_j is taken as torch.sigmoid(per_exit_confidence_logits[j]). The structure-aware
    mask is computed from current model predictions with no gradient (hard correctness).

    Citations: Kubaty et al. arXiv:2508.21495 (NeurIPS 2025) for the structure-aware
    relabeling; Franc et al. JMLR 2023 for the SELE pairwise rank form; Corbiere et al.
    NeurIPS 2019 (ConfidNet) for the TCP regression anchor; Geifman & El-Yaniv NeurIPS 2017
    for the selective-classification base.
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        # Regression / single-exit fallback: plain mean per-exit CE; the SCAR rank term
        # has no meaning without a discrete correctness label and at least one deeper exit.
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if len(per_exit_confidence_logits) != n:
        raise ValueError(
            f"Expected {n} confidence logit tensors to match {n} exit logit tensors; "
            f"got {len(per_exit_confidence_logits)}."
        )

    weights = exit_weights(n, weight_schedule, device=labels.device)

    # Hard predictions per exit (no grad through the structure-aware mask)
    with torch.no_grad():
        preds = torch.stack([z.argmax(dim=-1) for z in per_exit_logits], dim=0)  # [J, B]
        correct = (preds == labels.unsqueeze(0)).float()  # [J, B], 1 if correct
        # ybar_{j,i} = correct[j,i] OR (no l>j has correct[l,i]==1).
        # Compute "no deeper exit correct" via reverse-cumulative-sum:
        # deeper_correct_sum[j, i] = sum over l > j of correct[l, i].
        if n > 1:
            deeper_correct_sum = correct.flip(0).cumsum(dim=0).flip(0)
            # Subtract self contribution to get strictly l > j
            deeper_correct_sum = deeper_correct_sum - correct
        else:
            deeper_correct_sum = torch.zeros_like(correct)
        no_deeper_correct = (deeper_correct_sum == 0).float()
        ybar = torch.clamp(correct + (1.0 - correct) * no_deeper_correct, max=1.0)  # [J, B]

    total = labels.new_zeros((), dtype=torch.float32)
    per_sample_ce_stack = []
    for j in range(n):
        per_sample_ce_stack.append(
            F.cross_entropy(per_exit_logits[j], labels, reduction="none")
        )
    per_sample_ce_stack = torch.stack(per_sample_ce_stack, dim=0)  # [J, B]

    s_logits = torch.stack(per_exit_confidence_logits, dim=0)  # [J, B]
    s = torch.sigmoid(s_logits)  # [J, B] in (0, 1)

    # TCP target: true-class probability per exit, detached so the gradient on L_tcp
    # only flows through s, not through the classifier (the classifier already gets
    # gradient from L_CE).
    with torch.no_grad():
        probs_stack = torch.stack([F.softmax(z, dim=-1) for z in per_exit_logits], dim=0)
        tcp_target = probs_stack.gather(2, labels.view(1, -1, 1).expand(n, -1, 1)).squeeze(-1)
    L_tcp = ((s - tcp_target) ** 2).mean(dim=1)  # [J]

    # Pairwise rank surrogate per exit (batch-local).
    # L_rank,j = E_{i in pos_j, k in neg_j} sigma((s_j[k] - s_j[i]) / T_rank)
    # Vectorise across the batch via the outer product of (pos_mask, neg_mask).
    L_rank_per_exit = []
    for j in range(n):
        pos = ybar[j]  # [B], 1 for positive
        neg = 1.0 - pos
        n_pos = pos.sum().clamp_min(1.0)
        n_neg = neg.sum().clamp_min(1.0)
        # diff[i, k] = s_j[k] - s_j[i]
        diff = s_logits[j].unsqueeze(0) - s_logits[j].unsqueeze(1)  # [B, B]
        kernel = torch.sigmoid(diff / T_rank)
        # weight by pos_i * neg_k mask
        mask = pos.unsqueeze(1) * neg.unsqueeze(0)  # [B, B]
        L_rank_per_exit.append((kernel * mask).sum() / (n_pos * n_neg))
    L_rank = torch.stack(L_rank_per_exit, dim=0)  # [J]

    per_exit_ce = per_sample_ce_stack.mean(dim=1)  # [J]
    per_exit_loss = per_exit_ce + beta * L_rank + gamma * L_tcp  # [J]
    total = (weights * per_exit_loss).sum()
    return total


def litelaplace_loss(
    per_exit_logits, labels, *,
    regression: bool = False,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """LiteLaplace-EE (v4 candidate C4; epistemic-axis bet, meronen's mechanism class).

    This function computes ONLY the i-weighted per-exit CE term. The KL-to-unit-Gaussian
    regulariser (beta * sum_e KL(q(W_e) || N(0, I))) is computed by the model's
    ``litelaplace_kl_sum()`` and added to this base loss in ``jolt/train.py`` to keep the
    function signature stateless and model-agnostic. The model must be the variational variant
    ``resnet56_sdn_litelaplace_exit`` (or any backbone with the same litelaplace_kl_sum
    method); using LiteLaplace with a deterministic backbone reduces to plain i-weighted CE.
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    weights = exit_weights(n, weight_schedule, device=labels.device)
    per_exit_ce = _per_exit_ce(per_exit_logits, labels, regression=False)
    return (weights * per_exit_ce).sum()


def anytime_stable_distill_mono_loss(
    per_exit_logits, labels, *,
    regression: bool = False,
    tau: float = 1.0,
    eta: float = 0.5,
    alpha_min: float = 0.05,
    gamma: float = 0.5,
    distill_T: float = 1.0,
    rho_max: float = 0.5,
    anneal_fraction: float = 0.4,
    epoch_progress: float = 0.0,
    **_,
) -> torch.Tensor:
    """AnytimeStable-Distill + monotonicity hinge (v5 enhancement of v4 C7).

    Builds on anytime_stable_distill_loss (cost-aware adaptive alpha + alpha-modulated
    KL-to-final teacher) by adding the PoE-Anneal entropy-monotonicity penalty:

        L_mono = sum_{j>=1} max(0, H(p_j) - H(p_{j-1}))

    where p_j = softmax(z_j) (the raw per-exit prediction, not the temperature-softened
    distillation form). H(p_j) is the entropy of the j-th exit's prediction. The hinge
    penalises *increases* in entropy as the model goes deeper, encouraging early exits
    to be at least as confident as their successors. Combined with the cost-aware alpha
    (which already rewards early exits with high accuracy), this should shift the EMAR
    operating point earlier without breaking the AURC win from the KL distillation.

    rho(t) = rho_max * min(epoch_progress / anneal_fraction, 1). Defaults rho_max=0.5,
    anneal_fraction=0.4 (mirrors PoE-Anneal): rho ramps from 0 to 0.5 over the first
    40 % of epochs, after which the monotonicity penalty is at full strength.

    anytime_stable_distill is our 3rd-place HV candidate on WRN-28-10 at 3.189e+09 -- 3 %
    below AdaLoss's 3.281e+09. If the monotonicity term improves EMAR without regressing
    AURC, this composition could clear AdaLoss on HV.
    """
    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()

    # Reuse the anytime_stable_distill base.
    base = anytime_stable_distill_loss(
        per_exit_logits, labels, regression=regression,
        tau=tau, eta=eta, alpha_min=alpha_min, gamma=gamma, distill_T=distill_T,
    )

    # Compute per-exit entropies on the raw softmax (with grad).
    entropies = []
    for z_e in per_exit_logits:
        log_p = F.log_softmax(z_e, dim=-1)
        p = log_p.exp()
        entropies.append(-(p * log_p).sum(dim=-1).mean())
    ent = torch.stack(entropies, dim=0)  # [J]
    if n > 1:
        deltas = ent[1:] - ent[:-1]  # [J-1]
        L_mono = torch.clamp(deltas, min=0.0).sum()
    else:
        L_mono = labels.new_zeros((), dtype=torch.float32)

    rho_t = rho_max * min(epoch_progress / max(anneal_fraction, 1e-9), 1.0)
    return base + rho_t * L_mono


def anytime_stable_cold_loss(per_exit_logits, labels, *, regression: bool = False, **_) -> torch.Tensor:
    """AnytimeStable variant with tau=0.5 (sharper softmax over per-exit accuracies).

    Tests whether more aggressive adaptive weighting beats the default tau=1.0. If this wins,
    the adaptive bit is doing real work; if it ties or loses, the default tau is already
    capturing the available signal.
    """
    return anytime_stable_loss(per_exit_logits, labels, regression=regression, tau=0.5)


def anytime_stable_cheap_loss(per_exit_logits, labels, *, regression: bool = False, **_) -> torch.Tensor:
    """AnytimeStable variant with eta=1.0 (stronger cost penalty on deeper exits).

    Tests whether under-weighting of the cost preference (eta=0.5 default) is hiding
    additional EMAR gains by failing to route enough samples to cheap exits.
    """
    return anytime_stable_loss(per_exit_logits, labels, regression=regression, eta=1.0)


def asym_select_loss(
    per_exit_logits, labels, *,
    regression: bool = False,
    kappa: float = 2.0,
    tau_cov: float = 0.8,
    nu: float = 1.0,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """AsymSelect (v4 candidate C5; selective-classification probe; DG-EE-collapse-immune).

    Per exit:
      L_e = mean_i [ (correct_i + kappa * wrong_i) * CE(z_{e,i}, y_i) ]
            + nu * max(0, tau_cov - coverage_e)^2
    Total = sum_e w_e * L_e.

    Removes Deep Gamblers' abstention logit (the collapse source) and replaces it with a
    one-sided coverage-floor Lagrangian. Errors are weighted kappa = 2 x more than corrects
    (asymmetric selective risk). coverage_e in the faithful spec is an EMA across batches;
    this first-pass impl uses the CURRENT batch's mean max-softmax confidence as a stateless
    proxy.
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()

    weights = exit_weights(n, weight_schedule, device=labels.device)
    total_terms = []
    for z_e in per_exit_logits:
        with torch.no_grad():
            preds = z_e.argmax(dim=-1)
            correctness = (preds == labels).float()
            sample_weights = correctness + kappa * (1.0 - correctness)
        per_sample_ce = F.cross_entropy(z_e, labels, reduction="none")
        weighted_ce = (sample_weights * per_sample_ce).mean()

        # Coverage proxy: batch-mean max-softmax confidence (stateless approx of the spec's EMA).
        batch_confidence = F.softmax(z_e, dim=-1).max(dim=-1).values.mean()
        coverage_penalty = nu * F.relu(tau_cov - batch_confidence) ** 2

        total_terms.append(weighted_ce + coverage_penalty)
    return (weights * torch.stack(total_terms)).sum()


def asym_select_mac_loss(
    per_exit_logits, labels, *,
    per_exit_macs,
    regression: bool = False,
    kappa: float = 2.0,
    tau_cov: float = 0.8,
    nu: float = 1.0,
    lambda_mac: float = 0.15,
    T_gate: float = 0.05,
    tau: float = 0.5,
    weight_schedule: str = "increasing",
    **_,
) -> torch.Tensor:
    """AsymSelect with an additive cumulative-MAC penalty (v5 enhancement of v4 C5).

    AsymSelect (Candidate C5) gives the strongest AURC of our v4 candidates (0.058 on
    WRN-28-10) and near-canonical accuracy (0.792) but pays a 1235 M op_macs footprint --
    53 % above AdaLoss's 809 M. The HV loss is purely a compute trade-off; the metric
    quality is there. This variant grafts BudgetBoost's L_mac term onto the AsymSelect
    base so the deep-exit MAC cost is paid down without sacrificing the asymmetric
    selective-risk shaping that produced the AURC win.

    Math:
      L_asym,e = mean_i [ (correct_i + kappa * wrong_i) * CE(z_{e,i}, y_i) ]
                 + nu * max(0, tau_cov - mean(c_e))^2
      r_{j,i}  = sigmoid((c_j - tau)/T_gate) * prod_{l<j} (1 - sigmoid((c_l - tau)/T_gate))
                 with the standard residual at the final exit
      L_mac    = sum_i sum_j r_{j,i} * (m_j / m_J)
      L_total  = sum_e w_e L_asym,e + lambda_mac * L_mac

    The responsibility r is computed with gradient (unlike BudgetBoost's detached version)
    so the MAC penalty acts as a direct downward lever on op_macs; the AsymSelect base's
    correctness-weighted CE provides the counter-pressure against shallow collapse.

    Defaults: kappa=2.0, tau_cov=0.8, nu=1.0, lambda_mac=0.15, T_gate=0.05, tau=0.5.
    lambda_mac is intentionally softer than BudgetBoost's 0.3 because the asymmetric CE
    already provides selective shaping; less MAC pressure is needed to move the OP.
    """
    from .losses_zoo import exit_weights

    n = len(per_exit_logits)
    if regression or n < 2:
        return _per_exit_ce(per_exit_logits, labels, regression).mean()
    if per_exit_macs is None or len(per_exit_macs) != n:
        raise ValueError(
            f"asym_select_mac needs per_exit_macs of length {n}; got "
            f"{None if per_exit_macs is None else len(per_exit_macs)}."
        )

    device = labels.device
    weights = exit_weights(n, weight_schedule, device=device)
    mac_tensor = torch.as_tensor(per_exit_macs, dtype=torch.float32, device=device)
    mac_norm = mac_tensor / mac_tensor[-1].clamp_min(1.0)  # [J] in (0, 1]

    # AsymSelect base per exit (identical to asym_select_loss).
    asym_terms = []
    softmax_max_per_exit = []  # for the MAC term -- WITH gradient through softmax
    for z_e in per_exit_logits:
        with torch.no_grad():
            preds = z_e.argmax(dim=-1)
            correctness = (preds == labels).float()
            sample_weights = correctness + kappa * (1.0 - correctness)
        per_sample_ce = F.cross_entropy(z_e, labels, reduction="none")
        weighted_ce = (sample_weights * per_sample_ce).mean()
        batch_confidence = F.softmax(z_e, dim=-1).max(dim=-1).values  # [B], with gradient
        coverage_penalty = nu * F.relu(tau_cov - batch_confidence.mean()) ** 2
        asym_terms.append(weighted_ce + coverage_penalty)
        softmax_max_per_exit.append(batch_confidence)
    L_asym = (weights * torch.stack(asym_terms)).sum()

    # BudgetBoost-style differentiable MAC penalty. Use the WITH-gradient softmax max so
    # the MAC term backpropagates through the model's logits, directly rewarding higher
    # early-exit confidence (the asym CE counter-presses against making confidence
    # without correctness).
    confidences = torch.stack(softmax_max_per_exit, dim=0)  # [J, B], with gradient
    gates = torch.sigmoid((confidences - tau) / T_gate)  # [J, B]
    early_gates = gates[:-1]
    one_minus = 1.0 - early_gates + 1e-9
    cum_one_minus = torch.cumprod(one_minus, dim=0)
    prev_survival = torch.cat([
        torch.ones_like(early_gates[:1]),
        cum_one_minus[:-1],
    ], dim=0)
    r_early = early_gates * prev_survival
    r_final = cum_one_minus[-1:].clone()
    responsibility = torch.cat([r_early, r_final], dim=0)
    responsibility = responsibility / responsibility.sum(dim=0, keepdim=True).clamp_min(1e-6)
    L_mac = (responsibility * mac_norm.view(-1, 1)).sum(dim=0).mean()

    return L_asym + lambda_mac * L_mac


def boostnet_lin_loss(per_exit_logits, labels, *, regression: bool = False, **_) -> torch.Tensor:
    """Superseded workaround (kept for output-dir compatibility). The linspace-alpha
    schedule was a patch for the old undecayed-carry boostnet_loss (equivalent to the
    paper's degenerate t=1 setting, which failed at ~13% on Tiny-ImageNet). The main
    boostnet_loss now implements the paper's recursive t=0.5 decay, so this variant
    simply forwards to it.
    """
    return boostnet_loss(per_exit_logits, labels, regression=regression)


BASELINES: Dict[str, Callable] = {
    "adaloss": adaloss_loss,
    "branchynet": branchynet_loss,
    "eenet": eenet_loss,
    "td": td_loss,
    "meronen": meronen_loss,
    "meronen_laplace": meronen_loss,  # training loss matches meronen; post-hoc Laplace fit at end
    "deep_only": deep_only_loss,  # backbone-only training — Table 1 reference accuracy
    "boostnet": boostnet_loss,
    "boostnet_lin": boostnet_lin_loss,
    "ztw_cascade": ztw_cascade_loss,
    # v4 candidates dispatched as top-level baselines (semantically novel; technically share
    # the BASELINES dispatch because they are stateless top-level functions, not composite_loss
    # preset entries). C4 LiteLaplace-EE needs architectural variant (variational heads +
    # spectral norm); deferred.
    "cwet": cwet_loss,                       # C1
    "route_consensus": route_consensus_loss,  # C6 (headline EMAR bet)
    "sat_consensus": sat_consensus_loss,      # C3 (headline AURC bet; stateless approx)
    "anytime_stable": anytime_stable_loss,    # C2 (probe; bounded normalized aggregation)
    "anytime_stable_cold": anytime_stable_cold_loss,    # C2 tuning: tau=0.5
    "anytime_stable_cheap": anytime_stable_cheap_loss,  # C2 tuning: eta=1.0
    "anytime_stable_distill": anytime_stable_distill_loss,  # C7: AnytimeStable + KL-to-final
    "asym_select": asym_select_loss,          # C5 (probe; asymmetric selective risk)
    "litelaplace": litelaplace_loss,          # C4 (epistemic axis; KL added in trainer)
}


def baseline_loss(
    method: str, per_exit_logits: List[torch.Tensor], labels: torch.Tensor, *,
    regression: bool = False, **kwargs,
) -> torch.Tensor:
    if method not in BASELINES:
        raise ValueError(f"Unknown baseline method '{method}'. Options: {sorted(BASELINES)}.")
    return BASELINES[method](per_exit_logits, labels, regression=regression, **kwargs)


def is_composite_method(method: str) -> bool:
    return method == "composite" or method in CANDIDATE_PRESETS


def resolve_components(loss_cfg) -> LossComponents:
    """Build the LossComponents for a candidate preset or free-form composite, applying any
    ``loss_cfg.components`` overrides (validated against the LossComponents fields)."""
    if loss_cfg.method in CANDIDATE_PRESETS:
        base = CANDIDATE_PRESETS[loss_cfg.method]
    elif loss_cfg.method == "composite":
        base = LossComponents()
    else:
        raise ValueError(f"'{loss_cfg.method}' is not a composite/candidate method.")
    overrides = dict(getattr(loss_cfg, "components", {}) or {})
    valid = {f.name for f in dataclasses.fields(LossComponents)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"Unknown loss component overrides: {sorted(unknown)}")
    return dataclasses.replace(base, **overrides)


def method_loss(
    loss_cfg, per_exit_logits: List[torch.Tensor], labels: torch.Tensor, *,
    regression: bool = False, epoch_progress: float = 0.0,
) -> torch.Tensor:
    """Dispatch a non-JOLT training loss from a LossConfig (baseline or composite/candidate).

    JOLT (and AdaLoss) are handled separately in the runners because they own MultiTaskLoss
    parameters. ``epoch_progress`` is the fraction of training completed (epoch / total_epochs)
    and is forwarded to baselines that have epoch-dependent terms (e.g., EENet's late KL).
    """
    method = loss_cfg.method
    if method in BASELINES:
        return baseline_loss(
            method, per_exit_logits, labels,
            regression=regression, epoch_progress=epoch_progress,
        )
    if is_composite_method(method):
        return composite_loss(per_exit_logits, labels, resolve_components(loss_cfg), regression=regression)
    raise ValueError(f"method_loss does not handle method '{method}'.")
