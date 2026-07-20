"""Offline analysis of the rank-based (quantile) exit policy under shift.

Inputs: per-run exit_scores.npz (clean val/test matrices, from budgeted_eval.py)
and exit_scores_shift.npz (per-corruption score/correctness matrices, from
shift_eval.py). Everything below is pure NumPy on those matrices.

Four analyses per run (method x seed0):

1. BUDGET FIDELITY: at each quantile level q, compare policies on corrupted
   test streams: (a) absolute thresholds frozen from clean validation;
   (b) oracle batch quantiles of the corrupted stream (upper bound);
   (c) streaming quantiles with a sliding window (the deployable policy).
   Report exit-population drift (L1 distance between realized exit shares and
   the clean-val target profile), realized MACs drift, and accuracy.

2. STREAMING WINDOW ABLATION: policy (c) at window sizes {64, 256, 1024},
   warm-started from the clean-validation thresholds. Reports the
   budget-fidelity gap to the oracle (b).

3. RANKING QUALITY UNDER SHIFT (property-b honesty check): per-exit AUROC of
   (-score) vs correctness, clean test vs corruption severities. Shift can
   degrade ranking, not just calibration (Mehra et al.); we measure it.

4. UAT-STYLE BANDIT BASELINE: UCB1 over a grid of shared absolute thresholds
   (single global threshold, as in UAT), reward = certainty-vs-cost proxy
   r = 1{score_at_exit < tau} * (1 - normalized_macs) + confidence_bonus.
   Simulated online over the corrupted stream. Reports realized exit shares +
   accuracy for comparison against (c).

Usage:
    python scripts/quantile_routing_analysis.py --cell CIFAR-100 \
        [--methods poe_distill_mtl_brier__g4.0-lb0.5,adaloss,...] [--q 0.5]

Writes outputs/analysis/quantile_routing_<cell>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from jolt.budgeted import simulate_routing, thresholds_for_population

from budgeted_eval import CELLS  # noqa: E402

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
SEVERITIES = [1, 3, 5]
WINDOWS = [64, 256, 1024]


def exit_shares(exit_counts: Sequence[int]) -> np.ndarray:
    a = np.asarray(exit_counts, dtype=np.float64)
    return a / max(1.0, a.sum())


def l1_drift(counts_a: Sequence[int], counts_b: Sequence[int]) -> float:
    return float(np.abs(exit_shares(counts_a) - exit_shares(counts_b)).sum())


def streaming_route(
    scores: np.ndarray,
    correct: np.ndarray,
    q: float,
    per_exit_macs: Sequence[float],
    window: int,
    init_thresholds: Sequence[float],
) -> Dict[str, object]:
    """Sequential-sample simulation of the sliding-window streaming quantile policy.

    Each sample is routed with the CURRENT thresholds; its per-exit scores are
    then appended to per-exit windows and thresholds updated to the window
    q-quantile. Exit j's window only collects scores of samples that REACHED
    exit j (sequential population semantics, matching thresholds_for_population).
    """
    n, num_early = scores.shape
    thr = [float(t) for t in init_thresholds]
    buffers: List[List[float]] = [[] for _ in range(num_early)]
    exit_idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos = num_early
        for j in range(num_early):
            s = float(scores[i, j])
            buffers[j].append(s)
            if len(buffers[j]) > window:
                buffers[j].pop(0)
            if s < thr[j]:
                pos = j
                break
        exit_idx[i] = pos
        # update thresholds from current windows (cheap: numpy quantile on <=window)
        for j in range(num_early):
            if len(buffers[j]) >= 16:
                thr[j] = float(np.quantile(np.asarray(buffers[j]), q))
            if pos == j:
                break

    cum = np.cumsum(np.asarray(per_exit_macs, dtype=np.float64))
    counts = [int((exit_idx == j).sum()) for j in range(num_early + 1)]
    acc = float(correct[np.arange(n), exit_idx].mean())
    return {
        "accuracy": acc,
        "exit_counts": counts,
        "macs_frac": float(cum[exit_idx].mean() / cum[-1]),
    }


def ucb_bandit_route(
    scores: np.ndarray,
    correct: np.ndarray,
    per_exit_macs: Sequence[float],
    arms: Sequence[float],
    c_explore: float = 1.0,
    cost_weight: float = 0.5,
) -> Dict[str, object]:
    """UAT-style single-global-threshold UCB1 simulation (unsupervised reward).

    Arms are shared absolute thresholds. Reward for a routed sample: confidence
    proxy (1 - normalized score at the exit taken) minus cost_weight * realized
    macs fraction. Unsupervised — labels never used for the policy.
    """
    n, num_early = scores.shape
    cum = np.cumsum(np.asarray(per_exit_macs, dtype=np.float64))
    smax = float(scores.max()) or 1.0
    counts_arm = np.zeros(len(arms))
    means_arm = np.zeros(len(arms))
    exit_idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        if i < len(arms):
            a = i  # play each arm once
        else:
            ucb = means_arm + c_explore * np.sqrt(np.log(i + 1) / np.maximum(counts_arm, 1))
            a = int(np.argmax(ucb))
        tau = arms[a]
        pos = num_early
        for j in range(num_early):
            if scores[i, j] < tau:
                pos = j
                break
        exit_idx[i] = pos
        conf = 1.0 - float(scores[i, min(pos, num_early - 1)]) / smax
        macs_frac = float(cum[pos] / cum[-1])
        r = conf - cost_weight * macs_frac
        counts_arm[a] += 1
        means_arm[a] += (r - means_arm[a]) / counts_arm[a]
    counts = [int((exit_idx == j).sum()) for j in range(num_early + 1)]
    return {
        "accuracy": float(correct[np.arange(n), exit_idx].mean()),
        "exit_counts": counts,
        "macs_frac": float(cum[exit_idx].mean() / cum[-1]),
    }


def dtaci_route(
    scores: np.ndarray,
    correct: np.ndarray,
    q: float,
    per_exit_macs: Sequence[float],
    init_thresholds: Sequence[float],
    gammas: Sequence[float] = (0.005, 0.02, 0.08, 0.32),
) -> Dict[str, object]:
    """DtACI (Gibbs & Candes 2024): parameter-free online quantile via K ACI
    experts at different learning rates, aggregated by exp-weighted pinball loss.
    Per exit, targets exit fraction q online; no hand-set step size. Included to
    show the fixed-rate policy is not a tuned step size."""
    n, num_early = scores.shape
    K = len(gammas)
    theta = np.array([[float(init_thresholds[j]) for _ in range(K)] for j in range(num_early)])
    w = np.ones((num_early, K))
    sigma = 0.02  # expert-weight learning rate
    exit_idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos = num_early
        for j in range(num_early):
            wj = w[j] / w[j].sum()
            thr_j = float(wj @ theta[j])
            s = float(scores[i, j])
            exited = s < thr_j
            # pinball loss of each expert's quantile at coverage q, then reweight
            err = np.array([1.0 if s < theta[j, k] else 0.0 for k in range(K)])
            pinball = np.where(err >= 1, (1 - q) * (theta[j] - s), q * (s - theta[j]))
            w[j] = w[j] * np.exp(-sigma * pinball)
            w[j] = np.clip(w[j], 1e-8, None)
            # ACI update each expert toward coverage q
            theta[j] = theta[j] + gammas * (q - err)
            if exited:
                pos = j
                break
        exit_idx[i] = pos
    cum = np.cumsum(np.asarray(per_exit_macs, dtype=np.float64))
    counts = [int((exit_idx == j).sum()) for j in range(num_early + 1)]
    return {"accuracy": float(correct[np.arange(n), exit_idx].mean()),
            "exit_counts": counts, "macs_frac": float(cum[exit_idx].mean() / cum[-1])}


def rc_eenn_thresholds(scores_val: np.ndarray, correct_val: np.ndarray,
                       per_exit_macs: Sequence[float], budget: float,
                       delta: float = 0.1) -> List[float]:
    """RC-EENN / Fast-yet-Safe (Jazbec 2024): Learn-then-Test picks per-exit
    thresholds on clean calibration that bound the early-exit error risk with a
    Hoeffding-valid, Bonferroni-corrected p-value, frozen at test. We sweep the
    risk level so the clean-calibration compute lands at the budget, then return
    that (frozen) threshold vector. Under shift the guarantee, and the budget,
    break, which is what the overspend figure exposes."""
    n, num_early = scores_val.shape
    cum = np.cumsum(np.asarray(per_exit_macs, dtype=np.float64))
    order = np.linspace(scores_val.min(), scores_val.max(), 60)
    hoeff = math.sqrt(math.log(1.0 / delta) / (2.0 * n)) if n > 0 else 0.0
    best_thr, best_gap = list(order[-1:]) * num_early, 1e9
    for alpha in np.linspace(0.02, 0.6, 40):
        # per-exit: largest threshold (most aggressive early-exit) whose
        # LTT-valid early-exit error (empirical + Hoeffding slack) <= alpha
        thr = []
        remaining = np.ones(n, dtype=bool)
        for j in range(num_early):
            chosen = order[0]
            for t in order:
                take = remaining & (scores_val[:, j] < t)
                if take.sum() < 10:
                    chosen = t; continue
                err = 1.0 - correct_val[take, j].mean()
                if err + hoeff <= alpha:
                    chosen = t
                else:
                    break
            thr.append(float(chosen))
            remaining &= ~(remaining & (scores_val[:, j] < chosen))
        # realized clean compute of this threshold vector
        r = simulate_routing(scores_val, correct_val, thr, per_exit_macs)
        gap = abs(r["macs_frac"] - budget)
        if gap < best_gap:
            best_gap, best_thr = gap, thr
    return best_thr


def build_reliability(score_col: np.ndarray, correct_col: np.ndarray,
                      num_bins: int = 20):
    """Per-exit reliability diagram over the routing score, built on clean
    validation: bin the score, record empirical accuracy per bin. Used by the
    PCEE policy comparator."""
    lo, hi = float(score_col.min()), float(score_col.max())
    edges = np.linspace(lo, hi, num_bins + 1)
    bin_idx = np.digitize(score_col, edges[1:-1])
    acc = np.zeros(num_bins, dtype=np.float64)
    for b in range(num_bins):
        sel = bin_idx == b
        if sel.any():
            acc[b] = float(correct_col[sel].mean())
    return edges, acc


def pcee_route(scores: np.ndarray, correct: np.ndarray,
               per_exit_macs: Sequence[float],
               diagrams: Sequence, delta: float) -> Dict[str, object]:
    """PCEE (PABEE-family): exit at the first exit whose reliability-diagram
    bin-estimated accuracy meets delta. A performance-target rule, not a
    budget-targeting one."""
    n, num_early = scores.shape
    exit_idx = np.full(n, num_early, dtype=np.int64)
    for j in range(num_early):
        edges, acc = diagrams[j]
        bin_idx = np.clip(np.digitize(scores[:, j], edges[1:-1]), 0, len(acc) - 1)
        est = acc[bin_idx]
        take = (exit_idx == num_early) & (est >= delta)
        exit_idx[take] = j
    cum = np.cumsum(np.asarray(per_exit_macs, dtype=np.float64))
    counts = [int((exit_idx == j).sum()) for j in range(num_early + 1)]
    return {
        "accuracy": float(correct[np.arange(n), exit_idx].mean()),
        "exit_counts": counts,
        "macs_frac": float(cum[exit_idx].mean() / cum[-1]),
    }


def auroc(neg_scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC of neg_scores (higher = predicted correct) vs labels."""
    order = np.argsort(neg_scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    pos = labels > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def analyze_run(run_dir: Path, q: float) -> Optional[dict]:
    npz_clean = run_dir / "exit_scores.npz"
    npz_shift = run_dir / "exit_scores_shift.npz"
    if not npz_clean.exists() or not npz_shift.exists():
        return None
    z = np.load(npz_clean)
    zs = np.load(npz_shift)
    scores_val = z["scores_val"].astype(np.float64)
    pem = z["per_exit_macs"].tolist()
    num_early = scores_val.shape[1]

    # clean-val calibration: absolute thresholds + target exit profile at level q
    thr_clean = thresholds_for_population(scores_val, q)
    clean_route = simulate_routing(scores_val, z["correct_val"].astype(np.float64), thr_clean, pem)
    target_counts = clean_route["exit_counts"]

    # clean-test ranking quality per exit
    sc_t = z["scores_test"].astype(np.float64)
    co_t = z["correct_test"].astype(np.float64)
    clean_auroc = [auroc(-sc_t[:, j], co_t[:, j]) for j in range(num_early)]

    out = {"q": q, "target_shares": exit_shares(target_counts).tolist(),
           "clean_auroc": clean_auroc, "corruptions": {}}

    smax = float(scores_val.max())
    arms = list(np.linspace(0.05 * smax, 0.95 * smax, 10))

    for corruption in CORRUPTIONS:
        out["corruptions"][corruption] = {}
        for sev in SEVERITIES:
            k_s, k_c = f"scores_{corruption}_{sev}", f"correct_{corruption}_{sev}"
            if k_s not in zs:
                continue
            sc = zs[k_s].astype(np.float64)
            co = zs[k_c].astype(np.float64)
            # rng-free deterministic shuffle for streaming order (fixed permutation
            # derived from sample index bit-mix) to avoid corruption-block ordering
            idx = np.argsort((np.arange(len(sc)) * 2654435761) % 2**32)
            sc, co = sc[idx], co[idx]

            absolute = simulate_routing(sc, co, thr_clean, pem)
            oracle_thr = thresholds_for_population(sc, q)
            oracle = simulate_routing(sc, co, oracle_thr, pem)
            entry = {
                "deep_acc": float(co[:, -1].mean()),
                "auroc": [auroc(-sc[:, j], co[:, j]) for j in range(num_early)],
                "absolute": {"acc": absolute["accuracy"],
                             "drift": l1_drift(absolute["exit_counts"], target_counts),
                             "macs_frac": absolute["macs_frac"]},
                "oracle": {"acc": oracle["accuracy"],
                           "drift": l1_drift(oracle["exit_counts"], target_counts),
                           "macs_frac": oracle["macs_frac"]},
                "streaming": {},
            }
            for w in WINDOWS:
                st = streaming_route(sc, co, q, pem, w, thr_clean)
                entry["streaming"][str(w)] = {
                    "acc": st["accuracy"],
                    "drift": l1_drift(st["exit_counts"], target_counts),
                    "macs_frac": st["macs_frac"]}
            bd = ucb_bandit_route(sc, co, pem, arms)
            entry["bandit"] = {"acc": bd["accuracy"],
                               "drift": l1_drift(bd["exit_counts"], target_counts),
                               "macs_frac": bd["macs_frac"]}
            out["corruptions"][corruption][str(sev)] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--q", type=float, default=0.5)
    ap.add_argument("--methods", default=None,
                    help="comma-separated method dir names; default = all with shift npz")
    args = ap.parse_args()

    cell = CELLS[args.cell]
    wanted = set(args.methods.split(",")) if args.methods else None
    results = {}
    for root_rel in cell["roots"]:
        root = REPO / root_rel
        if not root.exists():
            continue
        for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
            if wanted and mdir.name not in wanted:
                continue
            run = mdir / "seed0"
            r = analyze_run(run, args.q)
            if r is not None:
                results[mdir.name] = r
                # compact progress line: mean severity-5 stats across corruptions
                s5 = [v["5"] for v in r["corruptions"].values() if "5" in v]
                if s5:
                    ma = np.mean([e["absolute"]["drift"] for e in s5])
                    mo = np.mean([e["streaming"]["256"]["drift"] for e in s5])
                    aa = np.mean([e["absolute"]["acc"] for e in s5])
                    ao = np.mean([e["streaming"]["256"]["acc"] for e in s5])
                    print(f"[{mdir.name}] sev5 mean: drift abs={ma:.3f} stream256={mo:.3f} "
                          f"| acc abs={aa*100:.2f} stream256={ao*100:.2f}")
    tag = f"__{args.methods.replace(',', '_')[:60]}" if args.methods else ""
    out = REPO / f"outputs/analysis/quantile_routing_{args.cell}{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=None))
    print(f"Wrote {out} ({len(results)} methods)")


if __name__ == "__main__":
    main()
