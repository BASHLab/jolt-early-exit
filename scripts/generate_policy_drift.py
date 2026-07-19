#!/usr/bin/env python3
"""Policy comparison under distribution shift, across all seven datasets.

Four inference-time exit policies are run on the SAME trained JOLT model, all
pinned to the same clean B=0.5 operating point, then evaluated at the severest
shift condition of each dataset:

  quantile  running-quantile policy (streaming, window 256) -- JOLT's
  frozen    MSDNet budgeted-batch: absolute thresholds frozen from clean val
  pcee      reliability-bin patience rule, delta pinned to the clean budget
  bandit    UCB threshold selection over a fixed arm grid

For each policy we report MAC overspend % = max(0, realized - budget)/budget,
where "budget" is the validation-calibrated compute at the operating point,
averaged over the dataset's severest-severity conditions and over seeds. The
running quantile holds the budget by construction; the others overspend.

Reads per-sample shift score matrices already on disk (no retrain). Writes
outputs/analysis/policy_drift.json and fig/policy_drift.pdf.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from quantile_routing_analysis import (  # noqa: E402  (torch-free primitives)
    build_reliability, pcee_route, simulate_routing, streaming_route,
    thresholds_for_population, ucb_bandit_route,
)

B = 0.5           # operating point (matches the shift figure)
WINDOW = 256      # running-quantile streaming window

# name, pick glob (candidate cell, all seeds), shift npz filename
CELLS = [
    ("UCI-HAR", "outputs/ucihar/jolt__g0.25-lb0.15-ce2.0*/seed*", "exit_scores_sshift.npz"),
    ("PAMAP2", "outputs/pamap2/jolt__g8.0-lb1.0*/seed*", "exit_scores_sshift.npz"),
    ("GSC v2", "outputs/gsc/jolt__g0.5-lb0.5*/seed*", "exit_scores_ashift.npz"),
    ("ESC-50", "outputs/esc50/jolt__g16.0-lb2.0*/seed*", "exit_scores_ashift.npz"),
    ("SST-2", "outputs/sst2/jolt__g16.0-lb0.5-ce0.5*/seed*", "exit_scores_tshift.npz"),
    ("CIFAR-100", "outputs/cifar100/jolt__g2.0-lb0.5-ce4.0*/seed*", "exit_scores_shift.npz"),
    ("Tiny-ImageNet", "outputs/tinyimagenet/jolt__g2.0-lb0.5*/seed*", "exit_scores_shift.npz"),
]

POLICIES = ["quantile", "frozen", "pcee", "bandit"]


def operating_point(run_dir: Path):
    """B=0.5 level and its validation-calibrated compute (the budget)."""
    bj = run_dir / "budgeted.json"
    if not bj.exists():
        return None
    d = json.loads(bj.read_text())
    ok = [r for r in d["curve"] if r["val"]["macs_frac"] <= B]
    if not ok:
        return None
    row = max(ok, key=lambda r: r["val"]["accuracy"])
    return float(row["q"]), float(row["val"]["macs_frac"])


def run_seed(run_dir: Path, npz_name: str):
    """Per-policy overspend % and accuracy at this seed's severest conditions."""
    clean = run_dir / "exit_scores.npz"
    shift = run_dir / npz_name
    op = operating_point(run_dir)
    if not clean.exists() or not shift.exists() or op is None:
        return None
    q, budget = op
    z, zs = np.load(clean), np.load(shift)
    scores_val = z["scores_val"].astype(np.float64)
    correct_val = z["correct_val"].astype(np.float64)
    pem = z["per_exit_macs"].tolist()
    num_early = scores_val.shape[1]

    thr_clean = thresholds_for_population(scores_val, q)
    smax = float(scores_val.max())
    arms = list(np.linspace(0.05 * smax, 0.95 * smax, 10))

    # PCEE delta pinned so its clean-val compute matches the budget.
    diagrams = [build_reliability(scores_val[:, j], correct_val[:, j]) for j in range(num_early)]
    pcee_delta, best = 0.5, 1e9
    for d in np.linspace(0.10, 0.99, 90):
        m = pcee_route(scores_val, correct_val, pem, diagrams, float(d))["macs_frac"]
        if abs(m - budget) < best:
            best, pcee_delta = abs(m - budget), float(d)

    sev5 = sorted(k[len("scores_"):] for k in zs.files
                  if k.startswith("scores_") and k.endswith("_5"))
    if not sev5:
        return None
    acc = {p: [] for p in POLICIES}
    over = {p: [] for p in POLICIES}
    for cond in sev5:
        sc = zs[f"scores_{cond}"].astype(np.float64)
        co = zs[f"correct_{cond}"].astype(np.float64)
        idx = np.argsort((np.arange(len(sc)) * 2654435761) % 2**32)
        sc, co = sc[idx], co[idx]
        routes = {
            "quantile": streaming_route(sc, co, q, pem, WINDOW, thr_clean),
            "frozen": simulate_routing(sc, co, thr_clean, pem),
            "pcee": pcee_route(sc, co, pem, diagrams, pcee_delta),
            "bandit": ucb_bandit_route(sc, co, pem, arms),
        }
        for p, r in routes.items():
            over[p].append((r["macs_frac"] - budget) / budget * 100.0)
            acc[p].append(r["accuracy"] * 100.0)
    return {"budget": budget, "q": q, "n_cond": len(sev5),
            "over": {p: float(np.mean(v)) for p, v in over.items()},
            "acc": {p: float(np.mean(v)) for p, v in acc.items()}}


def build():
    out = {}
    for name, glob_pat, npz_name in CELLS:
        seeds = []
        for run_dir in sorted(REPO.glob(glob_pat)):
            r = run_seed(run_dir, npz_name)
            if r is not None:
                seeds.append(r)
        if not seeds:
            print(f"[WARN] {name}: no usable seed")
            continue
        agg = {"n_seed": len(seeds), "n_cond": seeds[0]["n_cond"],
               "over": {p: float(np.mean([s["over"][p] for s in seeds])) for p in POLICIES},
               "acc": {p: float(np.mean([s["acc"][p] for s in seeds])) for p in POLICIES}}
        out[name] = agg
        o = agg["over"]
        print(f"{name:14s} seeds={agg['n_seed']} conds={agg['n_cond']}  overspend%: "
              + "  ".join(f"{p}={o[p]:+.1f}" for p in POLICIES))
    dst = REPO / "outputs/analysis/policy_drift.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"Wrote {dst} ({len(out)} cells)")
    return out


def render(data):
    if not data:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [c[0] for c in CELLS if c[0] in data]
    labels = {"quantile": "Running quantile (ours)", "frozen": "Frozen thresholds",
              "pcee": "PCEE", "bandit": "UCB bandit"}
    colors = {"quantile": "#d4a017", "frozen": "#4a6fa5",
              "pcee": "#c0504d", "bandit": "#7f7f7f"}
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for i, p in enumerate(POLICIES):
        # Overspend of a budget cap is max(0, realized - budget).
        vals = [max(0.0, data[n]["over"][p]) for n in names]
        ax.bar(x + (i - 1.5) * w, vals, w, label=labels[p], color=colors[p])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Budget overspend (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig_dir = REPO / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dst = fig_dir / "policy_drift.pdf"
    fig.savefig(dst, bbox_inches="tight")
    print(f"Wrote {dst}")


if __name__ == "__main__":
    render(build())
