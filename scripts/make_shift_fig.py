"""Shift figure: budget overspend versus severity for five exit policies on
JOLT's fixed exits, one panel per dataset (the paper's fig:shift_ladders).
The running quantile stays on the budget line under shift while frozen
thresholds, PCEE, RC-EENN, and the UCB bandit drift off it.

Routing is identical to generate_policy_drift (same calibration, same
primitives), so each line's severest endpoint matches the overspend numbers
in outputs/analysis/policy_drift.json and the paper text. Reads the
per-sample shift score matrices already on disk (no retrain). Writes
figures/shift_ladders.pdf.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import generate_policy_drift as gp  # noqa: E402
from quantile_routing_analysis import (  # noqa: E402
    build_reliability, pcee_route, simulate_routing, streaming_route,
    thresholds_for_population, ucb_bandit_route, rc_eenn_thresholds,
)

B, WINDOW, POLICIES, CELLS = gp.B, gp.WINDOW, gp.POLICIES, gp.CELLS


def run_seed_ladder(run_dir: Path, npz_name: str):
    """Per-policy deployed compute (macs_frac) at each severity bucket, plus
    the clean operating point. Severity buckets group all corruptions sharing
    a trailing integer severity; the severest bucket reproduces run_seed()."""
    clean, shift = run_dir / "exit_scores.npz", run_dir / npz_name
    op = gp.operating_point(run_dir)
    if not clean.exists() or not shift.exists() or op is None:
        return None
    q, budget = op
    z, zs = np.load(clean), np.load(shift)
    sv, cv = z["scores_val"].astype(np.float64), z["correct_val"].astype(np.float64)
    pem = z["per_exit_macs"].tolist()
    ne = sv.shape[1]
    thr = thresholds_for_population(sv, q)
    rc = rc_eenn_thresholds(sv, cv, pem, budget)
    smax = float(sv.max())
    arms = list(np.linspace(0.05 * smax, 0.95 * smax, 10))
    diag = [build_reliability(sv[:, j], cv[:, j]) for j in range(ne)]
    pd_, best = 0.5, 1e9
    for d in np.linspace(0.10, 0.99, 90):
        m = pcee_route(sv, cv, pem, diag, float(d))["macs_frac"]
        if abs(m - budget) < best:
            best, pd_ = abs(m - budget), float(d)

    by_sev: dict[int, list[str]] = {}
    for k in zs.files:
        if not k.startswith("scores_"):
            continue
        cond = k[len("scores_"):]
        parts = cond.rsplit("_", 1)
        if len(parts) < 2 or not parts[1].isdigit():
            continue  # skip non-severity OOD conditions (e.g. SST-2 imdb)
        by_sev.setdefault(int(parts[1]), []).append(cond)
    sevs = sorted(by_sev)
    if not sevs:
        return None

    mac = {p: [budget] for p in POLICIES}  # rung 0 = clean operating point
    for s in sevs:
        per = {p: [] for p in POLICIES}
        for cond in by_sev[s]:
            sc = zs[f"scores_{cond}"].astype(np.float64)
            co = zs[f"correct_{cond}"].astype(np.float64)
            idx = np.argsort((np.arange(len(sc)) * 2654435761) % 2**32)
            sc, co = sc[idx], co[idx]
            routes = {
                "quantile": streaming_route(sc, co, q, pem, WINDOW, thr),
                "frozen": simulate_routing(sc, co, thr, pem),
                "pcee": pcee_route(sc, co, pem, diag, pd_),
                "bandit": ucb_bandit_route(sc, co, pem, arms),
                "rc_eenn": simulate_routing(sc, co, rc, pem),
            }
            for p, r in routes.items():
                per[p].append(r["macs_frac"])
        for p in POLICIES:
            mac[p].append(float(np.mean(per[p])))
    return {"budget": budget, "sevs": sevs, "mac": mac}


def build_ladders():
    out = {}
    for name, glob_pat, npz_name in CELLS:
        seeds = [run_seed_ladder(rd, npz_name) for rd in sorted(REPO.glob(glob_pat))]
        seeds = [s for s in seeds if s]
        if not seeds:
            print(f"[WARN] {name}: no usable seed")
            continue
        sevs = seeds[0]["sevs"]
        k = len(sevs) + 1
        budget = float(np.mean([s["budget"] for s in seeds]))
        mac = {p: np.mean(np.array([s["mac"][p][:k] for s in seeds], float), 0).tolist()
               for p in POLICIES}
        out[name] = {"budget": budget, "sevs": sevs, "mac": mac}
        sev_over = {p: (mac[p][-1] - budget) / budget * 100 for p in POLICIES}
        print(f"{name:14s} rungs={k}  severest overspend%: "
              + " ".join(f"{p}={sev_over[p]:+.0f}" for p in POLICIES))
    return out


def render_ladders(data):
    import matplotlib
    matplotlib.use("Agg")
    # Type 42 (TrueType) rather than matplotlib's default Type 3, which
    # IEEE Xplore does not accept in camera-ready PDFs.
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    import matplotlib.pyplot as plt

    names = [c[0] for c in CELLS if c[0] in data]
    labels = {"quantile": "Running quantile (ours)", "frozen": "Frozen thresholds",
              "pcee": "PCEE", "bandit": "UCB bandit", "rc_eenn": "RC-EENN (risk control)"}
    colors = {"quantile": "#d4a017", "frozen": "#4a6fa5", "pcee": "#c0504d",
              "bandit": "#7f7f7f", "rc_eenn": "#59a14f"}
    order = ["quantile", "frozen", "pcee", "rc_eenn", "bandit"]
    rung_lab = {3: ["clean", "mild", "mod.", "sev."], 1: ["clean", "sev."]}

    fig, axes = plt.subplots(1, len(names), figsize=(7.1, 1.4))
    for i, n in enumerate(names):
        ax = axes[i]
        nsev = len(data[n]["sevs"])
        x = np.arange(nsev + 1)
        b = data[n]["budget"]
        # Budget is a compute cap: overspend = max(0, realized - budget);
        # coming in under the cap is zero overspend, not a negative one.
        ax.axhline(0.0, color="0.25", lw=0.7, ls=":", zorder=0)
        for p in order:
            over = [max(0.0, (m - b) / b * 100.0) for m in data[n]["mac"][p]]
            ax.plot(x, over, "-o", color=colors[p], lw=1.0, ms=2.2,
                    zorder=3 if p == "quantile" else 2, label=labels[p])
        ax.set_title(n, fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(rung_lab.get(nsev, ["clean"] + [str(s) for s in data[n]["sevs"]]),
                           fontsize=5, rotation=0)
        ax.tick_params(axis="y", labelsize=5)
        if i == 0:
            ax.set_ylabel("budget overspend (%)", fontsize=6.5)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, ncol=5, fontsize=6, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.97), columnspacing=1.2,
               handlelength=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.3)
    fig_dir = REPO / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dst = fig_dir / "shift_ladders.pdf"
    fig.savefig(dst, dpi=300, bbox_inches="tight")
    print(f"Wrote {dst}")


if __name__ == "__main__":
    render_ladders(build_ladders())
