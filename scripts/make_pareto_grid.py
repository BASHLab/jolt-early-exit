"""Per-comparison accuracy-compute Pareto grid: 7 datasets (columns) x 3
budgets (rows), the paper's fig:pareto_grid. Each panel plots the eleven
methods at that dataset-budget comparison in the (MAC saved, accuracy) plane;
JOLT is a star, gold where it is Pareto-undominated and red where a baseline
dominates it (higher-or-equal accuracy at equal-or-lower compute, one strict).
Comparisons whose budget lies below the first exit's cost are infeasible and
left blank.

Built from the same per-run statistics as the main table and paper_numbers,
so the count of red (dominated) panels matches the reported Pareto counts.
Writes figures/pareto_grid.pdf.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from generate_budget_table import (  # noqa: E402
    BASELINES, BUDGETS, CELLS, baseline_pick, candidate_at_contract,
    candidate_tag_pool, stat as gstat,
)


def dominated(ours, others):
    """JOLT dominated iff some baseline has acc >= ours and mac <= ours,
    one strict."""
    return any(
        x[0] >= ours[0] - 1e-9 and x[3] <= ours[3] + 1e-9
        and (x[0] > ours[0] + 1e-9 or x[3] < ours[3] - 1e-9)
        for x in others if x is not None)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [c["name"] for c in CELLS]
    C_OURS = "#d4a017"
    C_DOM = "#c0504d"
    C_BASE = "0.62"
    fig, axes = plt.subplots(len(BUDGETS), len(order), figsize=(7.1, 1.7),
                             squeeze=False)
    titled = set()
    n_dom = n_tot = 0
    pools = {c["name"]: candidate_tag_pool(c) for c in CELLS}
    for r, b in enumerate(BUDGETS):
        for c, cell in enumerate(CELLS):
            name = cell["name"]
            ax = axes[r][c]
            ours = gstat(candidate_at_contract(pools[name], b)[1], b)
            others = [st for m in BASELINES
                      if (st := baseline_pick(cell, m, b)[1]) is not None]
            if ours is None or not others:
                ax.axis("off")
                continue
            # Title each column at its first feasible budget (columns with no
            # B=0.3 comparison would otherwise be unlabelled).
            if name not in titled:
                ax.set_title(name, fontsize=6.5, pad=3)
                titled.add(name)
            xs = [(1.0 - x[3]) * 100 for x in others]  # MAC saved %
            ys = [x[0] for x in others]                # accuracy %
            ax.scatter(xs, ys, s=9, facecolors="none", edgecolors=C_BASE,
                       linewidths=0.6, zorder=2)
            n_tot += 1
            dom = dominated(ours, others)
            n_dom += int(dom)
            ax.scatter([(1.0 - ours[3]) * 100], [ours[0]], marker="*",
                       s=90, facecolors=(C_DOM if dom else C_OURS),
                       edgecolors="black", linewidths=0.5, zorder=4)
            ax.tick_params(labelsize=4.5, length=2, pad=1)
            if c == 0:
                ax.set_ylabel(f"$B{{=}}{b}$", fontsize=6.5)
    handles = [
        plt.Line2D([], [], marker="*", ls="none", ms=9, mfc=C_OURS,
                   mec="black", mew=0.5, label="JOLT, undominated"),
        plt.Line2D([], [], marker="*", ls="none", ms=9, mfc=C_DOM,
                   mec="black", mew=0.5, label="JOLT, dominated"),
        plt.Line2D([], [], marker="o", ls="none", ms=5, mfc="none",
                   mec=C_BASE, mew=0.8, label="baseline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.25, h_pad=0.25)
    fig_dir = REPO / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dst = fig_dir / "pareto_grid.pdf"
    fig.savefig(dst, dpi=300, bbox_inches="tight")
    print(f"Wrote {dst}; JOLT dominated in {n_dom} of {n_tot} comparisons")


if __name__ == "__main__":
    main()
