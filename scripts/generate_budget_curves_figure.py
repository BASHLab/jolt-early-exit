"""Accuracy-versus-budget curves figure (replaces the retired Pareto scatter).

One panel per dataset. For every method, the test accuracy of each
validation-feasible exit-fraction level is plotted against its realized test
compute, averaged across seeds on a common budget grid; the three contracts
are marked as vertical lines. JOLT is the thick gold line; baselines are thin
gray with the strongest baseline (highest accuracy at the loosest contract) in
blue for reference. Output: figures/budget_curves.pdf.

Data: the same budgeted.json pool the main table reads, via
generate_budget_table's registry, so figure and table stay in lock-step.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from generate_budget_table import (  # noqa: E402
    BASELINES, BUDGETS, CELLS, FIXED, LATEX, REPO as TREPO,
    candidate_at_contract, candidate_tag_pool, seed_jsons,
)

OUT = REPO / "figures/budget_curves.pdf"

mpl.rcParams.update({
    "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 7,
    "legend.fontsize": 5, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

GRID = np.linspace(0.15, 1.0, 60)


def method_curve(runs):
    """Mean best-feasible test accuracy on the shared budget grid.

    For each budget b, per seed: among levels whose validation compute is
    within b, the level with the highest validation accuracy (the same rule
    as the table); its test accuracy enters the mean. NaN where infeasible.
    """
    out = np.full(GRID.shape, np.nan)
    per_seed = []
    for r in runs:
        rows = sorted(r["curve"], key=lambda x: x["val"]["macs_frac"])
        vmacs = np.array([x["val"]["macs_frac"] for x in rows])
        vacc = np.array([x["val"]["accuracy"] for x in rows])
        tacc = np.array([x["test"]["accuracy"] for x in rows])
        seed_vals = np.full(GRID.shape, np.nan)
        for gi, b in enumerate(GRID):
            ok = vmacs <= b
            if ok.any():
                seed_vals[gi] = tacc[ok][np.argmax(vacc[ok])] * 100
        per_seed.append(seed_vals)
    if not per_seed:
        return out
    stack = np.vstack(per_seed)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=0)


def main() -> None:
    cells = [c for c in CELLS if not c.get("pending")]
    ncols = len(cells)
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.1, 1.55))
    axes = np.atleast_2d(axes)

    for idx, cell in enumerate(cells):
        ax = axes[idx // ncols][idx % ncols]
        method_runs = {}
        for m in BASELINES:
            if m in FIXED:
                method_runs[m] = seed_jsons(TREPO / cell["fix_root"], m)
            else:
                runs = []
                for r in cell["base_roots"]:
                    runs.extend(seed_jsons(TREPO / r, m))
                method_runs[m] = runs
        pool = candidate_tag_pool(cell)
        _, ours = candidate_at_contract(pool, BUDGETS[-1])

        # strongest baseline at the loosest contract, for the blue reference
        best_m, best_val = None, -1
        curves = {}
        for m, runs in method_runs.items():
            if not runs:
                continue
            c = method_curve(runs)
            curves[m] = c
            ref = c[np.searchsorted(GRID, BUDGETS[-1])]
            if np.isfinite(ref) and ref > best_val:
                best_m, best_val = m, ref

        for m, c in curves.items():
            if m == best_m:
                continue
            ax.plot(GRID, c, color="0.75", lw=0.7, zorder=1)
        if best_m:
            ax.plot(GRID, curves[best_m], color="#1f77b4", lw=1.1, zorder=2,
                    label=LATEX[best_m])
        ax.plot(GRID, method_curve(ours), color="#d4a017", lw=1.8, zorder=3,
                label="JOLT")
        for b in BUDGETS:
            ax.axvline(b, color="0.85", lw=0.6, ls=":", zorder=0)
        ax.set_title(cell["name"], fontsize=7)
        ax.set_xlim(0.12, 1.0)
        finite = np.concatenate([c[np.isfinite(c)] for c in curves.values()] +
                                [method_curve(ours)[np.isfinite(method_curve(ours))]])
        lo = np.percentile(finite, 5)
        ax.set_ylim(max(lo - 2, 0), finite.max() + 1)
        ax.legend(loc="lower right", frameon=False, handlelength=1.1,
                  fontsize=5, borderaxespad=0.1, labelspacing=0.2)
        if idx % ncols == 0:
            ax.set_ylabel("test accuracy (%)")

    for j in range(len(cells), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    OUT.parent.mkdir(exist_ok=True)
    fig.tight_layout(w_pad=0.5)
    fig.supxlabel("average compute (fraction of full-depth cost)", fontsize=7, y=0.02)
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
