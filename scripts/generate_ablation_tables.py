"""Generate the loss leave-one-out ablation table from budgeted.json files.

One row per removed term, one column per dataset; each entry is the change in
test accuracy at the dataset's tightest feasible budget (where the terms
matter most) when that term is removed from the pick, mean over three seeds.
The full-objective row carries absolute accuracy so the deltas have an anchor.

Prints LaTeX; --insert writes tables/ablation_loo.tex.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[1]
TEX = REPO / "tables"

# (dataset, pick tag, tightest feasible budget, pick roots, loo root)
CELLS = [
    ("UCI-HAR", "g0.25-lb0.15-ce2.0", 0.3, ["outputs/ucihar"], "outputs/ucihar_loo"),
    ("PAMAP2", "g8.0-lb1.0", 0.3, ["outputs/pamap2"], "outputs/pamap2_loo"),
    ("CIFAR-100", "g2.0-lb0.5-ce4.0", 0.5, ["outputs/cifar100"], "outputs/cifar100_loo"),
    ("GSC v2", "g0.5-lb0.5", 0.5, ["outputs/gsc"], "outputs/gsc_loo"),
    ("SST-2", "g16.0-lb0.5-ce0.5", 0.3, ["outputs/sst2"], "outputs/sst2_loo"),
    ("ESC-50", "g16.0-lb2.0", 0.3, ["outputs/esc50"], "outputs/esc50_loo"),
    ("Tiny-ImageNet", "g2.0-lb0.5", 0.5, ["outputs/tinyimagenet"], "outputs/tinyimagenet_loo"),
]
ARMS = [("poe_multitask_brier", r"$-$ distillation"),
        ("poe_distill_brier", r"$-$ learned weighting"),
        ("poe_distill_mtl", r"$-$ Brier anchor")]


def acc_at(path: Path, b: float):
    d = json.loads(path.read_text())
    ok = [r for r in d["curve"] if r["val"]["macs_frac"] <= b]
    if not ok:
        return None
    return max(ok, key=lambda r: r["val"]["accuracy"])["test"]["accuracy"] * 100


def runs_stat(patterns, b):
    vals = []
    for pat in patterns:
        for p in REPO.glob(pat):
            v = acc_at(p, b)
            if v is not None:
                vals.append(v)
        if len(vals) >= 3:
            break
    if not vals:
        return None
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def build() -> str:
    full_row, arm_rows = [], {label: [] for _, label in ARMS}
    for name, tag, b, pick_roots, loo_root in CELLS:
        pats = [f"{r}/poe_distill_mtl_brier__{tag}_seed*/seed0/budgeted.json" for r in pick_roots]
        pats += [f"{r}/poe_distill_mtl_brier__{tag}_rho0_seed*/seed0/budgeted.json" for r in pick_roots]
        pats += [f"{r}/poe_distill_mtl_brier__{tag}/seed*/budgeted.json" for r in pick_roots]
        full = runs_stat(pats, b)
        full_row.append(full)
        for arm, label in ARMS:
            a = runs_stat([f"{loo_root}/{arm}__{tag}_seed*/seed0/budgeted.json",
                           f"{loo_root}/{arm}__ce4_seed*/seed0/budgeted.json"], b)
            arm_rows[label].append((a[0] - full[0]) if (a and full) else None)

    out = []
    out.append(r"\begin{table*}[!tbp]")
    out.append(r"\centering")
    out.append(r"\caption{Leave-one-out ablation of the training objective. Each")
    out.append(r"entry is the change in test accuracy at the dataset's tightest")
    out.append(r"feasible contract when one term is removed from the selected")
    out.append(r"configuration and the model is retrained; the first row anchors the")
    out.append(r"deltas with the full objective's accuracy. Mean over three seeds.}")
    out.append(r"\label{tab:abl_loss_loo}")
    out.append(r"\setlength{\tabcolsep}{3pt}")
    out.append(r"{\scriptsize")
    out.append(r"\begin{tabular}{@{} l " + "r" * len(CELLS) + r" @{}}")
    out.append(r"\toprule")
    out.append(" & " + " & ".join(n for n, _, *_ in CELLS) + r" \\")
    out.append(" & " + " & ".join(f"$B{{=}}{b}$" for _, _, b, *_ in CELLS) + r" \\")
    out.append(r"\midrule")
    out.append(r"full objective & " + " & ".join(
        f"{f[0]:.2f}" + (r"\,{\tiny$\pm$" + f"{f[1]:.2f}" + "}" if f[2] > 1 else "")
        if f else "--" for f in full_row) + r" \\")
    out.append(r"\midrule")
    for _, label in ARMS:
        cells = []
        for d in arm_rows[label]:
            if d is None:
                cells.append("--")
            else:
                mark = r"\textbf{" + f"{d:+.2f}" + "}" if d <= -1.0 else f"{d:+.2f}"
                cells.append(mark)
        out.append(label + " & " + " & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}}")
    out.append(r"\end{table*}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--insert", action="store_true")
    args = ap.parse_args()
    tex_block = build()
    print(tex_block)
    if args.insert:
        OUT_TEX.parent.mkdir(exist_ok=True)
        OUT_TEX.write_text(tex_block + "\n")
        print(f"\nwrote {OUT_TEX}")


if __name__ == "__main__":
    main()
