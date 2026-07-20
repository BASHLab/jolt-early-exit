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
    ("CIFAR-100", "g2.0-lb0.5-ce2.0", 0.5, ["outputs/cifar100"], "outputs/cifar100_loo"),
    ("GSC v2", "g0.5-lb0.5-ce2.0", 0.5, ["outputs/gsc"], "outputs/gsc_loo"),
    ("SST-2", "g16.0-lb0.5-ce0.5", 0.3, ["outputs/sst2"], "outputs/sst2_loo"),
    ("ESC-50", "g16.0-lb2.0", 0.3, ["outputs/esc50"], "outputs/esc50_loo"),
    ("Tiny-ImageNet", "g2.0-lb0.5-ce4.0", 0.5, ["outputs/tinyimagenet"], "outputs/tinyimagenet_loo"),
]
ARMS = [("poe_multitask_brier", r"$-$ distillation"),
        ("poe_distill_brier", r"$-$ learned weighting"),
        ("poe_distill_mtl", r"$-$ Brier anchor")]


def tag_without_ce(tag: str) -> str:
    """The pick tag with its -ceX.Y component removed (the final-exit-anchor
    removal arm); returns None when the pick carries no anchor."""
    stripped = re.sub(r"-ce[0-9.]+", "", tag)
    return stripped if stripped != tag else None


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
    arm_rows[r"$-$ final-exit anchor"] = []
    arm_rows[r"$-$ monotonicity penalty"] = []
    for name, tag, b, pick_roots, loo_root in CELLS:
        pats = [f"{r}/poe_distill_mtl_brier__{tag}_seed*/seed0/budgeted.json" for r in pick_roots]
        pats += [f"{r}/poe_distill_mtl_brier__{tag}_rho0_seed*/seed0/budgeted.json" for r in pick_roots]
        pats += [f"{r}/poe_distill_mtl_brier__{tag}/seed*/budgeted.json" for r in pick_roots]
        full = runs_stat(pats, b)
        full_row.append(full)
        for arm, label in ARMS:
            a = runs_stat([f"{loo_root}/{arm}__{tag}_seed*/seed0/budgeted.json",
                           f"{loo_root}/{arm}__{tag}/seed*/budgeted.json"], b)
            arm_rows[label].append((a[0] - full[0]) if (a and full) else None)
        noce_tag = tag_without_ce(tag)
        noce = runs_stat(
            [f"{r}/poe_distill_mtl_brier__{noce_tag}_seed*/seed0/budgeted.json" for r in pick_roots]
            + [f"{r}/poe_distill_mtl_brier__{noce_tag}/seed*/budgeted.json" for r in pick_roots],
            b) if noce_tag else None
        arm_rows.setdefault(r"$-$ final-exit anchor", []).append(
            (noce[0] - full[0]) if (noce and full) else None)
        mono = runs_stat(
            [f"{loo_root}/poe_distill_mtl_brier__{tag}_rho0_seed*/seed0/budgeted.json",
             f"{loo_root}/poe_distill_mtl_brier__{tag}_rho0/seed*/budgeted.json"], b)
        arm_rows.setdefault(r"$-$ monotonicity penalty", []).append(
            (mono[0] - full[0]) if (mono and full) else None)

    out = []
    out.append(r"\begin{table*}[!tbp]")
    out.append(r"\centering")
    out.append(r"\caption{Leave-one-out ablation of the training objective. Each")
    out.append(r"entry is the change in test accuracy at the dataset's budget when")
    out.append(r"one term is removed from the selected configuration and the model")
    out.append(r"is retrained. Mean over three seeds. Dashes mark columns whose")
    out.append(r"configuration carries no final-exit anchor to remove.}")
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
    for label in list(arm_rows):
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
