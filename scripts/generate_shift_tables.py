"""Section IV G: budget fidelity and accuracy under distribution shift.

Two tables plus the exit-share figure, all from the per-run shift jsons.

Fidelity (tab:shift_fidelity): per dataset, at the B=0.5 pick's quantile
level, the realized compute on the most severe shift condition under (a)
thresholds frozen from clean validation and (b) population-quantile
thresholds at the same level on the shifted stream. Reported as absolute
deviation from the clean realized compute; quantile rows hold by
construction, frozen rows drift.

Coupling (tab:shift_coupling): per dataset, accuracy on the shifted stream
at the same level, ours versus the strongest baseline of the clean table.

Exit-share figure (fig:shift_exit_shares): exit-population shares against
severity, frozen versus quantile, for CIFAR-100 and GSC v2.

Prints the per-dataset drift numbers; --figure renders figures/shift_ladders.pdf.
--figure renders fig/shift_exit_shares.pdf.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
TEX = REPO / "tables"
FIG = REPO / "figures/shift_exit_shares.pdf"

# (dataset, shift json filename, severest condition key, pick glob, strongest-baseline glob)
SPECS = [
    ("UCI-HAR", "sensor_shift_eval.json", "sev5",
     "outputs/ucihar/jolt__g0.25-lb0.15-ce2.0*/seed*",
     "outputs/ucihar/adaloss*/seed*"),
    ("PAMAP2", "sensor_shift_eval.json", "sev5",
     "outputs/pamap2/jolt__g8.0-lb1.0*/seed*",
     "outputs/pamap2/jei_dnn__cl0.2*/seed*"),
    ("CIFAR-100", "shift_eval.json", "sev5",
     "outputs/cifar100/jolt__g2.0-lb0.5-ce2.0*/seed*",
     "outputs/cifar100/poe_jazbec*/seed*"),
    ("GSC v2", "audio_shift_eval.json", "sev5",
     "outputs/gsc/jolt__g0.5-lb0.5-ce2.0*/seed*",
     "outputs/gsc/meronen_laplace*/seed*"),
    ("SST-2", "text_shift_eval.json", "typo_5",
     "outputs/sst2/jolt__g16.0-lb0.5-ce0.5*/seed*",
     "outputs/sst2/boostnet*/seed*"),
    ("ESC-50", "audio_shift_eval.json", "sev5",
     "outputs/esc50/jolt__g16.0-lb2.0*/seed*",
     "outputs/esc50/meronen_laplace*/seed*"),
    ("Tiny-ImageNet", "shift_eval.json", "sev5",
     "outputs/tinyimagenet/jolt__g2.0-lb0.5-ce4.0*/seed*",
     "outputs/tinyimagenet/poe_jazbec*/seed*"),
]
B = 0.5


def load_rows(run_dir: Path, fname: str, cond: str):
    f = run_dir / fname
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    res = d.get("results", d)
    # condition keys vary: sevN / snrN / typo_N / imdb / corruption-averaged dicts
    for key in (cond, cond.replace("sev", "severity_"), cond.replace("snr", "snr_")):
        if key in res:
            return res[key].get("rows", res[key])
    # vision jsons: {corruption: {severity: {...}}} — average the severest severity
    sev = cond[-1]
    rows_acc = {}
    n = 0
    for corr, per_sev in res.items():
        if not isinstance(per_sev, dict) or sev not in per_sev:
            continue
        for row in per_sev[sev].get("rows", []):
            q = row["q"]
            agg = rows_acc.setdefault(q, {k: 0.0 for k in row if k != "q"})
            for k, v in row.items():
                if k != "q" and isinstance(v, (int, float)):
                    agg[k] += v
        n += 1
    if not rows_acc:
        return None
    return [{**{k: v / n for k, v in agg.items()}, "q": q} for q, agg in rows_acc.items()]


def clean_q_and_macs(run_dir: Path):
    """The B=0.5 level and its calibrated (validation) compute."""
    bj = run_dir / "budgeted.json"
    if not bj.exists():
        return None
    d = json.loads(bj.read_text())
    ok = [r for r in d["curve"] if r["val"]["macs_frac"] <= B]
    if not ok:
        return None
    row = max(ok, key=lambda r: r["val"]["accuracy"])
    return row["q"], row["val"]["macs_frac"]


COND_SETS = {
    "sensor_shift_eval.json": ["sev1", "sev3", "sev5"],
    "shift_eval.json": ["sev1", "sev3", "sev5"],
    "audio_shift_eval.json": ["sev1", "sev3", "sev5"],
    "text_shift_eval.json": ["typo_1", "typo_3", "typo_5"],
}


# Per-modality per-sample shift score matrices, for the deployable running
# (streaming) quantile policy. The batch quantile re-estimates thresholds from
# the whole shifted stream; the streaming policy holds a running-window quantile,
# which is what actually deploys and what fig:policy_drift reports.
_NPZ = {"shift_eval.json": "exit_scores_shift.npz",
        "sensor_shift_eval.json": "exit_scores_sshift.npz",
        "audio_shift_eval.json": "exit_scores_ashift.npz",
        "text_shift_eval.json": "exit_scores_tshift.npz"}
_STREAM_WINDOW = 256


def _streaming_by_rung(run_dir: Path, fname: str, q_star: float, rungs):
    """Streaming-quantile (window 256) accuracy% and MAC-frac per rung, from the
    per-sample shift matrices. {} if the npz is absent (caller falls back to the
    batch JSON values)."""
    import numpy as np
    from quantile_routing_analysis import streaming_route, thresholds_for_population
    clean = run_dir / "exit_scores.npz"
    shift = run_dir / _NPZ.get(fname, "")
    if not clean.exists() or not shift.exists():
        return {}
    z, zs = np.load(clean), np.load(shift)
    thr = thresholds_for_population(z["scores_val"].astype(np.float64), q_star)
    pem = z["per_exit_macs"].tolist()
    out = {}
    for c in rungs:
        n = c[-1]  # trailing severity digit (sev5 -> 5, typo_5 -> 5)
        accs, macs = [], []
        for k in zs.files:
            if not (k.startswith("scores_") and k.endswith(f"_{n}")):
                continue
            sc = zs[k].astype(np.float64)
            co = zs["correct_" + k[len("scores_"):]].astype(np.float64)
            idx = np.argsort((np.arange(len(sc)) * 2654435761) % 2**32)
            r = streaming_route(sc[idx], co[idx], q_star, pem, _STREAM_WINDOW, thr)
            accs.append(r["accuracy"] * 100)
            macs.append(r["macs_frac"])
        if accs:
            out[c] = (float(np.mean(accs)), float(np.mean(macs)))
    return out


def ladder_for(glob_pat: str, fname: str):
    """Per-rung mean accuracy (clean, mild, moderate, severe) for the
    quantile policy and, for the frozen row, the same rungs under
    thresholds frozen from clean validation. The quantile rows use the
    deployable running (streaming) policy where the per-sample matrices are
    present, falling back to the batch quantile otherwise."""
    rungs = COND_SETS[fname]
    clean, cal = [], []
    quant, froz = {c: [] for c in rungs}, {c: [] for c in rungs}
    qmac, fmac = {c: [] for c in rungs}, {c: [] for c in rungs}
    for run_dir in sorted(REPO.glob(glob_pat)):
        cq = clean_q_and_macs(run_dir)
        if cq is None:
            continue
        q_star, calib = cq
        cal.append(calib)
        stream = _streaming_by_rung(run_dir, fname, q_star, rungs)
        bj = json.loads((run_dir / "budgeted.json").read_text())
        ok = [r for r in bj["curve"] if r["val"]["macs_frac"] <= B]
        if ok:
            clean.append(max(ok, key=lambda r: r["val"]["accuracy"])["test"]["accuracy"] * 100)
        for c in rungs:
            rows = load_rows(run_dir, fname, c)
            if rows is None:
                continue
            row = min(rows, key=lambda r: abs(r["q"] - q_star))
            if c in stream:
                acc_c, mac_c = stream[c]
                quant[c].append(acc_c)
                qmac[c].append(mac_c)
            else:
                quant[c].append(row["q_accuracy"] * 100)
                qmac[c].append(row.get("q_macs_frac", row["macs_frac"]))
            froz[c].append(row["accuracy"] * 100)
            fmac[c].append(row["macs_frac"])
    if not clean:
        return None
    return (mean(clean),
            [mean(quant[c]) if quant[c] else None for c in rungs],
            [mean(froz[c]) if froz[c] else None for c in rungs],
            mean(cal),
            [mean(qmac[c]) if qmac[c] else None for c in rungs],
            [mean(fmac[c]) if fmac[c] else None for c in rungs])


def stat_for(glob_pat: str, fname: str, cond: str):
    """Fidelity at the severest condition; accuracy averaged over all
    conditions of the modality's benchmark; clean accuracy at the level."""
    frozen_dev, quant_dev, acc_frozen, acc_quant, acc_clean = [], [], [], [], []
    for run_dir in sorted(REPO.glob(glob_pat)):
        cq = clean_q_and_macs(run_dir)
        if cq is None:
            continue
        bj = json.loads((run_dir / "budgeted.json").read_text())
        ok = [r for r in bj["curve"] if r["val"]["macs_frac"] <= B]
        if ok:
            acc_clean.append(max(ok, key=lambda r: r["val"]["accuracy"])["test"]["accuracy"] * 100)
        q_star, clean_macs = cq
        rows = load_rows(run_dir, fname, cond)
        if rows is not None:
            row = min(rows, key=lambda r: abs(r["q"] - q_star))
            frozen_dev.append(abs(row["macs_frac"] - clean_macs) / clean_macs * 100)
            quant_dev.append(abs(row.get("q_macs_frac", row["macs_frac"]) - clean_macs) / clean_macs * 100)
        per_q, per_f = [], []
        for c in COND_SETS[fname]:
            crows = load_rows(run_dir, fname, c)
            if crows is None:
                continue
            crow = min(crows, key=lambda r: abs(r["q"] - q_star))
            per_q.append(crow["q_accuracy"] * 100)
            per_f.append(crow["accuracy"] * 100)
        if per_q:
            acc_quant.append(mean(per_q))
            acc_frozen.append(mean(per_f))
    if not frozen_dev:
        return None
    return (mean(frozen_dev), mean(quant_dev), mean(acc_frozen), mean(acc_quant),
            mean(acc_clean) if acc_clean else None)


def build_tables():
    fid_rows, cop_rows = [], []
    for name, fname, cond, pick_glob, base_glob in SPECS:
        ours = stat_for(pick_glob, fname, cond)
        base = stat_for(base_glob, fname, cond)
        if ours is None:
            fid_rows.append((name, None, None))
            cop_rows.append((name, None, None, None))
            continue
        fid_rows.append((name, ours[0], ours[1]))
        cop_rows.append((name, ours[4], ours[3], base[3] if base else None, ours[2]))

    fid = [r"\begin{table}[!tbp]", r"\centering",
           r"\caption{Budget fidelity under the most severe shift of each",
           r"modality. Each entry is how far the deployed compute moves from",
           r"the budget it was calibrated to at the $B{=}0.5$ level, as a",
           r"percentage of that budget: SST-2's $61.9$ means thresholds",
           r"frozen from clean validation spend nearly two thirds more",
           r"compute than the deployment was given. The quantile column",
           r"holds the same level on the shifted stream.}",
           r"\label{tab:shift_fidelity}", r"{\small",
           r"\begin{tabular}{@{} l rr @{}}", r"\toprule",
           r" & \multicolumn{2}{c}{budget deviation (\%)} \\",
           r"\cmidrule(l){2-3}",
           r" & frozen & quantile \\", r"\midrule"]
    for name, f, qd in fid_rows:
        if f is None:
            fid.append(f"{name} & -- & -- \\\\")
        else:
            fc, qc = f"{f:.1f}", f"{qd:.1f}"
            if qd <= f:
                qc = "\\textbf{" + qc + "}"
            else:
                fc = "\\textbf{" + fc + "}"
            fid.append(f"{name} & {fc} & {qc} \\\\")
    fid += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]

    base_names = {"UCI-HAR": "AdaLoss", "PAMAP2": "JEI-DNN",
                  "CIFAR-100": "PoE", "GSC v2": "Laplace",
                  "SST-2": "BoostNet", "ESC-50": "Laplace",
                  "Tiny-ImageNet": "PoE"}
    cop = [r"\begin{table}[!tbp]", r"\centering",
           r"\caption{Accuracy (\%) at the $B{=}0.5$ level on each",
           r"modality's shift benchmark, from the clean stream to the",
           r"severest condition (corruption severity five, 0\,dB",
           r"signal-to-noise, or a 20\,\% character error rate).",
           r"\ourapproach\ and the strongest clean baseline route with the",
           r"quantile policy; the frozen row is \ourapproach\ under",
           r"thresholds frozen from clean validation. Bold marks the better",
           r"of the two methods at each rung.}",
           r"\label{tab:shift_coupling}",
           r"\setlength{\tabcolsep}{4pt}",
           r"{\scriptsize",
           r"\begin{tabular}{@{} l l rrrr @{}}", r"\toprule",
           r" & & clean & mild & moderate & severe \\", r"\midrule"]
    for name, fname, cond, pick_glob, base_glob in SPECS:
        ours = ladder_for(pick_glob, fname)
        base = ladder_for(base_glob, fname)
        if ours is None:
            continue
        oc, oq, of = ours[:3]
        bc, bq = (base[0], base[1]) if base else (None, [None]*3)
        def cell(v, bold):
            if v is None:
                return "--"
            t = f"{v:.1f}"
            return r"\textbf{" + t + "}" if bold else t
        our_cells = [cell(oc, bc is None or oc >= bc)] + [
            cell(v, v is not None and (bq[i] is None or v >= bq[i])) for i, v in enumerate(oq)]
        base_cells = [cell(bc, bc is not None and bc > oc)] + [
            cell(v, v is not None and oq[i] is not None and v > oq[i]) for i, v in enumerate(bq)]
        froz_cells = [cell(oc, False)] + [cell(v, False) for v in of]
        cop.append(r"\multirow{3}{*}{" + name + r"} & \ourapproach & " + " & ".join(our_cells) + r" \\")
        cop.append(" & " + base_names[name] + " & " + " & ".join(base_cells) + r" \\")
        cop.append(r" & frozen & " + " & ".join(froz_cells) + r" \\")
        cop.append(r"\midrule")
    cop[-1] = r"\bottomrule"
    cop += [r"\end{tabular}}", r"\end{table}"]
    # both halves of the shift story ship as fig:shift_ladders (accuracy on
    # top, deployed compute below); nothing is spliced into the paper here
    for name, f, qd in fid_rows:
        if f is not None:
            print(f"%   {name}: frozen {f:.1f}%  quantile {qd:.1f}%")
    return "% budget-fidelity numbers live in fig:shift_ladders (bottom row); percentages above"



LADDER_FIG = REPO / "figures/shift_ladders.pdf"


def render_ladder_figure():
    import matplotlib
    matplotlib.use("Agg")
    # Type 42 (TrueType) rather than matplotlib's default Type 3, which
    # IEEE Xplore does not accept in camera-ready PDFs.
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 7, figsize=(7.1, 2.9))
    x = np.arange(4)
    for i, (name, fname, cond, pick_glob, base_glob) in enumerate(SPECS):
        ours = ladder_for(pick_glob, fname)
        if ours is None:
            continue
        oc, oq, of, calib, qmac, fmac = ours
        top, bot = axes[0][i], axes[1][i]
        top.plot(x, [oc] + oq, "o-", color="#d4a017", lw=1.6, ms=2.5, zorder=3, label="quantile")
        top.plot(x, [oc] + of, "^--", color="0.55", lw=0.9, ms=2, zorder=1, label="frozen")
        top.set_title(name, fontsize=7)
        top.set_xticks(x)
        top.set_xticklabels([])
        top.tick_params(labelsize=6)
        top.legend(frameon=False, fontsize=5, handlelength=1.1,
                   borderaxespad=0.1, labelspacing=0.2)

        bot.axhline(calib, color="0.3", lw=0.7, ls=":", zorder=0)
        bot.plot(x, [calib] + qmac, "o-", color="#d4a017", lw=1.6, ms=2.5, zorder=2)
        bot.plot(x, [calib] + fmac, "^--", color="0.55", lw=0.9, ms=2, zorder=1)
        bot.set_xticks(x)
        bot.set_xticklabels(["clean", "mild", "mod.", "sev."], fontsize=6, rotation=45)
        bot.tick_params(labelsize=6)
        bot.set_ylim(min([calib] + fmac + qmac) - 0.05, max([calib] + fmac + qmac) + 0.05)
        if i == 0:
            top.set_ylabel("test accuracy (%)", fontsize=7)
            bot.set_ylabel("deployed compute", fontsize=7)
    fig.tight_layout(w_pad=0.5, h_pad=0.6)
    fig.savefig(LADDER_FIG, dpi=300, bbox_inches="tight")
    print(f"wrote {LADDER_FIG}")


def render_figure():
    import matplotlib
    matplotlib.use("Agg")
    # Type 42 (TrueType) rather than matplotlib's default Type 3, which
    # IEEE Xplore does not accept in camera-ready PDFs.
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    import matplotlib.pyplot as plt
    import numpy as np
    panels = [("CIFAR-100 (gaussian noise)", "shift_eval.json", "gaussian_noise", ["1", "3", "5"],
               "outputs/cefinal_v1/cifar100_convmixer/poe_distill_mtl_brier__g2.0-lb0.5-ce4.0_seed0/seed0"),
              ("GSC v2 (background noise)", "audio_shift_eval.json", "background", ["1", "3", "5"],
               "outputs/pickfill_v1/gsc_matchboxnet/poe_distill_mtl_brier__g0.5-lb0.5_seed0/seed0")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    for ax, (name, fname, corr, sevs, rd) in zip(axes, panels):
        run_dir = REPO / rd
        cq = clean_q_and_macs(run_dir)
        if cq is None:
            continue
        q_star, _ = cq
        d = json.loads((run_dir / fname).read_text())
        res = d.get("results", d)
        frozen_e1, quant_e1 = [], []
        for sev in sevs:
            rows = res.get(corr, {}).get(sev, {}).get("rows")
            if not rows:
                frozen_e1.append(np.nan); quant_e1.append(np.nan); continue
            row = min(rows, key=lambda r: abs(r["q"] - q_star))
            ec, qec = row.get("exit_counts"), row.get("q_exit_counts")
            frozen_e1.append(ec[0] / sum(ec) if ec else np.nan)
            quant_e1.append(qec[0] / sum(qec) if qec else np.nan)
        x = np.arange(len(sevs))
        ax.plot(x, frozen_e1, "o--", color="0.4", label="frozen")
        ax.plot(x, quant_e1, "s-", color="#d4a017", label="quantile")
        ax.axhline(quant_e1[0] if quant_e1 and np.isfinite(quant_e1[0]) else 0,
                   color="0.85", lw=0.6, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(sevs)
        ax.set_title(name); ax.set_xlabel("shift severity")
        ax.set_ylabel("exit-1 fraction")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=300, bbox_inches="tight")
    print(f"wrote {FIG}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--insert", action="store_true")
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()
    block = build_tables()
    print(block)
    if args.insert:
        out = REPO / "tables/shift_fidelity.txt"
        out.parent.mkdir(exist_ok=True)
        out.write_text(block + "\n")
        print(f"wrote {out}")
    if args.figure:
        render_figure()
        # render_ladder_figure() is superseded by make_shift_fig.py's
        # five-policy overspend panel, which owns figures/shift_ladders.pdf.


if __name__ == "__main__":
    main()
