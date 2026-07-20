"""Recompute the paper's derived headline numbers from budgeted.json and
shift-eval files under outputs/.

Reports, with the sources the paper cites:
  1. Main-table envelope: JOLT's worst deficit and best lead against the
     strongest baseline per dataset-budget pairing, and the win/tie/loss
     partition, a tie being overlapping error bars (the gap to the
     strongest baseline within the sum of the two seed standard deviations).
  2. Per-baseline worst deficit against the strongest method anywhere.
  3. Shift envelope under the shared quantile policy: worst deficit and
     best lead against the strongest clean baseline across all severities.
  4. Loss-term interaction: distillation added to the bare PoE chain
     versus added to the chain with the weighting and Brier anchor.
  5. Operating-point calibration: ECE and NLL of the JOLT configuration
     versus the strongest baseline, and ECE with the Brier anchor removed.
  6. Pareto membership on the accuracy-compute plane, for JOLT and for the
     best baseline, with the comparisons where JOLT is dominated named.
  7. Policy comparison under shift: per-policy budget overspend at the
     severest condition, and the largest accuracy gain any rule buys over
     the running quantile together with its overspend cost.
  8. Baseline-collapse diagnoses: JEI-DNN's gate concentration and per-exit
     starvation, EENet/BEEM's untrained first exit, and the frozen rule's
     zero accuracy gain for its SST-2 overspend.

Every number is a mean over the seeds present; run the training and
evaluation commands in the README first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from generate_budget_table import (  # noqa: E402
    BASELINES, BUDGETS, CELLS, baseline_pick, candidate_at_contract,
    candidate_tag_pool, seed_jsons, stat as gstat,
)
from generate_shift_tables import (  # noqa: E402
    COND_SETS, SPECS, clean_q_and_macs, ladder_for, load_rows,
)


def acc_at(run: dict, b: float):
    ok = [r for r in run["curve"] if r["val"]["macs_frac"] <= b]
    if not ok:
        return None
    return max(ok, key=lambda r: r["val"]["accuracy"])["test"]["accuracy"] * 100


def stat(vals):
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)


def main_table_envelope():
    print("== 1. main-table envelope ==")
    deltas, partition = [], {"win": 0, "tie": 0, "loss": 0}
    base_worst = {m: 0.0 for m in BASELINES}
    for cell in CELLS:
        pool = candidate_tag_pool(cell)
        for b in BUDGETS:
            _, ours = candidate_at_contract(pool, b)
            ovals = [v for r in ours if (v := acc_at(r, b)) is not None]
            if not ovals:
                continue
            om, os_ = stat(ovals)
            best = None
            per_method = {}
            for m in BASELINES:
                # symmetric HP selection (validation-best over default + tuned
                # off-defaults), the same rule the candidate gets
                _, st, _ = baseline_pick(cell, m, b)
                if st is None:
                    continue
                per_method[m] = (st[0], st[1])
                if best is None or per_method[m][0] > best[0]:
                    best = per_method[m]
            if best is None:
                continue
            d = om - best[0]
            band = os_ + best[1]
            deltas.append((d, cell["name"], b))
            partition["win" if d > band else "loss" if d < -band else "tie"] += 1
            row_best = max([om] + [v[0] for v in per_method.values()])
            for m, (mm, _) in per_method.items():
                base_worst[m] = max(base_worst[m], row_best - mm)
    deltas.sort()
    print(f"  pairings: {len(deltas)}  partition: {partition}")
    print(f"  worst deficit {deltas[0][0]:+.2f} ({deltas[0][1]} B{deltas[0][2]})")
    print(f"  best lead     {deltas[-1][0]:+.2f} ({deltas[-1][1]} B{deltas[-1][2]})")
    print("== 2. per-baseline worst deficit vs strongest method ==")
    for m, w in sorted(base_worst.items(), key=lambda kv: kv[1]):
        print(f"  {m:16s} {w:6.2f}")


def shift_envelope():
    print("== 3. shift envelope (shared quantile policy) ==")
    deltas = []
    for name, fname, cond, pick, base in SPECS:
        def rungvals(pat):
            out = {}
            for run_dir in sorted(REPO.glob(pat)):
                cq = clean_q_and_macs(run_dir)
                if cq is None:
                    continue
                q_star, _ = cq
                for c in COND_SETS[fname]:
                    rows = load_rows(run_dir, fname, c)
                    if rows is None:
                        continue
                    r = min(rows, key=lambda r: abs(r["q"] - q_star))
                    out.setdefault(c, []).append(r["q_accuracy"] * 100)
            return out
        ov, bv = rungvals(pick), rungvals(base)
        for c in ov:
            if c in bv and ov[c] and bv[c]:
                deltas.append((mean(ov[c]) - mean(bv[c]), name, c))
    deltas.sort()
    print(f"  cells: {len(deltas)}")
    print(f"  worst {deltas[0][0]:+.2f} ({deltas[0][1]} {deltas[0][2]})")
    print(f"  best  {deltas[-1][0]:+.2f} ({deltas[-1][1]} {deltas[-1][2]})")


INTERACTION = {  # dataset -> (root, loo root, tag, budget)
    "GSC v2": ("outputs/gsc", "outputs/gsc_stepwise", "g0.5-lb0.5", 0.5),
    "CIFAR-100": ("outputs/cifar100", "outputs/cifar100_stepwise", "g2.0-lb0.5-ce4.0", 0.5),
}


def interaction():
    print("== 4. distillation interaction (stepwise arms) ==")
    for name, (root, srot, tag, b) in INTERACTION.items():
        def arm(base, method):
            vals = []
            for p in REPO.glob(f"{base}/{method}__{tag}*/seed*/budgeted.json"):
                v = acc_at(json.loads(p.read_text()), b)
                if v is not None:
                    vals.append(v)
            return mean(vals) if vals else None
        poe, pd = arm(srot, "poe_anneal"), arm(srot, "poe_distill")
        pmb, full = arm(srot, "poe_multitask_brier"), arm(root, "poe_distill_mtl_brier")
        if None in (poe, pd, pmb, full):
            print(f"  {name}: stepwise arms missing (train poe_anneal, poe_distill,")
            print(f"    poe_multitask_brier under {srot} at the same tag)")
            continue
        print(f"  {name}: into bare chain {pd - poe:+.2f} | into weighting+Brier {full - pmb:+.2f}")


def pareto():
    print("== 6. Pareto membership (accuracy-compute plane) ==")

    def dominated_by(point, others):
        return any(
            x[0] >= point[0] - 1e-9 and x[3] <= point[3] + 1e-9
            and (x[0] > point[0] + 1e-9 or x[3] < point[3] - 1e-9)
            for x in others)

    undom = total = 0
    dom_cells = []
    rows = []  # (cell, b, ours, per-method stats) for the baseline pass
    for cell in CELLS:
        pool = candidate_tag_pool(cell)
        for b in BUDGETS:
            ours = gstat(candidate_at_contract(pool, b)[1], b)
            if ours is None:
                continue
            per = {m: st for m in BASELINES
                   if (st := baseline_pick(cell, m, b)[1]) is not None}
            if not per:
                continue
            total += 1
            if dominated_by(ours, list(per.values())):
                dom_cells.append(f"{cell['name']} B{b}")
            else:
                undom += 1
            rows.append((ours, per))
    print(f"  JOLT Pareto-undominated in {undom} of {total} comparisons")
    if dom_cells:
        print(f"  dominated at: {', '.join(dom_cells)}")
    best_m, best_n = None, -1
    for m in BASELINES:
        n = 0
        for ours, per in rows:
            if m not in per:
                continue
            others = [ours] + [v for k, v in per.items() if k != m]
            if not dominated_by(per[m], others):
                n += 1
        if n > best_n:
            best_m, best_n = m, n
    print(f"  best baseline undominated in {best_n} comparisons ({best_m})")


def calibration():
    print("== 5. operating-point calibration (opcal.json) ==")
    def eces(pat, b="0.5"):
        e, n = [], []
        for p in REPO.glob(pat + "/opcal.json"):
            r = json.loads(p.read_text())["results"].get(b)
            if r:
                e.append(r["ece"]); n.append(r["nll"])
        return (mean(e), mean(n)) if e else None
    for name, fname, cond, pick, base in SPECS:
        o, b_ = eces(pick), eces(base)
        if o and b_:
            print(f"  {name:14s} ECE {o[0]:.3f} vs {b_[0]:.3f} | NLL {o[1]:.2f} vs {b_[1]:.2f}")
    print("== 5b. ECE with the Brier anchor removed (loo arms) ==")
    for name, fname, cond, pick, base in SPECS:
        ds = pick.split("/")[1]
        loo = eces(f"outputs/{ds}_loo/poe_distill_mtl__*/seed*")
        o = eces(pick)
        if loo and o:
            print(f"  {name:14s} full {o[0]:.3f} -> minus-Brier {loo[0]:.3f}")


def policy_comparison():
    """Section 7: policy overspend at the severest shift, from
    generate_policy_drift.py's outputs/analysis/policy_drift.json."""
    print("== 7. policy comparison under shift (budget overspend %) ==")
    path = REPO / "outputs/analysis/policy_drift.json"
    if not path.exists():
        print("  policy_drift.json absent; run generate_policy_drift.py")
        return
    d = json.loads(path.read_text())
    col = lambda p: {c: d[c]["over"][p] for c in d}
    q, fr, pc, ba = col("quantile"), col("frozen"), col("pcee"), col("bandit")
    rc = col("rc_eenn")
    within = [c for c in q if abs(q[c]) <= 0.5]
    print(f"  cells: {len(d)}")
    print(f"  frozen  max overspend: {max(fr.values()):.1f}%")
    print(f"  PCEE    max overspend: {max(pc.values()):.1f}%")
    print(f"  RC-EENN max overspend: {max(rc.values()):.1f}%")
    print(f"  bandit  max overspend: {max(ba.values()):.1f}%")
    print(f"  quantile within 0.4% on {len(within)} cells; "
          f"two-worst {max(abs(q[c]) for c in q if c not in within):.1f}%")
    # largest accuracy gain any rule buys over the running quantile, and cost
    best = (-1e9, "", "")
    for c in d:
        for pol in ("frozen", "pcee", "rc_eenn", "bandit"):
            g = d[c]["acc"][pol] - d[c]["acc"]["quantile"]
            if g > best[0]:
                best = (g, pol, c)
    g, pol, c = best
    print(f"  largest gain any rule buys: {g:+.1f} points by {pol} on {c}, "
          f"at {d[c]['over'][pol]:.0f}% overspend")


def collapse_diagnostics():
    """Section 8: why the collapsed baselines collapse, from the saved score
    matrices (no retraining)."""
    import numpy as np
    print("== 8. baseline-collapse diagnoses ==")
    # JEI-DNN gate concentration on CIFAR-100 (native gate scores)
    fr = []
    for f in sorted(REPO.glob("outputs/cifar100/jei_dnn*/seed*/native_scores.npz")):
        g = np.load(f)["g_clean"]
        n, em1 = g.shape
        fire = (1.0 / (1.0 + np.exp(-g))) >= 0.5
        idx = np.full(n, em1, dtype=int)
        for j in range(em1):
            take = (idx == em1) & fire[:, j]
            idx[take] = j
        fr.append(np.bincount(idx, minlength=em1 + 1).max() / n)
    if fr:
        print(f"  JEI-DNN CIFAR-100 modal-exit fraction: {min(fr):.2f} "
              f"(1.00 = every input to a single exit; min over runs)")

    def per_exit(pat):
        accs = [np.load(f)["correct_val"].mean(0) * 100
                for f in sorted(REPO.glob(pat))]
        return np.mean(np.array(accs), 0) if accs else None

    jei = per_exit("outputs/gsc/jei_dnn*/seed*/exit_scores.npz")
    if jei is not None:
        a = sorted(jei, reverse=True)
        print(f"  JEI-DNN GSC v2 per-exit accuracy: selected {a[0]:.1f}, "
              f"starved exits at {a[1]:.1f} or below (chance for 35 classes)")
    for m in ("eenet", "beem"):
        v = per_exit(f"outputs/gsc/{m}*/seed*/exit_scores.npz")
        if v is not None:
            print(f"  {m} GSC v2 first-exit accuracy: {v[0]:.1f}")
    ours = per_exit("outputs/gsc/jolt__g0.5-lb0.5-ce2.0*/seed*/exit_scores.npz")
    if ours is not None:
        print(f"  JOLT GSC v2 first-exit accuracy: {ours[0]:.1f}")
    # SST-2: frozen overspend buys no accuracy at the severest condition
    r = ladder_for("outputs/sst2/jolt__g16.0-lb0.5-ce0.5*/seed*",
                   "text_shift_eval.json")
    if r is not None:
        _, quant, froz, *_ = r
        if froz[-1] is not None and quant[-1] is not None:
            print(f"  SST-2 severest: frozen-minus-quantile accuracy "
                  f"{froz[-1] - quant[-1]:+.2f}")


if __name__ == "__main__":
    main_table_envelope()
    shift_envelope()
    interaction()
    calibration()
    pareto()
    policy_comparison()
    collapse_diagnostics()
