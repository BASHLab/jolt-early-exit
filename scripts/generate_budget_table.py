"""Generate the budgeted-batch main table from per-run budgeted.json files.

Per (dataset, budget B, method): among validation-feasible q levels (val
macs_frac <= B) take the level with maximum validation accuracy and report
the test accuracy of that level, mean +- std across seeds. Selection is
validation-only. Budgets below the first exit's cost are marked infeasible.
The candidate configuration per dataset is the one with the highest mean
validation accuracy across the three budgets.

Prints the LaTeX block to stdout; --insert writes tables/main_table.tex.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
OUT_TEX = REPO / "tables/main_table.tex"
BUDGETS = [0.3, 0.5, 0.7]

CELLS = [
    {"name": "UCI-HAR", "backbone": "TST",
     "base_roots": ["outputs/ucihar"], "fix_root": "outputs/ucihar",
     "cand_roots": ["outputs/ucihar"]},
    {"name": "PAMAP2", "backbone": "InceptionTime",
     "base_roots": ["outputs/pamap2"], "fix_root": "outputs/pamap2",
     "cand_roots": ["outputs/pamap2"]},
    {"name": "CIFAR-100", "backbone": "ConvMixer-256/8",
     "base_roots": ["outputs/cifar100"], "fix_root": "outputs/cifar100",
     "cand_roots": ["outputs/cifar100"]},
    {"name": "GSC v2", "backbone": "MatchboxNet",
     "base_roots": ["outputs/gsc"], "fix_root": "outputs/gsc",
     "cand_roots": ["outputs/gsc"]},
    {"name": "SST-2", "backbone": "BERT-base",
     "base_roots": ["outputs/sst2"], "fix_root": "outputs/sst2",
     "cand_roots": ["outputs/sst2"]},
    {"name": "ESC-50", "backbone": "EfficientNet-B0",
     "base_roots": ["outputs/esc50"], "fix_root": "outputs/esc50",
     "cand_roots": ["outputs/esc50"]},
    {"name": "Tiny-ImageNet", "backbone": "CCT-7/3x2",
     "base_roots": ["outputs/tinyimagenet"], "fix_root": "outputs/tinyimagenet",
     "cand_roots": ["outputs/tinyimagenet"]},
]

BASELINES = ["adaloss", "branchynet", "eenet", "td", "meronen_laplace",
             "ztw_cascade", "beem", "jei_dnn", "poe_jazbec", "boostnet"]
FIXED = {"td", "boostnet", "jei_dnn"}
LATEX = {"adaloss": "AdaLoss", "branchynet": "BranchyNet", "eenet": "EENet",
         "td": "TD", "meronen_laplace": "Laplace", "ztw_cascade": "ZTW",
         "beem": "BEEM", "jei_dnn": "JEI-DNN", "poe_jazbec": "PoE",
         "boostnet": "BoostNet"}


def seed_jsons(root: Path, subdir_prefix: str) -> List[dict]:
    """All budgeted.json under method dirs matching prefix, across seed layouts."""
    out = []
    if not root.exists():
        return out
    for mdir in sorted(root.iterdir()):
        if not mdir.is_dir():
            continue
        stem = mdir.name.split("__")[0] if "__" in mdir.name else mdir.name
        if mdir.name != subdir_prefix and stem != subdir_prefix \
           and not mdir.name.startswith(subdir_prefix + "__") \
           and not mdir.name.startswith(subdir_prefix + "_"):
            continue
        for seed_dir in sorted(mdir.glob("seed*")):
            bj = seed_dir / "budgeted.json"
            if bj.exists():
                out.append(json.loads(bj.read_text()))
    return out


def acc_at_budget(curve: List[dict], b: float) -> Optional[Tuple[float, float]]:
    """(test accuracy %, realized test macs_frac) of the val-selected level."""
    ok = [r for r in curve if r["val"]["macs_frac"] <= b]
    if not ok:
        return None
    best = max(ok, key=lambda r: r["val"]["accuracy"])
    return best["test"]["accuracy"] * 100, best["test"]["macs_frac"]


def val_acc_at_budget(curve: List[dict], b: float) -> Optional[float]:
    ok = [r for r in curve if r["val"]["macs_frac"] <= b]
    return max(r["val"]["accuracy"] for r in ok) * 100 if ok else None


def stat(runs: List[dict], b: float) -> Optional[Tuple[float, float, int, float]]:
    """(mean acc, std acc, n, mean realized macs_frac)."""
    pairs = [a for a in (acc_at_budget(r["curve"], b) for r in runs) if a is not None]
    if not pairs:
        return None
    vals = [p[0] for p in pairs]
    macs = [p[1] for p in pairs]
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0), len(vals), mean(macs)


def candidate_tag_pool(cell: dict) -> Dict[str, List[dict]]:
    """All candidate HP tags with >=3 seeds of budgeted evals."""
    # collect all candidate hp tags across cand_roots + globalhp roots
    by_tag: Dict[str, List[dict]] = {}
    roots = [REPO / r for r in cell["cand_roots"]]
    # sibling fill dirs (named <base>_globalhp)
    ghp_name = Path(cell["base_roots"][0]).name.replace("_tier2", "") + "_globalhp"
    for root in roots + [REPO / "outputs" / ghp_name]:
        if not root.exists():
            continue
        for mdir in sorted(root.iterdir()):
            if not mdir.is_dir() or not mdir.name.startswith("poe_distill_mtl_brier"):
                continue
            tag_full = mdir.name.replace("poe_distill_mtl_brier__", "")
            tag = re.sub(r"(_rho0)?_seed\d+$", "", tag_full)
            tag = re.sub(r"-rho0\.5$|-T1\.0-rho0\.5$", lambda m: m.group(0), tag)
            for seed_dir in sorted(mdir.glob("seed*")):
                bj = seed_dir / "budgeted.json"
                if bj.exists():
                    d = json.loads(bj.read_text())
                    d["_rho0"] = "_rho0" in mdir.name
                    by_tag.setdefault(tag, []).append(d)
    # Reported runs use the paper recipe (rho=0.5). rho0 runs are retained only
    # as selection data for tags that have no rho0.5 runs yet (pending retrain).
    for tag, runs in by_tag.items():
        rho05 = [r for r in runs if not r.get("_rho0")]
        if rho05:
            by_tag[tag] = rho05
    return {t: r for t, r in by_tag.items() if len(r) >= 3} or by_tag


def candidate_at_contract(pool: Dict[str, List[dict]], b: float) -> Tuple[str, List[dict]]:
    """Per-contract selection: the tag with max mean val accuracy at contract b."""
    def score(runs):
        vs = [v for v in (val_acc_at_budget(r["curve"], b) for r in runs) if v is not None]
        return mean(vs) if vs else -1
    best = max(pool, key=lambda t: score(pool[t]))
    return best, pool[best]


def fmt(entry: Optional[Tuple[float, float, int, float]], bold: bool) -> str:
    if entry is None:
        return "--"
    m, s, n, mc = entry
    core = f"{m:.2f}" + (r"\,{\tiny$\pm$" + f"{s:.2f}" + "}")
    if n < 3:
        core += r"$^{\dagger}$"
    core += r"$_{\mathresized{" + f"{mc:.2f}".lstrip("0") + r"}}$"
    return r"\textbf{" + core + "}" if bold else core


def undominated(entries: Dict[str, Optional[Tuple[float, float, int, float]]]) -> set:
    """Bold set for a row: the accuracy leader, plus any entry whose accuracy is
    whose error bar overlaps the leader's at materially lower realized compute."""
    valid = {m: e for m, e in entries.items() if e is not None}
    if not valid:
        return set()
    leader = max(valid, key=lambda m: valid[m][0])
    lm, ls, _, lmac = valid[leader]
    out = {leader}
    for m, (a, s, _, mac) in valid.items():
        if m == leader:
            continue
        if lm - a <= s + ls and mac < lmac - 0.01:
            out.add(m)
    return out


def build_tables() -> str:
    rows_out = []
    picked_hps = {}
    for cell in CELLS:
        if cell.get("pending"):
            continue
        # baseline runs
        method_runs: Dict[str, List[dict]] = {}
        for m in BASELINES:
            if m in FIXED:
                method_runs[m] = seed_jsons(REPO / cell["fix_root"], m)
            else:
                runs = []
                for r in cell["base_roots"]:
                    runs.extend(seed_jsons(REPO / r, m))
                method_runs[m] = runs
        pool = candidate_tag_pool(cell)
        feasible = [b for b in BUDGETS
                    if any(stat(runs, b) for runs in method_runs.values())
                    or stat(candidate_at_contract(pool, b)[1], b)]
        block = []
        for b in feasible:
            hp, cand = candidate_at_contract(pool, b)
            picked_hps[f"{cell['name']} B{b}"] = (hp, len(cand))
            entries = {m: stat(runs, b) for m, runs in method_runs.items()}
            entries["ours"] = stat(cand, b)
            bold_set = undominated(entries)
            cols = [fmt(entries[m], m in bold_set) for m in BASELINES + ["ours"]]
            lead = (r"\multirow{" + str(len(feasible))
                    + r"}{*}{\shortstack[l]{\textbf{" + cell["name"]
                    + r"}\\ " + cell["backbone"] + "}}") if b == feasible[0] else ""
            block.append(f"{lead} & {b:.1f} & " + " & ".join(cols) + r" \\")
        rows_out.append("\n".join(block))

    main = []
    main.append("% Candidate HP per (cell, contract) (rule: max val acc at that contract):")
    for name, (hp, n) in picked_hps.items():
        main.append(f"%   {name:14s} {hp}  (n={n})")
    main.append(r"\providecommand{\mathresized}[1]{\text{\fontsize{5.5}{6}\selectfont #1}}")
    main.append(r"\begin{table*}[!tbp]")
    main.append(r"\centering")
    main.append(r"\caption{Test accuracy (\%) within a shared compute budget. $B$ caps the")
    main.append(r"average per-sample MAC cost as a fraction of full-depth cost; for every")
    main.append(r"method, per-exit thresholds are solved on validation within the contract")
    dag_note = r"; $^{\dagger}$ marks entries with fewer seeds" if any(
        r"\dagger" in r for r in rows_out) else ""
    main.append(r"(Section~\ref{sec:metrics-eval}). Mean $\pm$ std over three seeds" + dag_note + r".")
    main.append(r"Contracts below the")
    main.append(r"first exit's cost of a backbone cannot be met and are omitted.")
    main.append(r"The subscript is the realized")
    main.append(r"test compute. Bold marks the row's accuracy leader and any method at")
    main.append(r"lower realized compute whose error bar overlaps the leader's.}")
    main.append(r"\label{tab:budget-main}")
    main.append(r"\renewcommand{\arraystretch}{1.08}")
    main.append(r"\setlength{\tabcolsep}{3pt}")
    main.append(r"\resizebox{\textwidth}{!}{%")
    main.append(r"\begin{tabular}{@{} l c " + "r" * 11 + r" @{}}")
    main.append(r"\toprule")
    main.append(r"\textbf{Dataset} & $B$ & " +
                " & ".join(r"\textbf{" + LATEX[m] + "}" for m in BASELINES) +
                r" & \textbf{\ourapproach} \\")
    main.append(r"\midrule")
    main.append(("\n" + r"\midrule" + "\n").join(rows_out))
    main.append(r"\bottomrule")
    main.append(r"\end{tabular}}")
    main.append(r"\end{table*}")

    return "\n".join(main)


def splice(tex: str, start: str, end: str, payload: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    block = f"{start}\n{payload}\n{end}"
    if pattern.search(tex):
        return pattern.sub(lambda _: block, tex)
    raise SystemExit(f"markers {start} .. {end} not found in tex")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--insert", action="store_true")
    args = ap.parse_args()
    main_tex = build_tables()
    print(main_tex)
    if args.insert:
        OUT_TEX.parent.mkdir(exist_ok=True)
        OUT_TEX.write_text(main_tex + "\n")
        print(f"\nwrote {OUT_TEX}")


if __name__ == "__main__":
    main()
