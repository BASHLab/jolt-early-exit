"""Margin-vs-entropy routing-score ablation for the quantile policy (CIFAR-100).

For six key models: recollect clean val/test matrices and severity-5 matrices of
five representative corruptions with score_type=margin (1 - (top1 - top2)), then
compare against the stored entropy-score results on (a) accuracy at the q=0.5
contract on clean test, (b) budget drift + accuracy under shift with streaming
quantiles, (c) exit-1 score-vs-correctness AUROC. Preempts "why entropy?".

Writes outputs/analysis/margin_score_ablation_cifar100.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from jolt.budgeted import collect_exit_matrix, simulate_routing, thresholds_for_population
from jolt.train import _build_dataloaders

from budgeted_eval import CELLS  # noqa: E402
from quantile_routing_analysis import auroc  # noqa: E402
from shift_eval import corrupted_loader, load_model  # noqa: E402

METHODS = [
    "poe_distill_mtl_brier__g4.0-lb0.5",
    "poe_distill_mtl_brier__g2.0-lb0.5",
    "poe_jazbec",
    "meronen_laplace",
    "adaloss",
    "ztw_cascade",
]
CORRUPTIONS = ["gaussian_noise", "motion_blur", "fog", "contrast", "jpeg_compression"]
SEV = 5
Q = 0.5


def find_run(cell: dict, method: str) -> Path | None:
    for root_rel in cell["roots"]:
        p = REPO / root_rel / method / "seed0"
        if (p / "checkpoint.pt").exists():
            return p
    return None


def eval_score(model, cfg, cutoff, pem, device, score_type: str, cell_name: str) -> dict:
    _, val_loader, test_loader = _build_dataloaders(cfg)
    mats_val = collect_exit_matrix(model, val_loader, device=device,
                                   cutoff_type=cutoff, score_type=score_type)
    mats_test = collect_exit_matrix(model, test_loader, device=device,
                                    cutoff_type=cutoff, score_type=score_type)
    thr = thresholds_for_population(mats_val["scores"], Q)
    clean = simulate_routing(mats_test["scores"], mats_test["correct"], thr, pem)
    target_counts = clean["exit_counts"]

    out = {
        "clean_acc": clean["accuracy"] * 100,
        "clean_macs_frac": clean["macs_frac"],
        "clean_auroc_exit1": auroc(-mats_test["scores"][:, 0].astype(np.float64),
                                   mats_test["correct"][:, 0].astype(np.float64)),
        "shift": {},
    }
    for corruption in CORRUPTIONS:
        loader = corrupted_loader(cell_name, corruption, SEV)
        mats_c = collect_exit_matrix(model, loader, device=device,
                                     cutoff_type=cutoff, score_type=score_type)
        # absolute (clean-val thresholds) vs quantile (corrupted-stream batch quantile)
        ab = simulate_routing(mats_c["scores"], mats_c["correct"], thr, pem)
        thr_q = thresholds_for_population(mats_c["scores"], Q)
        qr = simulate_routing(mats_c["scores"], mats_c["correct"], thr_q, pem)
        tgt = np.asarray(target_counts, dtype=np.float64)
        tgt = tgt / tgt.sum()
        def drift(counts):
            a = np.asarray(counts, dtype=np.float64)
            return float(np.abs(a / a.sum() - tgt).sum())
        out["shift"][corruption] = {
            "abs_acc": ab["accuracy"] * 100, "abs_drift": drift(ab["exit_counts"]),
            "abs_macs": ab["macs_frac"],
            "q_acc": qr["accuracy"] * 100, "q_drift": drift(qr["exit_counts"]),
            "q_macs": qr["macs_frac"],
            "auroc_exit1": auroc(-mats_c["scores"][:, 0].astype(np.float64),
                                 mats_c["correct"][:, 0].astype(np.float64)),
        }
    return out


def main() -> None:
    cell = CELLS["CIFAR-100"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for method in METHODS:
        run = find_run(cell, method)
        if run is None:
            print(f"[skip] {method}: no checkpoint")
            continue
        loaded = load_model(run, cell, device)
        if loaded is None:
            continue
        model, m, cutoff, pem, cfg = loaded
        results[method] = {}
        for score_type in ("entropy", "margin"):
            r = eval_score(model, cfg, cutoff, pem, device, score_type, "CIFAR-100")
            results[method][score_type] = r
            mean_q_acc = np.mean([v["q_acc"] for v in r["shift"].values()])
            print(f"[{method}] {score_type}: clean={r['clean_acc']:.2f} "
                  f"shift_q_acc={mean_q_acc:.2f} auroc1={r['clean_auroc_exit1']:.4f}")
    out = REPO / "outputs/analysis/margin_score_ablation_cifar100.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
