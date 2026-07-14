"""Run the MSDNet-style budgeted-batch evaluation over every checkpoint of a cell.

For each run dir (method x seed) under the cell's tier2 / multiseed / hpsweep
roots: rebuild the model from checkpoint.pt, collect per-exit score/correctness
matrices on the validation and test splits (one full forward pass each), sweep
MSDNet population thresholds on validation, and write

    <run_dir>/exit_scores.npz   raw matrices (reusable for any OP rule offline)
    <run_dir>/budgeted.json     q-grid budget curve (val + test per q)

Usage:
    python scripts/budgeted_eval.py --cell CIFAR-100 [--force]

Run on a GPU node.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jolt.budgeted import (
    budget_curve, collect_exit_matrix, collect_exit_matrix_text,
)
from jolt.config import ExperimentConfig
from jolt.train import _build_dataloaders, _build_model

CELLS = {
    "UCI-HAR": {
        "config": "configs/ucihar_tst.yaml",
        "data_root": str(REPO / "data/UCI HAR Dataset"),
        "roots": ["outputs/ucihar"],
    },
    "PAMAP2": {
        "config": "configs/pamap2_inceptiontime_canonical27.yaml",
        "data_root": str(REPO / "data/PAMAP2/canonical27"),
        "roots": ["outputs/pamap2"],
    },
    "CIFAR-100": {
        "config": "configs/cifar100_convmixer_256_8.yaml",
        "data_root": str(REPO / "data"),
        "roots": ["outputs/cifar100"],
    },
    "GSC-v2": {
        "config": "configs/gsc_v2_matchboxnet.yaml",
        "data_root": str(REPO / "data"),
        "roots": ["outputs/gsc"],
    },
    "SST-2": {
        "config": "configs/glue_sst2_bert_base_multiexit_e3612.yaml",
        "data_root": str(REPO / "data"),
        "roots": ["outputs/sst2"],
    },
    "ESC-50": {
        "config": "configs/esc50_efficientnet_b0_v2.yaml",
        "data_root": str(REPO / "data/ESC-50"),
        "roots": ["outputs/esc50"],
    },
    "Tiny-ImageNet": {
        "config": "configs/tinyimagenet_cct7.yaml",
        "data_root": str(REPO / "data"),
        "roots": ["outputs/tinyimagenet"],
    },
}

def fix_state_shapes(model: torch.nn.Module, state: dict) -> dict:
    """Reshape checkpoint tensors whose element count matches the model parameter
    but whose dims differ (older Linear-style exit heads [C_out, C_in] vs current
    1x1-conv heads [C_out, C_in, 1, 1]). Only reshapes when numel matches."""
    model_state = model.state_dict()
    out = {}
    for k, v in state.items():
        if (k in model_state and hasattr(v, "shape")
                and v.shape != model_state[k].shape
                and v.numel() == model_state[k].numel()):
            out[k] = v.reshape(model_state[k].shape)
        else:
            out[k] = v
    return out


POE_METHODS = {
    "poe_anneal", "poe_distill", "poe_multitask", "poe_anytime", "poe_asym",
    "poe_brier", "scar_poe", "poe_distill_mtl", "poe_distill_brier",
    "poe_distill_asym", "poe_distill_anytime", "poe_distill_mtl_brier",
    "poe_distill_mtl_asym", "poe_distill_mtl_brier_asym", "poe_distill_mtl_brier_mac",
    "poe_jazbec", "poe_multitask_brier",
}
# Methods whose routing needs machinery collect_exit_matrix does not implement.
SKIP_METHODS = {"moe_router", "scar", "tri_axis", "scar_poe"}


def run_seed(run_dir: Path) -> int:
    """The seed a run trained with, reconstructed from the layout: roots
    ending in `_multiseed` place seedK at base seed 43+K; everywhere else
    seedK trained at 42+K, and single-seed runs carry the offset as
    `_seedN` in the method-dir tag (base seed 42+N). Getting this wrong
    re-carves train-split validation sets so they overlap the run's
    training data."""
    inner = run_dir.name                      # seedK
    tag = run_dir.parent.name                 # method__hp_seedN
    k = int(re.search(r"^seed(\d+)$", inner).group(1))
    if run_dir.parent.parent.name.endswith("_multiseed"):
        return 43 + k
    if k == 0:
        m = re.search(r"_seed(\d+)$", tag)
        if m:
            k = int(m.group(1))
    return 42 + k


def process_run(run_dir: Path, cell: dict, device: torch.device, force: bool,
                uniform_alpha: bool = False, score_type: str = "entropy") -> None:
    mp = run_dir / "metrics.json"
    ckpt = run_dir / "checkpoint.pt"
    suffix = "_ua" if uniform_alpha else ""
    if score_type == "margin":
        suffix = "_margin"
    out_json = run_dir / f"budgeted{suffix}.json"
    out_npz = run_dir / f"exit_scores{suffix}.npz"
    if not mp.exists() or not ckpt.exists():
        return
    if out_json.exists() and out_npz.exists() and not force:
        print(f"  [skip] {run_dir.relative_to(REPO)} (already done)")
        return

    data = json.loads(mp.read_text())
    method = data.get("method")
    pem = list(data.get("per_exit_macs", []))
    if not method or not pem:
        print(f"  [skip] {run_dir.relative_to(REPO)}: missing method/per_exit_macs")
        return
    if method in SKIP_METHODS:
        print(f"  [skip] {run_dir.relative_to(REPO)}: unsupported routing for {method}")
        return

    cfg = ExperimentConfig.from_yaml(str(REPO / cell["config"]))
    cfg.loss.method = method
    cfg.data.root = cell["data_root"]
    cfg.data.download = False
    cfg.seed = run_seed(run_dir)

    cutoff = "poe_entropy" if method in POE_METHODS else "entropy"

    model = _build_model(cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if cutoff == "poe_entropy":
        n_exits = model.num_exits + 1
        if "poe_alphas" in state and not uniform_alpha:
            init_alphas = state["poe_alphas"].detach().clone().to(device)
        else:
            init_alphas = torch.ones(n_exits, device=device, dtype=torch.float32)
            state.pop("poe_alphas", None)  # keep load_state_dict from restoring it
        model.register_buffer("poe_alphas", init_alphas, persistent=True)
    missing, unexpected = model.load_state_dict(fix_state_shapes(model, state), strict=False)

    _, val_loader, test_loader = _build_dataloaders(cfg)
    if cfg.data.name == "glue":
        if score_type != "entropy":
            print(f"  [skip] {run_dir.relative_to(REPO)}: text path has no margin collector")
            return
        collect = lambda *a, **kw: collect_exit_matrix_text(*a, **{k: v for k, v in kw.items() if k != "score_type"})
    else:
        collect = collect_exit_matrix
    mats_val = collect(model, val_loader, device=device, cutoff_type=cutoff, score_type=score_type)
    mats_test = collect(model, test_loader, device=device, cutoff_type=cutoff, score_type=score_type)

    rows = budget_curve(mats_val, mats_test, pem)

    np.savez_compressed(
        out_npz,
        scores_val=mats_val["scores"], correct_val=mats_val["correct"], labels_val=mats_val["labels"],
        scores_test=mats_test["scores"], correct_test=mats_test["correct"], labels_test=mats_test["labels"],
        per_exit_macs=np.asarray(pem, dtype=np.float64),
    )
    out_json.write_text(json.dumps({
        "method": method,
        "cutoff_type": cutoff,
        "per_exit_macs": pem,
        "n_missing_keys": len(missing),
        "n_unexpected_keys": len(unexpected),
        "deep_acc_val": float(mats_val["correct"][:, -1].mean()),
        "deep_acc_test": float(mats_test["correct"][:, -1].mean()),
        "curve": rows,
    }, indent=1))
    print(f"  [done] {run_dir.relative_to(REPO)} method={method} "
          f"deep_test_acc={mats_test['correct'][:, -1].mean()*100:.2f} "
          f"(missing={len(missing)} unexpected={len(unexpected)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--score-type", default="entropy", choices=["entropy", "margin"],
                    help="routing score; margin writes budgeted_margin.json")
    ap.add_argument("--uniform-alpha", action="store_true",
                    help="override poe_alphas with ones at inference; outputs "
                         "budgeted_ua.json / exit_scores_ua.npz")
    ap.add_argument("--only", default=None,
                    help="substring filter on method dir names")
    args = ap.parse_args()

    cell = CELLS[args.cell]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Budgeted eval for {args.cell} on {device}")
    for root_rel in cell["roots"]:
        root = REPO / root_rel
        if not root.exists():
            print(f" [warn] root missing: {root_rel}")
            continue
        for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if args.only and args.only not in method_dir.name:
                continue
            for seed_dir in sorted(method_dir.glob("seed*")):
                try:
                    process_run(seed_dir, cell, device, args.force,
                                uniform_alpha=args.uniform_alpha, score_type=args.score_type)
                except Exception as exc:
                    print(f"  [error] {seed_dir.relative_to(REPO)}: {exc}")
    print("all done")


if __name__ == "__main__":
    main()
