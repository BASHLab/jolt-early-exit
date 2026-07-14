"""Calibration at the contract operating point: ECE and NLL of the deployed
routing configuration.

For each run with a budgeted.json: rebuild the model, forward the test split
once collecting per-exit probability vectors, route every sample with the
val-selected thresholds of each contract (the same rows budgeted.json
reports), and score the routed predictions with 15-bin ECE and NLL. Writes
<run_dir>/opcal.json with one entry per contract.

Usage:
    python scripts/opcal_eval.py --cell CIFAR-100 [--only poe_distill_mtl_brier] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from jolt.config import ExperimentConfig
from jolt.train import _build_dataloaders, _build_model

from budgeted_eval import (  # noqa: E402
    CELLS, POE_METHODS, SKIP_METHODS, fix_state_shapes, run_seed,
)

BUDGETS = [0.3, 0.5, 0.7]


def collect_probs(model: nn.Module, loader, device: torch.device, poe: bool):
    """Per-exit probability tensors for every sample, using the same iterative
    per-exit forward as jolt.budgeted.collect_exit_matrix."""
    softmax = nn.Softmax(dim=1)
    log_softmax = nn.LogSoftmax(dim=1)
    n_exits = model.num_exits + 1
    model.eval()
    per_exit = [[] for _ in range(n_exits)]
    labels = []
    with torch.no_grad():
        for images, y in loader:
            x = images.to(device)
            running = None
            for pos in range(n_exits):
                x, logits = model(x, exit_layer_idx=pos)
                if poe:
                    log_p = log_softmax(logits)
                    alpha = float(model.poe_alphas[pos]) if hasattr(model, "poe_alphas") else 1.0
                    running = alpha * log_p if running is None else running + alpha * log_p
                    probs = log_softmax(running).exp()
                else:
                    probs = softmax(logits)
                per_exit[pos].append(probs.cpu())
            labels.append(y)
    return [torch.cat(chunks) for chunks in per_exit], torch.cat(labels)


def route(scores: np.ndarray, thresholds) -> np.ndarray:
    """Exit index per sample. ``scores`` is [num_early, N] (early exits only);
    a sample below no early threshold falls through to the final exit."""
    n_early, n = scores.shape
    out = np.full(n, n_early, dtype=np.int64)
    undecided = np.ones(n, dtype=bool)
    for i in range(n_early):
        take = undecided & (scores[i] <= thresholds[i])
        out[take] = i
        undecided &= ~take
    return out


def ece15(conf: np.ndarray, correct: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, 16)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cell = CELLS[args.cell]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for root_rel in cell["roots"]:
        root = REPO / root_rel
        if not root.exists():
            continue
        for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
            if args.only and not mdir.name.startswith(args.only):
                continue
            for run_dir in sorted(mdir.glob("seed*")):
                bj = run_dir / "budgeted.json"
                out = run_dir / "opcal.json"
                if not bj.exists() or (out.exists() and not args.force):
                    continue
                data = json.loads(bj.read_text())
                method = data["method"]
                if method in SKIP_METHODS:
                    continue
                npz = run_dir / "exit_scores.npz"
                if not npz.exists():
                    continue
                mats = np.load(npz)
                # stored as [N, num_early]; exits-first is what routing wants
                scores_val = mats["scores_val"].T
                scores_test = mats["scores_test"].T
                pem = np.asarray(data["per_exit_macs"], dtype=np.float64)
                deep = pem.sum()

                cfg = ExperimentConfig.from_yaml(str(REPO / cell["config"]))
                cfg.loss.method = method
                cfg.data.root = cell["data_root"]
                cfg.data.download = False
                cfg.seed = run_seed(run_dir)
                poe = method in POE_METHODS
                model = _build_model(cfg).to(device)
                state = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                if poe:
                    n_exits = model.num_exits + 1
                    alphas = state.get("poe_alphas")
                    init = (alphas.detach().clone().to(device) if alphas is not None
                            else torch.ones(n_exits, device=device))
                    state.pop("poe_alphas", None)
                    model.register_buffer("poe_alphas", init, persistent=True)
                model.load_state_dict(fix_state_shapes(model, state), strict=False)
                _, _, test_loader = _build_dataloaders(cfg)
                if cfg.data.name == "glue":
                    print(f"  [skip] {run_dir.relative_to(REPO)}: text path")
                    continue
                probs, labels = collect_probs(model, test_loader, device, poe)
                labels_np = labels.numpy()

                results = {}
                for b in BUDGETS:
                    ok = [r for r in data["curve"] if r["val"]["macs_frac"] <= b]
                    if not ok:
                        continue
                    row = max(ok, key=lambda r: r["val"]["accuracy"])
                    q = row["q"]
                    # sequential re-solve on val, mirroring thresholds_for_population:
                    # one threshold per early exit over the still-routing samples
                    remaining = np.ones(scores_val.shape[1], dtype=bool)
                    thr = []
                    for i in range(scores_val.shape[0]):
                        t = float(np.quantile(scores_val[i][remaining], q)) if remaining.any() else -np.inf
                        thr.append(t)
                        remaining &= scores_val[i] > t
                    exits = route(scores_test, thr)
                    n = labels_np.shape[0]
                    routed_probs = np.empty((n, probs[0].shape[1]), dtype=np.float64)
                    for i in range(len(probs)):
                        m = exits == i
                        if m.any():
                            routed_probs[m] = probs[i][m].numpy()
                    conf = routed_probs.max(axis=1)
                    pred = routed_probs.argmax(axis=1)
                    correct = (pred == labels_np).astype(np.float64)
                    true_p = np.clip(routed_probs[np.arange(n), labels_np], 1e-12, 1.0)
                    results[str(b)] = {
                        "q": q,
                        "accuracy": float(correct.mean()),
                        "ece": ece15(conf, correct),
                        "nll": float(-np.log(true_p).mean()),
                        "macs_frac": float(sum((exits >= i).mean() * pem[i] for i in range(len(pem))) / deep),
                    }
                out.write_text(json.dumps({"method": method, "results": results}))
                print(f"  [done] {run_dir.relative_to(REPO)} {method}")
    print("all done")


if __name__ == "__main__":
    main()
