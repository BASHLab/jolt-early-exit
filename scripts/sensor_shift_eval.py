"""Sensor-domain distribution-shift evaluation for UCI-HAR and PAMAP2.

Applies Um-et-al-style IMU perturbations (ICMI 2017) to the test split at three
severities and evaluates clean-validation-calibrated routing under both the
absolute-threshold policy and the quantile policy (mirrors shift_eval.py, which
covers the CIFAR-C cells).

Perturbations operate on the model-input tensors (post-normalization), so they
model post-calibration sensor degradation:
  jitter    additive Gaussian noise, sigma in {0.05, 0.15, 0.30} (z-scored units)
  scale     per-window multiplicative gain drift, sigma in {0.05, 0.15, 0.30}
  time_warp smooth local speed variation via a 4-knot interpolated warp,
            knot sigma in {0.05, 0.15, 0.30}
  chan_drop k random channels zeroed per window (sensor dropout),
            k in {1, 3, 6} for 9-channel UCI-HAR / {3, 9, 15} for 27-ch PAMAP2

All perturbations are seeded deterministically per (perturbation, severity).

Usage:
    python scripts/sensor_shift_eval.py --cell UCI-HAR [--force]

Writes <run_dir>/sensor_shift_eval.json and <run_dir>/exit_scores_sshift.npz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from jolt.budgeted import (
    DEFAULT_Q_GRID, collect_exit_matrix, simulate_routing, thresholds_for_population,
)
from jolt.train import _build_dataloaders

from budgeted_eval import CELLS  # noqa: E402
from shift_eval import clean_val_matrices, load_model  # noqa: E402

SEVERITIES = [1, 3, 5]
_SIGMA = {1: 0.05, 3: 0.15, 5: 0.30}
_DROP_K = {"UCI-HAR": {1: 1, 3: 3, 5: 6}, "PAMAP2": {1: 3, 3: 9, 5: 15}}
PERTURBATIONS = ["jitter", "scale", "time_warp", "chan_drop"]


def _collect_test_tensors(loader) -> tuple:
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, 0), torch.cat(ys, 0)


def perturb(x: torch.Tensor, kind: str, sev: int, cell: str, seed: int) -> torch.Tensor:
    """x: (N, C, T) float tensor. Deterministic under (kind, sev)."""
    rng = np.random.default_rng(hash((kind, sev, seed)) % 2**31)
    n, c, t = x.shape
    out = x.clone()
    if kind == "jitter":
        noise = rng.normal(0.0, _SIGMA[sev], size=(n, c, t)).astype(np.float32)
        out = out + torch.from_numpy(noise)
    elif kind == "scale":
        gain = rng.normal(1.0, _SIGMA[sev], size=(n, 1, 1)).astype(np.float32)
        out = out * torch.from_numpy(gain)
    elif kind == "time_warp":
        knots = 4
        base = np.linspace(0, t - 1, knots)
        grid = np.arange(t, dtype=np.float64)
        warped = np.empty((n, c, t), dtype=np.float32)
        xn = x.numpy()
        for i in range(n):
            speeds = 1.0 + rng.normal(0.0, _SIGMA[sev], size=knots)
            speeds = np.clip(speeds, 0.3, 3.0)
            speed_t = np.interp(grid, base, speeds)
            pos = np.cumsum(speed_t)
            pos = (pos - pos[0]) / (pos[-1] - pos[0]) * (t - 1)
            for ch in range(c):
                warped[i, ch] = np.interp(pos, grid, xn[i, ch]).astype(np.float32)
        out = torch.from_numpy(warped)
    elif kind == "chan_drop":
        k = _DROP_K[cell][sev]
        for i in range(n):
            drop = rng.choice(c, size=k, replace=False)
            out[i, drop, :] = 0.0
    else:
        raise ValueError(kind)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, choices=["UCI-HAR", "PAMAP2"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on method dir names")
    args = ap.parse_args()

    cell = CELLS[args.cell]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sensor shift eval for {args.cell} on {device}")

    run_dirs = []
    for root_rel in cell["roots"]:
        root = REPO / root_rel
        if not root.exists():
            continue
        for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
            if args.only and args.only not in mdir.name:
                continue
            for seed_dir in sorted(mdir.glob("seed*")):
                if (seed_dir / "checkpoint.pt").exists() and (seed_dir / "metrics.json").exists():
                    run_dirs.append(seed_dir)

    test_cache = None  # (x, y) tensors shared across models of the cell
    for run_dir in run_dirs:
        out_json = run_dir / "sensor_shift_eval.json"
        if out_json.exists() and not args.force:
            print(f"  [skip] {run_dir.relative_to(REPO)}")
            continue
        try:
            loaded = load_model(run_dir, cell, device)
        except Exception as exc:
            print(f"  [error] {run_dir.relative_to(REPO)}: {exc}")
            continue
        if loaded is None:
            continue
        model, method, cutoff, pem, cfg = loaded

        if test_cache is None:
            _, _, test_loader = _build_dataloaders(cfg)
            test_cache = _collect_test_tensors(test_loader)
        x_test, y_test = test_cache

        try:
            mats_val = clean_val_matrices(run_dir, model, cfg, cutoff, device)
        except Exception as exc:
            print(f"  [error] {run_dir.relative_to(REPO)} (val matrices): {exc}")
            continue
        thr_by_q = {q: thresholds_for_population(mats_val["scores"], q) for q in DEFAULT_Q_GRID}

        results = {}
        npz_payload = {}
        for kind in PERTURBATIONS:
            results[kind] = {}
            for sev in SEVERITIES:
                x_p = perturb(x_test, kind, sev, args.cell, seed=1234)
                loader = DataLoader(TensorDataset(x_p, y_test), batch_size=256,
                                    shuffle=False, num_workers=0)
                mats_c = collect_exit_matrix(model, loader, device=device, cutoff_type=cutoff)
                npz_payload[f"scores_{kind}_{sev}"] = mats_c["scores"].astype(np.float16)
                npz_payload[f"correct_{kind}_{sev}"] = mats_c["correct"].astype(np.uint8)
                rows = []
                for q, thr in thr_by_q.items():
                    st = simulate_routing(mats_c["scores"], mats_c["correct"], thr, pem)
                    thr_q = thresholds_for_population(mats_c["scores"], q)
                    st_q = simulate_routing(mats_c["scores"], mats_c["correct"], thr_q, pem)
                    rows.append({"q": q,
                                 "accuracy": st["accuracy"],
                                 "exit_counts": st["exit_counts"],
                                 "macs_frac": st["macs_frac"],
                                 "q_accuracy": st_q["accuracy"],
                                 "q_exit_counts": st_q["exit_counts"],
                                 "q_macs_frac": st_q["macs_frac"]})
                results[kind][str(sev)] = {
                    "deep_acc": float(mats_c["correct"][:, -1].mean()),
                    "rows": rows,
                }
            print(f"  [{run_dir.relative_to(REPO)}] {method} {kind} done")
        np.savez_compressed(run_dir / "exit_scores_sshift.npz", **npz_payload)
        out_json.write_text(json.dumps({
            "method": method, "cutoff_type": cutoff, "per_exit_macs": pem,
            "severities": SEVERITIES, "results": results,
        }, indent=None))
        print(f"  [done] {run_dir.relative_to(REPO)} method={method}")
    print("all done")


if __name__ == "__main__":
    main()
