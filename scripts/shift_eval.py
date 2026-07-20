"""Distribution-shift evaluation: clean-validation thresholds on corrupted test sets.

For each run dir (method x seed0) of a CIFAR cell: rebuild the model from
checkpoint.pt, collect per-exit score/correctness matrices on every requested
CIFAR-C corruption x severity, and simulate routing with the SAME MSDNet
population thresholds solved on the CLEAN validation split (reusing
exit_scores.npz from budgeted_eval.py when present, else collecting clean
matrices first).

The interesting outputs per (corruption, severity, q):
  - accuracy under shift
  - exit populations (does routing move deeper as severity rises?)
  - realized compute (does the model gracefully spend more MACs?)

Usage:
    python scripts/shift_eval.py --cell CIFAR-10 [--severities 1,3,5] [--methods all]

Writes <run_dir>/shift_eval.json and <run_dir>/exit_scores_shift.npz (scores only,
compressed, one entry per corruption_severity).
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
from jolt.config import ExperimentConfig
from jolt.train import _build_dataloaders, _build_model

from budgeted_eval import CELLS, POE_METHODS, SKIP_METHODS, fix_state_shapes  # noqa: E402

CORRUPTION_ROOT = Path(__file__).resolve().parents[1] / "data/corruptions"
C_DIRS = {"CIFAR-10": "CIFAR-10-C", "CIFAR-100": "CIFAR-100-C",
          "Tiny-ImageNet": "Tiny-ImageNet-C"}

# Standard Hendrycks & Dietterich corruption set.
CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]

_STATS = {
    "CIFAR-10": (
        (0.49139968, 0.48215827, 0.44653124),
        (0.24703233, 0.24348505, 0.26158768),
    ),
    "CIFAR-100": (
        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
    ),
}


def corrupted_loader(cell_name: str, corruption: str, severity: int,
                     batch_size: int = 256) -> DataLoader:
    """Corrupted test loader with the cell's test-time normalization.

    CIFAR-C npy layout: [50000, 32, 32, 3] uint8, severities 1..5 stacked in
    blocks of 10000; labels.npy holds the matching 50000 labels.
    Tiny-ImageNet-C: ImageFolder JPEG layout <corruption>/<severity>/<wnid>/,
    same sorted-wnid class mapping as the training ImageFolder.
    """
    cdir = CORRUPTION_ROOT / C_DIRS[cell_name]
    if cell_name == "Tiny-ImageNet":
        import torchvision
        import torchvision.transforms as T
        tf = T.Compose([T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        ds = torchvision.datasets.ImageFolder(str(cdir / corruption / str(severity)),
                                              transform=tf)
        if len(ds.classes) != 200:
            raise ValueError(f"Tiny-ImageNet-C {corruption}/{severity}: "
                             f"expected 200 classes, got {len(ds.classes)}")
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    imgs = np.load(cdir / f"{corruption}.npy", mmap_mode="r")
    labels = np.load(cdir / "labels.npy")
    lo, hi = (severity - 1) * 10000, severity * 10000
    x = imgs[lo:hi].astype(np.float32) / 255.0          # [N, 32, 32, 3]
    x = torch.from_numpy(x).permute(0, 3, 1, 2)          # [N, 3, 32, 32]
    mean, std = _STATS[cell_name]
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    y = torch.from_numpy(labels[lo:hi].astype(np.int64))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False,
                      num_workers=0)


def load_model(run_dir: Path, cell: dict, device: torch.device):
    mp = run_dir / "metrics.json"
    ckpt = run_dir / "checkpoint.pt"
    data = json.loads(mp.read_text())
    method = data.get("method")
    pem = list(data.get("per_exit_macs", []))
    if not method or not pem or method in SKIP_METHODS:
        return None
    cfg = ExperimentConfig.from_yaml(str(REPO / cell["config"]))
    cfg.loss.method = method
    cfg.data.root = cell["data_root"]
    cfg.data.download = False
    cutoff = "poe_entropy" if method in POE_METHODS else "entropy"
    model = _build_model(cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if cutoff == "poe_entropy":
        n_exits = model.num_exits + 1
        alphas = (state["poe_alphas"].detach().clone().to(device)
                  if "poe_alphas" in state
                  else torch.ones(n_exits, device=device, dtype=torch.float32))
        model.register_buffer("poe_alphas", alphas, persistent=True)
    model.load_state_dict(fix_state_shapes(model, state), strict=False)
    return model, method, cutoff, pem, cfg


def clean_val_matrices(run_dir: Path, model, cfg, cutoff, device):
    npz_path = run_dir / "exit_scores.npz"
    if npz_path.exists():
        z = np.load(npz_path)
        return {"scores": z["scores_val"], "correct": z["correct_val"], "labels": z["labels_val"]}
    _, val_loader, _ = _build_dataloaders(cfg)
    return collect_exit_matrix(model, val_loader, device=device, cutoff_type=cutoff)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, choices=sorted(C_DIRS))
    ap.add_argument("--severities", default="1,3,5")
    ap.add_argument("--corruptions", default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="",
                    help="substring filter on the run-dir path (e.g. a method tag)")
    args = ap.parse_args()

    severities = [int(s) for s in args.severities.split(",")]
    corruptions = CORRUPTIONS if args.corruptions == "all" else args.corruptions.split(",")
    cell = CELLS[args.cell]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Shift eval for {args.cell}: {len(corruptions)} corruptions x {severities} on {device}")

    run_dirs = []
    for root_rel in cell["roots"]:
        root = REPO / root_rel
        if not root.exists():
            continue
        for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for seed_dir in sorted(method_dir.glob("seed*")):
                if args.only and args.only not in str(seed_dir):
                    continue
                if (seed_dir / "checkpoint.pt").exists() and (seed_dir / "metrics.json").exists():
                    run_dirs.append(seed_dir)

    for run_dir in run_dirs:
        out_json = run_dir / "shift_eval.json"
        if out_json.exists() and not args.force:
            print(f"  [skip] {run_dir.relative_to(REPO)} (already done)")
            continue
        try:
            loaded = load_model(run_dir, cell, device)
        except Exception as exc:
            print(f"  [error] {run_dir.relative_to(REPO)}: {exc}")
            continue
        if loaded is None:
            continue
        model, method, cutoff, pem, cfg = loaded

        try:
            mats_val = clean_val_matrices(run_dir, model, cfg, cutoff, device)
        except Exception as exc:
            print(f"  [error] {run_dir.relative_to(REPO)} (val matrices): {exc}")
            continue
        thr_by_q = {q: thresholds_for_population(mats_val["scores"], q) for q in DEFAULT_Q_GRID}

        results = {}
        npz_payload = {}
        for corruption in corruptions:
            results[corruption] = {}
            for sev in severities:
                loader = corrupted_loader(args.cell, corruption, sev)
                mats_c = collect_exit_matrix(model, loader, device=device, cutoff_type=cutoff)
                npz_payload[f"scores_{corruption}_{sev}"] = mats_c["scores"].astype(np.float16)
                npz_payload[f"correct_{corruption}_{sev}"] = mats_c["correct"].astype(np.uint8)
                rows = []
                for q, thr in thr_by_q.items():
                    # Absolute policy: thresholds frozen from CLEAN validation.
                    st = simulate_routing(mats_c["scores"], mats_c["correct"], thr, pem)
                    # Quantile policy: same q level, thresholds re-estimated from the
                    # corrupted score stream (idealized batch quantile; the streaming
                    # window version is analyzed offline from the saved matrices).
                    thr_q = thresholds_for_population(mats_c["scores"], q)
                    st_q = simulate_routing(mats_c["scores"], mats_c["correct"], thr_q, pem)
                    rows.append({"q": q,
                                 "accuracy": st["accuracy"],
                                 "exit_counts": st["exit_counts"],
                                 "macs_frac": st["macs_frac"],
                                 "q_accuracy": st_q["accuracy"],
                                 "q_exit_counts": st_q["exit_counts"],
                                 "q_macs_frac": st_q["macs_frac"]})
                results[corruption][str(sev)] = {
                    "deep_acc": float(mats_c["correct"][:, -1].mean()),
                    "exit1_acc_all": float(mats_c["correct"][:, 0].mean()),
                    "rows": rows,
                }
            print(f"  [{run_dir.relative_to(REPO)}] {method} {corruption} done")
        np.savez_compressed(run_dir / "exit_scores_shift.npz", **npz_payload)

        out_json.write_text(json.dumps({
            "method": method, "cutoff_type": cutoff, "per_exit_macs": pem,
            "severities": severities, "results": results,
        }, indent=None))
        print(f"  [done] {run_dir.relative_to(REPO)} method={method}")

    print("all done")


if __name__ == "__main__":
    main()
