"""Run the candidate-screening funnel on one dataset and write a ranked leaderboard.

    # local CPU smoke over a few candidates:
    python scripts/screen.py --config configs/ucihar_resnet18.yaml \
        --data-root "/path/to/UCI HAR Dataset" --seeds 1 \
        --candidates jolt,branchynet --output-root outputs/screen/ucihar --smoke

    # full stage (cluster): all candidates, 1 seed on UCI-HAR
    python scripts/screen.py --config configs/ucihar_resnet18.yaml \
        --data-root "/path/to/UCI HAR Dataset" --seeds 1 --output-root outputs/screen/ucihar

Ranks by a composite of EMAR(p=2) and AURC (see jolt.screening). Use the printed top-k to choose
which candidates advance to the next funnel stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from jolt.config import ExperimentConfig
from jolt.screening import write_leaderboard
from jolt.train import run_experiment

DEFAULT_CANDIDATES = [
    "jolt", "adaloss", "branchynet", "eenet", "td", "meronen",
    "candidate_a", "candidate_b", "candidate_c", "candidate_d", "candidate_e", "candidate_f",
    "focal_only", "byot_only", "ls", "s_avuc", "brier",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-leaderboard", action="store_true",
                        help="skip writing the leaderboard (for cluster array jobs; aggregate separately)")
    parser.add_argument(
        "--component", action="append", default=[],
        help="Override one entry in cfg.loss.components as KEY=VALUE (float). Repeatable, "
        "e.g. --component gamma_distill=1.0 --component rho_max=0.3. Useful for HP sweeps.",
    )
    parser.add_argument(
        "--variant-tag", default=None,
        help="Optional suffix appended to the output sub-dir (writes to <root>/<candidate>__<tag>/ "
        "instead of <root>/<candidate>/). Use with --component to sweep HPs side-by-side.",
    )
    parser.add_argument(
        "--num-early-exits", type=int, default=None,
        help="Override cfg.model.num_early_exits. Used for the exit-count ablation cells. "
        "Backbones that ignore this field train at their native exit count.",
    )
    args = parser.parse_args()

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    # Parse --component KEY=VALUE entries. All values cast to float (the existing
    # components dict accepts numeric scalars; string fields like weight_schedule must
    # still go through YAML or per-config edits).
    component_overrides = {}
    for entry in args.component:
        if "=" not in entry:
            raise ValueError(f"--component must be KEY=VALUE; got '{entry}'")
        k, v = entry.split("=", 1)
        try:
            component_overrides[k.strip()] = float(v.strip())
        except ValueError:
            raise ValueError(f"--component value must be a float; got '{v}' for key '{k}'")

    for candidate in candidates:
        for seed_idx in range(args.seeds):
            cfg = ExperimentConfig.from_yaml(args.config)
            cfg.loss.method = candidate
            cfg.seed = args.base_seed + seed_idx
            if args.data_root is not None:
                cfg.data.root = args.data_root
                cfg.data.download = False
            if component_overrides:
                # Merge into the existing components dict (preserves any defaults set in YAML).
                cfg.loss.components = {**(cfg.loss.components or {}), **component_overrides}
            if args.num_early_exits is not None:
                cfg.model.num_early_exits = args.num_early_exits
            limit_train = limit_eval = None
            targets = None
            if args.smoke:
                cfg.train.epochs = min(cfg.train.epochs, 1)
                cfg.data.num_workers = 0
                cfg.data.batch_size = 64
                limit_train = limit_eval = 5
                targets = [0.6, 0.8]
            dir_name = candidate if args.variant_tag is None else f"{candidate}__{args.variant_tag}"
            out_dir = root / dir_name / f"seed{seed_idx}"
            run_experiment(
                cfg, output_dir=str(out_dir), limit_train_batches=limit_train,
                limit_eval_batches=limit_eval, target_accuracies=targets,
            )
            print(f"[done] {dir_name} seed{seed_idx}")

    if args.skip_leaderboard:
        return
    rows = write_leaderboard(root, root / "leaderboard.csv")
    print(f"\nleaderboard -> {root / 'leaderboard.csv'}")
    for row in rows[:5]:
        print(f"  {row['candidate']:14s} composite={row['composite']:.2f} "
              f"emar={row['emar']:.4f} aurc={row['aurc']:.4f} acc={row['accuracy']:.4f}")


if __name__ == "__main__":
    main()
