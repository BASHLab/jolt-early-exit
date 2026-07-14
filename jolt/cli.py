"""Shared CLI for the experiment entrypoints in ``experiments/``.

Keeps each entrypoint a two-line wrapper so there is no per-experiment argument-parsing
duplication.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

from .config import ExperimentConfig
from .train import run_experiment


def run_experiment_cli(default_config: Union[str, Path], default_output: str) -> dict:
    parser = argparse.ArgumentParser(description="Run a JOLT experiment from a YAML config.")
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--method", default=None, help="override loss.method (jolt or a baseline)")
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument(
        "--smoke", action="store_true",
        help="fast end-to-end check: 1 epoch, few batches, two calibration targets, no workers",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.method is not None:
        cfg.loss.method = args.method
    if args.data_root is not None:
        cfg.data.root = args.data_root
        cfg.data.download = False

    target_accuracies = None
    limit_train_batches = args.limit_train_batches
    limit_eval_batches = None
    if args.smoke:
        cfg.train.epochs = min(cfg.train.epochs, 1)
        cfg.data.num_workers = 0
        cfg.data.batch_size = 64
        limit_train_batches = limit_train_batches or 5
        limit_eval_batches = 5
        target_accuracies = [0.6, 0.8]

    summary = run_experiment(
        cfg,
        output_dir=args.output_dir,
        limit_train_batches=limit_train_batches,
        limit_eval_batches=limit_eval_batches,
        target_accuracies=target_accuracies,
    )
    best = max(summary["curve"], key=lambda r: r["accuracy"]) if summary["curve"] else {}
    print(f"[{cfg.name}] epochs={cfg.train.epochs} per_exit_macs={summary['per_exit_macs']}")
    print(f"best curve point: {best}")
    return summary
