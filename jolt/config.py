"""Typed experiment configuration loaded from YAML, plus global seeding.

No source repo had config files or consistent seeding (CIFAR-10 set 42/100; CIFAR-100,
Imagenette, and UCI-HAR set nothing), which is why from-scratch paper numbers were not
reproducible. Every experiment now flows through one YAML and one seeding entry point.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml


@dataclass
class DataConfig:
    name: str = "cifar100"
    root: str = "./data"
    num_classes: int = 100
    in_channels: int = 3
    batch_size: int = 128
    num_workers: int = 2
    val_size: int = 5000
    download: bool = False
    # modern augmentation (vision)
    randaugment: bool = False
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9
    cutout: float = 0.0          # RandomErasing probability (0 disables)
    image_size: int = 32         # input resolution (Imagenette: 32 to reuse the CIFAR stem, or 160/192 native)
    # IMU augmentation (UCI-HAR, PAMAP2); 0 disables. Rotation/time-warp are a future addition.
    imu_jitter: float = 0.0      # Gaussian noise std added to the signal
    imu_scale: float = 0.0       # std of the per-window multiplicative scale
    # UCI-HAR layout: "2d" (legacy v3 reshape into (n_steps, n_length, 9) for ResNet-18)
    # or "1d" (channels-first (9, 128) for 1D-CNN backbones — the v4 canonical setup).
    layout: str = "2d"
    # PAMAP2 only: which subject is held out under LOSO (default subject106 per Pellegrini 2023).
    test_subject: str = "subject106"
    # GLUE/text only
    task: Optional[str] = None
    max_len: int = 128


@dataclass
class ModelConfig:
    name: str = "mobilenetv2_exit"
    stem_stride: int = 1                   # >1 = native-resolution downsampling stem (Imagenette)
    # GLUE/text only
    pretrained_name: Optional[str] = None
    exit_layers: Optional[List[int]] = None
    dropout: float = 0.1
    # Optional override of the number of early exits (default is whatever the model
    # backbone uses natively). Only used by the exit-count ablation cells.
    num_early_exits: Optional[int] = None


@dataclass
class LossConfig:
    # method: "jolt" (variance scaling + adaptive weighting), a baseline
    # (adaloss, branchynet, eenet, td, meronen), a candidate preset (candidate_a..f, focal_only,
    # byot_only, ls, s_avuc, brier), or "composite" (free-form, configured via `components`).
    method: str = "jolt"
    use_b: bool = True          # variance-based scaling (JOLT only)
    use_multitask: bool = True  # adaptive uncertainty weighting; off + use_b=True => collapse case
    divide_b: float = 10.0
    # Overrides for the composite loss (see jolt.losses_zoo.LossComponents); applied on top of a
    # preset or the default composite. Keys are validated against LossComponents fields.
    components: dict = field(default_factory=dict)


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 0.1
    schedule: str = "multistep"  # multistep | cosine
    milestones: List[int] = field(default_factory=lambda: [60, 120, 160])
    warmup_epochs: int = 1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    nesterov: bool = False
    eta_lr: float = 1e-4        # learning rate for MultiTaskLoss eta parameters
    # modern recipe (vision)
    ema_decay: float = 0.0      # 0 disables EMA
    mixup_alpha: float = 0.0    # 0 disables MixUp
    cutmix_alpha: float = 0.0   # 0 disables CutMix
    # text / transformer fine-tuning
    warmup_ratio: float = 0.1   # linear warmup fraction (GLUE)
    grad_clip: float = 0.0      # global-norm gradient clip; 0 disables
    bf16: bool = False          # bf16 autocast (Ampere/TPU). NEVER fp16 with MobileBERT (NaNs).
    two_stage: bool = False     # DeeBERT-style: train backbone+final exit, then freeze and train early exits
    alt_finetune: bool = False  # BERxiT-style: alternate per step between updating all params and only the exit heads
    # Generic curriculum (vision/HAR/audio): if curriculum_stage1_epochs > 0, the first N
    # epochs train the backbone + final classifier only (early heads + their aux heads
    # frozen), with loss = CE on the final exit. Remaining epochs unfreeze everything and
    # use the configured method as normal. Targets the WRN-28-10 / BudgetBoost collapse
    # mode by giving the backbone time to converge before early exits add gradient pressure.
    curriculum_stage1_epochs: int = 0


@dataclass
class EvalConfig:
    cutoff_type: str = "entropy"
    target_accuracies: List[float] = field(default_factory=lambda: [round(0.6 + 0.01 * i, 2) for i in range(40)])
    split_at: Optional[int] = None


@dataclass
class DiagnosticsConfig:
    enabled: bool = False
    log_class_distribution: bool = True
    log_gradient_norms: bool = True
    log_exit1_entropy: bool = True


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "ExperimentConfig":
        section_types = {
            "data": DataConfig,
            "model": ModelConfig,
            "loss": LossConfig,
            "train": TrainConfig,
            "eval": EvalConfig,
            "diagnostics": DiagnosticsConfig,
        }
        kwargs = {}
        for key, value in raw.items():
            if key in section_types and isinstance(value, dict):
                kwargs[key] = _build(section_types[key], value)
            else:
                kwargs[key] = value
        return cls(**kwargs)


def _build(dc_type, values: dict):
    valid = {f.name for f in fields(dc_type)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown keys for {dc_type.__name__}: {sorted(unknown)}")
    return dc_type(**values)


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch (CPU + CUDA) for reproducible runs.

    Note: this does not by itself make multi-worker DataLoaders deterministic. Pass a seeded
    generator and :func:`seed_worker` to each shuffling DataLoader (the dataset loaders here
    already do). Exact bitwise reproducibility still depends on hardware and library versions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` so each worker's NumPy/Python RNG is seeded deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A seeded ``torch.Generator`` for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
