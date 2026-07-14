"""Experiment runner: config in, trained model + EMAR/accuracy/ECE/NLL curve out.

One code path drives every (dataset, backbone) pair. Entry points in ``experiments/`` are
thin wrappers that load a YAML and call :func:`run_experiment`.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR, SequentialLR

from .augment import mixed_loss, mixup_cutmix
from .calibration import estimate_thresholds_for_accuracy
from .config import ExperimentConfig, seed_everything
from .diagnostics import CollapseLogger
from .datasets import cifar_dataloaders, imagenette_dataloaders, uci_har_dataloaders
from .ema import ModelEMA
from .inference import evaluate_dynamic_exit
from .losses import MultiTaskLoss, forward_all_exits, variance_scaled_losses
from .losses_zoo import selectivenet_loss
from .macs import per_exit_macs
from .methods import method_loss
from .models import (
    deepconvlstm_exit,
    mobilenetv2_exit,
    resnet18_exit,
    resnet56_exit,
    resnet56_sdn_exit,
    resnet56_sdn_litelaplace_exit,
    wideresnet28_10_sdn_exit,
)


def _build_dataloaders(cfg: ExperimentConfig):
    d = cfg.data
    aug = dict(randaugment=d.randaugment, randaugment_num_ops=d.randaugment_num_ops,
               randaugment_magnitude=d.randaugment_magnitude, cutout=d.cutout)
    if d.name in ("cifar10", "cifar100"):
        return cifar_dataloaders(
            name=d.name, root=d.root, batch_size=d.batch_size, num_workers=d.num_workers,
            val_size=d.val_size, download=d.download, seed=cfg.seed, **aug,
        )
    if d.name == "imagenette":
        return imagenette_dataloaders(
            root=d.root, batch_size=d.batch_size, num_workers=d.num_workers,
            val_size=d.val_size, download=d.download, seed=cfg.seed, image_size=d.image_size, **aug,
        )
    if d.name == "tiny_imagenet":
        from .datasets.tiny_imagenet import tiny_imagenet_dataloaders
        return tiny_imagenet_dataloaders(
            root=d.root, batch_size=d.batch_size, num_workers=d.num_workers,
            val_size=d.val_size, download=d.download, seed=cfg.seed, **aug,
        )
    if d.name == "uci_har":
        return uci_har_dataloaders(
            root=d.root, batch_size=d.batch_size, num_workers=d.num_workers,
            val_size=d.val_size, seed=cfg.seed, jitter=d.imu_jitter, scale=d.imu_scale,
            layout=getattr(d, "layout", "2d"),
        )
    if d.name == "pamap2":
        from .datasets.pamap2 import pamap2_dataloaders
        return pamap2_dataloaders(
            root=d.root, batch_size=d.batch_size, num_workers=d.num_workers,
            val_size=d.val_size, seed=cfg.seed, jitter=d.imu_jitter, scale=d.imu_scale,
            test_subject=getattr(d, "test_subject", "subject106"),
        )
    if d.name == "gsc_v2":
        from .datasets.gsc_v2 import gsc_v2_dataloaders
        return gsc_v2_dataloaders(
            root=d.root, batch_size=d.batch_size, num_workers=d.num_workers,
            seed=cfg.seed, download=d.download,
        )
    if d.name == "esc50":
        from .datasets.esc50 import esc50_dataloaders
        return esc50_dataloaders(
            root=d.root, batch_size=d.batch_size, num_workers=d.num_workers,
            seed=cfg.seed, download=d.download,
        )
    if d.name == "glue":
        from .datasets.glue import glue_dataloaders, build_tokenizer
        pretrained = cfg.model.pretrained_name
        if not pretrained:
            raise ValueError("GLUE requires cfg.model.pretrained_name (e.g., 'google/mobilebert-uncased').")
        tokenizer = build_tokenizer(pretrained)
        return glue_dataloaders(
            task=d.task, tokenizer=tokenizer, max_len=d.max_len,
            batch_size=d.batch_size, num_workers=d.num_workers, seed=cfg.seed,
        )
    raise ValueError(f"Unknown dataset '{d.name}'.")


def _build_model(cfg: ExperimentConfig) -> nn.Module:
    name = cfg.model.name
    nc, ic = cfg.data.num_classes, cfg.data.in_channels
    if name == "mobilenetv2_exit":
        base = mobilenetv2_exit(num_classes=nc, in_channels=ic, stem_stride=cfg.model.stem_stride)
    elif name == "resnet18_exit":
        base = resnet18_exit(num_classes=nc, in_channels=ic)
    elif name == "resnet56_exit":
        base = resnet56_exit(num_classes=nc, in_channels=ic)
    elif name == "resnet56_sdn_exit":
        base = resnet56_sdn_exit(num_classes=nc, in_channels=ic)
    elif name == "resnet56_sdn_litelaplace_exit":
        base = resnet56_sdn_litelaplace_exit(num_classes=nc, in_channels=ic)
    elif name == "wideresnet28_10_sdn_exit":
        base = wideresnet28_10_sdn_exit(num_classes=nc, in_channels=ic)
    elif name == "wideresnet28_10_sdn_jei_exit":
        from .models import wideresnet28_10_sdn_jei_exit
        base = wideresnet28_10_sdn_jei_exit(num_classes=nc, in_channels=ic)
    elif name == "deepconvlstm_exit":
        base = deepconvlstm_exit(num_classes=nc, in_channels=ic)
    elif name == "uci_har_1d_cnn_exit":
        from .models import uci_har_1d_cnn_exit
        k = cfg.model.num_early_exits if cfg.model.num_early_exits is not None else 3
        base = uci_har_1d_cnn_exit(num_classes=nc, in_channels=ic, num_early_exits=k)
    elif name == "resnet1d_har_exit":
        from .models.resnet1d_har_exit import resnet1d_har_exit
        base = resnet1d_har_exit(num_classes=nc, in_channels=ic)
    elif name == "inceptiontime_exit":
        from .models import inceptiontime_exit
        base = inceptiontime_exit(num_classes=nc, in_channels=ic)
    elif name == "tcn_har_exit":
        from .models import tcn_har_exit
        base = tcn_har_exit(num_classes=nc, in_channels=ic)
    elif name == "reslstm_exit":
        from .models import reslstm_exit
        base = reslstm_exit(num_classes=nc, in_channels=ic)
    elif name == "tst_exit":
        from .models import tst_exit
        base = tst_exit(num_classes=nc, in_channels=ic)
    elif name == "convmixer1d_exit":
        from .models import convmixer1d_exit
        base = convmixer1d_exit(num_classes=nc, in_channels=ic)
    elif name == "bcresnet8_exit":
        from .models import bcresnet8_exit
        base = bcresnet8_exit(num_classes=nc, in_channels=ic)
    elif name == "matchboxnet_exit":
        from .models.matchboxnet_exit import matchboxnet_exit
        base = matchboxnet_exit(num_classes=nc, in_channels=ic)
    elif name == "efficientnet_b0_exit":
        from .models.efficientnet_b0_exit import efficientnet_b0_exit
        base = efficientnet_b0_exit(num_classes=nc, in_channels=ic)
    elif name == "cct7_exit":
        from .models import cct7_exit
        base = cct7_exit(num_classes=nc, in_channels=ic)
    elif name == "mobilenetv3_large_exit":
        from .models import mobilenetv3_large_exit
        base = mobilenetv3_large_exit(num_classes=nc, in_channels=ic, stem_stride=cfg.model.stem_stride)
    elif name == "mobilenetv3_small_exit":
        from .models import mobilenetv3_small_exit
        base = mobilenetv3_small_exit(num_classes=nc, in_channels=ic, stem_stride=cfg.model.stem_stride)
    elif name == "densenet_bc_100_exit":
        from .models import densenet_bc_100_exit
        base = densenet_bc_100_exit(num_classes=nc, in_channels=ic)
    elif name == "resnet110_exit":
        from .models import resnet110_exit
        base = resnet110_exit(num_classes=nc, in_channels=ic)
    elif name == "wideresnet28_2_exit":
        from .models import wideresnet28_2_exit
        base = wideresnet28_2_exit(num_classes=nc, in_channels=ic)
    elif name == "convmixer_256_8_exit":
        from .models import convmixer_256_8_exit
        base = convmixer_256_8_exit(num_classes=nc, in_channels=ic)
    elif name == "mobilevit_xxs_exit":
        from .models import mobilevit_xxs_exit
        base = mobilevit_xxs_exit(num_classes=nc, in_channels=ic)
    elif name == "bert_early_exit":
        from .models.bert_early_exit import BertEarlyExit
        from .datasets.glue import glue_num_labels
        if not cfg.model.pretrained_name:
            raise ValueError("bert_early_exit requires cfg.model.pretrained_name.")
        if not cfg.model.exit_layers:
            raise ValueError("bert_early_exit requires cfg.model.exit_layers.")
        num_labels = glue_num_labels(cfg.data.task)
        base = BertEarlyExit(
            model_name=cfg.model.pretrained_name,
            num_labels=num_labels,
            exit_layers=list(cfg.model.exit_layers),
            dropout=cfg.model.dropout,
            task=cfg.data.task,
        )
    else:
        raise ValueError(f"Unknown model '{name}'.")
    if cfg.loss.method == "candidate_c":  # SelectiveNet adds per-exit selection/auxiliary heads
        from .models.selective import SelectiveExitModel
        return SelectiveExitModel(base, num_classes=nc)
    return base


def _build_scheduler(optimizer, cfg: ExperimentConfig):
    if cfg.train.schedule == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=max(cfg.train.epochs - cfg.train.warmup_epochs, 1))
    else:
        main = MultiStepLR(optimizer, milestones=cfg.train.milestones, gamma=0.2)
    if cfg.train.warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.train.warmup_epochs)
        return SequentialLR(optimizer, [warmup, main], milestones=[cfg.train.warmup_epochs])
    return main


# Prefixes of submodule names that count as "early-exit auxiliary heads" for the
# curriculum freeze. Backbone (stem, chunks), final classifier (fc), and any other
# non-prefixed module stays trainable in stage 1.
_EARLY_HEAD_PREFIXES = ("exit_head_", "gate_head_", "confidence_head_")


def _is_early_head(name: str) -> bool:
    """True iff ``name`` is a top-level module name of an early-exit auxiliary head."""
    return any(name.startswith(p) for p in _EARLY_HEAD_PREFIXES)


def _curriculum_freeze_early(model: nn.Module) -> None:
    """Curriculum stage 1: freeze every early-exit head (and aux head) on the model.

    The backbone (stem + chunks) and the final classifier head (``self.fc``) stay
    trainable. The frozen heads still produce logits (used for monitoring) but their
    parameters get no gradient. Applies to any SDN-style backbone whose early exits are
    named with the prefixes in ``_EARLY_HEAD_PREFIXES``.
    """
    for name, module in model.named_children():
        if _is_early_head(name):
            for p in module.parameters():
                p.requires_grad_(False)


def _curriculum_unfreeze_all(model: nn.Module) -> None:
    """Curriculum stage 2: reverse the freeze; restore requires_grad=True for everything."""
    for p in model.parameters():
        p.requires_grad_(True)


def train_one_epoch(
    model, loader, criterion, multitask, optimizer, cfg, device,
    limit_batches=None, diagnostics=None, epoch=0, ema=None,
    per_exit_macs=None, poe_state=None, curriculum_stage1=False,
):
    method = cfg.loss.method
    text_mode = cfg.data.name == "glue"
    model.train()
    if multitask is not None:
        multitask.train()
    rng = np.random.default_rng(cfg.seed + epoch)
    total, correct, loss_sum = 0, 0, 0.0
    last_logits = None
    for batch_index, raw_batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        if text_mode:
            # Text batch: dict with input_ids / attention_mask / token_type_ids / labels.
            # No MixUp / CutMix (the discrete-token analog isn't applicable). The forward
            # path depends on the method: SCAR uses forward_with_confidences, JEI-DNN uses
            # forward_with_gates, every other method uses the plain forward (logits_per_exit).
            batch = {k: v.to(device) for k, v in raw_batch.items() if isinstance(v, torch.Tensor)}
            labels = batch["labels"]
            optimizer.zero_grad()
            jei_gate_logits = None
            scar_confidence_logits = None
            if method == "jei_dnn" and hasattr(model, "forward_with_gates"):
                per_exit_logits, jei_gate_logits = model.forward_with_gates(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    token_type_ids=batch.get("token_type_ids"),
                )
            elif method in ("scar", "tri_axis", "scar_distill", "tri_axis_distill", "scar_brier", "tri_axis_brier", "scar_poe", "moe_router") and hasattr(model, "forward_with_confidences"):
                per_exit_logits, scar_confidence_logits = model.forward_with_confidences(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    token_type_ids=batch.get("token_type_ids"),
                )
            else:
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    token_type_ids=batch.get("token_type_ids"),
                )
                per_exit_logits = out["logits_per_exit"]
            primary_labels = labels

            def text_loss_for(target):
                # Curriculum stage 1: train the backbone + final classifier only via
                # plain CE on the final exit. Early heads are frozen (no gradient flows
                # to them) and their losses would not contribute meaningfully anyway.
                if curriculum_stage1:
                    return F.cross_entropy(per_exit_logits[-1], target)
                if method == "jei_dnn":
                    from .methods import jei_dnn_loss
                    components = cfg.loss.components or {}
                    cost_lambda = float(components.get("cost_lambda", 0.1))
                    K = len(per_exit_logits)
                    costs = [float(i + 1) / K for i in range(K)]
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    # some backbones (reslstm_exit) emit a gate for the final exit too;
                    # the routing parameterization uses K-1 gates with a residual final.
                    gate_logits = jei_gate_logits[: len(per_exit_logits) - 1]
                    return jei_dnn_loss(
                        per_exit_logits, gate_logits, target,
                        cost_lambda=cost_lambda, per_exit_costs=costs,
                        epoch_progress=epoch_progress,
                        warmup_frac=float(components.get("warmup_frac", 0.2)),
                    )
                if method == "scar":
                    from .methods import scar_loss
                    components = cfg.loss.components or {}
                    return scar_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        beta=float(components.get("beta", 1.0)),
                        gamma=float(components.get("gamma", 0.5)),
                        T_rank=float(components.get("T_rank", 0.1)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "tri_axis":
                    from .methods import tri_axis_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("tri_axis needs per_exit_macs threaded.")
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return tri_axis_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        per_exit_macs=per_exit_macs, epoch_progress=epoch_progress,
                        beta_rank=float(components.get("beta_rank", 1.0)),
                        gamma_tcp=float(components.get("gamma_tcp", 0.5)),
                        T_rank=float(components.get("T_rank", 0.1)),
                        lambda_mac=float(components.get("lambda_mac", 0.15)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        rho_max=float(components.get("rho_max", 0.3)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "budget_boost":
                    from .methods import budget_boost_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("BudgetBoost needs per_exit_macs threaded.")
                    return budget_boost_loss(
                        per_exit_logits, target, per_exit_macs=per_exit_macs,
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        eta=float(components.get("eta", 0.2)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "poe_anneal":
                    from .methods import poe_anneal_loss
                    if poe_state is None:
                        raise RuntimeError("PoE-Anneal needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_anneal_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "poe_distill":
                    from .methods import poe_distill_loss
                    if poe_state is None:
                        raise RuntimeError("PoE-Distill needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "poe_multitask":
                    from .methods import poe_multitask_loss
                    if poe_state is None or multitask is None:
                        raise RuntimeError("poe_multitask needs poe_state AND multitask threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_multitask_loss(
                        per_exit_logits, target, alphas=poe_state(), multitask=multitask,
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                    )
                if method == "poe_multitask_brier":
                    from .methods import poe_multitask_brier_loss
                    if poe_state is None or multitask is None:
                        raise RuntimeError("poe_multitask_brier needs poe_state AND multitask threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_multitask_brier_loss(
                        per_exit_logits, target, alphas=poe_state(), multitask=multitask,
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
    lambda_ce_final=float(components.get("lambda_ce_final", 0.0)),
                                    )
                if method == "scar_distill":
                    from .methods import scar_distill_loss
                    components = cfg.loss.components or {}
                    return scar_distill_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        beta=float(components.get("beta", 1.0)),
                        gamma_tcp=float(components.get("gamma_tcp", 0.5)),
                        T_rank=float(components.get("T_rank", 0.1)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "budget_boost_distill":
                    from .methods import budget_boost_distill_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("budget_boost_distill needs per_exit_macs threaded.")
                    return budget_boost_distill_loss(
                        per_exit_logits, target, per_exit_macs=per_exit_macs,
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        eta=float(components.get("eta", 0.2)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "budget_boost_distill_brier":
                    from .methods import budget_boost_distill_brier_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("budget_boost_distill_brier needs per_exit_macs threaded.")
                    return budget_boost_distill_brier_loss(
                        per_exit_logits, target, per_exit_macs=per_exit_macs,
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        eta=float(components.get("eta", 0.2)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "budget_boost_distill_asym":
                    from .methods import budget_boost_distill_asym_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("budget_boost_distill_asym needs per_exit_macs threaded.")
                    return budget_boost_distill_asym_loss(
                        per_exit_logits, target, per_exit_macs=per_exit_macs,
                        kappa=float(components.get("kappa", 2.0)),
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        eta=float(components.get("eta", 0.2)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_asym=float(components.get("lambda_asym", 0.5)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "budget_boost_distill_mtl":
                    from .methods import budget_boost_distill_mtl_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None or multitask is None:
                        raise RuntimeError("budget_boost_distill_mtl needs per_exit_macs AND multitask threaded.")
                    return budget_boost_distill_mtl_loss(
                        per_exit_logits, target, multitask=multitask, per_exit_macs=per_exit_macs,
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        eta=float(components.get("eta", 0.2)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "budget_boost_distill_mtl_brier":
                    from .methods import budget_boost_distill_mtl_brier_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None or multitask is None:
                        raise RuntimeError("budget_boost_distill_mtl_brier needs per_exit_macs AND multitask threaded.")
                    return budget_boost_distill_mtl_brier_loss(
                        per_exit_logits, target, multitask=multitask, per_exit_macs=per_exit_macs,
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        eta=float(components.get("eta", 0.2)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "poe_anytime":
                    from .methods import poe_anytime_loss
                    if poe_state is None:
                        raise RuntimeError("poe_anytime needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_anytime_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        tau=float(components.get("tau", 1.0)),
                        eta_cost=float(components.get("eta_cost", 0.5)),
                        alpha_min=float(components.get("alpha_min", 0.05)),
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                    )
                if method == "poe_asym":
                    from .methods import poe_asym_loss
                    if poe_state is None:
                        raise RuntimeError("poe_asym needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_asym_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        kappa=float(components.get("kappa", 2.0)),
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "budget_boost_multitask":
                    from .methods import budget_boost_multitask_loss
                    if multitask is None:
                        raise RuntimeError("budget_boost_multitask needs multitask threaded.")
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("budget_boost_multitask needs per_exit_macs threaded.")
                    return budget_boost_multitask_loss(
                        per_exit_logits, target, per_exit_macs=per_exit_macs, multitask=multitask,
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        eta_floor=float(components.get("eta_floor", 0.2)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                    )
                if method == "tri_axis_distill":
                    from .methods import tri_axis_distill_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("tri_axis_distill needs per_exit_macs threaded.")
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return tri_axis_distill_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        per_exit_macs=per_exit_macs, epoch_progress=epoch_progress,
                        beta_rank=float(components.get("beta_rank", 1.0)),
                        gamma_tcp=float(components.get("gamma_tcp", 0.5)),
                        T_rank=float(components.get("T_rank", 0.1)),
                        lambda_mac=float(components.get("lambda_mac", 0.15)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        rho_max=float(components.get("rho_max", 0.3)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "poe_brier":
                    from .methods import poe_brier_loss
                    if poe_state is None:
                        raise RuntimeError("poe_brier needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_brier_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "scar_brier":
                    from .methods import scar_brier_loss
                    components = cfg.loss.components or {}
                    return scar_brier_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        beta=float(components.get("beta", 1.0)),
                        gamma=float(components.get("gamma", 0.5)),
                        T_rank=float(components.get("T_rank", 0.1)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "tri_axis_brier":
                    from .methods import tri_axis_brier_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("tri_axis_brier needs per_exit_macs threaded.")
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return tri_axis_brier_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        per_exit_macs=per_exit_macs, epoch_progress=epoch_progress,
                        beta_rank=float(components.get("beta_rank", 1.0)),
                        gamma_tcp=float(components.get("gamma_tcp", 0.5)),
                        T_rank=float(components.get("T_rank", 0.1)),
                        lambda_mac=float(components.get("lambda_mac", 0.15)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        rho_max=float(components.get("rho_max", 0.3)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "scar_poe":
                    from .methods import scar_poe_loss
                    if poe_state is None:
                        raise RuntimeError("scar_poe needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return scar_poe_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        alphas=poe_state(), epoch_progress=epoch_progress,
                        beta=float(components.get("beta", 1.0)),
                        gamma_tcp=float(components.get("gamma_tcp", 0.5)),
                        T_rank=float(components.get("T_rank", 0.1)),
                        rho_max=float(components.get("rho_max", 0.3)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "moe_router":
                    from .methods import moe_router_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("moe_router needs per_exit_macs threaded.")
                    return moe_router_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        per_exit_macs=per_exit_macs,
                        lambda_mac=float(components.get("lambda_mac", 0.1)),
                        lambda_ent=float(components.get("lambda_ent", 0.05)),
                        eta_floor=float(components.get("eta_floor", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "poe_jazbec":
                    from .methods import poe_jazbec_loss
                    return poe_jazbec_loss(per_exit_logits, target)
                if method == "poe_distill_mtl":
                    from .methods import poe_distill_mtl_loss
                    if poe_state is None or multitask is None:
                        raise RuntimeError("poe_distill_mtl needs poe_state AND multitask threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_mtl_loss(
                        per_exit_logits, target, alphas=poe_state(), multitask=multitask,
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
    lambda_ce_final=float(components.get("lambda_ce_final", 0.0)),
                                    )
                if method == "poe_distill_brier":
                    from .methods import poe_distill_brier_loss
                    if poe_state is None:
                        raise RuntimeError("poe_distill_brier needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_brier_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
    lambda_ce_final=float(components.get("lambda_ce_final", 0.0)),
                                    )
                if method == "poe_distill_asym":
                    from .methods import poe_distill_asym_loss
                    if poe_state is None:
                        raise RuntimeError("poe_distill_asym needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_asym_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        kappa=float(components.get("kappa", 2.0)),
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "poe_distill_anytime":
                    from .methods import poe_distill_anytime_loss
                    if poe_state is None:
                        raise RuntimeError("poe_distill_anytime needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_anytime_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        tau=float(components.get("tau", 1.0)),
                        eta_cost=float(components.get("eta_cost", 0.5)),
                        alpha_min=float(components.get("alpha_min", 0.05)),
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                    )
                if method == "poe_distill_mtl_brier":
                    from .methods import poe_distill_mtl_brier_loss
                    if poe_state is None or multitask is None:
                        raise RuntimeError("poe_distill_mtl_brier needs poe_state AND multitask threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_mtl_brier_loss(
                        per_exit_logits, target, alphas=poe_state(), multitask=multitask,
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        lambda_ce_final=float(components.get("lambda_ce_final", 0.0)),
                    )
                if method == "distill_mtl_brier":
                    from .methods import distill_mtl_brier_loss
                    if multitask is None:
                        raise RuntimeError("distill_mtl_brier needs multitask threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return distill_mtl_brier_loss(
                        per_exit_logits, target, multitask=multitask,
                        epoch_progress=epoch_progress,
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                    )
                if method == "poe_distill_mtl_asym":
                    from .methods import poe_distill_mtl_asym_loss
                    if poe_state is None or multitask is None:
                        raise RuntimeError("poe_distill_mtl_asym needs poe_state AND multitask threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_mtl_asym_loss(
                        per_exit_logits, target, alphas=poe_state(), multitask=multitask,
                        epoch_progress=epoch_progress,
                        kappa=float(components.get("kappa", 2.0)),
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                    )
                if method == "poe_distill_mtl_brier_asym":
                    from .methods import poe_distill_mtl_brier_asym_loss
                    if poe_state is None or multitask is None:
                        raise RuntimeError("poe_distill_mtl_brier_asym needs poe_state AND multitask threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_mtl_brier_asym_loss(
                        per_exit_logits, target, alphas=poe_state(), multitask=multitask,
                        epoch_progress=epoch_progress,
                        kappa=float(components.get("kappa", 2.0)),
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                    )
                if method == "poe_distill_mtl_brier_mac":
                    from .methods import poe_distill_mtl_brier_mac_loss
                    if poe_state is None or multitask is None:
                        raise RuntimeError("poe_distill_mtl_brier_mac needs poe_state AND multitask threaded.")
                    if per_exit_macs is None:
                        raise RuntimeError("poe_distill_mtl_brier_mac needs per_exit_macs threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_mtl_brier_mac_loss(
                        per_exit_logits, target, alphas=poe_state(), multitask=multitask,
                        per_exit_macs=per_exit_macs,
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                    )
                if method == "asym_select_mac":
                    from .methods import asym_select_mac_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("asym_select_mac needs per_exit_macs threaded.")
                    return asym_select_mac_loss(
                        per_exit_logits, target, per_exit_macs=per_exit_macs,
                        kappa=float(components.get("kappa", 2.0)),
                        tau_cov=float(components.get("tau_cov", 0.8)),
                        nu=float(components.get("nu", 1.0)),
                        lambda_mac=float(components.get("lambda_mac", 0.15)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "anytime_stable_distill_mono":
                    from .methods import anytime_stable_distill_mono_loss
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return anytime_stable_distill_mono_loss(
                        per_exit_logits, target,
                        tau=float(components.get("tau", 1.0)),
                        eta=float(components.get("eta", 0.5)),
                        alpha_min=float(components.get("alpha_min", 0.05)),
                        gamma=float(components.get("gamma", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        epoch_progress=epoch_progress,
                    )
                if method == "jolt":
                    per_exit_losses, _ = variance_scaled_losses(per_exit_logits, target, criterion, use_b=cfg.loss.use_b)
                    return multitask(per_exit_losses)[1] if multitask is not None else torch.stack(per_exit_losses).mean()
                if method == "adaloss":
                    per_exit_losses, _ = variance_scaled_losses(per_exit_logits, target, criterion, use_b=False)
                    return multitask(per_exit_losses)[1] if multitask is not None else torch.stack(per_exit_losses).mean()
                epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                return method_loss(cfg.loss, per_exit_logits, target, epoch_progress=epoch_progress)

            loss = text_loss_for(labels)
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            preds = per_exit_logits[-1].argmax(dim=1)
            correct += (preds == primary_labels).sum().item()
            total += labels.size(0)
            loss_sum += loss.item() * labels.size(0)
            last_logits = per_exit_logits
            continue
        # Vision / HAR / audio path (the original tuple-unpack path).
        images, labels = raw_batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if method == "candidate_c":
            # SelectiveNet path (no MixUp: selective risk needs hard labels).
            logits_list, selection_list, aux_list = model.forward_selective(images)
            comp = cfg.loss.components or {}
            coverage = [float(comp.get("coverage_target", 0.7))] * (model.num_exits + 1)
            loss = selectivenet_loss(
                logits_list, selection_list, aux_list, labels, coverage_targets=coverage,
                lam_coverage=float(comp.get("lam_coverage", 32.0)), alpha=float(comp.get("alpha", 0.5)),
            )
            per_exit_logits, primary_labels = logits_list, labels
        else:
            mixed_images, labels_a, labels_b, lam = mixup_cutmix(
                images, labels, mixup_alpha=cfg.train.mixup_alpha, cutmix_alpha=cfg.train.cutmix_alpha, rng=rng,
            )
            jei_gate_logits = None
            scar_confidence_logits = None
            if method == "jei_dnn" and hasattr(model, "forward_with_gates"):
                per_exit_logits, jei_gate_logits = model.forward_with_gates(mixed_images)
            elif method in ("scar", "tri_axis", "scar_distill", "tri_axis_distill", "scar_brier", "tri_axis_brier", "scar_poe", "moe_router") and hasattr(model, "forward_with_confidences"):
                per_exit_logits, scar_confidence_logits = model.forward_with_confidences(mixed_images)
            else:
                per_exit_logits = forward_all_exits(model, mixed_images)

            def loss_for(target):
                # Curriculum stage 1: backbone + final classifier only; CE on final exit.
                if curriculum_stage1:
                    return F.cross_entropy(per_exit_logits[-1], target)
                if method == "jei_dnn":
                    from .methods import jei_dnn_loss
                    components = cfg.loss.components or {}
                    cost_lambda = float(components.get("cost_lambda", 0.1))
                    K = len(per_exit_logits)
                    costs = [float(i + 1) / K for i in range(K)]
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    # some backbones (reslstm_exit) emit a gate for the final exit too;
                    # the routing parameterization uses K-1 gates with a residual final.
                    gate_logits = jei_gate_logits[: len(per_exit_logits) - 1]
                    return jei_dnn_loss(
                        per_exit_logits, gate_logits, target,
                        cost_lambda=cost_lambda, per_exit_costs=costs,
                        epoch_progress=epoch_progress,
                        warmup_frac=float(components.get("warmup_frac", 0.2)),
                    )
                if method == "scar":
                    from .methods import scar_loss
                    components = cfg.loss.components or {}
                    return scar_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        beta=float(components.get("beta", 1.0)),
                        gamma=float(components.get("gamma", 0.5)),
                        T_rank=float(components.get("T_rank", 0.1)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "tri_axis":
                    from .methods import tri_axis_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError(
                            "tri_axis requires per_exit_macs threaded into train_one_epoch."
                        )
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return tri_axis_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        per_exit_macs=per_exit_macs, epoch_progress=epoch_progress,
                        beta_rank=float(components.get("beta_rank", 1.0)),
                        gamma_tcp=float(components.get("gamma_tcp", 0.5)),
                        T_rank=float(components.get("T_rank", 0.1)),
                        lambda_mac=float(components.get("lambda_mac", 0.15)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        rho_max=float(components.get("rho_max", 0.3)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "budget_boost":
                    from .methods import budget_boost_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError(
                            "BudgetBoost requires per_exit_macs threaded into train_one_epoch; "
                            "got None. Check that run_experiment passes macs through."
                        )
                    return budget_boost_loss(
                        per_exit_logits, target, per_exit_macs=per_exit_macs,
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        eta=float(components.get("eta", 0.2)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "poe_anneal":
                    from .methods import poe_anneal_loss
                    if poe_state is None:
                        raise RuntimeError(
                            "PoE-Anneal requires poe_state threaded into train_one_epoch; got None."
                        )
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_anneal_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "poe_distill":
                    from .methods import poe_distill_loss
                    if poe_state is None:
                        raise RuntimeError(
                            "PoE-Distill requires poe_state threaded into train_one_epoch; got None."
                        )
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "poe_multitask":
                    from .methods import poe_multitask_loss
                    if poe_state is None or multitask is None:
                        raise RuntimeError("poe_multitask needs poe_state AND multitask threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_multitask_loss(
                        per_exit_logits, target, alphas=poe_state(), multitask=multitask,
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                    )
                if method == "poe_multitask_brier":
                    from .methods import poe_multitask_brier_loss
                    if poe_state is None or multitask is None:
                        raise RuntimeError("poe_multitask_brier needs poe_state AND multitask threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_multitask_brier_loss(
                        per_exit_logits, target, alphas=poe_state(), multitask=multitask,
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
    lambda_ce_final=float(components.get("lambda_ce_final", 0.0)),
                                    )
                if method == "scar_distill":
                    from .methods import scar_distill_loss
                    components = cfg.loss.components or {}
                    return scar_distill_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        beta=float(components.get("beta", 1.0)),
                        gamma_tcp=float(components.get("gamma_tcp", 0.5)),
                        T_rank=float(components.get("T_rank", 0.1)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "budget_boost_distill":
                    from .methods import budget_boost_distill_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("budget_boost_distill needs per_exit_macs threaded.")
                    return budget_boost_distill_loss(
                        per_exit_logits, target, per_exit_macs=per_exit_macs,
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        eta=float(components.get("eta", 0.2)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "budget_boost_distill_brier":
                    from .methods import budget_boost_distill_brier_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("budget_boost_distill_brier needs per_exit_macs threaded.")
                    return budget_boost_distill_brier_loss(
                        per_exit_logits, target, per_exit_macs=per_exit_macs,
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        eta=float(components.get("eta", 0.2)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "budget_boost_distill_asym":
                    from .methods import budget_boost_distill_asym_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("budget_boost_distill_asym needs per_exit_macs threaded.")
                    return budget_boost_distill_asym_loss(
                        per_exit_logits, target, per_exit_macs=per_exit_macs,
                        kappa=float(components.get("kappa", 2.0)),
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        eta=float(components.get("eta", 0.2)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_asym=float(components.get("lambda_asym", 0.5)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "budget_boost_distill_mtl":
                    from .methods import budget_boost_distill_mtl_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None or multitask is None:
                        raise RuntimeError("budget_boost_distill_mtl needs per_exit_macs AND multitask threaded.")
                    return budget_boost_distill_mtl_loss(
                        per_exit_logits, target, multitask=multitask, per_exit_macs=per_exit_macs,
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        eta=float(components.get("eta", 0.2)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "budget_boost_distill_mtl_brier":
                    from .methods import budget_boost_distill_mtl_brier_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None or multitask is None:
                        raise RuntimeError("budget_boost_distill_mtl_brier needs per_exit_macs AND multitask threaded.")
                    return budget_boost_distill_mtl_brier_loss(
                        per_exit_logits, target, multitask=multitask, per_exit_macs=per_exit_macs,
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        eta=float(components.get("eta", 0.2)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "poe_anytime":
                    from .methods import poe_anytime_loss
                    if poe_state is None:
                        raise RuntimeError("poe_anytime needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_anytime_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        tau=float(components.get("tau", 1.0)),
                        eta_cost=float(components.get("eta_cost", 0.5)),
                        alpha_min=float(components.get("alpha_min", 0.05)),
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                    )
                if method == "poe_asym":
                    from .methods import poe_asym_loss
                    if poe_state is None:
                        raise RuntimeError("poe_asym needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_asym_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        kappa=float(components.get("kappa", 2.0)),
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "budget_boost_multitask":
                    from .methods import budget_boost_multitask_loss
                    if multitask is None:
                        raise RuntimeError("budget_boost_multitask needs multitask threaded.")
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("budget_boost_multitask needs per_exit_macs threaded.")
                    return budget_boost_multitask_loss(
                        per_exit_logits, target, per_exit_macs=per_exit_macs, multitask=multitask,
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        eta_floor=float(components.get("eta_floor", 0.2)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                    )
                if method == "tri_axis_distill":
                    from .methods import tri_axis_distill_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("tri_axis_distill needs per_exit_macs threaded.")
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return tri_axis_distill_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        per_exit_macs=per_exit_macs, epoch_progress=epoch_progress,
                        beta_rank=float(components.get("beta_rank", 1.0)),
                        gamma_tcp=float(components.get("gamma_tcp", 0.5)),
                        T_rank=float(components.get("T_rank", 0.1)),
                        lambda_mac=float(components.get("lambda_mac", 0.15)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        rho_max=float(components.get("rho_max", 0.3)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "poe_brier":
                    from .methods import poe_brier_loss
                    if poe_state is None:
                        raise RuntimeError("poe_brier needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_brier_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "scar_brier":
                    from .methods import scar_brier_loss
                    components = cfg.loss.components or {}
                    return scar_brier_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        beta=float(components.get("beta", 1.0)),
                        gamma=float(components.get("gamma", 0.5)),
                        T_rank=float(components.get("T_rank", 0.1)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "tri_axis_brier":
                    from .methods import tri_axis_brier_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("tri_axis_brier needs per_exit_macs threaded.")
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return tri_axis_brier_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        per_exit_macs=per_exit_macs, epoch_progress=epoch_progress,
                        beta_rank=float(components.get("beta_rank", 1.0)),
                        gamma_tcp=float(components.get("gamma_tcp", 0.5)),
                        T_rank=float(components.get("T_rank", 0.1)),
                        lambda_mac=float(components.get("lambda_mac", 0.15)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        rho_max=float(components.get("rho_max", 0.3)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "scar_poe":
                    from .methods import scar_poe_loss
                    if poe_state is None:
                        raise RuntimeError("scar_poe needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return scar_poe_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        alphas=poe_state(), epoch_progress=epoch_progress,
                        beta=float(components.get("beta", 1.0)),
                        gamma_tcp=float(components.get("gamma_tcp", 0.5)),
                        T_rank=float(components.get("T_rank", 0.1)),
                        rho_max=float(components.get("rho_max", 0.3)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "moe_router":
                    from .methods import moe_router_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError("moe_router needs per_exit_macs threaded.")
                    return moe_router_loss(
                        per_exit_logits, scar_confidence_logits, target,
                        per_exit_macs=per_exit_macs,
                        lambda_mac=float(components.get("lambda_mac", 0.1)),
                        lambda_ent=float(components.get("lambda_ent", 0.05)),
                        eta_floor=float(components.get("eta_floor", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "poe_jazbec":
                    from .methods import poe_jazbec_loss
                    return poe_jazbec_loss(per_exit_logits, target)
                if method == "poe_distill_mtl":
                    from .methods import poe_distill_mtl_loss
                    if poe_state is None or multitask is None:
                        raise RuntimeError("poe_distill_mtl needs poe_state AND multitask threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_mtl_loss(
                        per_exit_logits, target, alphas=poe_state(), multitask=multitask,
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
    lambda_ce_final=float(components.get("lambda_ce_final", 0.0)),
                                    )
                if method == "poe_distill_brier":
                    from .methods import poe_distill_brier_loss
                    if poe_state is None:
                        raise RuntimeError("poe_distill_brier needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_brier_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
    lambda_ce_final=float(components.get("lambda_ce_final", 0.0)),
                                    )
                if method == "poe_distill_asym":
                    from .methods import poe_distill_asym_loss
                    if poe_state is None:
                        raise RuntimeError("poe_distill_asym needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_asym_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        kappa=float(components.get("kappa", 2.0)),
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "poe_distill_anytime":
                    from .methods import poe_distill_anytime_loss
                    if poe_state is None:
                        raise RuntimeError("poe_distill_anytime needs poe_state threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_anytime_loss(
                        per_exit_logits, target, alphas=poe_state(),
                        epoch_progress=epoch_progress,
                        tau=float(components.get("tau", 1.0)),
                        eta_cost=float(components.get("eta_cost", 0.5)),
                        alpha_min=float(components.get("alpha_min", 0.05)),
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                    )
                if method == "poe_distill_mtl_brier":
                    from .methods import poe_distill_mtl_brier_loss
                    if poe_state is None or multitask is None:
                        raise RuntimeError("poe_distill_mtl_brier needs poe_state AND multitask threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_mtl_brier_loss(
                        per_exit_logits, target, alphas=poe_state(), multitask=multitask,
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        lambda_ce_final=float(components.get("lambda_ce_final", 0.0)),
                    )
                if method == "distill_mtl_brier":
                    from .methods import distill_mtl_brier_loss
                    if multitask is None:
                        raise RuntimeError("distill_mtl_brier needs multitask threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return distill_mtl_brier_loss(
                        per_exit_logits, target, multitask=multitask,
                        epoch_progress=epoch_progress,
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                    )
                if method == "poe_distill_mtl_asym":
                    from .methods import poe_distill_mtl_asym_loss
                    if poe_state is None or multitask is None:
                        raise RuntimeError("poe_distill_mtl_asym needs poe_state AND multitask threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_mtl_asym_loss(
                        per_exit_logits, target, alphas=poe_state(), multitask=multitask,
                        epoch_progress=epoch_progress,
                        kappa=float(components.get("kappa", 2.0)),
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                    )
                if method == "poe_distill_mtl_brier_asym":
                    from .methods import poe_distill_mtl_brier_asym_loss
                    if poe_state is None or multitask is None:
                        raise RuntimeError("poe_distill_mtl_brier_asym needs poe_state AND multitask threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_mtl_brier_asym_loss(
                        per_exit_logits, target, alphas=poe_state(), multitask=multitask,
                        epoch_progress=epoch_progress,
                        kappa=float(components.get("kappa", 2.0)),
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                    )
                if method == "poe_distill_mtl_brier_mac":
                    from .methods import poe_distill_mtl_brier_mac_loss
                    if poe_state is None or multitask is None:
                        raise RuntimeError("poe_distill_mtl_brier_mac needs poe_state AND multitask threaded.")
                    if per_exit_macs is None:
                        raise RuntimeError("poe_distill_mtl_brier_mac needs per_exit_macs threaded.")
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return poe_distill_mtl_brier_mac_loss(
                        per_exit_logits, target, alphas=poe_state(), multitask=multitask,
                        per_exit_macs=per_exit_macs,
                        epoch_progress=epoch_progress,
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        gamma_distill=float(components.get("gamma_distill", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        lambda_brier=float(components.get("lambda_brier", 0.2)),
                        lambda_mac=float(components.get("lambda_mac", 0.3)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                    )
                if method == "asym_select_mac":
                    from .methods import asym_select_mac_loss
                    components = cfg.loss.components or {}
                    if per_exit_macs is None:
                        raise RuntimeError(
                            "asym_select_mac requires per_exit_macs threaded into train_one_epoch; got None."
                        )
                    return asym_select_mac_loss(
                        per_exit_logits, target, per_exit_macs=per_exit_macs,
                        kappa=float(components.get("kappa", 2.0)),
                        tau_cov=float(components.get("tau_cov", 0.8)),
                        nu=float(components.get("nu", 1.0)),
                        lambda_mac=float(components.get("lambda_mac", 0.15)),
                        T_gate=float(components.get("T_gate", 0.05)),
                        tau=float(components.get("tau", 0.5)),
                        weight_schedule=str(components.get("weight_schedule", "increasing")),
                    )
                if method == "anytime_stable_distill_mono":
                    from .methods import anytime_stable_distill_mono_loss
                    components = cfg.loss.components or {}
                    epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                    return anytime_stable_distill_mono_loss(
                        per_exit_logits, target,
                        tau=float(components.get("tau", 1.0)),
                        eta=float(components.get("eta", 0.5)),
                        alpha_min=float(components.get("alpha_min", 0.05)),
                        gamma=float(components.get("gamma", 0.5)),
                        distill_T=float(components.get("distill_T", 1.0)),
                        rho_max=float(components.get("rho_max", 0.5)),
                        anneal_fraction=float(components.get("anneal_fraction", 0.4)),
                        epoch_progress=epoch_progress,
                    )
                if method == "jolt":
                    per_exit_losses, _ = variance_scaled_losses(per_exit_logits, target, criterion, use_b=cfg.loss.use_b)
                    return multitask(per_exit_losses)[1] if multitask is not None else torch.stack(per_exit_losses).mean()
                if method == "adaloss":
                    # Faithful AdaLoss (train_anytime.py): per-exit CE (no variance scaling) fed
                    # to Kendall et al. 2018 MultiTaskLoss with learnable eta. Falls back to mean
                    # CE only if MultiTaskLoss is disabled in config.
                    per_exit_losses, _ = variance_scaled_losses(per_exit_logits, target, criterion, use_b=False)
                    return multitask(per_exit_losses)[1] if multitask is not None else torch.stack(per_exit_losses).mean()
                epoch_progress = float(epoch) / max(int(cfg.train.epochs), 1)
                base = method_loss(cfg.loss, per_exit_logits, target, epoch_progress=epoch_progress)
                # LiteLaplace-EE (C4): add the KL-to-unit-Gaussian regulariser computed by the
                # variational model. The loss function itself stays model-agnostic; the KL
                # passes through here so resnet56_sdn_litelaplace_exit (or any backbone exposing
                # litelaplace_kl_sum) is the only required architectural coupling. Standard
                # ELBO scaling: divide the KL by N_train so the per-batch contribution is the
                # canonical ELBO expectation. Effective penalty per batch is beta * KL_sum /
                # N_train, which makes beta=1e-2 a sensible default (KL ~ 240 k, N_train ~ 50 k
                # gives KL/N_train ~ 5, beta * 5 ~ 0.05 -- the right order of magnitude relative
                # to the per-exit CE).
                if method == "litelaplace" and hasattr(model, "litelaplace_kl_sum"):
                    components = cfg.loss.components or {}
                    beta = float(components.get("beta", 1e-2))
                    n_train = max(len(loader.dataset), 1)
                    base = base + (beta / float(n_train)) * model.litelaplace_kl_sum()
                return base

            loss = mixed_loss(loss_for, labels_a, labels_b, lam)
            primary_labels = labels_a
        loss.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        last_logits = per_exit_logits

        preds = per_exit_logits[-1].argmax(dim=1)
        correct += (preds == primary_labels).sum().item()  # primary-label train accuracy (approx under mixing)
        total += labels.size(0)
        loss_sum += loss.item() * labels.size(0)

    # Record after the last backward so exit-block gradients are still populated.
    if diagnostics is not None and last_logits is not None:
        diagnostics.record_epoch(epoch, model, last_logits)
    return loss_sum / max(total, 1), correct / max(total, 1)


def evaluate_curve(model, val_loader, test_loader, macs, cfg, device, target_accuracies=None):
    targets = target_accuracies if target_accuracies is not None else cfg.eval.target_accuracies
    # SCAR routes on its learned confidence head s_j; PoE-Anneal routes on the cumulative
    # product-of-experts entropy. Auto-override the cutoff_type here so the config file
    # doesn't have to mirror the method choice.
    if cfg.loss.method in ("scar", "tri_axis", "scar_distill", "tri_axis_distill", "scar_brier", "tri_axis_brier"):
        cutoff_type = "learned_confidence"
    elif cfg.loss.method in ("poe_anneal", "poe_distill", "poe_multitask", "poe_anytime", "poe_asym", "poe_brier", "poe_jazbec", "poe_distill_mtl", "poe_distill_brier", "poe_distill_asym", "poe_distill_anytime", "poe_distill_mtl_brier", "poe_distill_mtl_asym", "poe_distill_mtl_brier_asym", "poe_distill_mtl_brier_mac"):
        cutoff_type = "poe_entropy"
    elif cfg.loss.method == "scar_poe":
        cutoff_type = "poe_scar_entropy"  # hybrid: weighted combination of poe_entropy + learned_confidence
    elif cfg.loss.method == "moe_router":
        cutoff_type = "router_argmax"  # joint softmax over confidence heads; argmax selects exit
    else:
        cutoff_type = cfg.eval.cutoff_type

    # GLUE / transformer text path: BertEarlyExit threads (hidden_state, cache) between
    # exits, so the routing loop must subselect both. jolt/text_inference.py owns this
    # text-specific loop; the vision/HAR/audio cells stay on the standard inference path.
    if cfg.data.name == "glue":
        from .text_inference import (
            estimate_text_thresholds, evaluate_text_dynamic_exit, bert_macs_per_exit,
        )
        seq_len = int(getattr(cfg.data, "max_len", 128))
        text_macs = bert_macs_per_exit(model, seq_len)
        results = []
        for target in targets:
            thresholds = estimate_text_thresholds(
                target, model=model, val_loader=val_loader, device=device, cutoff_type=cutoff_type,
            )
            ev = evaluate_text_dynamic_exit(
                model, thresholds, test_loader, device=device, cutoff_type=cutoff_type,
            )
            # Tier 2: also evaluate val_loader at the same thresholds so the OP can be
            # selected by val_emar downstream.
            ev_val = evaluate_text_dynamic_exit(
                model, thresholds, val_loader, device=device, cutoff_type=cutoff_type,
            )
            results.append({
                "target_accuracy": float(target),
                "accuracy": ev.accuracy, "emar": ev.emar, "ece": ev.ece, "nll": ev.nll,
                "brier": ev.brier, "aurc": ev.aurc,
                "val_accuracy": ev_val.accuracy, "val_emar": ev_val.emar,
                "val_aurc": ev_val.aurc,
                "val_exit_counts": ev_val.exit_counts,
                "exit_counts": ev.exit_counts,
                "per_exit_accuracy": [float(x) for x in ev.per_exit_accuracy],
                "per_exit_support": [float(x) for x in ev.per_exit_support],
            })
        return results

    results = []
    for target in targets:
        thresholds = estimate_thresholds_for_accuracy(
            target, model=model, val_loader=val_loader,
            num_layers=model.num_exits, device=device, cutoff_type=cutoff_type,
        )
        ev = evaluate_dynamic_exit(
            model, thresholds, test_loader, macs=macs, num_layers=model.num_exits,
            device=device, cutoff_type=cutoff_type, split_at=cfg.eval.split_at,
        )
        # Tier 2: also evaluate val_loader at the same validation-calibrated thresholds so
        # the OP can be selected by val_emar (not test_emar) downstream. val_acc / val_emar
        # / val exit_counts are stored alongside the test metrics in the same curve row.
        ev_val = evaluate_dynamic_exit(
            model, thresholds, val_loader, macs=macs, num_layers=model.num_exits,
            device=device, cutoff_type=cutoff_type, split_at=cfg.eval.split_at,
        )
        results.append({
            "target_accuracy": float(target),
            "accuracy": ev.accuracy, "emar": ev.emar, "ece": ev.ece, "nll": ev.nll,
            "brier": ev.brier, "aurc": ev.aurc,
            "val_accuracy": ev_val.accuracy, "val_emar": ev_val.emar,
            "val_aurc": ev_val.aurc,
            "val_exit_counts": ev_val.exit_counts,
            "exit_counts": ev.exit_counts,
            "per_exit_accuracy": [float(x) for x in ev.per_exit_accuracy],
            "per_exit_support": [float(x) for x in ev.per_exit_support],
        })
    return results


def run_experiment(
    cfg: ExperimentConfig,
    *,
    device: Optional[torch.device] = None,
    output_dir: str = "outputs",
    limit_train_batches: Optional[int] = None,
    limit_eval_batches: Optional[int] = None,
    target_accuracies: Optional[List[float]] = None,
) -> dict:
    seed_everything(cfg.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = _build_dataloaders(cfg)
    model = _build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()

    multitask = None
    poe_state = None
    param_groups = [{"params": model.parameters()}]
    if cfg.loss.method in ("jolt", "adaloss") and cfg.loss.use_multitask:
        multitask = MultiTaskLoss([1.0] * (model.num_exits + 1)).to(device)
        param_groups.append({"params": multitask.parameters(), "lr": cfg.train.eta_lr, "weight_decay": 0.0})
    # poe_multitask + budget_boost_multitask use MultiTaskLoss regardless of cfg.loss.use_multitask
    # (the mechanism IS the adaptive weighting). Without it these methods reduce to their
    # non-MTL counterparts (poe_anneal / budget_boost).
    if cfg.loss.method in ("poe_multitask", "poe_multitask_brier", "budget_boost_multitask", "poe_distill_mtl", "poe_distill_mtl_brier", "poe_distill_mtl_asym", "poe_distill_mtl_brier_asym", "budget_boost_distill_mtl", "budget_boost_distill_mtl_brier", "poe_distill_mtl_brier_mac", "distill_mtl_brier"):
        multitask = MultiTaskLoss([1.0] * (model.num_exits + 1)).to(device)
        param_groups.append({"params": multitask.parameters(), "lr": cfg.train.eta_lr, "weight_decay": 0.0})
    if cfg.loss.method in ("poe_anneal", "poe_distill", "poe_multitask", "poe_multitask_brier", "poe_anytime", "poe_asym", "poe_brier", "scar_poe", "poe_distill_mtl", "poe_distill_brier", "poe_distill_asym", "poe_distill_anytime", "poe_distill_mtl_brier", "poe_distill_mtl_asym", "poe_distill_mtl_brier_asym", "poe_distill_mtl_brier_mac"):
        from .losses import PoEStateModule
        poe_state = PoEStateModule(num_exits=model.num_exits + 1).to(device)
        # Alphas use the same lr as the eta of MultiTaskLoss (small step, no weight decay).
        param_groups.append({"params": poe_state.parameters(), "lr": cfg.train.eta_lr, "weight_decay": 0.0})
    # GLUE / transformer cells: AdamW (canonical for BERT fine-tuning, and for CCT-7
    # per Hassani et al. WACV 2022). All other cells stay on SGD + momentum + Nesterov
    # per the modern CIFAR / HAR / audio recipe.
    if cfg.data.name == "glue" or cfg.model.name == "cct7_exit":
        optimizer = torch.optim.AdamW(
            param_groups, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.999), eps=1e-8,
        )
    else:
        optimizer = torch.optim.SGD(
            param_groups, lr=cfg.train.lr, momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay, nesterov=cfg.train.nesterov,
        )
    scheduler = _build_scheduler(optimizer, cfg)
    # PoE-Anneal: pre-register the poe_alphas buffer with initial values so EMA's deepcopy
    # captures it. EMA's update() will then EMA-average the buffer as poe_state.alphas
    # changes. At end of training we copy poe_state.alphas into model.poe_alphas in-place.
    # Without this, the post-training register_buffer breaks EMA's swapped_in (missing key).
    if cfg.loss.method in ("poe_anneal", "poe_distill", "poe_multitask", "poe_anytime", "poe_asym", "poe_brier", "scar_poe", "poe_distill_mtl", "poe_distill_brier", "poe_distill_asym", "poe_distill_anytime", "poe_distill_mtl_brier", "poe_distill_mtl_asym", "poe_distill_mtl_brier_asym", "poe_distill_mtl_brier_mac") and poe_state is not None:
        model.register_buffer("poe_alphas", poe_state.alphas.detach().clone(), persistent=True)
    # poe_jazbec: post-hoc PoE baseline (Jazbec et al. NeurIPS 2023). No PoEStateModule;
    # alphas are fixed at uniform 1.0. The summed-CE training is meronen-equivalent; PoE
    # is purely the eval-time transform via the poe_entropy cutoff.
    if cfg.loss.method == "poe_jazbec":
        n_exits = model.num_exits + 1
        model.register_buffer(
            "poe_alphas", torch.ones(n_exits, device=device, dtype=torch.float32), persistent=True,
        )
    ema = ModelEMA(model, decay=cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None

    # MAC profiling: vision/HAR/audio cells need the per-exit cumulative MACs at the
    # model's native input shape. Text cells (BertEarlyExit) thread a (hidden_state, cache)
    # tuple rather than a single tensor, so the standard profiler doesn't apply -- the
    # text_inference path derives a layer-count-based MAC proxy from model.exit_layers.
    if cfg.data.name == "glue":
        from .text_inference import bert_macs_per_exit
        seq_len = int(getattr(cfg.data, "max_len", 128))
        macs = bert_macs_per_exit(model, seq_len)
    else:
        sample_x = next(iter(train_loader))[0]
        macs = per_exit_macs(model, sample_x.shape[1:], device)

    collapse_logger = None
    if cfg.diagnostics.enabled:
        collapse_logger = CollapseLogger(
            num_classes=cfg.data.num_classes,
            log_class_distribution=cfg.diagnostics.log_class_distribution,
            log_gradient_norms=cfg.diagnostics.log_gradient_norms,
            log_exit1_entropy=cfg.diagnostics.log_exit1_entropy,
        )

    # Generic curriculum (vision/HAR/audio): train backbone + final classifier ONLY for
    # the first ``curriculum_stage1_epochs`` epochs, then unfreeze early heads + their
    # auxiliary heads. Helps when the configured method risks shallow collapse on a
    # large backbone (e.g. BudgetBoost / Meronen on WRN-28-10).
    stage1_epochs = int(getattr(cfg.train, "curriculum_stage1_epochs", 0))
    if stage1_epochs > 0:
        _curriculum_freeze_early(model)

    history = []
    for epoch in range(cfg.train.epochs):
        if stage1_epochs > 0 and epoch == stage1_epochs:
            _curriculum_unfreeze_all(model)
        in_stage1 = stage1_epochs > 0 and epoch < stage1_epochs
        loss, acc = train_one_epoch(
            model, train_loader, criterion, multitask, optimizer, cfg, device,
            limit_train_batches, diagnostics=collapse_logger, epoch=epoch, ema=ema,
            per_exit_macs=macs, poe_state=poe_state, curriculum_stage1=in_stage1,
        )
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": loss, "train_acc": acc})
        # Per-component loss diagnostic (NaN detection, dead-component detection).
        # Empty dict if no components were stashed this epoch (e.g. baseline methods).
        from .methods import pop_loss_components
        comps = pop_loss_components()
        comp_str = ""
        if comps.get("n", 0) > 0:
            comp_str = (f"  L_poe={comps['L_poe']:.3f}  L_mono={comps['L_mono']:.3f}"
                        f"  L_distill={comps['L_distill']:.3f}  L_brier={comps['L_brier']:.3f}")
        print(f"  epoch {epoch:>3}  train_loss={loss:.4f}  train_acc={acc:.4f}{comp_str}", flush=True)

    if limit_eval_batches is not None:
        val_eval = list(itertools.islice(val_loader, limit_eval_batches))
        test_eval = list(itertools.islice(test_loader, limit_eval_batches))
    else:
        val_eval, test_eval = val_loader, test_loader

    # PoE-Anneal: register the trained alpha vector onto the model so the eval routing
    # (jolt/calibration.py + jolt/inference.py poe_entropy branches) can recover the
    # cumulative product-of-experts prediction without needing the PoEStateModule itself.
    if cfg.loss.method in ("poe_anneal", "poe_distill", "poe_multitask", "poe_anytime", "poe_asym", "poe_brier", "scar_poe", "poe_distill_mtl", "poe_distill_brier", "poe_distill_asym", "poe_distill_anytime", "poe_distill_mtl_brier", "poe_distill_mtl_asym", "poe_distill_mtl_brier_asym", "poe_distill_mtl_brier_mac") and poe_state is not None:
        model.register_buffer("poe_alphas", poe_state.alphas.detach().clone(), persistent=True)

    # Meronen-Laplace post-hoc step: after MAP training, swap every exit head's last linear
    # for a LaplaceLinear, fit a diagonal-GGN posterior on the training data, and activate
    # the MacKay-probit correction. The downstream evaluate_curve uses the existing entropy
    # cutoff -- under Laplace, epistemic uncertainty broadens softmax and the entropy
    # cutoff defers automatically. EMA (when active) is the source of MAP weights.
    if cfg.loss.method == "meronen_laplace":
        from .posthoc_laplace import (
            convert_last_layers_to_laplace, fit_diagonal_laplace, activate_laplace,
        )
        components = cfg.loss.components or {}
        prior_prec = float(components.get("prior_prec", 1.0))
        max_fit_batches = components.get("max_fit_batches")
        max_fit_batches = int(max_fit_batches) if max_fit_batches is not None else None
        # Use EMA weights as the MAP point if available, then PERMANENTLY adopt them as the
        # model's weights. We can't run the conversion + fit + eval inside ema.swapped_in()
        # because LaplaceLinear adds buffers (posterior_prec_W/b) that the pre-conversion
        # state_dict backup doesn't know about — the context manager's restore would error
        # with missing-key state_dict mismatch.
        if ema is not None:
            ema.copy_into(model)
        convert_last_layers_to_laplace(model, prior_prec=prior_prec)
        fit_diagonal_laplace(model, train_loader, device, max_batches=max_fit_batches)
        activate_laplace(model)
        curve = evaluate_curve(model, val_eval, test_eval, macs, cfg, device, target_accuracies)
    # Evaluate with EMA-smoothed weights when EMA is enabled.
    elif ema is not None:
        with ema.swapped_in(model):
            curve = evaluate_curve(model, val_eval, test_eval, macs, cfg, device, target_accuracies)
    else:
        curve = evaluate_curve(model, val_eval, test_eval, macs, cfg, device, target_accuracies)

    summary = {
        "config": cfg.name,
        "dataset": cfg.data.name,
        "model": cfg.model.name,
        "method": cfg.loss.method,
        "use_b": cfg.loss.use_b,
        "use_multitask": cfg.loss.use_multitask,
        "per_exit_macs": macs,
        "history": history,
        "curve": curve,
    }
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    # Tier 3 infrastructure: save checkpoint so we can re-evaluate later without re-training
    # (e.g. when new OP-selection methods, calibration recipes, or analysis are added).
    try:
        torch.save(model.state_dict(), out_path / "checkpoint.pt")
    except Exception as exc:
        print(f"  warning: checkpoint save failed ({exc}); metrics.json still written")
    if collapse_logger is not None:
        summary["diagnostics"] = collapse_logger.records
        with (out_path / "diagnostics.json").open("w", encoding="utf-8") as handle:
            json.dump(collapse_logger.records, handle, indent=2)
    return summary
