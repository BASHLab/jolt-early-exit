"""GLUE training + evaluation for the multi-exit transformer (MobileBERT).

Mirrors the vision runner but for text: AdamW with linear warmup, the two-switch loss
(logit-variance scaling via ``use_b`` and adaptive weighting via MultiTaskLoss), and the
text dynamic-exit evaluation. Network-gated; ``transformers``/``datasets`` are imported
lazily. Not runnable in an offline environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import ExperimentConfig, seed_everything
from .datasets.glue import build_tokenizer, glue_dataloaders, glue_num_labels
from .glue_metrics import glue_primary_metric, is_regression
from .losses import MultiTaskLoss
from .methods import method_loss
from .text_inference import estimate_text_thresholds, evaluate_text_dynamic_exit


def text_cascading_loss(logits_per_exit, labels, criterion, *, use_b: bool, regression: bool):
    """Per-exit loss with optional running logit-variance scaling (text variant of use_b)."""
    total_b = None
    per_exit: List[torch.Tensor] = []
    total = logits_per_exit[0].new_zeros(())
    for i, logits in enumerate(logits_per_exit):
        if regression:
            loss = criterion(logits.squeeze(-1), labels.float())
            scaled = loss
        else:
            loss = criterion(logits, labels)
            b_layer = torch.var(logits.detach(), dim=1).mean()
            total_b = b_layer if total_b is None else total_b + b_layer
            true_b = total_b / (i + 1)
            scaled = loss / (true_b if use_b else logits.new_ones(()))
        per_exit.append(scaled)
        total = total + scaled
    return total / len(logits_per_exit), per_exit


def _model_inputs(batch, device):
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device) if "attention_mask" in batch else None,
        "token_type_ids": batch["token_type_ids"].to(device) if "token_type_ids" in batch else None,
    }


def _train_phase(*, model, train_loader, param_groups, loss_fn, cfg, device, limit, schedule_fn, tag):
    """One AdamW phase: linear warmup+decay, gradient clipping, optional bf16 autocast.

    MobileBERT was pretrained in bf16 and produces NaNs under fp16 AMP, so fp16 is never used;
    bf16 is opt-in via ``train.bf16`` and only on CUDA.
    """
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps_per_epoch = limit or len(train_loader)
    total_steps = max(cfg.train.epochs * steps_per_epoch, 1)
    scheduler = schedule_fn(optimizer, int(cfg.train.warmup_ratio * total_steps), total_steps)
    use_bf16 = cfg.train.bf16 and device.type == "cuda"
    clip_params = [p for group in param_groups for p in group["params"]]

    model.train()
    history: List[dict] = []
    for epoch in range(cfg.train.epochs):
        running, seen = 0.0, 0
        for batch_index, batch in enumerate(train_loader):
            if limit is not None and batch_index >= limit:
                break
            optimizer.zero_grad()
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = loss_fn(batch)
            else:
                loss = loss_fn(batch)
            loss.backward()
            if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(clip_params, cfg.train.grad_clip)
            optimizer.step()
            scheduler.step()
            batch_size = batch["labels"].size(0)
            running += loss.item() * batch_size
            seen += batch_size
        history.append({"phase": tag, "epoch": epoch, "train_loss": running / max(seen, 1)})
    return history


def final_exit_glue_metric(model, loader, task: str, device) -> dict:
    """No-early-exit baseline: run every sample to the final exit and score with the task's
    official GLUE metric. This is the comparable "MobileBERT GLUE score" for the task."""
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(**_model_inputs(batch, device))["logits_per_exit"][-1]
            labels = batch["labels"].to(device)
            preds = logits.squeeze(-1) if model.regression else logits.argmax(dim=-1)
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    value, name = glue_primary_metric(task, preds, labels)
    return {"glue_metric": value, "glue_metric_name": name}


def _alt_phase(*, model, train_loader, loss_fn, head_params, cfg, device, limit, schedule_fn):
    """BERxiT ALT fine-tune: alternate per step between updating all params and only the exit
    heads (off-ramps). Stabilizes multi-exit transformer fine-tuning (Xin et al. 2021)."""
    all_params = [p for p in model.parameters() if p.requires_grad]
    opt_all = torch.optim.AdamW(all_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    opt_heads = torch.optim.AdamW(head_params, lr=cfg.train.lr, weight_decay=0.0)
    steps_per_epoch = limit or len(train_loader)
    total_steps = max(cfg.train.epochs * steps_per_epoch, 1)
    warmup = int(cfg.train.warmup_ratio * total_steps)
    sched_all = schedule_fn(opt_all, warmup, total_steps)
    sched_heads = schedule_fn(opt_heads, warmup, total_steps)

    model.train()
    history: List[dict] = []
    step = 0
    for epoch in range(cfg.train.epochs):
        running, seen = 0.0, 0
        for batch_index, batch in enumerate(train_loader):
            if limit is not None and batch_index >= limit:
                break
            model.zero_grad()
            loss = loss_fn(batch)
            loss.backward()
            if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(all_params, cfg.train.grad_clip)
            if step % 2 == 0:                 # even: update everything
                opt_all.step(); sched_all.step()
            else:                             # odd: update only the off-ramps (backbone frozen in effect)
                opt_heads.step(); sched_heads.step()
            step += 1
            batch_size = batch["labels"].size(0)
            running += loss.item() * batch_size
            seen += batch_size
        history.append({"phase": "alt", "epoch": epoch, "train_loss": running / max(seen, 1)})
    return history


def run_glue(
    cfg: ExperimentConfig,
    *,
    device: Optional[torch.device] = None,
    output_dir: str = "outputs",
    limit_train_batches: Optional[int] = None,
    target_accuracies: Optional[List[float]] = None,
) -> dict:
    from transformers import get_linear_schedule_with_warmup

    from .models.bert_early_exit import BertEarlyExit, freeze_for_stage1, freeze_for_stage2

    seed_everything(cfg.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = cfg.data.task
    pretrained = cfg.model.pretrained_name
    exit_layers = cfg.model.exit_layers
    if not (task and pretrained and exit_layers):
        raise ValueError("GLUE config needs data.task, model.pretrained_name, and model.exit_layers.")
    num_labels = glue_num_labels(task)

    tokenizer = build_tokenizer(pretrained)
    model = BertEarlyExit(pretrained, num_labels, exit_layers, dropout=cfg.model.dropout, task=task).to(device)
    train_loader, calib_loader, eval_loader = glue_dataloaders(
        task=task, tokenizer=tokenizer, max_len=cfg.data.max_len,
        batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, seed=cfg.seed,
    )

    criterion = nn.MSELoss() if model.regression else nn.CrossEntropyLoss()

    def forward_logits(batch):
        return model(**_model_inputs(batch, device))["logits_per_exit"]

    history: List[dict] = []
    if cfg.train.alt_finetune:
        # BERxiT ALT: alternate updating all params vs only the exit heads. Uses the method loss
        # without MultiTaskLoss (ALT is a stabilizer for the joint objective).
        def alt_loss(batch):
            logits_per_exit = forward_logits(batch)
            labels = batch["labels"].to(device)
            if cfg.loss.method == "jolt":
                convex, _ = text_cascading_loss(
                    logits_per_exit, labels, criterion, use_b=cfg.loss.use_b, regression=model.regression
                )
                return convex
            return method_loss(cfg.loss, logits_per_exit, labels, regression=model.regression)

        head_params = [p for head in model.heads.values() for p in head.parameters()]
        history += _alt_phase(
            model=model, train_loader=train_loader, loss_fn=alt_loss, head_params=head_params,
            cfg=cfg, device=device, limit=limit_train_batches, schedule_fn=get_linear_schedule_with_warmup,
        )
    elif cfg.train.two_stage:
        # DeeBERT-style two-stage fine-tune: stabilizes multi-exit training and overrides the
        # method (intermediate exits get plain summed CE after the backbone is frozen).
        freeze_for_stage1(model)

        def stage1_loss(batch):
            final = forward_logits(batch)[-1]
            labels = batch["labels"].to(device)
            return criterion(final.squeeze(-1), labels.float()) if model.regression else criterion(final, labels)

        history += _train_phase(
            model=model, train_loader=train_loader,
            param_groups=[{"params": [p for p in model.parameters() if p.requires_grad]}],
            loss_fn=stage1_loss, cfg=cfg, device=device, limit=limit_train_batches,
            schedule_fn=get_linear_schedule_with_warmup, tag="stage1",
        )

        freeze_for_stage2(model)

        def stage2_loss(batch):
            early = forward_logits(batch)[:-1]
            labels = batch["labels"].to(device)
            if model.regression:
                return torch.stack([criterion(l.squeeze(-1), labels.float()) for l in early]).sum()
            return torch.stack([criterion(l, labels) for l in early]).sum()

        history += _train_phase(
            model=model, train_loader=train_loader,
            param_groups=[{"params": [p for p in model.parameters() if p.requires_grad]}],
            loss_fn=stage2_loss, cfg=cfg, device=device, limit=limit_train_batches,
            schedule_fn=get_linear_schedule_with_warmup, tag="stage2",
        )
    else:
        multitask = None
        param_groups = [{"params": list(model.parameters())}]
        if cfg.loss.method == "jolt" and cfg.loss.use_multitask and not model.regression:
            multitask = MultiTaskLoss([1.0] * len(exit_layers)).to(device)
            param_groups.append({"params": list(multitask.parameters()), "lr": cfg.train.eta_lr, "weight_decay": 0.0})

        def joint_loss(batch):
            logits_per_exit = forward_logits(batch)
            labels = batch["labels"].to(device)
            if cfg.loss.method == "jolt":
                convex, per_exit = text_cascading_loss(
                    logits_per_exit, labels, criterion, use_b=cfg.loss.use_b, regression=model.regression
                )
                return multitask(per_exit)[1] if multitask is not None else convex
            return method_loss(cfg.loss, logits_per_exit, labels, regression=model.regression)

        history += _train_phase(
            model=model, train_loader=train_loader, param_groups=param_groups,
            loss_fn=joint_loss, cfg=cfg, device=device, limit=limit_train_batches,
            schedule_fn=get_linear_schedule_with_warmup, tag="joint",
        )

    # Baseline GLUE metric (final exit, official per-task metric). Always well-defined.
    baseline = final_exit_glue_metric(model, eval_loader, task, device)

    # Dynamic early-exit curve uses confidence/entropy routing, which is only defined for
    # classification. STS-B is regression, so it has no early-exit curve.
    curve = []
    if not is_regression(task):
        targets = target_accuracies if target_accuracies is not None else cfg.eval.target_accuracies
        for target in targets:
            thresholds = estimate_text_thresholds(
                target, model=model, val_loader=calib_loader, device=device, cutoff_type=cfg.eval.cutoff_type
            )
            ev = evaluate_text_dynamic_exit(
                model, thresholds, eval_loader, device=device,
                cutoff_type=cfg.eval.cutoff_type,
            )
            curve.append({
                "target_accuracy": float(target),
                "accuracy": ev.accuracy, "emar": ev.emar, "ece": ev.ece, "nll": ev.nll,
                "brier": ev.brier, "aurc": ev.aurc,
                "exit_counts": ev.exit_counts,
            })

    summary = {
        "config": cfg.name, "task": task, "model": pretrained, "exit_layers": exit_layers,
        "method": cfg.loss.method,
        "two_stage": cfg.train.two_stage,
        "alt_finetune": cfg.train.alt_finetune,
        "use_b": cfg.loss.use_b, "use_multitask": cfg.loss.use_multitask,
        "regression": model.regression,
        "baseline_glue_metric": baseline["glue_metric"],
        "glue_metric_name": baseline["glue_metric_name"],
        "history": history, "curve": curve,
    }
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
