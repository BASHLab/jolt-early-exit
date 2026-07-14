"""Text early-exit dynamic inference for GLUE.

Separate from :mod:`jolt.inference` because a transformer threads ``(hidden_state, cache)``
rather than a single activation tensor: when samples exit early, both the hidden state and
the cached extended attention mask must be subselected. EMAR/ECE/NLL are shared with the
vision path; the compute proxy per exit is the transformer-layer count (e.g. 8/16/24 for
MobileBERT), so incremental "MACs" are the gaps between exit layers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .inference import ExitEvaluation, build_exit_evaluation


def macs_from_exit_layers(exit_layers: Sequence[int]) -> List[float]:
    """Incremental compute proxy per exit: first exit's depth, then gaps between exits.

    Legacy proxy that uses encoder-layer-count as the cost unit. Kept for back-compat;
    new code should call :func:`bert_macs_per_exit` which adds embedding and classifier
    head costs.
    """
    incr = [float(exit_layers[0])]
    for i in range(1, len(exit_layers)):
        incr.append(float(exit_layers[i] - exit_layers[i - 1]))
    return incr


def _bert_layer_macs(hidden: int, intermediate: int, num_heads: int, seq_len: int) -> float:
    """Per-encoder-layer MAC count for a standard BERT-style block.

    Counts:
        QKV linear:    3 * seq_len * hidden * hidden
        attention QK^T: seq_len * seq_len * hidden     (no per-head explosion at MAC level)
        attention softmax * V: seq_len * seq_len * hidden
        attention output projection: seq_len * hidden * hidden
        feed-forward intermediate: seq_len * hidden * intermediate
        feed-forward output:       seq_len * intermediate * hidden
    LayerNorm and residual additions are negligible (linear in hidden, no matrix mult).
    """
    attn = 3 * seq_len * hidden * hidden        # QKV
    attn += 2 * seq_len * seq_len * hidden      # QK^T then * V
    attn += seq_len * hidden * hidden           # attention output proj
    ffn = 2 * seq_len * hidden * intermediate   # intermediate + output
    return float(attn + ffn)


def _bert_embedding_macs(hidden: int, seq_len: int, vocab_size: int) -> float:
    """Embedding lookup is technically gather + add (no matrix multiply); LayerNorm is
    negligible. We charge a small constant approximating the post-embedding LayerNorm and
    position-id lookup. In practice <1% of total encoder cost for BERT-base seq=128, so
    the exact value barely affects MAC-saved% reporting.
    """
    # LayerNorm: ~2 * hidden ops per token (mean/var + scale/shift); approximate as MAC.
    return float(2.0 * seq_len * hidden)


def _bert_head_macs(hidden: int, num_labels: int, seq_len: int = 1) -> float:
    """Per-exit classifier head MAC: pooled [CLS] (seq_len=1 token) projected to num_labels.

    seq_len defaults to 1 because BERT heads operate on the [CLS] token only (sequence
    pooling is just an index lookup, not a matmul).
    """
    return float(seq_len * hidden * num_labels)


def bert_macs_per_exit(model, seq_len: int) -> List[float]:
    """Realistic per-exit incremental MACs for a BertEarlyExit model.

    For exit_layers = [L_0, L_1, ..., L_K] with L_K the deep exit:
        macs[0] = embedding + L_0 encoder layers + head
        macs[i] = (L_i - L_{i-1}) encoder layers + head   for i > 0

    Heads include the classifier; gate and confidence heads are an aux cost only
    incurred for inference with the corresponding cutoff_type. The classifier head cost
    is constant across exits because every exit's head is the same Linear(hidden,
    num_labels). The sum across this list is the total backbone+heads inference cost.
    """
    cfg = model.config
    hidden = cfg.hidden_size
    intermediate = getattr(cfg, "intermediate_size", 4 * hidden)
    num_heads = getattr(cfg, "num_attention_heads", 12)
    vocab_size = getattr(cfg, "vocab_size", 30522)
    num_labels = getattr(model, "num_labels", 2)

    layer_cost = _bert_layer_macs(hidden, intermediate, num_heads, seq_len)
    emb_cost = _bert_embedding_macs(hidden, seq_len, vocab_size)
    head_cost = _bert_head_macs(hidden, num_labels)

    exit_layers = list(model.exit_layers)
    macs: List[float] = []
    prev_depth = 0
    for i, depth in enumerate(exit_layers):
        layers_this_chunk = depth - prev_depth
        chunk = layers_this_chunk * layer_cost + head_cost
        if i == 0:
            chunk += emb_cost
        macs.append(chunk)
        prev_depth = depth
    return macs


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * (probs + 1e-12).log()).sum(dim=-1)


def _move(batch: Dict, device) -> Dict:
    out = {}
    for key in ("input_ids", "attention_mask", "token_type_ids", "labels"):
        if key in batch and batch[key] is not None:
            out[key] = batch[key].to(device)
    return out


def _run_exit(model, *, first, batch, pos, hidden_state, cache):
    return model(
        input_ids=batch["input_ids"] if first else None,
        attention_mask=batch.get("attention_mask") if first else None,
        token_type_ids=batch.get("token_type_ids") if first else None,
        exit_layer_idx=pos,
        hidden_state=hidden_state,
        cache=cache,
    )


_ENTROPY_LIKE = {"entropy", "poe_entropy"}  # ascending sort: low entropy = most confident


def _bert_confidence_head(model: nn.Module, position: int) -> nn.Module:
    """Look up the SCAR confidence head for a given exit position on a BertEarlyExit."""
    layer = model.exit_layers[position]
    heads = getattr(model, "confidence_heads", None)
    if heads is None or str(layer) not in heads:
        raise AttributeError(
            f"Model has no confidence_heads[{layer}]; learned_confidence cutoff requires "
            "per-exit SCAR confidence heads (BertEarlyExit constructs these by default)."
        )
    return heads[str(layer)]


def estimate_text_thresholds(
    target_accuracy: float,
    *,
    model: nn.Module,
    val_loader,
    device: torch.device,
    cutoff_type: str = "entropy",
) -> List[float]:
    """Per-early-exit thresholds that meet ``target_accuracy`` on the validation split.

    Supports four cutoff scores: ``entropy`` (default; softmax entropy), ``confidence``
    (max-softmax), ``learned_confidence`` (SCAR's per-exit Linear-1 head on [CLS]), and
    ``poe_entropy`` (PoE-Anneal cumulative product-of-experts entropy across exits using
    ``model.poe_alphas``).
    """
    if cutoff_type not in _ENTROPY_LIKE and cutoff_type not in ("confidence", "learned_confidence"):
        raise ValueError(f"Unsupported cutoff_type '{cutoff_type}'.")
    if cutoff_type == "poe_entropy":
        poe_alphas = getattr(model, "poe_alphas", None)
        if poe_alphas is None:
            raise AttributeError(
                "Model has no 'poe_alphas' buffer; poe_entropy cutoff requires the "
                "PoE-Anneal training path to register per-exit alphas on the model."
            )
    log_softmax_fn = nn.LogSoftmax(dim=1)
    model.eval()
    early = model.num_exits
    softmax = nn.Softmax(dim=1)
    data: List[list] = [[] for _ in range(early)]

    with torch.no_grad():
        for raw in val_loader:
            batch = _move(raw, device)
            labels = batch["labels"]
            hidden_state, cache = None, None
            running_sum = None  # cumulative log-PoE (poe_entropy only)
            for pos in range(early):
                out = _run_exit(model, first=(pos == 0), batch=batch, pos=pos, hidden_state=hidden_state, cache=cache)
                hidden_state, cache = out["hidden_state"], out["cache"]
                logits = out["logits"]
                if cutoff_type == "entropy":
                    probs = softmax(logits)
                    preds = probs.argmax(dim=-1)
                    metric = _entropy(probs)
                elif cutoff_type == "confidence":
                    probs = softmax(logits)
                    preds = probs.argmax(dim=-1)
                    metric = probs.max(dim=-1).values
                elif cutoff_type == "learned_confidence":
                    probs = softmax(logits)
                    preds = probs.argmax(dim=-1)
                    conf = _bert_confidence_head(model, pos)
                    metric = torch.sigmoid(conf(hidden_state[:, 0]).squeeze(-1))
                else:  # poe_entropy
                    log_p = log_softmax_fn(logits)
                    alpha = poe_alphas[pos]
                    running_sum = alpha * log_p if running_sum is None else running_sum + alpha * log_p
                    log_ptilde = log_softmax_fn(running_sum)
                    ptilde = log_ptilde.exp()
                    preds = log_ptilde.argmax(dim=-1)
                    metric = _entropy(ptilde)
                correct = (preds == labels).float().cpu().numpy()
                data[pos].extend(zip(metric.cpu().numpy().tolist(), correct.tolist()))

    thresholds: List[float] = []
    for pos in range(early):
        if not data[pos]:
            thresholds.append(float("-inf") if cutoff_type in _ENTROPY_LIKE else float("inf"))
            continue
        arr = np.array(data[pos], dtype=np.float32)
        # entropy-like: ascending sort (low entropy = confident); confidence-like: descending
        order = np.argsort(arr[:, 0]) if cutoff_type in _ENTROPY_LIKE else np.argsort(arr[:, 0])[::-1]
        correct_sorted = arr[order, 1]
        cumulative = np.cumsum(correct_sorted) / (np.arange(len(correct_sorted)) + 1)
        valid = np.where(cumulative >= target_accuracy)[0]
        idx = int(valid[-1]) if valid.size > 0 else int(np.argmax(cumulative))
        thresholds.append(float(arr[order, 0][idx]))
    return thresholds


def _subselect_cache(cache: Dict, keep: torch.Tensor) -> Dict:
    return {
        "extended_attention_mask": cache["extended_attention_mask"][keep],
        "head_mask": cache["head_mask"],
        "current_depth": cache["current_depth"],
    }


def evaluate_text_dynamic_exit(
    model: nn.Module,
    thresholds: Sequence[float],
    loader,
    *,
    device: torch.device,
    cutoff_type: str = "entropy",
) -> ExitEvaluation:
    """Dynamic early-exit evaluation for text, returning accuracy/EMAR/ECE/NLL.

    Supports the same four cutoff_type values as estimate_text_thresholds: entropy (default),
    confidence (max-softmax), learned_confidence (SCAR), poe_entropy (PoE-Anneal cumulative
    prediction).
    """
    if cutoff_type == "poe_entropy":
        poe_alphas = getattr(model, "poe_alphas", None)
        if poe_alphas is None:
            raise AttributeError("Model has no 'poe_alphas' buffer for poe_entropy cutoff.")
    model.eval()
    num_pos = len(model.exit_layers)
    early = model.num_exits
    # Use the realistic per-exit MAC count (embedding + per-layer + classifier head)
    # at the actual sequence length of the batch. Falls back to layer-count proxy if
    # the loader is empty (no batches) or the model lacks a config.
    seq_len_guess = 128
    try:
        first = next(iter(loader))
        if isinstance(first, dict) or (hasattr(first, "keys") and "input_ids" in first):
            seq_len_guess = int(first["input_ids"].shape[1])
    except Exception:
        pass
    macs = bert_macs_per_exit(model, seq_len_guess) if hasattr(model, "config") else macs_from_exit_layers(model.exit_layers)
    softmax = nn.Softmax(dim=1)
    log_softmax_fn = nn.LogSoftmax(dim=1)

    pred_y: List[list] = [[] for _ in range(num_pos)]
    true_y: List[list] = [[] for _ in range(num_pos)]
    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    total = 0

    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            remaining_labels = batch["labels"]
            hidden_state, cache = None, None
            poe_running_sum = None  # for poe_entropy cutoff
            for pos in range(num_pos):
                out = _run_exit(model, first=(pos == 0), batch=batch, pos=pos, hidden_state=hidden_state, cache=cache)
                hidden_state, cache = out["hidden_state"], out["cache"]
                logits = out["logits"]
                if cutoff_type == "poe_entropy":
                    log_p = log_softmax_fn(logits)
                    alpha = poe_alphas[pos]
                    poe_running_sum = alpha * log_p if poe_running_sum is None else poe_running_sum + alpha * log_p
                    log_ptilde = log_softmax_fn(poe_running_sum)
                    probs = log_ptilde.exp()
                    logits_for_pred = log_ptilde
                else:
                    probs = softmax(logits)
                    logits_for_pred = logits
                if pos < early:
                    if cutoff_type == "entropy":
                        exit_mask = _entropy(probs) < thresholds[pos]
                    elif cutoff_type == "confidence":
                        exit_mask = probs.max(dim=-1).values > thresholds[pos]
                    elif cutoff_type == "learned_confidence":
                        conf = _bert_confidence_head(model, pos)
                        s = torch.sigmoid(conf(hidden_state[:, 0]).squeeze(-1))
                        exit_mask = s > thresholds[pos]
                    else:  # poe_entropy
                        exit_mask = _entropy(probs) < thresholds[pos]
                else:
                    exit_mask = torch.ones(remaining_labels.size(0), dtype=torch.bool, device=device)

                accepted_labels = remaining_labels[exit_mask]
                if accepted_labels.numel() > 0:
                    pred_y[pos].extend(logits_for_pred[exit_mask].argmax(dim=-1).cpu().numpy().tolist())
                    true_y[pos].extend(accepted_labels.cpu().numpy().tolist())
                    probs_all.append(probs[exit_mask].cpu().numpy())
                    labels_all.append(accepted_labels.cpu().numpy())
                    total += accepted_labels.size(0)

                keep = ~exit_mask
                remaining_labels = remaining_labels[keep]
                if remaining_labels.numel() == 0:
                    break
                hidden_state = hidden_state[keep]
                cache = _subselect_cache(cache, keep)
                if poe_running_sum is not None:
                    poe_running_sum = poe_running_sum[keep]

    if total == 0:
        return ExitEvaluation(0.0, 0.0, 0.0, 0.0, 0.0)
    return build_exit_evaluation(
        num_pos, pred_y, true_y, macs, total, probs_all, labels_all,
    )
