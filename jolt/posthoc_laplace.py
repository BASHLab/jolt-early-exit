"""Post-hoc last-layer Laplace approximation per Meronen et al. NeurIPS 2024
("Fixing overconfidence in dynamic neural networks").

After MAP training of an SDN-style multi-exit network with summed cross-entropy across
exits (the canonical "meronen" / anytime training recipe), fit a diagonal generalised
Gauss-Newton Laplace posterior on the last-layer weights of each exit head. At eval, the
posterior predictive is approximated by MacKay's probit scaling:

    softmax(z / sqrt(1 + pi/8 * Var(z)))

where Var(z) is the diagonal predictive variance arising from the diagonal Laplace
posterior on (W, b). The scaled softmax has higher entropy on epistemically uncertain
samples, which the EE routing policy reads as "defer to a deeper exit." This is the
Meronen mechanism: epistemic-uncertainty-aware routing without changing the training loss.

The implementation lives outside the model definitions so it composes with any backbone
whose exit heads' final classifier is a plain ``nn.Linear`` (which is true for all SDN
backbones in ``jolt/models/``). The conversion walks ``named_modules`` and replaces every
module ending in ``.fc`` with a ``LaplaceLinear`` that carries the diagonal posterior.

Fitting uses an Empirical-Bayes-style single pass through the data: the contribution to
the diagonal GGN at exit head e is sum_i p_{i,e}(1 - p_{i,e}) * x_{i,e}^2, where
p_{i,e} is the predicted probability vector at exit e for sample i and x_{i,e} is the
input feature to the head's last layer. Prior precision is added before evaluation.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn


_PI_OVER_8 = math.pi / 8.0


class LaplaceLinear(nn.Linear):
    """``nn.Linear`` with an attached diagonal Laplace posterior on (W, b).

    When ``laplace_active`` is True and the posterior has been fit, forward returns
    MacKay-probit-scaled logits. Otherwise, behaves exactly like ``nn.Linear``.

    The posterior precision buffers default to ``prior_prec`` so unfit modules behave
    as if the prior is the only source of curvature (large prior_prec = tight posterior;
    small = loose posterior, more shrinkage of confidence).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 prior_prec: float = 1.0):
        super().__init__(in_features, out_features, bias=bias)
        self.prior_prec = float(prior_prec)
        self.register_buffer(
            "posterior_prec_W", torch.full_like(self.weight, self.prior_prec)
        )
        if bias:
            self.register_buffer(
                "posterior_prec_b", torch.full_like(self.bias, self.prior_prec)
            )
        else:
            self.register_buffer("posterior_prec_b", None)
        self.laplace_active: bool = False
        self.laplace_fitted: bool = False

    @classmethod
    def from_linear(cls, linear: nn.Linear, prior_prec: float = 1.0) -> "LaplaceLinear":
        """Create a LaplaceLinear that copies (W, b) from an existing nn.Linear."""
        has_bias = linear.bias is not None
        new = cls(linear.in_features, linear.out_features, bias=has_bias, prior_prec=prior_prec)
        with torch.no_grad():
            new.weight.copy_(linear.weight)
            if has_bias:
                new.bias.copy_(linear.bias)
        new.to(linear.weight.device)
        return new

    def reset_posterior(self) -> None:
        """Reset the diagonal posterior precision to the prior. Call before re-fitting."""
        with torch.no_grad():
            self.posterior_prec_W.fill_(self.prior_prec)
            if self.posterior_prec_b is not None:
                self.posterior_prec_b.fill_(self.prior_prec)
        self.laplace_fitted = False

    def add_contributions(self, features: torch.Tensor, probs: torch.Tensor) -> None:
        """Accumulate one batch's contribution to the diagonal GGN Hessian.

        For softmax CE at a Linear(D, C), the per-sample diagonal Hessian w.r.t. W_{c,d}
        is p_c (1 - p_c) * x_d^2. Summed over the batch and added to the running buffer.
        """
        with torch.no_grad():
            pv = probs * (1.0 - probs)  # [B, C]
            x2 = features.pow(2)  # [B, D]
            self.posterior_prec_W.add_(pv.t() @ x2)
            if self.posterior_prec_b is not None:
                self.posterior_prec_b.add_(pv.sum(dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = nn.functional.linear(x, self.weight, self.bias)
        if not (self.laplace_active and self.laplace_fitted):
            return logits
        var_w_contrib = x.pow(2) @ (1.0 / self.posterior_prec_W).t()  # [B, C]
        if self.posterior_prec_b is not None:
            var_b_contrib = 1.0 / self.posterior_prec_b
        else:
            var_b_contrib = 0.0
        var_z = var_w_contrib + var_b_contrib
        scale = (1.0 + _PI_OVER_8 * var_z).rsqrt()
        return logits * scale


def convert_last_layers_to_laplace(model: nn.Module, prior_prec: float = 1.0) -> List[LaplaceLinear]:
    """Replace every classifier-shaped Linear submodule with a ``LaplaceLinear``.

    Recognised classifier-head patterns:
      * ``fc`` or any ``...fc`` (SDN / 1D-CNN / BC-ResNet / CCT-7 SeqPool ``.fc`` heads)
      * ``heads.<layer>.<idx>`` (BertEarlyExit per-exit ``Sequential(Dropout, Linear)``)
      * ``conv2`` 1x1 Conv2d classifier (MobileNetV2 ``_ExitBlock`` final head). The 1x1
        Conv2d on a (B, C, 1, 1) feature map after AdaptiveAvgPool2d is mathematically
        identical to a Linear(C, num_classes); we swap in a Linear-shaped LaplaceLinear and
        replace the conv with an in-place ``LinearAsConv`` wrapper so the surrounding
        forward (with view(B, -1) at the exit) remains correct.

    Excludes auxiliary heads (gates, confidences) by filtering on ``out_features >= 2``.
    """
    converted: List[LaplaceLinear] = []
    named = dict(model.named_modules())
    targets: List[Tuple[nn.Module, str, str]] = []  # (parent, child_name, kind)
    for name, module in list(named.items()):
        # Linear-shaped classifiers
        if isinstance(module, nn.Linear) and not isinstance(module, LaplaceLinear):
            is_fc = name == "fc" or name.endswith(".fc")
            is_bert_head = name.startswith("heads.") and name.count(".") >= 2
            if not (is_fc or is_bert_head):
                continue
            if module.out_features < 2:
                continue
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = named[parent_name]
            else:
                parent = model
                child_name = name
            targets.append((parent, child_name, "linear"))
            continue
        # MobileNetV2-style 1x1 Conv2d classifier: ``conv2`` at the end of an ``_ExitBlock``.
        # Operates on (B, C, 1, 1) after the explicit AdaptiveAvgPool2d in forward().
        if isinstance(module, nn.Conv2d) and name.endswith(".conv2"):
            if module.kernel_size != (1, 1) or module.out_channels < 2:
                continue
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = named[parent_name]
            else:
                parent = model
                child_name = name
            targets.append((parent, child_name, "conv2_1x1"))
    for parent, child_name, kind in targets:
        original = getattr(parent, child_name)
        if kind == "linear":
            new = LaplaceLinear.from_linear(original, prior_prec=prior_prec)
        else:  # conv2_1x1
            new = _LaplaceConv2dAs1x1.from_conv2d(original, prior_prec=prior_prec)
        setattr(parent, child_name, new)
        converted.append(new)
    return converted


class _LaplaceConv2dAs1x1(LaplaceLinear):
    """LaplaceLinear that exposes a Conv2d-1x1 surface for MN2's ``conv2`` classifier.

    Inputs arrive as (B, C, 1, 1) post-AdaptiveAvgPool2d; we flatten to (B, C), run the
    LaplaceLinear, then reshape back to (B, num_classes, 1, 1) so the surrounding
    forward()'s ``y.view(y.size(0), -1)`` continues to work unchanged.

    Hooks registered on this module fire with the (B, C) flattened input and (B, K) logits,
    matching what ``fit_diagonal_laplace`` expects.
    """

    def forward(self, x):  # type: ignore[override]
        # x: (B, C, 1, 1)
        b, c, h, w = x.shape
        flat = x.view(b, c)
        out = super().forward(flat)  # (B, K)
        return out.view(b, -1, 1, 1)

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, prior_prec: float = 1.0):
        out_features = conv.out_channels
        in_features = conv.in_channels
        new = cls(in_features, out_features, bias=conv.bias is not None, prior_prec=prior_prec)
        with torch.no_grad():
            new.weight.copy_(conv.weight.view(out_features, in_features))
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
        new.to(conv.weight.device)
        return new


def fit_diagonal_laplace(model: nn.Module, loader, device, max_batches: int = None) -> None:
    """Single-pass diagonal-GGN fit on every ``LaplaceLinear`` in ``model``.

    Runs ``forward_all_exits`` for each batch, registers forward hooks on each LaplaceLinear
    to capture its input features and output logits, computes softmax probs, and calls
    ``add_contributions(features, probs)``. ``max_batches`` caps the fit; ``None`` uses the
    full loader.
    """
    laplace_modules = [m for m in model.modules() if isinstance(m, LaplaceLinear)]
    if not laplace_modules:
        raise RuntimeError(
            "fit_diagonal_laplace called on a model with no LaplaceLinear; "
            "call convert_last_layers_to_laplace(model) first."
        )
    for m in laplace_modules:
        m.reset_posterior()

    captures: dict = {id(m): {"features": None, "logits": None} for m in laplace_modules}

    def make_pre_hook(m):
        def pre_hook(module, inputs):
            captures[id(m)]["features"] = inputs[0].detach()
        return pre_hook

    def make_hook(m):
        def hook(module, inputs, output):
            captures[id(m)]["logits"] = output.detach()
        return hook

    handles = []
    for m in laplace_modules:
        handles.append(m.register_forward_pre_hook(make_pre_hook(m)))
        handles.append(m.register_forward_hook(make_hook(m)))

    from .losses import forward_all_exits  # local import to avoid cycle at module load

    def _forward_batch(batch):
        """Dispatch the per-batch forward across vision tuples and dict-shaped GLUE batches.

        Vision/HAR/audio loaders yield ``(images, labels)``; the standard forward_all_exits
        helper consumes the image tensor. GLUE/SST-2 loaders yield ``BatchEncoding`` (HF
        tokenizer output) which is dict-LIKE but doesn't satisfy ``isinstance(dict)``. We
        detect text batches by probing for known keys (``input_ids``). BertEarlyExit needs
        keyword inputs threaded through ``exit_layer_idx``. We don't need the logits return
        value here (hooks capture features/logits at each LaplaceLinear); we just need the
        model to traverse every exit so each LaplaceLinear fires once per sample.
        """
        # BatchEncoding from HF supports ``in`` checks and key access but is not dict-typed.
        is_text_batch = (hasattr(batch, "keys") and "input_ids" in batch) or isinstance(batch, dict)
        if is_text_batch:
            batch_dev = {k: batch[k].to(device) for k in batch.keys() if hasattr(batch[k], "to")}
            # BertEarlyExit threads (hidden_state, cache) across exits; reuse the
            # text_inference loop.
            from .text_inference import _run_exit
            hidden_state, cache = None, None
            num_exits = int(getattr(model, "num_exits", 0))
            for pos in range(num_exits + 1):
                out = _run_exit(
                    model, first=(pos == 0), batch=batch_dev, pos=pos,
                    hidden_state=hidden_state, cache=cache,
                )
                hidden_state = out.get("hidden_state")
                cache = out.get("cache")
            return
        # Vision/HAR/audio: (images, labels)
        images = batch[0].to(device)
        _ = forward_all_exits(model, images)

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            _forward_batch(batch)
            for m in laplace_modules:
                cap = captures[id(m)]
                features = cap["features"]
                logits = cap["logits"]
                if features is None or logits is None:
                    continue
                # _LaplaceConv2dAs1x1 case: hooks fire on the wrapper module and capture
                # the 4D (B, C, 1, 1) pre-forward input and 4D (B, K, 1, 1) post-forward
                # output. add_contributions expects 2D tensors; flatten the singleton
                # spatial dims before passing through.
                if features.dim() == 4 and features.shape[-2:] == (1, 1):
                    features = features.view(features.size(0), -1)
                if logits.dim() == 4 and logits.shape[-2:] == (1, 1):
                    logits = logits.view(logits.size(0), -1)
                probs = torch.softmax(logits, dim=-1)
                m.add_contributions(features, probs)
                cap["features"] = None
                cap["logits"] = None

    for h in handles:
        h.remove()
    for m in laplace_modules:
        m.laplace_fitted = True


def activate_laplace(model: nn.Module) -> None:
    """Turn on Laplace correction at every LaplaceLinear in ``model``."""
    for m in model.modules():
        if isinstance(m, LaplaceLinear):
            m.laplace_active = True


def deactivate_laplace(model: nn.Module) -> None:
    """Turn off Laplace correction (return to MAP logits)."""
    for m in model.modules():
        if isinstance(m, LaplaceLinear):
            m.laplace_active = False
