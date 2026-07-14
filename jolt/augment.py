"""MixUp / CutMix batch transforms and the mixed-label loss helper.

MixUp (Zhang et al. 2018) and CutMix (Yun et al. 2019) mix two examples and their labels. The
multi-exit loss is then a convex combination of the loss against each label, computed by
:func:`mixed_loss`. With mixing on, the calibration regularizers and the monotonicity hinge are
applied per label set inside ``composite_loss`` (each is computed twice and combined).

# RESEARCH GAP: the exact composition of MixUp/CutMix with calibration regularizers (which lack a
# single hard correctness signal on mixed labels) is unsettled; here we combine the full loss for
# each label by lam. Disabling calibration on mixed batches is an alternative to evaluate.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch


def mixup_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return ``(mixed_images, labels_a, labels_b, lam)``. Identity (lam=1) when both alphas are 0."""
    use_mixup = mixup_alpha > 0.0
    use_cutmix = cutmix_alpha > 0.0
    if not (use_mixup or use_cutmix):
        return images, labels, labels, 1.0

    rng = rng or np.random.default_rng()
    do_cutmix = use_cutmix and (not use_mixup or rng.random() < 0.5)
    alpha = cutmix_alpha if do_cutmix else mixup_alpha
    lam = float(rng.beta(alpha, alpha))
    index = torch.randperm(images.size(0), device=images.device)
    labels_a, labels_b = labels, labels[index]

    if do_cutmix and images.dim() == 4:
        mixed = images.clone()
        _, _, height, width = images.shape
        ratio = float(np.sqrt(1.0 - lam))
        cut_h, cut_w = int(height * ratio), int(width * ratio)
        cy, cx = int(rng.integers(height)), int(rng.integers(width))
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
        mixed[..., y1:y2, x1:x2] = images[index][..., y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (height * width))
        return mixed, labels_a, labels_b, lam

    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, labels_a, labels_b, lam


def mixed_loss(loss_fn: Callable[[torch.Tensor], torch.Tensor], labels_a, labels_b, lam: float):
    """``lam * loss_fn(labels_a) + (1 - lam) * loss_fn(labels_b)``; short-circuits when unmixed."""
    if lam >= 1.0 or labels_b is labels_a:
        return loss_fn(labels_a)
    return lam * loss_fn(labels_a) + (1.0 - lam) * loss_fn(labels_b)
