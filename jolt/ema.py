"""Exponential moving average of model weights (evaluate with the averaged weights).

Model EMA is part of the modern training recipe and typically improves both accuracy and
calibration. Build it when ``train.ema_decay > 0``, call :meth:`update` after each optimizer
step, and evaluate under :meth:`swapped_in`.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for key, ema_value in self.ema.state_dict().items():
            value = model_state[key]
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                ema_value.copy_(value)  # integer buffers (e.g. BN num_batches_tracked)

    @contextmanager
    def swapped_in(self, model: nn.Module):
        """Temporarily load the EMA weights into ``model`` for evaluation, then restore."""
        backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.ema.state_dict())
        try:
            yield model
        finally:
            model.load_state_dict(backup)

    @torch.no_grad()
    def copy_into(self, model: nn.Module) -> None:
        """Permanently overwrite ``model``'s weights with the EMA's weights.

        Use this when the model is about to be mutated structurally (e.g. swapping in
        ``LaplaceLinear`` modules for post-hoc Laplace inference) and the swapped_in
        context manager's restore step would fail with a state_dict mismatch.
        """
        model.load_state_dict(self.ema.state_dict())
