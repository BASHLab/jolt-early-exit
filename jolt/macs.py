"""Per-exit MAC accounting for EMAR.

EMAR weights each exit by ``macs[0] / sum(macs[:i+1])``, where ``macs`` is the list of
*incremental* MACs per exit (the cost added by reaching that exit from the previous one).
Because the :class:`jolt.models.base.ExitModel` forward runs exactly one exit block per call,
the incremental cost of exit ``i`` is the cost of ``model(intermediate_{i-1}, exit_layer_idx=i)``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from ptflops import get_model_complexity_info


class _ExitFlopWrapper(nn.Module):
    def __init__(self, model: nn.Module, exit_idx: int):
        super().__init__()
        self.model = model
        self.exit_idx = exit_idx

    def forward(self, x):
        _, logits = self.model(x, exit_layer_idx=self.exit_idx)
        return logits


def per_exit_macs(model: nn.Module, input_shape: Sequence[int], device: torch.device) -> List[float]:
    """Incremental MACs per exit for an input of shape ``input_shape`` (channels-first, no batch)."""
    model = model.to(device).eval()
    intermediate_shapes: List[Tuple[int, ...]] = []
    carry = torch.zeros(1, *input_shape, device=device)
    with torch.no_grad():
        for exit_idx in range(model.num_exits + 1):
            intermediate_shapes.append(tuple(carry.shape[1:]))
            carry, _ = model(carry, exit_layer_idx=exit_idx)

    macs: List[float] = []
    for exit_idx, shape in enumerate(intermediate_shapes):
        wrapper = _ExitFlopWrapper(model, exit_idx).to(device).eval()
        flops, _ = get_model_complexity_info(
            wrapper, tuple(shape), as_strings=False, print_per_layer_stat=False, verbose=False
        )
        macs.append(float(flops))
    return macs
