"""Shared early-exit model interface.

Every JOLT backbone implements the same contract so the loss, calibration, inference, and
diagnostics code is backbone-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ExitModel(nn.Module, ABC):
    """Multi-exit model contract.

    ``num_exits`` is the number of EARLY exits. The final head is exit index ``num_exits``,
    so a model has ``num_exits + 1`` classifiers in total.

    ``forward`` threads the intermediate activation: ``forward(x, exit_layer_idx=i)`` treats
    ``x`` as the activation carried from exit ``i - 1`` (or the raw input when ``i == 0``),
    runs the ``i``-th exit block, and returns ``(intermediate_for_next_exit, logits_i)``. The
    early-exit loop in :mod:`jolt.inference` and the threshold calibration in
    :mod:`jolt.calibration` both rely on this threading, so callers always pass an explicit
    ``exit_layer_idx``.
    """

    num_exits: int

    @abstractmethod
    def forward(
        self, x: torch.Tensor, exit_layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
