"""SelectiveNet-style per-exit selection heads (Candidate C; Geifman & El-Yaniv 2019).

Wraps any :class:`ExitModel`, adding per-exit selection heads ``g`` (sigmoid scalar) and
auxiliary classification heads ``h``. Inference and threshold calibration use the base
prediction heads ``f`` unchanged (``forward`` delegates), so the early-exit policy and EMAR/AURC
evaluation are identical to the other candidates. Training uses :meth:`forward_selective`.

# RESEARCH GAP: the selection/auxiliary heads ideally consume each exit's pooled features; here
# they take the exit's class logits to stay backbone-agnostic, which is an approximation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .base import ExitModel


class SelectiveExitModel(ExitModel):
    def __init__(self, base: ExitModel, num_classes: int, hidden: int = 128):
        super().__init__()
        self.base = base
        self.num_exits = base.num_exits
        n_heads = base.num_exits + 1
        self.selection_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(num_classes, hidden), nn.ReLU(), nn.Linear(hidden, 1)) for _ in range(n_heads)]
        )
        self.aux_heads = nn.ModuleList([nn.Linear(num_classes, num_classes) for _ in range(n_heads)])

    def forward(self, x: torch.Tensor, exit_layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Inference / calibration use the base prediction head f_e.
        return self.base(x, exit_layer_idx=exit_layer_idx)

    def forward_selective(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Return ``(per_exit_logits, per_exit_selection, per_exit_aux)`` for SelectiveNet training."""
        logits_list, selection_list, aux_list = [], [], []
        carry = x
        for i in range(self.num_exits + 1):
            carry, logits = self.base(carry, exit_layer_idx=i)
            logits_list.append(logits)
            selection_list.append(torch.sigmoid(self.selection_heads[i](logits)))
            aux_list.append(self.aux_heads[i](logits))
        return logits_list, selection_list, aux_list
