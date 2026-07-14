"""WideResNet-28-10-SDN with JEI-DNN per-exit gate heads (Regol et al., ICLR 2024).

Each of the 6 early exits gets a binary GATE head alongside its classifier head. The gate is
a per-sample logit indicating whether to take this exit; chained sigmoid gates produce a
per-sample routing distribution pi_i over the K=7 exits. Final exit has no gate (it is the
absorbing state: pi_{K-1} = prod_{j<K-1} (1 - sigmoid(g_j))).

Training uses ``forward_with_gates`` to return both classifier logits and gate logits at
every early exit. Evaluation reuses the inherited entropy-threshold sweep on classifier
heads -- the gates are training-time-only here, which keeps the (accuracy, MACs) curve
comparable to all other baselines. (A gate-routed single OP is in scope for the appendix.)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .wideresnet28_10_sdn_exit import WideResNet2810SDNExit


class _GateHead(nn.Module):
    """Per-sample binary gate: AvgPool -> Linear(C, 1) -> (B,) logits."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1)).squeeze(-1)


class WideResNet2810SDNJEIExit(WideResNet2810SDNExit):
    """WRN-28-10-SDN with six per-exit gate heads (JEI-DNN training; classifier-only eval)."""

    def __init__(self, num_classes: int = 100, in_channels: int = 3) -> None:
        super().__init__(num_classes=num_classes, in_channels=in_channels)
        k = 10
        c1, c2, c3 = 16 * k, 32 * k, 64 * k  # 160, 320, 640
        self.gate_head_0 = _GateHead(c1)
        self.gate_head_1 = _GateHead(c1)
        self.gate_head_2 = _GateHead(c2)
        self.gate_head_3 = _GateHead(c2)
        self.gate_head_4 = _GateHead(c3)
        self.gate_head_5 = _GateHead(c3)

    def forward_with_gates(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Full forward returning (classifier_logits_per_exit, gate_logits_per_early_exit).

        Threads the carry through all chunks like ``forward_all_exits`` but also collects
        gate logits at each of the 6 early exits. The final exit has no gate.
        """
        per_exit_logits: List[torch.Tensor] = []
        per_exit_gate_logits: List[torch.Tensor] = []

        x = self.stem(x)
        x = self.chunk_0(x)
        per_exit_logits.append(self.exit_head_0(x))
        per_exit_gate_logits.append(self.gate_head_0(x))

        x = self.chunk_1(x)
        per_exit_logits.append(self.exit_head_1(x))
        per_exit_gate_logits.append(self.gate_head_1(x))

        x = self.chunk_2(x)
        per_exit_logits.append(self.exit_head_2(x))
        per_exit_gate_logits.append(self.gate_head_2(x))

        x = self.chunk_3(x)
        per_exit_logits.append(self.exit_head_3(x))
        per_exit_gate_logits.append(self.gate_head_3(x))

        x = self.chunk_4(x)
        per_exit_logits.append(self.exit_head_4(x))
        per_exit_gate_logits.append(self.gate_head_4(x))

        x = self.chunk_5(x)
        per_exit_logits.append(self.exit_head_5(x))
        per_exit_gate_logits.append(self.gate_head_5(x))

        x = self.chunk_6(x)
        per_exit_logits.append(self._final_classify(x))

        return per_exit_logits, per_exit_gate_logits


def wideresnet28_10_sdn_jei_exit(
    num_classes: int = 100, in_channels: int = 3
) -> WideResNet2810SDNJEIExit:
    return WideResNet2810SDNJEIExit(num_classes=num_classes, in_channels=in_channels)
