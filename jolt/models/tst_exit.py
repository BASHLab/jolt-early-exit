"""Multi-exit Time Series Transformer (TST, Zerveas et al. KDD 2021).

Channel projection -> sinusoidal positional encoding -> N transformer encoder
blocks -> classification head. Multi-exit splits: chunks of (N/4) blocks
each, with 3 internal exits + final.

Input convention: ``(B, in_channels, time)``.
~0.6M params at d_model=64, n_blocks=8, n_heads=4, num_classes=6.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import math

import torch
import torch.nn as nn

from .base import ExitModel


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class _Head1d(nn.Module):
    """Pool over time + Linear."""
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return self.fc(x.mean(dim=1))


class _GateHead1d(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=1))


class _ConfidenceHead1d(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=1))


class TSTExit(ExitModel):
    """TST with three early exits + final classifier (transformer)."""

    num_exits = 3

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 9,
        d_model: int = 64,
        n_blocks: int = 8,
        n_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.d_model = d_model

        # Channel projection: (B, C, T) -> (B, T, d_model)
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=max_len)

        # Split transformer blocks into 4 chunks
        per_chunk = max(1, n_blocks // 4)
        def make_chunk(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)

        self.chunk_0 = make_chunk(per_chunk)
        self.chunk_1 = make_chunk(per_chunk)
        self.chunk_2 = make_chunk(per_chunk)
        # last chunk takes the remainder so total = n_blocks
        remainder = max(1, n_blocks - 3 * per_chunk)
        self.chunk_3 = make_chunk(remainder)

        self.exit_head_0 = _Head1d(d_model, num_classes)
        self.exit_head_1 = _Head1d(d_model, num_classes)
        self.exit_head_2 = _Head1d(d_model, num_classes)
        self.exit_head_final = _Head1d(d_model, num_classes)

        self.gate_head_0 = _GateHead1d(d_model)
        self.confidence_head_0 = _ConfidenceHead1d(d_model)
        self.gate_head_1 = _GateHead1d(d_model)
        self.confidence_head_1 = _ConfidenceHead1d(d_model)
        self.gate_head_2 = _GateHead1d(d_model)
        self.confidence_head_2 = _ConfidenceHead1d(d_model)
        self.confidence_head_final = _ConfidenceHead1d(d_model)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, d_model)
        x = x.permute(0, 2, 1).contiguous()
        x = self.input_proj(x)
        x = self.pos_enc(x)
        return x

    def forward(
        self, x: torch.Tensor, exit_layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exit_layer_idx is None:
            exit_layer_idx = 0
        if exit_layer_idx == 0:
            h = self._embed(x); h = self.chunk_0(h); return h, self.exit_head_0(h)
        if exit_layer_idx == 1:
            h = self.chunk_1(x); return h, self.exit_head_1(h)
        if exit_layer_idx == 2:
            h = self.chunk_2(x); return h, self.exit_head_2(h)
        if exit_layer_idx == 3:
            h = self.chunk_3(x); return h, self.exit_head_final(h)
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
        )

    def forward_with_gates(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        h = self._embed(x)
        h = self.chunk_0(h); logits.append(self.exit_head_0(h)); gates.append(self.gate_head_0(h))
        h = self.chunk_1(h); logits.append(self.exit_head_1(h)); gates.append(self.gate_head_1(h))
        h = self.chunk_2(h); logits.append(self.exit_head_2(h)); gates.append(self.gate_head_2(h))
        h = self.chunk_3(h); logits.append(self.exit_head_final(h))
        return logits, gates

    def forward_with_confidences(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        confs: List[torch.Tensor] = []
        h = self._embed(x)
        h = self.chunk_0(h); logits.append(self.exit_head_0(h)); confs.append(self.confidence_head_0(h))
        h = self.chunk_1(h); logits.append(self.exit_head_1(h)); confs.append(self.confidence_head_1(h))
        h = self.chunk_2(h); logits.append(self.exit_head_2(h)); confs.append(self.confidence_head_2(h))
        h = self.chunk_3(h); logits.append(self.exit_head_final(h)); confs.append(self.confidence_head_final(h))
        return logits, confs


def tst_exit(num_classes: int = 6, in_channels: int = 9) -> TSTExit:
    return TSTExit(num_classes=num_classes, in_channels=in_channels)
