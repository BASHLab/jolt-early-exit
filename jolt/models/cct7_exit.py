"""CCT-7/3x2 (Compact Convolutional Transformer) with three early exits + final.

Source: Hassani, Walton, Shah, Abuduweili, Li & Shi, "Escaping the Big Data Paradigm with
Compact Transformers" (WACV 2022). CCT-N/KxC denotes N transformer encoder layers, KxK
convolution kernels, and C conv tokenizer blocks. CCT-7/3x2 is the variant Hassani et al.
report on Tiny-ImageNet at 64x64 input: 2-layer conv tokenizer with stride-2 max-pool
between layers, 7 transformer layers at embed_dim=256, num_heads=4, mlp_ratio=2, with
learnable positional embeddings and a SeqPool (learnable attention pooling) classifier head.

~4.1M params at num_classes=200, under the 5M on-device cap. CCT-14 is the 224x224
ImageNet variant (~22M params); not used here.

Exits placed after transformer layers 2 / 4 / 6 (three internal exits at ~29 / 57 / 86%
compute fractions per the SDN convention), plus the final exit after layer 7 -> LayerNorm
-> SeqPool -> Linear. ``num_exits = 3`` (count of EARLY exits) per the ExitModel contract;
4 classifiers total.

Internal-exit / gate / confidence heads all use SeqPool. The SDN-style mixed-pool head
isn't well-defined on a 1D token sequence; SeqPool is the CCT-native pooling primitive
and Hassani's published numbers use it for the final exit, so we use it everywhere.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ExitModel


class _ConvTokenizer(nn.Module):
    """CCT 2-layer conv tokenizer.

    Two 3x3 convolutions, each followed by ReLU and 3x3 max-pool with stride 2 (pad 1).
    For 64x64 input this yields a 16x16 = 256-token sequence at embed_dim channels. The
    intermediate width is embed_dim // 4.
    """

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        mid = embed_dim // 4
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(mid, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x), inplace=True)
        x = self.pool1(x)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.pool2(x)
        return x.flatten(2).transpose(1, 2)  # [B, N, D]


class _TransformerEncoderLayer(nn.Module):
    """Pre-LN ViT encoder block: norm -> MHA -> add | norm -> MLP -> add."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float, attn_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class _SeqPoolHead(nn.Module):
    """CCT SeqPool internal classifier: softmax-attention over the token axis -> Linear(D, nc).

    Hassani et al. (WACV 2022) Eq. 1: scores = softmax(Linear(D, 1)(x)) over N tokens;
    pooled = scores^T x; logits = Linear(D, num_classes)(pooled).
    """

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.attn(x).squeeze(-1), dim=1)
        pooled = (w.unsqueeze(-1) * x).sum(dim=1)
        return self.fc(pooled)


class _SeqGateHead(nn.Module):
    """JEI-DNN gate (CCT variant): SeqPool -> Linear(D, 1).

    Returns one logit per sample; sigmoid(logit) is the Bernoulli routing probability for
    this exit. Architecturally identical to _SeqConfidenceHead but kept separate to avoid
    confusing JEI-DNN gates with SCAR confidence scores at use sites.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.fc = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.attn(x).squeeze(-1), dim=1)
        pooled = (w.unsqueeze(-1) * x).sum(dim=1)
        return self.fc(pooled).squeeze(-1)


class _SeqConfidenceHead(nn.Module):
    """SCAR confidence head (CCT variant): SeqPool -> Linear(D, 1)."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.fc = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.attn(x).squeeze(-1), dim=1)
        pooled = (w.unsqueeze(-1) * x).sum(dim=1)
        return self.fc(pooled).squeeze(-1)


def _build_block_stack(
    depth: int, dim: int, num_heads: int, mlp_ratio: float, dropout: float, attn_dropout: float,
) -> nn.Sequential:
    return nn.Sequential(*[
        _TransformerEncoderLayer(dim, num_heads, mlp_ratio, dropout, attn_dropout) for _ in range(depth)
    ])


class CCT7Exit(ExitModel):
    """CCT-7/3x2 with three internal SeqPool classifiers + final classifier.

    Forward pass threads the [B, N, D] token sequence through 4 chunks per the ExitModel
    contract. Exit indices 0..2 emit internal SeqPool logits after layers 2 / 4 / 6;
    index 3 runs the final transformer layer + LayerNorm + SeqPool + Linear.
    """

    num_exits = 3

    def __init__(
        self,
        num_classes: int = 200,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        num_tokens: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        self.tokenizer = _ConvTokenizer(in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        # chunk_0: transformer layers 1-2 (~29% compute fraction).
        self.chunk_0 = _build_block_stack(2, embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
        self.exit_head_0 = _SeqPoolHead(embed_dim, num_classes)
        self.gate_head_0 = _SeqGateHead(embed_dim)
        self.confidence_head_0 = _SeqConfidenceHead(embed_dim)

        # chunk_1: transformer layers 3-4 (~57%).
        self.chunk_1 = _build_block_stack(2, embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
        self.exit_head_1 = _SeqPoolHead(embed_dim, num_classes)
        self.gate_head_1 = _SeqGateHead(embed_dim)
        self.confidence_head_1 = _SeqConfidenceHead(embed_dim)

        # chunk_2: transformer layers 5-6 (~86%).
        self.chunk_2 = _build_block_stack(2, embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
        self.exit_head_2 = _SeqPoolHead(embed_dim, num_classes)
        self.gate_head_2 = _SeqGateHead(embed_dim)
        self.confidence_head_2 = _SeqConfidenceHead(embed_dim)

        # chunk_3: transformer layer 7 + final LayerNorm + canonical CCT SeqPool head. No
        # gate on the final exit (routing probability is the residual). SCAR DOES use a
        # confidence head at every exit including the final one.
        self.chunk_3 = _build_block_stack(1, embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_head = _SeqPoolHead(embed_dim, num_classes)
        self.confidence_head_final = _SeqConfidenceHead(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        if tokens.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"Tokenizer produced {tokens.shape[1]} tokens but positional embedding has "
                f"{self.pos_embed.shape[1]} slots. CCT-7/3x2 expects 64x64 input."
            )
        return self.pos_drop(tokens + self.pos_embed)

    def _final_classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_head(self.final_norm(x))

    def forward(
        self, x: torch.Tensor, exit_layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exit_layer_idx is None:
            exit_layer_idx = 0
        if exit_layer_idx == 0:
            x = self._embed(x)
            x = self.chunk_0(x)
            return x, self.exit_head_0(x)
        if exit_layer_idx == 1:
            x = self.chunk_1(x)
            return x, self.exit_head_1(x)
        if exit_layer_idx == 2:
            x = self.chunk_2(x)
            return x, self.exit_head_2(x)
        if exit_layer_idx == 3:
            x = self.chunk_3(x)
            return x, self._final_classify(x)
        raise ValueError(
            f"exit_layer_idx {exit_layer_idx} out of range for {self.num_exits} early exits"
        )

    def forward_with_gates(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Single-pass forward returning per-exit classifier logits AND per-exit gate logits.

        Returns 4 classifier logits and 3 gate logits (final exit has no gate; its routing
        probability is the residual). Used by the JEI-DNN training path.
        """
        per_exit_logits: List[torch.Tensor] = []
        per_exit_gate_logits: List[torch.Tensor] = []
        x = self._embed(x)
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
        per_exit_logits.append(self._final_classify(x))
        return per_exit_logits, per_exit_gate_logits

    def forward_with_confidences(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Single-pass forward returning per-exit classifier logits AND per-exit confidence
        logits s_j (one per exit, including the final). Used by the SCAR training path.
        """
        per_exit_logits: List[torch.Tensor] = []
        per_exit_confidence_logits: List[torch.Tensor] = []
        x = self._embed(x)
        x = self.chunk_0(x)
        per_exit_logits.append(self.exit_head_0(x))
        per_exit_confidence_logits.append(self.confidence_head_0(x))
        x = self.chunk_1(x)
        per_exit_logits.append(self.exit_head_1(x))
        per_exit_confidence_logits.append(self.confidence_head_1(x))
        x = self.chunk_2(x)
        per_exit_logits.append(self.exit_head_2(x))
        per_exit_confidence_logits.append(self.confidence_head_2(x))
        x = self.chunk_3(x)
        per_exit_logits.append(self._final_classify(x))
        per_exit_confidence_logits.append(self.confidence_head_final(x))
        return per_exit_logits, per_exit_confidence_logits


def cct7_exit(num_classes: int = 200, in_channels: int = 3) -> CCT7Exit:
    return CCT7Exit(num_classes=num_classes, in_channels=in_channels)
