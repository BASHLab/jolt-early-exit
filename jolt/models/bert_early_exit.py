"""Multi-exit transformer for GLUE (MobileBERT and other HuggingFace encoders).

Ported faithfully from the Bravery monorepo. Unlike the conv/IMU backbones, a transformer
exit threads two pieces of state between exits: the hidden state and a ``cache`` holding the
extended attention mask, head mask, and current encoder depth. It therefore does not use the
single-tensor ``ExitModel`` contract; the text dynamic-exit loop in
:mod:`jolt.text_inference` knows how to subselect both the hidden state and the cached mask
when samples exit early.

Exit positions are transformer layer indices (MobileBERT uses [8, 16, 24]). Following the
project convention, the last exit layer is the "final" exit, so ``num_exits`` (the number of
early exits) is ``len(exit_layers) - 1``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


def init_classifier_(module: nn.Module, std: float = 0.02) -> None:
    """BERT-style init for a freshly-attached head (Linear ~ N(0, std), LayerNorm to 1/0).

    MobileBERT's default SequenceClassification head can emit logits in the millions with the
    pretrained pooler; we use the raw [CLS] state with a fresh head and initialize it small to
    keep logits in a sane range and avoid saturated-softmax / zero-gradient at the start.
    """
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            nn.init.normal_(sub.weight, mean=0.0, std=std)
            if sub.bias is not None:
                nn.init.zeros_(sub.bias)
        elif isinstance(sub, nn.LayerNorm):
            nn.init.ones_(sub.weight)
            nn.init.zeros_(sub.bias)


class ECECalibrator(nn.Module):
    """Temperature scaling for optional per-exit calibration."""

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / (torch.exp(self.log_temp) + 1e-6)


class BertEarlyExit(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        exit_layers: List[int],
        *,
        use_layernorm_head: bool = False,
        dropout: float = 0.1,
        task: str = "sst2",
        local_files_only: bool = False,
    ):
        super().__init__()
        self.task = task
        self.num_labels = num_labels
        self.regression = num_labels == 1

        self.config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config, local_files_only=local_files_only)

        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            self.encoder_layers = self.backbone.encoder.layer
        elif hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "layer"):
            self.encoder_layers = self.backbone.transformer.layer
        else:
            raise ValueError("Backbone must expose stacked encoder layers for early exits.")

        self.num_encoder_layers = len(self.encoder_layers)
        config_layers = getattr(self.config, "num_hidden_layers", None) or self.num_encoder_layers

        self.exit_layers = sorted(int(layer) for layer in exit_layers)
        if not self.exit_layers:
            raise ValueError("At least one exit layer must be provided.")
        for layer_idx in self.exit_layers:
            if layer_idx < 1 or layer_idx > config_layers:
                raise ValueError(f"Exit layer {layer_idx} must be within [1, {config_layers}].")

        self._exit_to_position = {layer: pos for pos, layer in enumerate(self.exit_layers)}
        hidden_size = self.config.hidden_size
        self.heads = nn.ModuleDict()
        self.calibrators = nn.ModuleDict()
        for layer_idx in self.exit_layers:
            head_layers = [nn.Dropout(dropout), nn.Linear(hidden_size, num_labels)]
            if use_layernorm_head:
                head_layers = [nn.LayerNorm(hidden_size)] + head_layers
            head = nn.Sequential(*head_layers)
            init_classifier_(head)
            self.heads[str(layer_idx)] = head
            self.calibrators[str(layer_idx)] = ECECalibrator()

        # SCAR / JEI-DNN / learned-confidence auxiliary heads. SCAR has one confidence head
        # PER exit (including the final exit -- s_j is the routing score at every operating
        # point on the curve). JEI-DNN has one gate head per EARLY exit only (the final
        # exit's routing probability is the residual of the prior gates).
        self.confidence_heads = nn.ModuleDict()
        for layer_idx in self.exit_layers:
            head = nn.Linear(hidden_size, 1)
            init_classifier_(head)
            self.confidence_heads[str(layer_idx)] = head
        self.gate_heads = nn.ModuleDict()
        for layer_idx in self.exit_layers[:-1]:
            head = nn.Linear(hidden_size, 1)
            init_classifier_(head)
            self.gate_heads[str(layer_idx)] = head

    @property
    def num_exits(self) -> int:
        """Number of early exits (the final exit layer is the full-depth head)."""
        return len(self.exit_layers) - 1

    def _resolve_exit_position(self, exit_layer_idx: int) -> int:
        if exit_layer_idx in self._exit_to_position:
            return self._exit_to_position[exit_layer_idx]
        if isinstance(exit_layer_idx, int) and 0 <= exit_layer_idx < len(self.exit_layers):
            return exit_layer_idx
        raise ValueError(f"exit_layer_idx {exit_layer_idx} is not a valid exit specification.")

    def _prepare_inputs(self, input_ids, attention_mask, token_type_ids):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        extended = self.backbone.get_extended_attention_mask(attention_mask, input_ids.shape, input_ids.device)
        head_mask = self.backbone.get_head_mask(None, self.num_encoder_layers)
        embedding_kwargs = {"input_ids": input_ids}
        if token_type_ids is not None and hasattr(self.backbone.embeddings, "token_type_embeddings"):
            embedding_kwargs["token_type_ids"] = token_type_ids
        hidden_state = self.backbone.embeddings(**embedding_kwargs)
        cache = {"extended_attention_mask": extended, "head_mask": head_mask, "current_depth": 0}
        return hidden_state, cache

    def _run_encoder_layers(self, hidden_state, start_layer, end_layer, cache):
        attention_mask = cache["extended_attention_mask"]
        head_mask = cache["head_mask"]
        for layer_idx in range(start_layer, end_layer):
            layer_head_mask = None if head_mask is None else head_mask[layer_idx]
            hidden_state = self.encoder_layers[layer_idx](
                hidden_state, attention_mask=attention_mask, head_mask=layer_head_mask
            )[0]
        return hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        exit_layer_idx: Optional[int] = None,
        hidden_state: Optional[torch.Tensor] = None,
        cache: Optional[Dict] = None,
        apply_calibration: bool = False,
    ) -> Dict:
        """Run one exit (``exit_layer_idx`` set) or all exits (``exit_layer_idx is None``).

        When ``exit_layer_idx`` is set, returns ``{logits, hidden_state, cache, exit_layer}``;
        pass the returned ``hidden_state`` and ``cache`` back in to resume from that depth.
        """
        if exit_layer_idx is not None:
            position = self._resolve_exit_position(exit_layer_idx)
            hidden_state, logits, cache = self._forward_from(
                input_ids, attention_mask, token_type_ids, hidden_state, cache, position
            )
            if apply_calibration:
                logits = self.calibrators[str(self.exit_layers[position])](logits)
            return {"logits": logits, "hidden_state": hidden_state, "cache": cache, "exit_layer": self.exit_layers[position]}

        logits_per_exit: List[torch.Tensor] = []
        hs, ca = hidden_state, cache
        for position in range(len(self.exit_layers)):
            hs, logits, ca = self._forward_from(input_ids, attention_mask, token_type_ids, hs, ca, position)
            if apply_calibration:
                logits = self.calibrators[str(self.exit_layers[position])](logits)
            logits_per_exit.append(logits)
        return {"logits_per_exit": logits_per_exit, "hidden_state": hs, "cache": ca}

    def forward_with_gates(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Single-pass forward returning per-exit classifier logits AND per-exit gate logits.

        Used by the JEI-DNN training path. Returns ``len(exit_layers)`` classifier logits
        (shape ``(B, num_labels)``) and ``len(exit_layers) - 1`` gate logits (shape ``(B,)``;
        the final exit has no gate, residual probability).
        """
        per_exit_logits: List[torch.Tensor] = []
        per_exit_gate_logits: List[torch.Tensor] = []
        hs, ca = None, None
        for position in range(len(self.exit_layers)):
            hs, logits, ca = self._forward_from(
                input_ids, attention_mask, token_type_ids, hs, ca, position
            )
            per_exit_logits.append(logits)
            layer_key = str(self.exit_layers[position])
            if layer_key in self.gate_heads:
                per_exit_gate_logits.append(self.gate_heads[layer_key](hs[:, 0]).squeeze(-1))
        return per_exit_logits, per_exit_gate_logits

    def forward_with_confidences(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Single-pass forward returning per-exit classifier logits AND per-exit confidence logits.

        Used by the SCAR training path. Returns ``len(exit_layers)`` classifier logits
        plus ``len(exit_layers)`` confidence logits of shape ``(B,)`` -- one s_j per exit
        including the final.
        """
        per_exit_logits: List[torch.Tensor] = []
        per_exit_confidence_logits: List[torch.Tensor] = []
        hs, ca = None, None
        for position in range(len(self.exit_layers)):
            hs, logits, ca = self._forward_from(
                input_ids, attention_mask, token_type_ids, hs, ca, position
            )
            per_exit_logits.append(logits)
            layer_key = str(self.exit_layers[position])
            per_exit_confidence_logits.append(
                self.confidence_heads[layer_key](hs[:, 0]).squeeze(-1)
            )
        return per_exit_logits, per_exit_confidence_logits

    def _forward_from(
        self, input_ids, attention_mask, token_type_ids, hidden_state, cache, position
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if hidden_state is None or cache is None:
            if input_ids is None:
                raise ValueError("input_ids required when no cached hidden state is supplied.")
            hidden_state, cache = self._prepare_inputs(input_ids, attention_mask, token_type_ids)
        start_depth = cache.get("current_depth", 0)
        target_depth = self.exit_layers[position]
        if target_depth < start_depth:
            raise ValueError(f"Exit depth {target_depth} is behind current depth {start_depth}.")
        hidden_state = self._run_encoder_layers(hidden_state, start_depth, target_depth, cache)
        cache["current_depth"] = target_depth
        logits = self.heads[str(self.exit_layers[position])](hidden_state[:, 0])
        return hidden_state, logits, cache


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def freeze_for_stage1(model: "BertEarlyExit") -> None:
    """DeeBERT stage 1: train the backbone and the final-exit head; early heads are inactive."""
    set_requires_grad(model.backbone, True)
    final = str(model.exit_layers[-1])
    for key, head in model.heads.items():
        set_requires_grad(head, key == final)


def freeze_for_stage2(model: "BertEarlyExit") -> None:
    """DeeBERT stage 2: freeze the backbone and final head; train only the early-exit heads."""
    set_requires_grad(model.backbone, False)
    final = str(model.exit_layers[-1])
    for key, head in model.heads.items():
        set_requires_grad(head, key != final)


__all__ = [
    "ECECalibrator",
    "BertEarlyExit",
    "init_classifier_",
    "freeze_for_stage1",
    "freeze_for_stage2",
    "set_requires_grad",
]
