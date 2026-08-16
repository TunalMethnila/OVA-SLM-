"""LoRA (Low-Rank Adaptation) for LEAFv5 -- parameter-efficient fine-tuning.

`apply_lora(model, rank)` wraps the memory + FFN + gate Linear layers in
low-rank adapters (A: in->r, B: r->out, B init 0 -> output identical to the
base at start, so fine-tuning starts from the exact pretrained behavior).
Only the LoRA params (+ biases/scales that are already params) train; the
base weights stay frozen -- typically ~1-3% of the model's params.

`merge_lora(model)` folds the adapters back into the base weights and removes
the wrappers, producing a PLAIN LeafLM state_dict that works with every
existing tool (generate, serve, grow, quantize).

Usage (finetune.py):  --lora-rank 16
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrapped Linear: y = base(x) + (x@A^T)@B^T * (alpha/r).  B=0 at init."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(in_f, rank) * (1.0 / math.sqrt(in_f)))
        self.B = nn.Parameter(torch.zeros(rank, out_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A) @ self.B * self.scale

    def merge_into_base(self):
        """base.weight [out, in] += (A @ B)^T * scale  (y = base(x) + x@A@B)."""
        with torch.no_grad():
            delta = (self.A @ self.B) * self.scale  # [in, out]
            self.base.weight.add_(delta.t())
        return self.base


_TARGETS = ["wk", "wv", "wq", "wo", "w_write", "w_forget", "w_read", "w_decay",
            "mix_gate", "out_gate"]


def _walk_get(parent: nn.Module, path: str) -> nn.Module:
    """Traverse 'blocks.0.memory.wk' supporting ModuleList/Sequential index."""
    for part in path.split("."):
        if part.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent


def apply_lora(model: nn.Module, rank: int, alpha: float = 1.0) -> int:
    """Wrap target Linear layers in LoRA; freeze base weights everywhere.
    Returns the number of trainable params (LoRA only)."""
    if rank <= 0:
        return 0
    replaced = 0
    for name, module in model.named_modules():
        if name.endswith(tuple("." + t for t in _TARGETS)):
            parent_name, _, attr = name.rpartition(".")
            parent = _walk_get(model, parent_name) if parent_name else model
            cur = getattr(parent, attr)
            if isinstance(cur, nn.Linear) and not isinstance(cur, LoRALinear):
                setattr(parent, attr, LoRALinear(cur, rank, alpha))
                replaced += 1
    # freeze all non-LoRA params
    for name, p in model.named_parameters():
        if ".A" not in name and ".B" not in name:
            p.requires_grad_(False)
    return replaced


def lora_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if ".A" in n or ".B" in n]


def merge_lora(model: nn.Module) -> int:
    """Fold every LoRA adapter into its base Linear and replace the wrapper.
    Returns the number of adapters merged."""
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            base = module.merge_into_base()
            parent_name, _, attr = name.rpartition(".")
            parent = model
            if parent_name:
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            setattr(parent, attr, base)
            n += 1
    # unfreeze everything (back to a normal trainable model)
    for p in model.parameters():
        p.requires_grad_(True)
    return n
