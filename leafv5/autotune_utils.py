"""Small, testable helpers for the "easiest to train" guarantees:
LR autotune smoke and loss-spike recovery.  (The trainer wires these in;
this module keeps the logic unit-testable without a full training run.)
"""
from __future__ import annotations

from typing import Tuple

import torch


def spike_recover(shadow_sd, avg_loss: float, loss_ema: float, lr: float,
                  threshold: float = 3.0, extra: float = 0.5,
                  max_recoveries: int = 5, n_recoveries: int = 0,
                  model=None) -> Tuple[float, bool]:
    """If avg_loss >> loss_ema (divergence), roll the model back to the shadow
    weights and halve the LR.  Returns (new_lr, rolled_back)."""
    if avg_loss > threshold * loss_ema + extra and n_recoveries < max_recoveries:
        if model is not None and shadow_sd is not None:
            model.load_state_dict(shadow_sd)
        return lr * 0.5, True
    return lr, False


def nan_guard(model) -> bool:
    """True if any gradient is non-finite (NaN/Inf).  The trainer skips the
    step and rolls back when this fires -- a NaN batch can never corrupt
    AdamW's moments."""
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return True
    return False


def autotune_smoke() -> float:
    """Minimal reproducible autotune: probe 3 LRs on a tiny fixed problem,
    return the best.  Mirrors train.autotune_lr logic for unit tests."""
    import torch
    import torch.nn.functional as F

    from .config import preset_config
    from .model import LeafLM

    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=2, dim=96, d_h=32,
                        scale_init=0.1)
    x = torch.randint(0, 256, (8, 32))
    y = torch.randint(0, 256, (8, 32))
    best_lr, best_loss = 1e-3, float("inf")
    for c in (0.3, 1.0, 3.0):
        lr = 5e-4 * c
        m = LeafLM(cfg)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
        losses = []
        for _ in range(6):
            opt.zero_grad()
            lg, _ = m(x, m.init_states(8, torch.device("cpu")))
            loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        last = sum(losses[-3:]) / 3
        if all(map(lambda v: v == v, losses)) and last < best_loss:
            best_loss, best_lr = last, lr
    return best_lr
