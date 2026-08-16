"""ablate_suite.py — fixed-compute ablation: what actually matters (Tier-3 #9).

Every variant gets the SAME steps, batch, seq, dim, layers and LR on the same
held-out split (Tiny Shakespeare char-LM), so the comparison is at fixed
compute.  Axes:
  A. multi-timescale vs single-timescale (all heads in one group)
  B. StateNorm on vs off
  C. read-query vs write-key readout
  D. identity-start (scale_init=0) vs small scale_init
  E. SWA hybrid ratio (off / every 4 / every 2 / every layer)
  F. input decay on vs off
  G. surprise-gated writes on vs off   (Tier-1)

Run:  python -m leafv5.ablate_suite [--steps 150]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import preset_config
from .model import LeafLM
from .speed_demo import get_batch, load_shakespeare


def train_eval(kw, train_ids, val_x, val_y, V, steps, bs, seq, seed):
    torch.manual_seed(seed)
    cfg = preset_config("micro", vocab_size=V, rope_dim=0, **kw)
    m = LeafLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        x, y = get_batch(train_ids, bs, seq, rng)
        lg, _ = m(x)
        F.cross_entropy(lg.reshape(-1, V), y.reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    with torch.no_grad():
        lg, _ = m(val_x)
        return F.cross_entropy(lg.reshape(-1, V).float(),
                               val_y.reshape(-1)).item(), m.n_params


BASE = dict(dim=128, n_layers=2, d_h=48, scale_init=0.1)
VARIANTS = [
    ("A  multi-timescale (default)", dict(BASE)),
    ("A' single-timescale          ", dict(BASE, fast_heads=6, medium_heads=0,
                                           slow_heads=0, write_strength=(1.0, 1.0, 1.0),
                                           forget_strength=(1.0, 1.0, 1.0))),
    ("B  StateNorm OFF             ", dict(BASE, state_norm=False)),
    ("C  read-query OFF            ", dict(BASE, use_read_query=False)),
    ("D  identity-start (si=0)     ", dict(BASE, scale_init=0.0)),
    ("E  +SWA every 2              ", dict(BASE, use_swa=True, swa_every=2,
                                           swa_window=32)),
    ("E' +SWA every layer          ", dict(BASE, use_swa=True, swa_every=1,
                                           swa_window=32)),
    ("F  +input-decay              ", dict(BASE, input_decay=True)),
    ("G  +surprise-gate            ", dict(BASE, surprise_gate=True)),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bs", type=int, default=12)
    p.add_argument("--seq", type=int, default=32)
    p.add_argument("--only", type=str, default=None)
    args = p.parse_args()

    train_ids, val_ids, V = load_shakespeare()
    vr = np.random.default_rng(1234)
    val_x, val_y = get_batch(val_ids, args.bs, args.seq, vr)

    print("=" * 70)
    print(f"FIXED-COMPUTE ABLATION | Shakespeare char-LM | {args.steps} steps "
          f"| dim 128 L2 | held-out loss (lower better)")
    print("=" * 70)
    results = []
    for name, kw in VARIANTS:
        if args.only and args.only not in name:
            continue
        t0 = time.time()
        vl, np_ = train_eval(kw, train_ids, val_x, val_y, V, args.steps,
                             args.bs, args.seq, args.seed)
        results.append((name, vl, np_))
        print(f"  {name}  loss={vl:.4f}  ({np_/1e6:.2f}M)  ({time.time()-t0:.0f}s)",
              flush=True)
    print("-" * 70)
    base = next(vl for n, vl, _ in results if n.startswith("A "))
    for name, vl, _ in results:
        if name.startswith("A"):
            continue
        tag = "worse" if vl - base > 0.005 else ("better" if vl - base < -0.005
                                                 else "tie")
        print(f"  {name.strip():30s} Δvs-base {vl - base:+.4f}  ({tag})")


if __name__ == "__main__":
    main()
