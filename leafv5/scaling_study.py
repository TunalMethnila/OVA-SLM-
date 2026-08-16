"""scaling_study.py — honest micro-scale scaling evidence (Tier-1 #2).

At three sizes, LEAFv5 vs a same-size Transformer++ on Tiny Shakespeare
char-LM, matched data/optimizer/steps.  Reports:
  * params
  * held-out loss at matched steps
  * loss-vs-params trend (the "scaling" that micro scale can show)

This is NOT a production scaling law (that needs 100M-1B runs on a real
corpus); it is the honest, runnable-anywhere precursor that shows the
architecture's loss-vs-size trend and the head-to-head at each size.

Run:  python -m leafv5.scaling_study [--steps 150] [--seed 0]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import preset_config
from .model import LeafLM
from .recall_demo import TinyTransformer
from .speed_demo import get_batch, load_shakespeare

SIZES = [
    ("S  (128, L2)", dict(dim=128, n_layers=2)),
    ("M  (192, L4)", dict(dim=192, n_layers=4)),
    ("L  (256, L6)", dict(dim=256, n_layers=6)),
]


def run_size(kw, train_ids, val_x, val_y, V, steps, bs, seq, seed):
    torch.manual_seed(seed)
    cfg = preset_config("micro", vocab_size=V, rope_dim=0, scale_init=0.1, **kw)
    m = LeafLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    losses = []
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        x, y = get_batch(train_ids, bs, seq, rng)
        lg, _ = m(x)
        F.cross_entropy(lg.reshape(-1, V), y.reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if len(losses) < 5 or step % 25 == 0:
            losses.append((step, F.cross_entropy(
                m(x, m.init_states(bs, torch.device("cpu")))[0].reshape(-1, V),
                y.reshape(-1)).item()))
    m.eval()
    with torch.no_grad():
        lg, _ = m(val_x)
        vl = F.cross_entropy(lg.reshape(-1, V).float(), val_y.reshape(-1)).item()
    return m.n_params, vl, time.time() - t0, losses


def run_transformer(kw, train_ids, val_x, val_y, V, steps, bs, seq, seed):
    torch.manual_seed(seed)
    m = TinyTransformer(V, dim=kw["dim"], layers=kw["n_layers"])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        x, y = get_batch(train_ids, bs, seq, rng)
        lg = m(x)
        F.cross_entropy(lg.reshape(-1, V), y.reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    with torch.no_grad():
        lg = m(val_x)
        vl = F.cross_entropy(lg.reshape(-1, V).float(), val_y.reshape(-1)).item()
    return sum(p.numel() for p in m.parameters()), vl, time.time() - t0, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bs", type=int, default=12)
    p.add_argument("--seq", type=int, default=32)
    p.add_argument("--arch", choices=["leaf", "trans", "both"], default="both")
    args = p.parse_args()

    train_ids, val_ids, V = load_shakespeare()
    vr = np.random.default_rng(1234)
    val_x, val_y = get_batch(val_ids, args.bs, args.seq, vr)

    print("=" * 70)
    print(f"MICRO-SCALE SCALING STUDY | Shakespeare char-LM | "
          f"{args.steps} steps | seed {args.seed}")
    print("=" * 70)
    rows = []
    for name, kw in SIZES:
        if args.arch in ("leaf", "both"):
            np_, vl, dt, hist = run_size(
                kw, train_ids, val_x, val_y, V, args.steps, args.bs, args.seq,
                args.seed)
            rows.append((name, "LEAFv5", np_, vl))
            print(f"  {name} LEAFv5   : {np_/1e6:5.2f}M params  "
                  f"val_loss={vl:.4f}  ({dt:.0f}s)", flush=True)
        if args.arch in ("trans", "both"):
            np_, vl, dt, _ = run_transformer(
                kw, train_ids, val_x, val_y, V, args.steps, args.bs, args.seq,
                args.seed)
            rows.append((name, "Transformer", np_, vl))
            print(f"  {name} Transf.  : {np_/1e6:5.2f}M params  "
                  f"val_loss={vl:.4f}  ({dt:.0f}s)", flush=True)
    print("-" * 70)
    print("  loss-vs-size trend (LEAFv5):")
    leaf = [r for r in rows if r[1] == "LEAFv5"]
    for (_, _, p1, l1), (_, _, p2, l2) in zip(leaf, leaf[1:]):
        dl = l1 - l2
        print(f"    {p1/1e6:.2f}M -> {p2/1e6:.2f}M : loss {l1:.4f} -> {l2:.4f} "
              f"(Δ={dl:+.4f})")
    print("  Honest note: micro-scale trend only; the production scaling law "
          "needs the T4 runs (README §32).")


if __name__ == "__main__":
    main()
