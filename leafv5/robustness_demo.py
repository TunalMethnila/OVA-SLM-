"""LR-robustness sweep: the quantitative "easiest to train" claim.

An architecture is easy to train if it works across a WIDE range of learning
rates (no tuning), stays stable at high LR (no divergence/NaN), and learns from
step 1 (no dead zone).

Runs same-size LEAFv5 / Transformer / GatedRNN (Mamba-lite) at LRs spanning
4 orders of magnitude on the same recall task; reports final held-out accuracy
and whether each run diverged.

Run:  python -m leafv5.robustness_demo [--steps 60]
"""
from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F

from .config import preset_config
from .model import LeafLM
from .recall_demo import TinyTransformer
from .benchmark_world import GatedRNN


def make_batch(bs, V, P, Q, rng, device):
    T = 2 * P + Q
    xs, ys, masks = [], [], []
    for _ in range(bs):
        keys = rng.sample(range(2, V), P)
        vals = rng.sample(range(2, V), P)
        qi = rng.sample(range(P), Q)
        ids = []
        for k, v in zip(keys, vals):
            ids += [k, v]
        for j in qi:
            ids.append(keys[j])
        y = ids[1:] + [1]
        m = torch.zeros(T, dtype=torch.bool)
        for j, pos in enumerate(range(2 * P, T)):
            m[pos] = True
            y[pos] = vals[qi[j]]
        xs.append(torch.tensor(ids))
        ys.append(torch.tensor(y))
        masks.append(m)
    return (torch.stack(xs).to(device), torch.stack(ys).to(device),
            torch.stack(masks).to(device))


@torch.no_grad()
def heldout(model, V, P, Q, device, n=128, kind="leaf"):
    rng = random.Random(999)
    x, y, m = make_batch(n, V, P, Q, rng, device)
    model.eval()
    lg = model(x, model.init_states(n, device))[0] if kind == "leaf" else model(x)
    model.train()
    return 100.0 * (((lg.argmax(-1) == y) & m).float().sum() / m.float().sum()).item()


def run_one(model, kind, V, P, Q, lr, steps, batch, device, seed=0):
    """Returns (final held-out acc, diverged)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    rng = random.Random(seed)
    diverged = False
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        x, y, m = make_batch(batch, V, P, Q, rng, device)
        if kind == "leaf":
            lg = model(x, model.init_states(batch, device))[0]
        else:
            lg = model(x)
        loss = F.cross_entropy(lg.reshape(-1, V)[m.reshape(-1)],
                               y.reshape(-1)[m.reshape(-1)])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not torch.isfinite(loss):
            diverged = True
            break
    if diverged:
        return 0.0, True
    return heldout(model, V, P, Q, device, kind=kind), False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lrs", type=str, default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    lrs = [float(x) for x in args.lrs.split(",")]
    V, P, Q, BATCH, DIM, LAYERS = 64, 2, 1, 64, 128, 2

    torch.manual_seed(0)
    print("LR-ROBUSTNESS SWEEP (store-2/query-1 recall, held-out %, "
          f"{args.steps} steps): wider stable range = easier to train\n")
    print(f"  {'LR':>8s} | " + " | ".join(f"{n:>11s}" for n in
          ["LEAFv5", "Transformer", "GatedRNN"]))
    results = {"LEAFv5": {}, "Transformer": {}, "GatedRNN": {}}
    for lr in lrs:
        row = []
        for name, kind in [("LEAFv5", "leaf"), ("Transformer", "trans"),
                           ("GatedRNN", "rnn")]:
            if kind == "leaf":
                m = LeafLM(preset_config("micro", vocab_size=V, n_layers=LAYERS,
                                         dim=DIM, d_h=48, rope_dim=0,
                                         scale_init=0.1)).to(device)
            elif kind == "trans":
                m = TinyTransformer(V, dim=DIM, layers=LAYERS).to(device)
            else:
                m = GatedRNN(V, dim=DIM, layers=LAYERS).to(device)
            acc, div = run_one(m, kind, V, P, Q, lr, args.steps, BATCH, device)
            results[name][lr] = acc if not div else float("nan")
            row.append("DIVERGED" if div else f"{acc:5.1f}%")
        print(f"  {lr:>8.0e} | " + " | ".join(f"{r:>11s}" for r in row))

    print("\n  stable LR range (no divergence, acc > 5%):")
    for name in results:
        stable = [lr for lr, acc in results[name].items() if acc > 5.0]
        print(f"    {name:12s}: {min(stable):.0e} .. {max(stable):.0e} "
              f"({len(stable)}/{len(lrs)} LRs usable)")


if __name__ == "__main__":
    main()
