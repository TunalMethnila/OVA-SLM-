"""Rapid adaptation & continual learning (paper sec. 6).

Protocol:
  A) Train the recall task with keys/values drawn from pool A (~500 steps).
     Measure held-out recall on pool A.
  B) Train K gradient steps on task B (disjoint key/value pool).
     Measure: B recall (absorption speed: few cycles) and A recall
     (retention / catastrophic forgetting -- the slow heads' protection).
  C) One-cycle: after A, a SINGLE gradient step on one B example, then
     measure B recall on fresh examples.

Runs the same protocol on LEAFv5 and a same-size Transformer for comparison.

Run:  python -m leafv5.adapt_demo [--steps-a 500] [--steps-b 20]
"""
from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F

from .config import preset_config
from .model import LeafLM
from .recall_demo import TinyTransformer


def make_pool_batch(bs, pool, P, Q, rng, device):
    """Recall-batch with keys/values sampled from `pool` (disjoint from other pools)."""
    T = 2 * P + Q
    xs, ys, masks = [], [], []
    for _ in range(bs):
        keys = rng.sample(pool, P)
        vals = rng.sample(pool, P)
        qi = rng.sample(range(P), Q)
        ids = []
        for k, v in zip(keys, vals):
            ids += [k, v]
        for j in qi:
            ids.append(keys[j])
        y = ids[1:] + [1]
        mask = torch.zeros(T, dtype=torch.bool)
        for j, pos in enumerate(range(2 * P, T)):
            mask[pos] = True
            y[pos] = vals[qi[j]]
        xs.append(torch.tensor(ids))
        ys.append(torch.tensor(y))
        masks.append(mask)
    return (torch.stack(xs).to(device), torch.stack(ys).to(device),
            torch.stack(masks).to(device))


@torch.no_grad()
def recall_acc(model, pool, P, Q, rng, device, n=64, is_leaf=True):
    model.eval()
    x, y, mask = make_pool_batch(n, pool, P, Q, rng, device)
    if is_leaf:
        logits, _ = model(x, model.init_states(n, device))
    else:
        logits = model(x)
    pred = logits.argmax(-1)
    hits = ((pred == y) & mask).sum().item()
    model.train()
    return hits / max(1, mask.sum().item())


def run_protocol(args, device, label, is_leaf):
    V, P, Q = args.vocab, args.pairs, args.queries
    poolA = list(range(2, V // 2))
    poolB = list(range(V // 2, V))
    rng = random.Random(args.seed)

    if is_leaf:
        cfg = preset_config("micro", vocab_size=V, n_layers=args.layers,
                            dim=args.dim, d_h=args.d_h, rope_dim=0)
        model = LeafLM(cfg).to(device)
        fwd = lambda x: model(x, model.init_states(x.shape[0], device))[0]
    else:
        model = TinyTransformer(V, dim=args.dim, layers=args.layers).to(device)
        fwd = model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    def step_on(pool, steps):
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            x, y, _ = make_pool_batch(args.batch, pool, P, Q, rng, device)
            logits = fwd(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            loss.backward()
            opt.step()

    # phase A
    step_on(poolA, args.steps_a)
    a0 = recall_acc(model, poolA, P, Q, rng, device, is_leaf=is_leaf)
    # one-cycle: 1 gradient step on ONE B example, then test B
    step_on(poolB, 1)
    b1 = recall_acc(model, poolB, P, Q, rng, device, is_leaf=is_leaf)
    # continue B to args.steps_b total
    step_on(poolB, args.steps_b - 1)
    bK = recall_acc(model, poolB, P, Q, rng, device, is_leaf=is_leaf)
    # retention on A
    aK = recall_acc(model, poolA, P, Q, rng, device, is_leaf=is_leaf)
    print(f"[{label}] A recall={100*a0:5.1f}% | after 1 B-step: B={100*b1:5.1f}% "
          f"| after {args.steps_b} B-steps: B={100*bK:5.1f}%  A(retained)={100*aK:5.1f}%")
    return dict(a0=a0, b1=b1, bK=bK, aK=aK)


def main():
    p = argparse.ArgumentParser(description="LEAFv5 rapid adaptation / continual learning.")
    p.add_argument("--steps-a", type=int, default=500)
    p.add_argument("--steps-b", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--queries", type=int, default=2)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--d-h", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"task: store {args.pairs}, recall {args.queries} | "
          f"pools A=[2..{args.vocab//2}) B=[{args.vocab//2}..{args.vocab}) | "
          f"chance={100.0/args.vocab:.1f}%")
    leaf = run_protocol(args, device, "LEAFv5  ", True)
    trans = run_protocol(args, device, "Transformer", False)
    print("\nsummary:")
    print(f"  absorption after 1 step (B):  LEAFv5 {100*leaf['b1']:.1f}%  vs  "
          f"Transformer {100*trans['b1']:.1f}%")
    print(f"  retention of A after B train:  LEAFv5 {100*leaf['aK']:.1f}%  vs  "
          f"Transformer {100*trans['aK']:.1f}%")


if __name__ == "__main__":
    main()
