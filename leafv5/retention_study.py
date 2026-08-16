"""retention_study.py — which lever fixes long-range retention?  (Tier-1 #1)

Task (the paper's long-range memory test): store P (key -> value) pairs, then
D DISTRACTOR tokens, then query one key.  Held-out accuracy vs distance D.
Random chance = 1/|V| (a model that simply forgot the pairs scores chance).

Levers compared at matched steps/params-as-close-as-possible:
  0. baseline (micro, d_h=48)
  1. + surprise-gated writes (novelty suppresses redundant writes)
  2. + d_h=96  (more state capacity per head)
  3. + input decay (Gated-DeltaNet-style a_t)
  4. + SWA hybrid every 2 layers (Mistral/GatedDeltaNet-H style)

Everything is measured in this repo; run with --steps (default 250).

Run:  python -m leafv5.retention_study [--steps 250] [--seed 0]
"""
from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import preset_config
from .model import LeafLM


def make_batch(bs, V, P, D, rng, device):
    """Store P pairs, D distractors, query 1 key.  Returns (x, y, mask)."""
    T = 2 * P + D + 1
    xs, ys, masks = [], [], []
    for _ in range(bs):
        keys = rng.sample(range(2, V), P)
        vals = rng.sample(range(2, V), P)
        qi = rng.randrange(P)
        ids = []
        for k, v in zip(keys, vals):
            ids += [k, v]
        ids += [rng.randrange(2, V) for _ in range(D)]   # distractors
        ids.append(keys[qi])                             # query
        y = ids[1:] + [1]
        m = torch.zeros(T, dtype=torch.bool)
        m[-1] = True
        y[-1] = vals[qi]
        xs.append(torch.tensor(ids))
        ys.append(torch.tensor(y))
        masks.append(m)
    return (torch.stack(xs).to(device), torch.stack(ys).to(device),
            torch.stack(masks).to(device))


@torch.no_grad()
def heldout_acc(model, V, P, D, device, n=32):
    """Held-out accuracy at distance D.  n is small: the scan materializes
    [B*H, T, d_h] buffers, so long D with a big batch can OOM low-RAM boxes
    (T=1033 at D=1024 -> keep n modest)."""
    rng = random.Random(999)
    x, y, m = make_batch(n, V, P, D, rng, device)
    model.eval()
    lg, _ = model(x, model.init_states(n, device))
    model.train()
    acc = 100.0 * (((lg.argmax(-1) == y) & m).float().sum() / m.float().sum()).item()
    del x, y, m, lg
    return acc


CONFIGS = [
    ("baseline        ", dict(dim=128, n_layers=2, d_h=48)),
    ("+surprise-gate  ", dict(dim=128, n_layers=2, d_h=48, surprise_gate=True)),
    ("+d_h=96         ", dict(dim=128, n_layers=2, d_h=96)),
    ("+input-decay    ", dict(dim=128, n_layers=2, d_h=48, input_decay=True)),
    ("+SWA every 2    ", dict(dim=128, n_layers=2, d_h=48, use_swa=True,
                              swa_every=2, swa_window=32)),
]


def run(seed, steps, P, distances, bs, device="cpu", only=None, distractors=32):
    torch.manual_seed(seed)
    V = 64
    configs = CONFIGS if only is None else [c for c in CONFIGS if only in c[0]]
    print("=" * 72)
    print(f"RETENTION STUDY  |  store {P} pairs + D distractors + query  "
          f"| chance {100/V:.1f}%  | {steps} steps, seed {seed}")
    print("=" * 72)
    results = {}
    for label, kw in configs:
        t0 = time.time()
        rng = random.Random(seed)
        cfg = preset_config("micro", vocab_size=V, rope_dim=0, scale_init=0.2,
                            **kw)
        m = LeafLM(cfg).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.95))
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            x, y, mask = make_batch(bs, V, P, distractors, rng, device)
            lg, _ = m(x, m.init_states(bs, device))
            loss = F.cross_entropy(lg.reshape(-1, V)[mask.reshape(-1)],
                                   y.reshape(-1)[mask.reshape(-1)])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        accs = {d: heldout_acc(m, V, P, d, device) for d in distances}
        results[label] = accs
        row = "  ".join(f"D={d}:{accs[d]:5.1f}%" for d in distances)
        print(f"  {label}  {row}  ({time.time()-t0:.0f}s)", flush=True)
        del m, opt
        torch.cuda.empty_cache() if device.startswith("cuda") else None
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--distractors", type=int, default=32,
                   help="distractor tokens between store and query (difficulty; "
                        "32 = hard/clobbering, 8 = easy)")
    p.add_argument("--distances", type=str, default="64,256,1024")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--device", default="cpu")
    p.add_argument("--only", type=str, default=None,
                   help="run a single config substring (e.g. 'surprise') "
                        "in its own process, to bound peak memory")
    args = p.parse_args()
    distances = [int(d) for d in args.distances.split(",")]
    run(args.seed, args.steps, args.pairs, distances, args.batch, args.device,
        only=args.only, distractors=args.distractors)


if __name__ == "__main__":
    main()
