"""Controlled validation of LEAFv5's one-shot associative recall (paper sec. 6).

Task: each training sequence stores P random (key -> value) pairs in the
recurrent state, then queries Q of them:   k1 v1 k2 v2 ... kP vP | k2 k4 ...
The model must answer with the stored value for each queried key (the value
does NOT appear after the query in the input, so copying the next token fails).

Keys/values/pair-order are randomized per example, so position-based lookup
cannot solve it — only the delta memory's content-addressable write/read can.

With a working Multi-Timescale Delta Memory, recall accuracy should jump from
~1/V (chance) to >90% within a few hundred training steps.
"""
from __future__ import annotations

import argparse
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import preset_config
from .model import LeafLM


def make_batch(bs: int, V: int, P: int, Q: int, rng: random.Random, device):
    """(x [bs,T], y [bs,T], mask [bs,T]); mask=True at query positions whose
    target is the stored value v_q (not the next input token)."""
    T = 2 * P + Q
    xs, ys, masks = [], [], []
    for _ in range(bs):
        keys = rng.sample(range(2, V), P)
        vals = rng.sample(range(2, V), P)
        query_idx = rng.sample(range(P), Q)          # which pairs get queried
        ids: list[int] = []
        for k, v in zip(keys, vals):
            ids += [k, v]
        for qi in query_idx:
            ids.append(keys[qi])
        x = ids
        y = ids[1:] + [1]                            # default: predict next token
        mask = torch.zeros(T, dtype=torch.bool)
        for qi, pos in enumerate(range(2 * P, T)):   # query positions (last Q)
            mask[pos] = True
            y[pos] = vals[query_idx[qi]]             # target = stored value
        xs.append(torch.tensor(x))
        ys.append(torch.tensor(y))
        masks.append(mask)
    return (torch.stack(xs).to(device), torch.stack(ys).to(device),
            torch.stack(masks).to(device))


class TinyTransformer(nn.Module):
    """Minimal decoder-only Transformer baseline (learned positions)."""

    def __init__(self, V: int, dim: int = 128, layers: int = 3, heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(V, dim)
        self.pos = nn.Parameter(torch.randn(4096, dim) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                dropout=0.0, batch_first=True, norm_first=True,
                activation="gelu"))
        self.head = nn.Linear(dim, V)

    def forward(self, x):
        x = self.emb(x) + self.pos[:x.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--queries", type=int, default=2)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--d-h", type=int, default=32)
    p.add_argument("--arch", choices=["leaf", "trans"], default="leaf")
    p.add_argument("--rope-dim", type=int, default=None,
                   help="fraction of width to rotate (None = full width)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--print-every", type=int, default=100)
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.arch == "leaf":
        cfg = preset_config("micro", vocab_size=args.vocab, n_layers=args.layers,
                            dim=args.dim, d_h=args.d_h, rope_dim=args.rope_dim)
        model = LeafLM(cfg).to(device)
        print(f"[recall] LEAFv5 {model.n_params/1e6:.1f}M params | rope_dim={args.rope_dim if args.rope_dim is not None else 'full'}")
    else:
        model = TinyTransformer(args.vocab, dim=args.dim, layers=args.layers).to(device)
        n = sum(p.numel() for p in model.parameters())
        print(f"[recall] Transformer {n/1e6:.1f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    rng = random.Random(args.seed)

    print(f"        | store {args.pairs}, recall {args.queries} | chance={100.0/args.vocab:.1f}%")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        x, y, mask = make_batch(args.batch, args.vocab, args.pairs, args.queries, rng, device)
        if args.arch == "leaf":
            logits, _ = model(x, model.init_states(args.batch, device))
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab), y.reshape(-1))
        loss.backward()
        opt.step()

        if step % args.print_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                x, y, mask = make_batch(64, args.vocab, args.pairs, args.queries, rng, device)
                if args.arch == "leaf":
                    logits, _ = model(x, model.init_states(64, device))
                else:
                    logits = model(x)
                pred = logits.argmax(-1)
                nq = mask.float().sum().item()
                acc = ((pred == y) & mask).float().sum().item() / max(nq, 1.0)
                # query-only cross-entropy
                lg = logits.reshape(-1, args.vocab)[mask.reshape(-1)]
                tg = y.reshape(-1)[mask.reshape(-1)]
                qloss = F.cross_entropy(lg, tg).item()
            model.train()
            print(f"step {step:5d}  loss={loss.item():.4f}  qloss={qloss:.4f}  "
                  f"recall_acc={100*acc:5.1f}%   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
