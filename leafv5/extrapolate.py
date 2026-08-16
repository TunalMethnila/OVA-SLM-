"""Length extrapolation: train SHORT, evaluate LONG.

A classic limit: models trained at seq S often degrade at seq >> S because
their position encoding (RoPE / learned positions) never saw long offsets and
the mixer never saw long context.

LEAFv5's delta memory is position-agnostic (rope_dim=0: no rotation at all) and
its local convs are local by construction, so it should extrapolate almost
flat.  With full RoPE it degrades like a Transformer.  Measured here on the
same char-LM: train at seq=64, eval at 64..2048.

Run:  python -m leafv5.extrapolate [--steps 150]
"""
from __future__ import annotations

import argparse
import os
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

from .config import preset_config
from .model import LeafLM
from .recall_demo import TinyTransformer


def load_shakespeare():
    cache = os.path.join("data_cache", "tinyshakespeare.txt")
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        req = urllib.request.Request(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
            "tinyshakespeare/input.txt", headers={"User-Agent": "leafv5/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r, open(cache, "wb") as f:
            f.write(r.read())
    text = open(cache).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    n = len(ids)
    n_val = int(0.05 * n)
    return ids[:n - n_val], ids[n - n_val:], len(chars)


def get_batch(arr, bs, seq, rng):
    offs = rng.integers(0, len(arr) - seq - 1, size=bs)
    return (torch.from_numpy(np.stack([arr[o:o + seq] for o in offs])),
            torch.from_numpy(np.stack([arr[o + 1:o + seq + 1] for o in offs])))


@torch.no_grad()
def eval_at(model, val_ids, V, seq, device, kind, bs=2):
    vx, vy = get_batch(val_ids, bs, seq, np.random.default_rng(1234))
    model.eval()
    lg = model(vx.to(device))[0] if kind == "leaf" else model(vx.to(device))
    model.train()
    return F.cross_entropy(lg.reshape(-1, V).float(), vy.to(device).reshape(-1)).item()


def train(model, train_ids, V, steps, seq, device, kind, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    rng = np.random.default_rng(0)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        x, y = get_batch(train_ids, 16, seq, rng)
        lg = model(x.to(device))[0] if kind == "leaf" else model(x.to(device))
        F.cross_entropy(lg.reshape(-1, V).float(), y.to(device).reshape(-1)).backward()
        opt.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--train-seq", type=int, default=64)
    p.add_argument("--eval-seqs", type=str, default="64,128,256,512,1024")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_seqs = [int(x) for x in args.eval_seqs.split(",")]

    torch.manual_seed(0)
    train_ids, val_ids, V = load_shakespeare()
    DIM, LAYERS = 128, 2

    print("LENGTH EXTRAPOLATION: train at seq=%d, eval at longer seq (char-LM "
          "held-out loss, lower=better)" % args.train_seq)

    models = [
        ("LEAFv5 (rope off)", "leaf",
         LeafLM(preset_config("micro", vocab_size=V, n_layers=LAYERS, dim=DIM,
                              d_h=48, rope_dim=0, scale_init=0.1)).to(device),
         1e-3),
        ("LEAFv5 (rope full)", "leaf",
         LeafLM(preset_config("micro", vocab_size=V, n_layers=LAYERS, dim=DIM,
                              d_h=48, rope_dim=DIM, scale_init=0.1)).to(device),
         1e-3),
        ("Transformer", "trans",
         TinyTransformer(V, dim=DIM, layers=LAYERS).to(device), 1e-3),
    ]
    print(f"  {'model':<20s} " + "  ".join(f"s{s}" for s in eval_seqs))
    for name, kind, m, lr in models:
        train(m, train_ids, V, args.steps, args.train_seq, device, kind, lr)
        row = []
        for s in eval_seqs:
            row.append(f"{eval_at(m, val_ids, V, s, device, kind):.3f}")
        print(f"  {name:<20s} " + "  ".join(row))

    # extrapolation ratio: loss at 1024 vs 64 (1.0 = perfect extrapolation)
    print("\n  extrapolation ratio (loss@1024 / loss@64; 1.0 = perfect):")
    for name, kind, m, lr in models:
        l64 = eval_at(m, val_ids, V, 64, device, kind)
        l1024 = eval_at(m, val_ids, V, 1024, device, kind)
        print(f"    {name:<20s} {l1024/l64:.2f}x")


if __name__ == "__main__":
    main()
