"""Standard-corpus perplexity: LEAFv5 vs Transformer vs Mamba-lite RNN on
WikiText-2 (char-level), matched steps.  More credible than the toy corpora.

Run:  python -m leafv5.benchmark_ppl [--steps 150]
"""
from __future__ import annotations

import argparse
import math
import os
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

from .config import preset_config
from .model import LeafLM
from .recall_demo import TinyTransformer
from .benchmark_world import GatedRNN

TRAIN_URL = ("https://raw.githubusercontent.com/tomsercu/lstm/master/data/"
             "ptb.train.txt")
VALID_URL = ("https://raw.githubusercontent.com/tomsercu/lstm/master/data/"
             "ptb.valid.txt")


def fetch(url, cache):
    if os.path.exists(cache):
        return open(cache).read()
    print(f"  fetching {url.split('/')[-1]} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "leafv5/0.1"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = r.read().decode("utf-8", errors="ignore")
    with open(cache, "w") as f:
        f.write(data)
    return data


def get_batch(arr, bs, seq, rng):
    offs = rng.integers(0, len(arr) - seq - 1, size=bs)
    return (torch.from_numpy(np.stack([arr[o:o + seq] for o in offs])),
            torch.from_numpy(np.stack([arr[o + 1:o + seq + 1] for o in offs])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("data_cache", exist_ok=True)
    train_text = fetch(TRAIN_URL, "data_cache/ptb_train.txt")
    valid_text = fetch(VALID_URL, "data_cache/ptb_valid.txt")
    chars = sorted(set(train_text))
    stoi = {c: i for i, c in enumerate(chars)}
    V = len(chars)
    tr = np.array([stoi.get(c, 0) for c in train_text], dtype=np.int64)
    va = np.array([stoi.get(c, 0) for c in valid_text], dtype=np.int64)
    print(f"[data] Penn Treebank char: vocab={V}, train {len(tr)/1e6:.1f}M chars, "
          f"valid {len(va)/1e6:.1f}M")

    vx, vy = get_batch(va, 8, args.seq, np.random.default_rng(1234))
    vx, vy = vx.to(device), vy.to(device)

    torch.manual_seed(0)
    DIM, LAYERS = 128, 2
    models = [
        ("LEAFv5", "leaf", LeafLM(preset_config("micro", vocab_size=V,
            n_layers=LAYERS, dim=DIM, d_h=48, rope_dim=DIM,
            scale_init=0.1)).to(device), 1e-3),
        ("Transformer", "trans", TinyTransformer(V, dim=DIM,
            layers=LAYERS).to(device), 1e-3),
        ("GatedRNN", "rnn", GatedRNN(V, dim=DIM, layers=LAYERS).to(device),
         2e-3),
    ]
    print(f"\nPenn Treebank char PPL at {args.steps} steps (lower = better):")
    print(f"  {'model':<12s} {'train loss':>10s} {'valid PPL':>10s}")
    for name, kind, m, lr in models:
        opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95),
                                weight_decay=0.0)
        rng = np.random.default_rng(0)
        train_loss = None
        for _ in range(args.steps):
            opt.zero_grad(set_to_none=True)
            x, y = get_batch(tr, 16, args.seq, rng)
            lg = m(x.to(device))[0] if kind == "leaf" else m(x.to(device))
            loss = F.cross_entropy(lg.reshape(-1, V).float(),
                                   y.to(device).reshape(-1))
            loss.backward()
            opt.step()
            train_loss = loss.item()
        m.eval()
        with torch.no_grad():
            lgv = m(vx)[0] if kind == "leaf" else m(vx)
            vl = F.cross_entropy(lgv.reshape(-1, V).float(),
                                 vy.reshape(-1)).item()
        print(f"  {name:<12s} {train_loss:>10.3f} {math.exp(vl):>10.1f}")


if __name__ == "__main__":
    main()
