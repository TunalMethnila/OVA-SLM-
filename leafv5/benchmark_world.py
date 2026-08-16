"""World-class benchmark: LEAFv5 vs Transformer vs Mamba-family (gated RNN).

Runs the same tasks on same-size models and prints the table that backs the
"world-class" claim in research/world-class.md:
  1. few-step associative recall (held-out accuracy vs gradient steps)
  2. char-LM held-out loss vs steps (Tiny Shakespeare)
  3. params + FLOPs/token

Run:  python -m leafv5.benchmark_world [--steps 20]
"""
from __future__ import annotations

import argparse
import os
import random
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import preset_config
from .model import LeafLM
from .recall_demo import TinyTransformer


# ---------------------------------------------------------------------------
# Mamba-family-lite baseline: gated linear RNN (short conv + per-channel decay)
# ---------------------------------------------------------------------------
class GatedRNN(nn.Module):
    """Mamba2-family representative (no selective scan; per-channel decay a_t):
        h_t = a_t * h_{t-1} + (1 - a_t) * SiLU(conv1d(x_t))
    with a residual SwiGLU FFN per layer.  State h reset per sequence."""

    def __init__(self, vocab: int, dim: int = 128, layers: int = 2, ff_mult: int = 4):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                "conv": nn.Conv1d(dim, dim, 3, padding=1, groups=dim, bias=False),
                "a_gate": nn.Linear(dim, dim, bias=False),   # decay logit
                "act": nn.SiLU(),
                "norm1": nn.LayerNorm(dim),
                "w1": nn.Linear(dim, dim * ff_mult),
                "w2": nn.Linear(dim, dim * ff_mult),
                "w3": nn.Linear(dim * ff_mult, dim),
            }))
        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.emb.weight  # tie

    def forward(self, x: torch.Tensor):
        B, T = x.shape
        h = x.new_zeros(B, self.dim)
        out = []
        for t in range(T):
            xin = self.emb(x[:, t])                                  # [B,D]
            for L in self.layers:
                xin = L["conv"](xin.unsqueeze(-1)).squeeze(-1)
                a = torch.sigmoid(L["a_gate"](xin))
                h = a * h + (1 - a) * L["act"](xin)
                xin = h
                hn = L["norm1"](xin)
                xin = xin + L["w3"](F.silu(L["w2"](hn)) * L["w1"](hn))
                h = xin
            out.append(xin)
        o = torch.stack(out, 1)
        return self.head(self.norm_f(o))


# ---------------------------------------------------------------------------
# shared helpers (recall + LM)
# ---------------------------------------------------------------------------
def make_recall_batch(bs, V, P, Q, rng, device):
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
def recall_heldout(model, V, P, Q, device, n=256, kind="leaf"):
    rng = random.Random(999)
    x, y, m = make_recall_batch(n, V, P, Q, rng, device)
    model.eval()
    if kind == "leaf":
        lg = model(x, model.init_states(n, device))[0]
    else:
        lg = model(x)
    model.train()
    return 100.0 * (((lg.argmax(-1) == y) & m).float().sum() / m.float().sum()).item()


def recall_race(model, V, P, Q, lr, steps, batch, device, kind, seed=0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    rng = random.Random(seed)
    res = {}
    for s in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        x, y, m = make_recall_batch(batch, V, P, Q, rng, device)
        if kind == "leaf":
            lg = model(x, model.init_states(batch, device))[0]
        else:
            lg = model(x)
        F.cross_entropy(lg.reshape(-1, V)[m.reshape(-1)],
                        y.reshape(-1)[m.reshape(-1)]).backward()
        opt.step()
        if s in (1, 3, 5, 10, 20):
            res[s] = recall_heldout(model, V, P, Q, device, kind=kind)
    return res


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


def lm_race(model, train_ids, val_x, val_y, V, lr, steps, device, kind):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    rng = np.random.default_rng(0)
    res = {}
    for s in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        x, y = get_batch(train_ids, 16, 64, rng)
        if kind == "leaf":
            lg = model(x)[0]
        else:
            lg = model(x)
        F.cross_entropy(lg.reshape(-1, V).float(), y.reshape(-1)).backward()
        opt.step()
        if s in (20, 60, 120):
            model.eval()
            with torch.no_grad():
                lgv = model(val_x)[0] if kind == "leaf" else model(val_x)
                res[s] = round(F.cross_entropy(lgv.reshape(-1, V).float(),
                                               val_y.reshape(-1)).item(), 3)
            model.train()
    return res


def model_flops(m, V, T=512):
    """Rough per-token FLOPs (2*MACs) for the LM forward at seq T."""
    # count Linear MACs from weight shapes + recurrence/attention extra
    macs = 0
    for name, p in m.named_parameters():
        if p.ndim == 2:
            macs += p.shape[0] * p.shape[1]  # per token for most
    # recurrence/attention per-token extra
    if hasattr(m, "cfg"):  # LEAFv5: delta scan matvecs
        c = m.cfg
        macs += c.n_layers * c.n_heads * 6 * c.d_h * c.d_h
    else:
        for L in m.layers if isinstance(m, GatedRNN) else m.blocks:
            macs += 0
    return 2 * macs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    DIM, LAYERS, V_R = 128, 2, 64

    print("=" * 78)
    print("WORLD-CLASS BENCHMARK: LEAFv5 vs Transformer vs Mamba-family")
    print("=" * 78)

    # ---- build models ----
    cfg = preset_config("micro", vocab_size=V_R, n_layers=LAYERS, dim=DIM,
                        d_h=48, rope_dim=0, scale_init=0.1)
    leaf = LeafLM(cfg).to(device)
    trans = TinyTransformer(V_R, dim=DIM, layers=LAYERS).to(device)
    rnn = GatedRNN(V_R, dim=DIM, layers=LAYERS).to(device)
    print(f"\nparams: LEAFv5={leaf.n_params/1e6:.2f}M  "
          f"Transformer={sum(x.numel() for x in trans.parameters())/1e6:.2f}M  "
          f"GatedRNN={sum(x.numel() for x in rnn.parameters())/1e6:.2f}M")

    # ---- 1) recall race (store-2/query-1) ----
    print("\n[1] few-step associative recall, store-2/query-1, held-out %:")
    P, Q = 2, 1
    lr_map = {"leaf": 2e-2, "trans": 1e-3, "rnn": 2e-3}
    res = {}
    for name, m, kind in [("LEAFv5", leaf, "leaf"),
                          ("Transformer", trans, "trans"),
                          ("GatedRNN", rnn, "rnn")]:
        res[name] = recall_race(m, V_R, P, Q, lr_map[kind], args.steps, 64,
                                device, kind)
        print(f"  {name:12s}: " + "  ".join(
            f"s{k}:{v:.0f}%" for k, v in res[name].items()))

    # ---- 2) char-LM race ----
    print("\n[2] char-LM held-out loss (Tiny Shakespeare), lower better:")
    train_ids, val_ids, V_LM = load_shakespeare()
    vx, vy = get_batch(val_ids, 16, 64, np.random.default_rng(1234))
    # rebuild models with LM vocab (char)
    cfg2 = preset_config("micro", vocab_size=V_LM, n_layers=LAYERS, dim=DIM,
                         d_h=48, rope_dim=DIM, scale_init=0.1)
    leaf2 = LeafLM(cfg2).to(device)
    trans2 = TinyTransformer(V_LM, dim=DIM, layers=LAYERS).to(device)
    rnn2 = GatedRNN(V_LM, dim=DIM, layers=LAYERS).to(device)
    lr_map2 = {"leaf": 1e-3, "trans": 1e-3, "rnn": 2e-3}
    for name, m, kind in [("LEAFv5", leaf2, "leaf"),
                          ("Transformer", trans2, "trans"),
                          ("GatedRNN", rnn2, "rnn")]:
        r = lm_race(m, train_ids, vx.to(device), vy.to(device), V_LM,
                    lr_map2[kind], 120, device, kind)
        print(f"  {name:12s}: " + "  ".join(f"s{k}:{v}" for k, v in r.items()))

    # ---- 3) FLOPs ----
    print("\n[3] rough per-token FLOPs (all layers, LM):")
    for name, m in [("LEAFv5", leaf2), ("Transformer", trans2), ("GatedRNN", rnn2)]:
        print(f"  {name:12s}: {model_flops(m, V_LM)/1e6:.1f}M/token")


if __name__ == "__main__":
    main()
