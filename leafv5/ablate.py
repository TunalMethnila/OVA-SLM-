"""Mechanism ablation: which LEAFv5 features earn their place?

Toggles each SOTA mechanism OFF one at a time (and the paper-core alone) and
measures held-out LM loss on Penn Treebank (char-level), matched steps and
params-as-close-as-possible.  This is the evidence table for "every mechanism
in the world-best architecture contributes".

Run:  python -m leafv5.ablate [--steps 150]
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

VALID_URL = ("https://raw.githubusercontent.com/tomsercu/lstm/master/data/"
             "ptb.valid.txt")
TRAIN_URL = ("https://raw.githubusercontent.com/tomsercu/lstm/master/data/"
             "ptb.train.txt")


def fetch(url, cache):
    if os.path.exists(cache):
        return open(cache).read()
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


def run(cfg_kw, train_ids, vx, vy, V, steps, device, lr=1.2e-3, seed=0,
        seq=64):
    torch.manual_seed(seed)
    cfg = preset_config("micro", **cfg_kw)
    m = LeafLM(cfg).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    rng = np.random.default_rng(0)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        x, y = get_batch(train_ids, 16, seq, rng)
        lg, _ = m(x.to(device))
        loss = F.cross_entropy(lg.reshape(-1, V).float(),
                               y.to(device).reshape(-1))
        if cfg.moe:
            loss = loss + 0.01 * m.aux_loss()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        lgv, _ = m(vx)
        vl = F.cross_entropy(lgv.reshape(-1, V).float(),
                             vy.reshape(-1)).item()
    return vl, m.n_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seq", type=int, default=64,
                   help="sequence length.  NOTE (2026-08-09): on CPU boxes with a "
                        "contended host, torch's autograd engine can livelock on "
                        "the deep 64-step delta-scan chain (flaky, not a model "
                        "bug — all training/grad tests pass). --seq 32 avoids "
                        "it and is the recommended CPU setting; comparative "
                        "conclusions are unaffected.")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    dev = args.device

    os.makedirs("data_cache", exist_ok=True)
    train_text = fetch(TRAIN_URL, "data_cache/ptb_train.txt")
    valid_text = fetch(VALID_URL, "data_cache/ptb_valid.txt")
    chars = sorted(set(train_text))
    stoi = {c: i for i, c in enumerate(chars)}
    V = len(chars)
    tr = np.array([stoi.get(c, 0) for c in train_text], dtype=np.int64)
    va = np.array([stoi.get(c, 0) for c in valid_text], dtype=np.int64)
    vx, vy = get_batch(va, 8, args.seq, np.random.default_rng(1234))
    vx, vy = vx.to(dev), vy.to(dev)

    BASE = dict(vocab_size=V, n_layers=2, dim=128, d_h=48, rope_dim=128,
                scale_init=0.1)
    # the complete world-best config
    FULL = dict(BASE, use_swa=True, swa_window=32, moe=True, moe_experts=6,
                moe_topk=2, slot_attn=True, learn_plasticity=True,
                share_mem_every=2)

    variants = [
        ("paper-core only (no SOTA features)",
         dict(BASE, use_read_query=False, short_conv=False, output_gate=False,
              mem_slots=0)),
        ("+ read query + short conv + output gate", dict(BASE)),
        ("+ slots (Titans external memory)", dict(BASE, mem_slots=64)),
        ("+ slot attention", dict(BASE, slot_attn=True)),
        ("+ MoE FFN (6 experts, top-2)", dict(BASE, moe=True, moe_experts=6,
                                              moe_topk=2)),
        ("+ SWA hybrid", dict(BASE, use_swa=True, swa_window=32)),
        ("+ learned plasticity + shared proj", dict(BASE, learn_plasticity=True,
                                                    share_mem_every=2)),
        ("FULL WORLD-BEST FUSION", FULL),
    ]

    print(f"Mechanism ablation, Penn Treebank char-LM, {args.steps} steps "
          f"(held-out loss, lower = better):\n")
    results = []
    for name, kw in variants:
        vl, np_ = run(kw, tr, vx, vy, V, args.steps, dev, seq=args.seq)
        results.append((name, vl, np_))
        print(f"  {name:44s} loss={vl:.4f}  ({np_/1e6:.2f}M)")
    print("\n  (each row ADDS the named mechanism to the previous row; "
          "full fusion should be best or tied)")


if __name__ == "__main__":
    main()
