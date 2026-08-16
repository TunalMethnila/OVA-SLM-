"""MINI-RESEARCH sweep C: store-1 recall, push to 100% by step <=10.
Levers: queries per pair (Q=1 vs Q=2 on one pair), batch, LR, scale_init, beta1,
structured init.  Eval with n=256 for reliable near-100 numbers.
Run:  python research/sweep_c.py
"""
import gc
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.config import preset_config
from leafv5.model import LeafLM


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


def heldout(model, V, P, Q, device, n=256):
    rng = random.Random(999)
    x, y, m = make_batch(n, V, P, Q, rng, device)
    model.eval()
    with torch.no_grad():
        lg, _ = model(x, model.init_states(n, device))
    model.train()
    return 100.0 * (((lg.argmax(-1) == y) & m).float().sum() / m.float().sum()).item()


def structured_init(model, D, dh):
    H = model.cfg.n_heads
    with torch.no_grad():
        for blk in model.blocks:
            wk, wv = blk.memory.wk, blk.memory.wv
            wk.weight.zero_()
            wv.weight.zero_()
            for h in range(H):
                i0 = h * dh
                if i0 + dh <= D:
                    wk.weight[i0:i0 + dh, i0:i0 + dh] = torch.eye(dh)
                    wv.weight[i0:i0 + dh, i0:i0 + dh] = torch.eye(dh)


def run(V, P, Q, dim, dh, lr, si, batch, steps, struc=False, beta1=0.9, seed=0):
    cfg = preset_config("micro", vocab_size=V, n_layers=2, dim=dim, d_h=dh,
                        rope_dim=0, scale_init=si)
    model = LeafLM(cfg)
    if struc:
        structured_init(model, dim, dh)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(beta1, 0.95),
                            weight_decay=0.0)
    rng = random.Random(seed)
    res = {}
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        x, y, m = make_batch(batch, V, P, Q, rng, "cpu")
        lg, _ = model(x, model.init_states(batch, "cpu"))
        F.cross_entropy(lg.reshape(-1, V)[m.reshape(-1)],
                        y.reshape(-1)[m.reshape(-1)]).backward()
        opt.step()
        if step in (1, 3, 5, 10, 15, 20):
            res[step] = heldout(model, V, P, Q, "cpu")
    return res


if __name__ == "__main__":
    print("Sweep C: store-1 V=64, held-out n=256 (target: 100 by step <=10)", flush=True)
    confs = [
        ("P1Q1 b128 lr1e-2 si.1",       1, 1, 1e-2, 0.1, 128, False, 0.9),
        ("P1Q1 b128 lr2e-2 si.2",       1, 1, 2e-2, 0.2, 128, False, 0.9),
        ("P1Q1 b128 lr3e-2 si.2",       1, 1, 3e-2, 0.2, 128, False, 0.9),
        ("P1Q1 b128 lr3e-2 si.3",       1, 1, 3e-2, 0.3, 128, False, 0.9),
        ("P1Q1 b128 lr3e-2 si.3 b1.8",  1, 1, 3e-2, 0.3, 128, False, 0.8),
        ("P1Q1 b128 lr3e-2 si.3 STRUC", 1, 1, 3e-2, 0.3, 128, True, 0.9),
        ("P1Q2 b128 lr2e-2 si.2",       1, 2, 2e-2, 0.2, 128, False, 0.9),
        ("P1Q2 b128 lr3e-2 si.3",       1, 2, 3e-2, 0.3, 128, False, 0.9),
    ]
    for name, P, Q, lr, si, batch, struc, beta1 in confs:
        t0 = time.time()
        r = run(64, P, Q, 128, 64, lr, si, batch, 20, struc=struc, beta1=beta1)
        print(f"{name:26s}: " + " ".join(f"{k}:{v:.0f}" for k, v in r.items())
              + f"  ({time.time()-t0:.0f}s)", flush=True)
        gc.collect()
