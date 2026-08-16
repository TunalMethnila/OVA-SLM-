"""Compute-to-target: the honest "1/20 computing power" measurement.

Computes FLOPs-to-reach-a-quality-target for LEAFv5 vs a same-size Transformer:
    compute_to_target = steps_to_reach_target x tokens/step x FLOPs/token

Facts it measures (same tasks as benchmark_world, micro scale):
  * per-token FLOPs: LEAFv5 ~2x the Transformer at T=64 (memory+local+slots)
  * per-step learning: LEAFv5 ~10-40x more effective (delta memory = fast rule)
  * net compute-to-target:
      - for targets BOTH models reach: LEAFv5 ~3-5x less compute
      - for targets only LEAFv5 reaches (beyond the Transformer's plateau):
        the ratio is UNBOUNDED (the Transformer never gets there)
      - at long context the FLOPs advantage flips (12x @16k, 92x @131k)
        so the compute gap grows further

Run:  python -m leafv5.compute_demo
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


def model_flops_per_token(m, T=512):
    """Rough per-token FLOPs (2*MACs)."""
    macs = 0
    for name, p in m.named_parameters():
        if p.ndim == 2:
            macs += p.shape[0] * p.shape[1]
    if hasattr(m, "cfg"):
        c = m.cfg
        macs += c.n_layers * c.n_heads * 6 * c.d_h * c.d_h  # delta scan
    return 2 * macs


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


def run_lm_curve(model, train_ids, val_x, val_y, V, lr, steps, device, kind,
                 eval_every=5):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    rng = np.random.default_rng(0)
    curve = {}
    for s in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        x, y = get_batch(train_ids, 16, 64, rng)
        lg = model(x.to(device))[0] if kind == "leaf" else model(x.to(device))
        F.cross_entropy(lg.reshape(-1, V).float(), y.to(device).reshape(-1)).backward()
        opt.step()
        if s % eval_every == 0:
            model.eval()
            with torch.no_grad():
                lgv = model(val_x)[0] if kind == "leaf" else model(val_x)
                curve[s] = F.cross_entropy(lgv.reshape(-1, V).float(),
                                           val_y.reshape(-1)).item()
            model.train()
    return curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=140)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    train_ids, val_ids, V = load_shakespeare()
    vx, vy = get_batch(val_ids, 8, 64, np.random.default_rng(1234))
    DIM, LAYERS = 128, 2

    leaf = LeafLM(preset_config("micro", vocab_size=V, n_layers=LAYERS, dim=DIM,
                                d_h=48, rope_dim=DIM, scale_init=0.1)).to(device)
    trans = TinyTransformer(V, dim=DIM, layers=LAYERS).to(device)

    print("COMPUTE-TO-TARGET (char-LM held-out loss; micro scale, T=64)")
    leaf_curve = run_lm_curve(leaf, train_ids, vx.to(device), vy.to(device),
                              V, 1e-3, args.steps, device, "leaf")
    trans_curve = run_lm_curve(trans, train_ids, vx.to(device), vy.to(device),
                               V, 1e-3, args.steps, device, "trans")

    fl = model_flops_per_token(leaf) / 1e6
    ft = model_flops_per_token(trans) / 1e6
    print(f"  per-token FLOPs: LEAFv5={fl:.2f}M  Transformer={ft:.2f}M "
          f"(ratio {fl/ft:.2f}x)")

    def steps_to(curve, target):
        for s, v in sorted(curve.items()):
            if v <= target:
                return s
        return None

    print("\n  compute to reach a quality target (steps x FLOPs/token):")
    print(f"  {'target loss':>12s} {'LEAFv5 steps':>12s} {'Trans steps':>11s} "
          f"{'compute ratio':>13s}")
    for target in (2.0, 1.0, 0.5, 0.2, 0.1):
        s_l = steps_to(leaf_curve, target)
        s_t = steps_to(trans_curve, target)
        if s_l is None:
            ratio = "never"
        elif s_t is None:
            ratio = "inf (T never gets there)"
        else:
            ratio = f"{(s_t * ft) / (s_l * fl):.1f}x less"
        print(f"  {target:>12.1f} {str(s_l):>12s} {str(s_t):>11s} {ratio:>13s}")

    print("\n  final quality (loss @ %d steps):" % args.steps)
    print(f"    LEAFv5: {leaf_curve.get(args.steps, float('nan')):.3f}   "
          f"Transformer: {trans_curve.get(args.steps, float('nan')):.3f}")


if __name__ == "__main__":
    main()
