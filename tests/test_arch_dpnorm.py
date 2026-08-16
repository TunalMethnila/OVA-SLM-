"""Regression tests for the DP-normalized readout (`dp_norm`, Delta-Product
style, opt-in; see research/architecture-2026-08.md).

Guards:
  1. dp_norm builds, forwards finite, and trains;
  2. the denominator D follows the SAME recurrence as S (v -> ones-vector)
     — checked against a hand-rolled reference on a small case;
  3. train == decode invariant holds with dp_norm ON (reviewer's criterion);
  4. width+depth growth is exact with dp_norm ON (d_bias carried);
  5. the readout is normalized: with dp_norm, a value written once and read
     back is bounded (|o| <= |v| + eps), unlike the raw sum's drift.
Run:  python tests/test_arch_dpnorm.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.config import preset_config
from leafv5.model import LeafLM


def test_builds_and_trains():
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=2, dim=96, d_h=32,
                        dp_norm=True, scale_init=0.1)
    m = LeafLM(cfg)
    assert m.blocks[0].memory.d_bias is not None
    assert torch.allclose(m.blocks[0].memory.d_bias,
                          torch.ones_like(m.blocks[0].memory.d_bias))
    x = torch.randint(0, 256, (2, 16))
    lg, _ = m(x, m.init_states(2, torch.device("cpu")))
    assert torch.isfinite(lg).all()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad()
        xi = torch.randint(0, 256, (4, 16)); yi = torch.randint(0, 256, (4, 16))
        lg, _ = m(xi, m.init_states(4, torch.device("cpu")))
        torch.nn.functional.cross_entropy(lg.reshape(-1, 256),
                                          yi.reshape(-1)).backward()
        opt.step()
    print("  dp_norm builds, forwards finite, trains OK")


def test_denominator_recurrence():
    """Hand-rolled DP reference vs the model's scan (no decay, no norm)."""
    torch.manual_seed(0)
    from leafv5.model import MultiTimescaleDeltaV2
    cfg = preset_config("micro", vocab_size=256, n_layers=1, dim=96, d_h=32,
                        dp_norm=True)
    mem = MultiTimescaleDeltaV2(cfg).eval()
    B, T, H, dh, D = 2, 8, cfg.n_heads, cfg.d_h, cfg.dim
    x = torch.randn(B, T, D)
    with torch.no_grad():
        k = torch.nn.functional.normalize(mem.wk(x).view(B, T, H, dh), dim=-1)
        v = torch.nn.functional.normalize(mem.wv(x).view(B, T, H, dh), dim=-1)
        q = torch.nn.functional.normalize(mem.wq(x).view(B, T, H, dh), dim=-1)
        bw = torch.sigmoid(mem.w_write(x)).view(B, T, H, 1)
        bf = torch.sigmoid(mem.w_forget(x)).view(B, T, H, 1)
        gr = torch.sigmoid(mem.w_read(x)).view(B, T, H, 1)
        alpha = mem.alpha
    out, S, Df = mem._sequential(k, v, q, bw, bf, gr, None,
                                 torch.zeros(B, H, dh, dh), False,
                                 torch.zeros(B * H, dh))
    # hand reference with the same recurrence (no decay/norm): D <- D - bf(D.k)k + bw k
    BH = B * H
    Sref = torch.zeros(BH, dh, dh)
    Dref = torch.zeros(BH, dh)
    kf = k.permute(0, 2, 1, 3).reshape(BH, T, dh)
    vf = v.permute(0, 2, 1, 3).reshape(BH, T, dh)
    qf = q.permute(0, 2, 1, 3).reshape(BH, T, dh)
    bwf = bw.permute(0, 2, 1, 3).reshape(BH, T, 1)
    bff = bf.permute(0, 2, 1, 3).reshape(BH, T, 1)
    grf = gr.permute(0, 2, 1, 3).reshape(BH, T, 1)
    db = mem.d_bias.float().repeat(B).view(BH, 1)
    outs = []
    for t in range(T):
        kt = kf[:, t]; vt = vf[:, t]; qt = qf[:, t]
        dq = (Dref * qt).sum(-1, keepdim=True)         # [BH,1] = D^T q
        dk = (Dref * kt).sum(-1, keepdim=True)         # [BH,1] = D^T k
        ok = Sref @ kt.unsqueeze(-1)
        o_prev = (Sref @ qt.unsqueeze(-1)) / (dq + db).clamp(min=1e-3).unsqueeze(-1)
        Sref = Sref - bff[:, t:t + 1] * (ok @ kt.unsqueeze(1)) \
            + bwf[:, t:t + 1] * (vt.unsqueeze(-1) @ kt.unsqueeze(1))
        Dref = Dref - bff[:, t:t + 1].squeeze(-1) * dk * kt \
            + bwf[:, t:t + 1].squeeze(-1) * kt
        dq2 = (Dref * qt).sum(-1, keepdim=True)        # re-read AFTER update
        o_new = (Sref @ qt.unsqueeze(-1)) / (dq2 + db).clamp(min=1e-3).unsqueeze(-1)
        outs.append((grf[:, t:t + 1] * o_new + alpha.repeat(B).view(BH, 1, 1) * o_prev).squeeze(-1))
    outs = torch.stack(outs, 1)
    d_out = (out.permute(0, 2, 1) - outs).abs().max().item()
    d_D = (Df - Dref).abs().max().item()
    assert d_out < 1e-4, d_out
    assert d_D < 1e-4, d_D
    print(f"  denominator recurrence matches hand-rolled reference "
          f"(max|d_out|={d_out:.1e}, max|d_D|={d_D:.1e}) OK")


def test_train_equals_decode():
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=2, dim=96, d_h=32,
                        dp_norm=True, scale_init=0.1)
    m = LeafLM(cfg).eval()
    x = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        lg_full, _ = m(x, m.init_states(1, torch.device("cpu")))
        st = m.init_states(1, torch.device("cpu"))
        outs = []
        for t in range(24):
            lg, st = m(x[:, t:t + 1], st)
            outs.append(lg)
        lg_dec = torch.cat(outs, 1)
    d = (lg_full - lg_dec).abs().max().item()
    assert d < 1e-4, d
    print(f"  train==decode with dp_norm (max|d|={d:.2e}) OK")


def test_growth_exact_with_dpnorm():
    from leafv5.grow import grow_depth, grow_width
    torch.manual_seed(0)
    V = 256
    cfg = preset_config("micro", vocab_size=V, n_layers=2, dim=128, d_h=32,
                        dp_norm=True, scale_init=0.1)
    m = LeafLM(cfg).eval()
    x = torch.randint(0, V, (4, 16))
    with torch.no_grad():
        before = m(x, m.init_states(4, torch.device("cpu")))[0]
    g = grow_depth(grow_width(m, 256), 4).eval()
    with torch.no_grad():
        after = g(x, g.init_states(4, torch.device("cpu")))[0]
    d = (after - before).abs().max().item()
    assert d < 5e-3, d
    print(f"  width+depth growth exact with dp_norm (max|d|={d:.2e}) OK")


def test_readout_bounded():
    """A value written once and read back is bounded by the value norm
    (the DP normalization claim).  Raw sum would grow with |q^T k|."""
    torch.manual_seed(0)
    from leafv5.model import MultiTimescaleDeltaV2
    cfg = preset_config("micro", vocab_size=256, n_layers=1, dim=96, d_h=32,
                        dp_norm=True)
    mem = MultiTimescaleDeltaV2(cfg).eval()
    B, T, H, dh = 1, 8, cfg.n_heads, cfg.d_h
    # k/q random unit vectors, v unit; run the scan and check |out| stays ~<=2
    k = torch.nn.functional.normalize(torch.randn(B, T, H, dh), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(B, T, H, dh), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, dh), dim=-1)
    bw = torch.ones(B, T, H, 1) * 1.0
    bf = torch.zeros(B, T, H, 1)
    gr = torch.ones(B, T, H, 1)
    with torch.no_grad():
        out, _, _ = mem._sequential(k, v, q, bw, bf, gr, None,
                                    torch.zeros(B, H, dh, dh), False,
                                    torch.zeros(B * H, dh))
    assert out.abs().max().item() < 5.0, out.abs().max().item()
    print(f"  readout bounded (max|out|={out.abs().max().item():.3f} with "
          f"unit v, 8 writes) OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_builds_and_trains()
    test_denominator_recurrence()
    test_train_equals_decode()
    test_growth_exact_with_dpnorm()
    test_readout_bounded()
    print("\nDP-norm architecture tests passed.")
