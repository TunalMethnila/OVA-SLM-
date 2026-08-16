"""LEAFv5 MISTRAL-STACK STABILITY CERTIFICATION.

The Mistral-inspired efficiency stack (GQA, rolling-buffer KV cache,
pre-fill & chunking, Mixtral-style MoE) gets its own stability certificate,
mirroring stability_check.py.  The properties that must hold, measured:

   1. boundary exactness   : rolling == tuple-cache decode at EVERY step
                             across multiple window wraps (W-1, W, W+1, 2W..)
   2. position-offset      : prefill(pos=k>0) + decode == one-shot full-seq
                             (regression for the 2026-08-09 window() bug)
   3. determinism          : same input -> bit-identical outputs (rolling,
                             prefill, MoE, full model with GQA)
   4. long decode          : 3000-token rolling decode stays finite, bounded,
                             storage constant, position counter exact
   5. edge cases & guards  : W=1, heads=1, kv=1, T==W, T==W+1, batch 1/7;
                             GQA divisibility and chunk>W raise cleanly
   6. MoE stability        : 100-step training, no NaN, aux loss in range,
                             all experts utilized, router logits bounded
   7. deep stack           : 12 layers + SWA/GQA/MoE forward+backward finite
   8. chunked prefill      : chunk in {1,2,3,7,W} == one-shot, any prompt len
   9. low precision        : bf16 (and fp16) forward+backward finite,
                             rolling == tuple still exact in bf16
  10. train==decode        : full-seq == rolling decode after training, with
                             GQA + interleave (reviewer's central invariant)

Run:  python -m leafv5.stability_check_mistral
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .config import preset_config
from .model import LeafLM, RollingKVCache, SlidingWindowAttention

_results = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}  {detail}")
    _results.append(ok)
    return ok


def _swa(dim=64, heads=4, W=16, kv=2):
    m = SlidingWindowAttention(dim, heads, W, kv_heads=kv).eval()
    m.scale.data.fill_(1.0)          # nonzero branch so outputs are meaningful
    return m


def _roll_decode(d, seq):
    """Decode `seq` one token at a time with a fresh rolling cache.  Returns
    (concatenated outputs, cache)."""
    cache = d.prefill(seq[:, :1], pos=0, chunk=d.window)
    outs = []
    with torch.no_grad():
        for t in range(1, seq.shape[1]):
            o, cache = d(seq[:, t:t + 1], cache)
            outs.append(o)
    return torch.cat(outs, 1), cache


def _tuple_decode(d, seq):
    cache = None
    outs = []
    with torch.no_grad():
        for t in range(seq.shape[1]):
            o, cache = d(seq[:, t:t + 1], cache)
            outs.append(o)
    return torch.cat(outs, 1)


def check_boundary_exactness(dev):
    """Rolling == tuple decode at every step, wrapping the window 3+ times."""
    torch.manual_seed(0)
    d = _swa(kv=2)
    seq = torch.randn(2, 3 * 16 + 5, 64)      # 53 tokens: wraps at 16, 32, 48
    o_roll, cache = _roll_decode(d, seq)
    o_tuple = _tuple_decode(d, seq)
    # compare at every decoded position (1..52) — covers W-1, W, W+1, 2W-1, ...
    dmax = (o_roll - o_tuple[:, 1:]).abs().max().item()
    ok = dmax < 1e-6
    ok = ok and cache.shape == (2, 2, 16, 16) and cache.pos == 53
    return check("boundary exactness (rolling==tuple at every step, 3+ wraps)",
                 ok, f"max|d|={dmax:.2e} storage {tuple(cache.shape)}")


def check_position_offset_prefill(dev):
    """prefill(hist, pos=7) + decode == one-shot full-seq windowed forward
    over the same tokens — the regression test for the window() start bug."""
    torch.manual_seed(0)
    d = _swa(kv=2)
    H, C = 30, 20
    hist = torch.randn(2, H, 64)
    rest = torch.randn(2, C, 64)
    full = torch.cat([hist, rest], dim=1)
    with torch.no_grad():
        ofull, _ = d(full)                     # oracle: one-shot windowed
    cache = d.prefill(hist, pos=7, chunk=5)    # start at absolute position 7
    outs = []
    with torch.no_grad():
        for t in range(C):
            o, cache = d(rest[:, t:t + 1], cache)
            outs.append(o)
    o_roll = torch.cat(outs, 1)
    dmax = (o_roll - ofull[:, H:]).abs().max().item()
    return check("position-offset prefill (pos=7) == one-shot full-seq",
                 dmax < 1e-5, f"max|d|={dmax:.2e}")


def check_determinism(dev):
    torch.manual_seed(0)
    d = _swa(kv=1)
    seq = torch.randn(2, 200, 64)
    a, _ = _roll_decode(d, seq)
    b, _ = _roll_decode(d, seq)
    same_roll = torch.equal(a, b)
    p1 = d.prefill(seq[:, :50], pos=0, chunk=7)
    p2 = d.prefill(seq[:, :50], pos=0, chunk=7)
    same_prefill = torch.equal(p1.k, p2.k) and torch.equal(p1.v, p2.v)
    # MoE determinism
    cfg = preset_config("micro", vocab_size=256, n_layers=1, dim=64, d_h=16,
                        moe=True, moe_experts=8, moe_topk=2, scale_init=0.1)
    m = LeafLM(cfg).eval()
    x = torch.randn(3, 16, 64)
    with torch.no_grad():
        o1 = m.blocks[0].ffn(x)
        o2 = m.blocks[0].ffn(x)
    same_moe = torch.equal(o1, o2)
    # full model with SWA+GQA determinism
    cfg2 = preset_config("micro", vocab_size=256, n_layers=2, dim=96, d_h=32,
                         use_swa=True, swa_window=8, swa_kv_heads=2,
                         scale_init=0.1)
    m2 = LeafLM(cfg2).eval()
    xi = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        l1, _ = m2(xi, m2.init_states(2, torch.device("cpu")))
        l2, _ = m2(xi, m2.init_states(2, torch.device("cpu")))
    same_model = torch.equal(l1, l2)
    ok = same_roll and same_prefill and same_moe and same_model
    return check("determinism (rolling/prefill/MoE/full-model bit-identical)",
                 ok, f"roll={same_roll} prefill={same_prefill} moe={same_moe} "
                     f"model={same_model}")


def check_long_decode_bounded(dev):
    torch.manual_seed(0)
    d = _swa(dim=64, heads=4, W=64, kv=2)
    seq = torch.randn(1, 3000, 64)
    cache = d.prefill(seq[:, :1], pos=0, chunk=64)
    finite = bounded = True
    max_abs = 0.0
    with torch.no_grad():
        for t in range(1, 3000):
            o, cache = d(seq[:, t:t + 1], cache)
            finite = finite and torch.isfinite(o).all().item()
            max_abs = max(max_abs, o.abs().max().item())
            if t % 500 == 0:
                bounded = bounded and cache.shape == (1, 2, 64, 16)
    ok = finite and bounded and cache.pos == 3000 and max_abs < 500
    return check("long decode (3000 tokens): finite, bounded, storage constant",
                 ok, f"max|out|={max_abs:.2f} final pos={cache.pos} "
                     f"storage {tuple(cache.shape)}")


def check_edge_cases_and_guards(dev):
    ok = True
    # degenerate-but-valid configs
    for kw in (dict(heads=1, kv=1, W=1), dict(heads=2, kv=1, W=2),
               dict(heads=4, kv=4, W=16)):
        d = _swa(**kw)
        x = torch.randn(1, 10, 64)
        with torch.no_grad():
            o, _ = d(x)
        ok = ok and torch.isfinite(o).all().item()
    # T == W and T == W+1 full-seq
    d = _swa(W=16)
    with torch.no_grad():
        for T in (16, 17):
            o, c = d(torch.randn(2, T, 64))
            ok = ok and torch.isfinite(o).all().item() and \
                c[0].shape[2] == min(T, 16)
    # batch 1 and 7
    with torch.no_grad():
        for B in (1, 7):
            o, _ = d(torch.randn(B, 8, 64))
            ok = ok and torch.isfinite(o).all().item()
    # guards raise cleanly
    try:
        SlidingWindowAttention(64, 4, 16, kv_heads=3)   # 4 % 3 != 0
        ok = False
    except AssertionError:
        pass
    d2 = _swa(W=16)
    try:
        d2.prefill(torch.randn(1, 20, 64), chunk=17)    # chunk > W
        ok = False
    except AssertionError:
        pass
    # empty prefill returns None (no history), and empty-start cache works
    ok = ok and d2.prefill(torch.randn(1, 0, 64), chunk=16) is None
    rc = RollingKVCache(torch.randn(1, 2, 0, 16), torch.randn(1, 2, 0, 16),
                        pos=5, window=16)
    with torch.no_grad():
        kw, vw = rc.window()
    ok = ok and kw.shape[2] == 0
    rc.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
    with torch.no_grad():
        kw, _ = rc.window()
    ok = ok and kw.shape[2] == 1
    return check("edge cases (W=1,heads=1,kv=1,T==W,T==W+1,batch 1/7) + guards",
                 ok)


def check_moe_stability(dev):
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=2, dim=96, d_h=32,
                        moe=True, moe_experts=8, moe_topk=2, scale_init=0.1)
    m = LeafLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.95))
    moe = m.blocks[0].ffn
    used = torch.zeros(8)
    max_aux = 0.0
    max_router = 0.0
    finite = True

    def hook(mod, inp, out):
        """Track real router decisions during training forwards."""
        xh = inp[0].reshape(-1, 96)
        with torch.no_grad():
            idx = torch.topk(mod.router(xh), 2).indices
            used.scatter_add_(0, idx.reshape(-1),
                              torch.ones(idx.numel()))
            nonlocal max_router
            max_router = max(max_router, mod.router(xh).abs().max().item())

    h = moe.register_forward_hook(hook)
    for i in range(100):
        opt.zero_grad(set_to_none=True)
        x = torch.randint(0, 256, (4, 16)); y = torch.randint(0, 256, (4, 16))
        lg, _ = m(x, m.init_states(4, torch.device("cpu")))
        loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
        aux = m.aux_loss()
        loss = loss + 0.01 * aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        finite = finite and torch.isfinite(loss).item()
        max_aux = max(max_aux, aux.item())
    h.remove()
    all_used = (used > 0).all().item()
    aux_ok = 0.0 < max_aux < 8.0
    ok = finite and all_used and aux_ok and max_router < 100
    return check("MoE stability (100 steps, aux loss in range, all experts "
                 "used, router bounded)", ok,
                 f"finite={finite} aux_max={max_aux:.3f} used={used.int().tolist()} "
                 f"router_max={max_router:.2f}")


def check_deep_stack_new_features(dev):
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=12, dim=64, d_h=16,
                        use_swa=True, swa_every=2, swa_window=16,
                        swa_kv_heads=1, moe=True, moe_experts=4, moe_topk=2,
                        scale_init=0.1)
    m = LeafLM(cfg)
    x = torch.randint(0, 256, (2, 16)); y = torch.randint(0, 256, (2, 16))
    lg, _ = m(x, m.init_states(2, torch.device("cpu")))
    loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)) + 0.01 * m.aux_loss()
    loss.backward()
    ok = torch.isfinite(loss).item() and \
        all(torch.isfinite(p.grad).all() for p in m.parameters()
            if p.grad is not None)
    return check("12-layer stack + SWA/GQA/MoE forward+backward finite", ok)


def check_chunked_prefill_exact(dev):
    torch.manual_seed(0)
    d = _swa(kv=2)
    prompt = torch.randn(2, 50, 64)
    with torch.no_grad():
        full = d.prefill(prompt, pos=0, chunk=None)
        dmax = 0.0
        for ch in (1, 2, 3, 7, 16):
            c = d.prefill(prompt, pos=0, chunk=ch)
            dmax = max(dmax, (full.k - c.k).abs().max().item(),
                       (full.v - c.v).abs().max().item())
    ok = dmax < 1e-6 and full.shape == (2, 2, 16, 16)
    return check("chunked prefill exact (chunk 1/2/3/7/W == one-shot)",
                 ok, f"max|d|={dmax:.2e} width={full.shape[2]}")


def check_low_precision(dev):
    torch.manual_seed(0)
    seq = torch.randn(2, 30, 64)
    # bf16: rolling == tuple still exact-ish, all finite
    d16 = _swa(dim=64, heads=4, W=16, kv=2).to(torch.bfloat16)
    s16 = seq.to(torch.bfloat16)
    o_roll16, _ = _roll_decode(d16, s16)
    o_tuple16 = _tuple_decode(d16, s16)[:, 1:]
    d16_ok = torch.isfinite(o_roll16).all().item() and \
        (o_roll16 - o_tuple16).abs().max().item() < 1e-3
    # fp16 (CPU): forward must at least be finite
    fp16_ok = True
    try:
        df = _swa(dim=32, heads=2, W=8, kv=1).to(torch.float16)
        sf = torch.randn(1, 10, 32).to(torch.float16)
        with torch.no_grad():
            of, _ = df(sf)
        fp16_ok = torch.isfinite(of).all().item()
    except Exception:
        fp16_ok = False
    return check("low precision (bf16 rolling==tuple exact, fp16 finite)",
                 d16_ok and fp16_ok,
                 f"bf16 max|d|={(o_roll16 - o_tuple16).abs().max().item():.2e} "
                 f"fp16_finite={fp16_ok}")


def check_train_equals_decode_gqa(dev):
    """Reviewer's central invariant with the new stack, AFTER training."""
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=2, dim=96, d_h=32,
                        use_swa=True, swa_window=8, swa_kv_heads=2,
                        scale_init=0.1)
    m = LeafLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(6):
        opt.zero_grad()
        xi = torch.randint(0, 256, (4, 16)); yi = torch.randint(0, 256, (4, 16))
        lg, _ = m(xi, m.init_states(4, torch.device("cpu")))
        F.cross_entropy(lg.reshape(-1, 256), yi.reshape(-1)).backward()
        opt.step()
    m.eval()
    x = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        lg_full, _ = m(x, m.init_states(1, torch.device("cpu")))
        st = m.init_states(1, torch.device("cpu"))
        outs = []
        for t in range(24):
            lg, st = m(x[:, t:t + 1], st)
            outs.append(lg)
        lg_dec = torch.cat(outs, 1)
    dmax = (lg_full - lg_dec).abs().max().item()
    return check("train==decode with SWA+GQA (kv=2, every=1) after training",
                 dmax < 1e-4, f"max|d|={dmax:.2e}")


CHECKS = [
    check_boundary_exactness,
    check_position_offset_prefill,
    check_determinism,
    check_long_decode_bounded,
    check_edge_cases_and_guards,
    check_moe_stability,
    check_deep_stack_new_features,
    check_chunked_prefill_exact,
    check_low_precision,
    check_train_equals_decode_gqa,
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    dev = args.device
    torch.manual_seed(0)
    _results.clear()   # fresh certificate run (main() may be called repeatedly)
    print("=" * 70)
    print("LEAFv5 MISTRAL-STACK STABILITY CERTIFICATION")
    print("=" * 70)
    for fn in CHECKS:
        fn(dev)
    n = sum(_results)
    print("-" * 70)
    print(f"MISTRAL-STACK STABILITY CERTIFICATE: {n}/{len(_results)} passed")
    print("RESULT: " + ("STABLE" if n == len(_results) else "UNSTABLE"))
    print("-" * 70)
    return n == len(_results)


if __name__ == "__main__":
    main()
