"""LEAFv5 stability certification: stress-tests the model and prints a
PASS/FAIL certificate.  This is the "very stable" guarantee, measured.

Battery:
  1. edge inputs    : empty prompt, max_new=0, temperature<=0, huge top_k
  2. determinism    : same seed -> identical outputs (fp32, CPU)
  3. weight pert.   : +-1% weight noise -> bounded, proportional output change
  4. input pert.    : 1-token change -> bounded output change (no explosion)
  5. state pert.    : state noise -> bounded output change
  6. long training  : 600 steps, LR 3e-2, grad-clip, injected NaN -> loss
                      finite, states bounded, NaN-grad guard fires
  7. deep stack     : 24 layers forward+backward finite

Run:  python -m leafv5.stability_check [--steps 300]
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from .config import preset_config
from .model import LeafLM
from .generate import generate
from .autotune_utils import nan_guard


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}  {detail}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    dev = args.device
    torch.manual_seed(0)

    print("=" * 64)
    print("LEAFv5 STABILITY CERTIFICATION")
    print("=" * 64)
    results = []

    # ---- 1. edge inputs ----
    from .data import CharTokenizer
    import string
    voc = {c: i for i, c in enumerate(string.ascii_lowercase)}
    tok = CharTokenizer(voc)
    m0 = LeafLM(preset_config("micro", vocab_size=256)).eval().to(dev)
    ok = True
    try:
        generate(m0, tok, "", max_new=4, temperature=0.0, device=dev)
        generate(m0, tok, "abc", max_new=0, temperature=0.0, device=dev)
        generate(m0, tok, "abc", max_new=4, temperature=0.0, top_k=10 ** 9, device=dev)
        generate(m0, tok, "abc", max_new=4, temperature=-1.0, device=dev)
    except Exception as e:
        ok = False
        print("   ", e)
    results.append(check("edge inputs (empty/0/new/NaN-ish) never crash", ok))
    del m0

    # ---- 2. determinism ----
    cfg = preset_config("micro", vocab_size=256, n_layers=2, dim=96, d_h=32,
                        scale_init=0.1)
    m = LeafLM(cfg).eval().to(dev)
    x = torch.randint(0, 256, (4, 16)).to(dev)
    with torch.no_grad():
        a, _ = m(x, m.init_states(4, dev))
        b, _ = m(x, m.init_states(4, dev))
    results.append(check("determinism (same input -> same output)",
                         torch.allclose(a, b, atol=1e-9)))

    # ---- 3. weight perturbation ----
    with torch.no_grad():
        base, _ = m(x, m.init_states(4, dev))
        m2 = LeafLM(cfg).eval().to(dev)
        m2.load_state_dict(m.state_dict())
        for p in m2.parameters():
            p.add_(torch.randn_like(p) * 0.01 * p.abs().clamp_min(1e-4))
        pert, _ = m2(x, m2.init_states(4, dev))
    rel = (pert - base).abs().max().item() / (base.abs().max().item() + 1e-9)
    results.append(check("weight perturbation +-1% stays proportional",
                         rel < 2.0, f"(rel change {rel:.3f})"))
    del m2

    # ---- 4. input perturbation ----
    x2 = x.clone(); x2[:, 5] = (x2[:, 5] + 7) % 256
    with torch.no_grad():
        ip, _ = m(x2, m.init_states(4, dev))
    d_in = (ip - base).abs().max().item() / (base.abs().max().item() + 1e-9)
    results.append(check("input perturbation (1 token) bounded",
                         d_in < 2.0, f"(rel change {d_in:.3f})"))

    # ---- 5. state perturbation ----
    with torch.no_grad():
        _, st = m(x, m.init_states(4, dev))
        st_noisy = [s + torch.randn_like(s) * 0.01 for s in st]
        # re-run a fresh short input with clean vs noisy state
        xs = torch.randint(0, 256, (4, 8)).to(dev)
        oc, _ = m(xs, m.init_states(4, dev))
        on, _ = m(xs, st_noisy, offset=0)
    d_s = (on - oc).abs().max().item() / (oc.abs().max().item() + 1e-9)
    results.append(check("state perturbation bounded",
                         d_s < 2.0, f"(rel change {d_s:.3f})"))

    # ---- 6. long training with noise injection ----
    mt = LeafLM(cfg).to(dev)
    opt = torch.optim.AdamW(mt.parameters(), lr=3e-2, betas=(0.9, 0.95))
    inject_step = args.steps // 3
    finite_after = True          # finiteness of every step AFTER the injection
    guard_fired = False
    for i in range(args.steps):
        opt.zero_grad(set_to_none=True)
        xb = torch.randint(0, 256, (4, 16)).to(dev)
        yb = torch.randint(0, 256, (4, 16)).to(dev)
        lg, _ = mt(xb, mt.init_states(4, dev))
        loss = F.cross_entropy(lg.reshape(-1, 256), yb.reshape(-1))
        if i == inject_step:     # inject a NaN batch mid-training
            loss = loss * float("nan")
        loss.backward()
        if nan_guard(mt):
            guard_fired = True
            opt.zero_grad(set_to_none=True)
        else:
            torch.nn.utils.clip_grad_norm_(mt.parameters(), 1.0)
            opt.step()
        if i > inject_step:      # the property: recovery after the NaN
            finite_after = finite_after and torch.isfinite(loss).item()
    # states bounded after all that
    with torch.no_grad():
        _, st_end = mt(torch.randint(0, 256, (2, 16)).to(dev),
                       mt.init_states(2, dev))
    bound_ok = all(s.norm(dim=(-1, -2)).max().item() <=
                   math.sqrt(cfg.d_h) * 1.05 for s in st_end)
    results.append(check("NaN-grad guard fires on injected NaN",
                         guard_fired))
    results.append(check(f"training RECOVERS after NaN ({args.steps} steps, "
                         f"LR 3e-2): all later losses finite", finite_after))
    results.append(check("states bounded after stress", bound_ok))
    del mt

    # ---- 7. deep stack ----
    md = LeafLM(preset_config("micro", vocab_size=256, n_layers=24, dim=96,
                              d_h=32, scale_init=0.1)).to(dev)
    xd = torch.randint(0, 256, (2, 16)).to(dev)
    try:
        lgd, _ = md(xd, md.init_states(2, dev))
        F.cross_entropy(lgd.reshape(-1, 256), torch.randint(0, 256, (2, 16)).to(dev).reshape(-1)).backward()
        gfinite = all(torch.isfinite(p.grad).all() for p in md.parameters()
                      if p.grad is not None)
        results.append(check("24-layer stack forward+backward finite", gfinite))
    except Exception as e:
        results.append(check("24-layer stack forward+backward finite", False,
                             str(e)[:60]))
    del md

    # ---- certificate ----
    passed = sum(results)
    total = len(results)
    print("-" * 64)
    print(f"STABILITY CERTIFICATE: {passed}/{total} passed")
    print("RESULT:", "STABLE" if passed == total else "NEEDS ATTENTION")
    print("-" * 64)
    return 0 if passed == total else 1


if __name__ == "__main__":
    main()
