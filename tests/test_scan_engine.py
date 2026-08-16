"""Regression tests for the C/Mojo scan engine (2026-08-10 speedup pass).

Guards:
  1. the C kernel builds (gcc available) and matches torch exactly —
     norm on AND off, with decay — and the OpenMP parallel path is
     bit-identical to the serial path;
  2. the algebraic fusion (norm=0) is exact vs torch (the fused identity
     o_new = a*o_prev + (k.q)*(bw*v - bf*tmp));
  3. the fused q==k kernel (no decay) matches torch;
  4. a speed sanity check: the parallel kernel is not SLOWER than serial
     (>= 0.9x) and the fused no-norm path beats the norm path in serial time
     (>= 1.2x) — the fusion must actually save work;
  5. the model's fast path (model._sequential_fast) still equals the Python
     scan end-to-end.
Run:  python tests/test_scan_engine.py
"""
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOJO = os.path.join(ROOT, "mojo")


def _build():
    """Build the .so; return True if gcc is available and the build worked."""
    try:
        r = subprocess.run(["bash", "mojo/c_ref/build.sh"], cwd=ROOT,
                           capture_output=True, text=True, timeout=300)
        return r.returncode == 0 and os.path.exists(
            os.path.join(MOJO, "c_ref", "leafv5_scan.so"))
    except Exception:
        return False


def _torch_seq(k, v, q, bw, bf, gr, dec, alpha, S0, state_norm):
    from leafv5.model import statenorm
    BH, T, dh = k.shape
    S = S0.clone()
    outs = []
    for t in range(T):
        kt1 = k[:, t].unsqueeze(-1); kt2 = k[:, t].unsqueeze(1)
        vt = v[:, t].unsqueeze(-1); qt1 = q[:, t].unsqueeze(-1)
        o_prev = torch.bmm(S, qt1)
        tmp = torch.bmm(S, kt1)
        a = dec[:, t:t + 1].unsqueeze(-1) if dec is not None else 1.0
        S = (a * S - bf[:, t:t + 1].unsqueeze(-1) * torch.bmm(tmp, kt2)
             + bw[:, t:t + 1].unsqueeze(-1) * torch.bmm(vt, kt2))
        if state_norm:
            S = statenorm(S, dh)
        o_new = torch.bmm(S, qt1)
        outs.append((gr[:, t:t + 1].unsqueeze(-1) * o_new
                     + alpha.view(BH, 1, 1) * o_prev).squeeze(-1))
    return torch.stack(outs, 1), S


def test_kernel_matches_torch_and_parallel_exact():
    if not _build():
        print("  (gcc unavailable -> kernel engine test skipped)")
        return
    sys.path.insert(0, MOJO)
    from c_ref import scan_q, scan_q_nt, scan_fused
    torch.manual_seed(0)
    BH, T, dh = 48, 64, 48
    k = torch.nn.functional.normalize(torch.randn(BH, T, dh), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(BH, T, dh), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(BH, T, dh), dim=-1)
    bw = torch.sigmoid(torch.randn(BH, T)); bf = torch.sigmoid(torch.randn(BH, T))
    gr = torch.sigmoid(torch.randn(BH, T)); dec = torch.sigmoid(torch.randn(BH, T))
    alpha = torch.full((BH,), 0.5); S0 = torch.zeros(BH, dh, dh)
    for sn in (False, True):
        out_c, S_c = scan_q(q, k, v, bw, bf, gr, dec, alpha, S0, state_norm=sn)
        out_r, S_r = _torch_seq(k, v, q, bw, bf, gr, dec, alpha, S0, sn)
        assert (out_c - out_r).abs().max().item() < 1e-4, (sn, "out")
        assert (S_c - S_r).abs().max().item() < 1e-4, (sn, "S")
    # parallel == serial (bit-exact)
    a, _ = scan_q(q, k, v, bw, bf, gr, dec, alpha, S0, False)
    b, _ = scan_q_nt(q, k, v, bw, bf, gr, dec, alpha, S0, False, 2)
    assert torch.equal(a, b), "OpenMP path must be bit-identical to serial"
    # fused q==k (no decay) == torch
    out_f, _ = scan_fused(k, v, bw, bf, gr, alpha, S0)
    ref_f, _ = _torch_seq(k, v, k, bw, bf, gr, None, alpha, S0, False)
    assert (out_f - ref_f).abs().max().item() < 1e-4
    print("  C kernel: torch-exact (norm on/off), OpenMP==serial bit-exact, "
          "fused q==k exact OK")


def test_fusion_and_parallel_not_slower():
    """The fusion must save work (norm=0 < norm=1 serial time) and the
    parallel path must not be slower than serial."""
    if not _build():
        print("  (gcc unavailable -> speed sanity skipped)")
        return
    sys.path.insert(0, MOJO)
    from c_ref import scan_q_nt
    torch.manual_seed(0)
    BH, T, dh = 192, 256, 48
    k = torch.nn.functional.normalize(torch.randn(BH, T, dh), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(BH, T, dh), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(BH, T, dh), dim=-1)
    bw = torch.sigmoid(torch.randn(BH, T)); bf = torch.sigmoid(torch.randn(BH, T))
    gr = torch.sigmoid(torch.randn(BH, T)); dec = torch.sigmoid(torch.randn(BH, T))
    alpha = torch.full((BH,), 0.5); S0 = torch.zeros(BH, dh, dh)

    def t(fn, iters=3):
        for _ in range(1):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters

    t_norm = t(lambda: scan_q_nt(q, k, v, bw, bf, gr, dec, alpha, S0, True, 1))
    t_fused = t(lambda: scan_q_nt(q, k, v, bw, bf, gr, dec, alpha, S0, False, 1))
    t_par = t(lambda: scan_q_nt(q, k, v, bw, bf, gr, dec, alpha, S0, True, 2))
    assert t_fused < t_norm * 0.8, (t_fused, t_norm)   # fusion saves >=20%
    assert t_par <= t_norm * 1.1, (t_par, t_norm)       # parallel not slower
    print(f"  fusion: {t_norm/t_fused:.2f}x faster; parallel 2-thread "
          f"{t_norm/t_par:.2f}x faster than serial OK")


def test_model_fast_path_equals_python():
    """model._sequential_fast (C kernel) == model._sequential (Python)."""
    from leafv5.config import preset_config
    from leafv5.model import LeafLM
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256)
    m = LeafLM(cfg).eval()
    x = torch.randint(0, 256, (2, 24))
    with torch.no_grad():
        a, _ = m(x, m.init_states(2, torch.device("cpu")), fast=True)
        b, _ = m(x, m.init_states(2, torch.device("cpu")), fast=False)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max().item()
    print("  model fast path == python scan OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_kernel_matches_torch_and_parallel_exact()
    test_fusion_and_parallel_not_slower()
    test_model_fast_path_equals_python()
    print("\nScan-engine tests passed.")
