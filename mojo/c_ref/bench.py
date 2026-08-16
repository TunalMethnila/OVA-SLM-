#!/usr/bin/env python3
"""Validate the C kernel (and by construction the Mojo port) against the
PyTorch sequential scan, then benchmark all scan variants on this machine.

v2 (2026-08-10): the C kernel is OpenMP-parallel over batch*heads with SIMD
(AVX2/AVX-512 via -march=native) and an algebraic fusion of the post-update
read when StateNorm is off.  This script validates parallel == serial == torch
and measures the multi-thread scaling.

Usage:  bash mojo/c_ref/build.sh && OMP_NUM_THREADS=4 python mojo/c_ref/bench.py
"""
import os
import sys
import time

import numpy as np
import torch

_MOJO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _MOJO)                      # so `import c_ref` works
sys.path.insert(0, os.path.dirname(_MOJO))     # so `import leafv5` works
from c_ref import (scan_fused, scan_fused_nt, scan_q, scan_q_nt, version)  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)


def torch_seq_scan(k, v, q, bw, bf, gr, dec, alpha, S0, state_norm):
    """Reference: the exact recurrence from leafv5.model._sequential."""
    from leafv5.model import statenorm
    BH, T, dh = k.shape
    S = S0.clone()
    outs = []
    for t in range(T):
        kt1 = k[:, t].unsqueeze(-1)
        kt2 = k[:, t].unsqueeze(1)
        vt = v[:, t].unsqueeze(-1)
        qt1 = q[:, t].unsqueeze(-1)
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
    return torch.stack(outs, dim=1), S


def bench(fn, iters=5):
    for _ in range(2):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main():
    print(f"kernel: {version()}")

    # ---- 1. numerical validation: C == torch (norm on AND off), par == ser
    BH, T, dh = 48, 64, 48
    k = torch.nn.functional.normalize(torch.randn(BH, T, dh), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(BH, T, dh), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(BH, T, dh), dim=-1)
    bw = torch.sigmoid(torch.randn(BH, T))
    bf = torch.sigmoid(torch.randn(BH, T))
    gr = torch.sigmoid(torch.randn(BH, T))
    dec = torch.sigmoid(torch.randn(BH, T))
    alpha = torch.full((BH,), 0.5)
    S0 = torch.zeros(BH, dh, dh)

    for sn in (False, True):
        out_c, S_c = scan_q(q, k, v, bw, bf, gr, dec, alpha, S0, state_norm=sn)
        out_r, S_r = torch_seq_scan(k, v, q, bw, bf, gr, dec, alpha, S0, sn)
        d_o = (out_c - out_r).abs().max().item()
        d_S = (S_c - S_r).abs().max().item()
        assert d_o < 1e-4 and d_S < 1e-4, (sn, d_o, d_S)
        print(f"[validate] C vs torch (norm={int(sn)}): "
              f"max|d_out|={d_o:.2e} max|d_S|={d_S:.2e} OK")
    # parallel == serial (same norm setting!)
    out_c0, _ = scan_q(q, k, v, bw, bf, gr, dec, alpha, S0, False)
    out_p, _ = scan_q_nt(q, k, v, bw, bf, gr, dec, alpha, S0, False, 2)
    d = (out_p - out_c0).abs().max().item()
    assert d < 1e-6, d
    print(f"[validate] OpenMP(2) == serial: max|d|={d:.2e} OK")
    # fused q==k is only valid with NO decay (a=1) and norm off.
    # torch_seq_scan(k, v, q, ...) -> pass q=k explicitly (k, v, k, ...)
    out_f, _ = scan_fused(k, v, bw, bf, gr, alpha, S0)
    ref_f, _ = torch_seq_scan(k, v, k, bw, bf, gr, None, alpha, S0, False)
    d = (out_f - ref_f).abs().max().item()
    assert d < 1e-4, d
    print(f"[validate] fused q==k (no decay) vs torch: max|d|={d:.2e} OK")

    # ---- 2. benchmark: T4-realistic shapes, serial vs parallel scaling
    print("\n[bench] T4-shape BH=192, T=512, d_h=48 (98304 positions)")
    BH, T, dh = 192, 512, 48
    k = torch.nn.functional.normalize(torch.randn(BH, T, dh), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(BH, T, dh), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(BH, T, dh), dim=-1)
    bw = torch.sigmoid(torch.randn(BH, T)); bf = torch.sigmoid(torch.randn(BH, T))
    gr = torch.sigmoid(torch.randn(BH, T)); dec = torch.sigmoid(torch.randn(BH, T))
    alpha = torch.full((BH,), 0.5); S0 = torch.zeros(BH, dh, dh)
    flops = 2 * BH * T * dh * dh * 3 * 2
    import os as _os
    ncores = os.cpu_count() or 1
    rows = []
    for name, fn in [
        ("general norm=1 t=1", lambda: scan_q_nt(q, k, v, bw, bf, gr, dec, alpha, S0, True, 1)),
        ("general norm=1 t=2", lambda: scan_q_nt(q, k, v, bw, bf, gr, dec, alpha, S0, True, 2)),
        ("general norm=1 t=4", lambda: scan_q_nt(q, k, v, bw, bf, gr, dec, alpha, S0, True, 4)),
        ("general norm=0 t=1", lambda: scan_q_nt(q, k, v, bw, bf, gr, dec, alpha, S0, False, 1)),
        ("general norm=0 t=4", lambda: scan_q_nt(q, k, v, bw, bf, gr, dec, alpha, S0, False, 4)),
        ("fused q==k  t=1", lambda: scan_fused_nt(k, v, bw, bf, gr, alpha, S0, 1)),
        ("fused q==k  t=4", lambda: scan_fused_nt(k, v, bw, bf, gr, alpha, S0, 4)),
    ]:
        dt = bench(fn, iters=3)
        rows.append((name, dt))
        print(f"  {name:18s}: {dt*1e3:9.2f} ms  {flops/dt/1e9:7.1f} GFLOP/s  "
              f"{BH*T/dt/1e3:7.0f}k pos/s")
    # torch SCAN-ONLY reference at the same shape (honest apples-to-apples)
    t_scan = bench(lambda: torch_seq_scan(
        k, v, q, bw, bf, gr, dec, alpha, S0, True), 2)
    print(f"  {'torch scan-only (Python loop)':18s}: {t_scan*1e3:9.0f} ms")
    dt1 = dict(rows)["general norm=1 t=1"]
    dt2 = dict(rows)["general norm=1 t=2"]
    dtf = dict(rows)["general norm=0 t=4"]
    print(f"\n  vs torch scan-only: C(t=2) {t_scan/dt2:.0f}x, "
          f"C fused(t=4) {t_scan/dtf:.0f}x faster")
    print(f"  scaling 1->2 threads (general norm=1): {dt1/dt2:.2f}x "
          f"(host cores: {ncores})")


if __name__ == "__main__":
    main()
