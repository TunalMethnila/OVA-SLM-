"""ctypes wrapper for the C reference kernel (leafv5_scan.so).

The C kernel matches the CURRENT LEAFv5 memory (SOTA-upgraded): read with a
separate query q, erase along k, optional input decay + StateNorm.  It is the
validated twin of leafv5.model.MultiTimescaleDeltaV2._sequential and of the
Mojo port (mojo/leafv5.mojo).

v2 (2026-08-10): OpenMP parallel over batch*heads, SIMD (AVX2/AVX-512 via
-march=native), aligned per-thread stack scratch (no per-call malloc), and an
algebraic fusion that drops the post-update matvec when StateNorm is off.

Usage:
    from c_ref import scan_q, scan_fused, scan_q_nt, scan_fused_nt
    out, S = scan_q(q, k, v, bw, bf, gr, dec, alpha, S0, state_norm=False)
    out, S = scan_q_nt(..., state_norm=False, threads=4)   # explicit threads
"""
from __future__ import annotations

import ctypes
import os

import torch

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        here = os.path.dirname(os.path.abspath(__file__))
        so = os.path.join(here, "leafv5_scan.so")
        if not os.path.exists(so):
            raise RuntimeError("build the kernel first: bash mojo/c_ref/build.sh")
        _LIB = ctypes.CDLL(so)
        for name in ("leafv5_scan_q", "leafv5_scan_q_nt",
                     "leafv5_scan_q_s", "leafv5_scan_q_s_nt"):
            f = getattr(_LIB, name)
            f.restype = None
            f.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_void_p] * 2 + [
                ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_int]
            if name.endswith("_s"):
                f.argtypes += [ctypes.c_void_p, ctypes.c_void_p]
            if name.endswith("_nt"):
                f.argtypes.append(ctypes.c_int)
        for name in ("leafv5_scan_fused", "leafv5_scan_fused_nt"):
            g = getattr(_LIB, name)
            g.restype = None
            g.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_void_p, ctypes.c_void_p,
                                                  ctypes.c_long, ctypes.c_long,
                                                  ctypes.c_long]
            if name.endswith("_nt"):
                g.argtypes.append(ctypes.c_int)
        v = _LIB.leafv5_scan_version
        v.restype = ctypes.c_char_p
        _LIB._version = v().decode()
    return _LIB


def version() -> str:
    return _lib()._version


def _t(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().float()


def scan_q(q, k, v, bw, bf, gr, dec, alpha, S0, state_norm: bool = False):
    """Query-read delta scan (current architecture).  Returns (out, S)."""
    return _scan_q(q, k, v, bw, bf, gr, dec, alpha, S0, state_norm, None)


def scan_q_nt(q, k, v, bw, bf, gr, dec, alpha, S0, state_norm: bool = False,
              threads: int = 0):
    """Query-read delta scan with an explicit OpenMP thread count (0=auto)."""
    return _scan_q(q, k, v, bw, bf, gr, dec, alpha, S0, state_norm, threads)


def scan_q_s(q, k, v, bw, bf, gr, dec, alpha, S0, state_norm=False,
             sw=None, sb=None):
    """Query-read delta scan with novelty-gated writes (Tier-1 retention
    fix).  sw/sb: per-head [BH] w/b factors (None -> gate off)."""
    return _scan_q(q, k, v, bw, bf, gr, dec, alpha, S0, state_norm,
                   None, sw=sw, sb=sb)


def scan_q_s_nt(q, k, v, bw, bf, gr, dec, alpha, S0, state_norm=False,
                sw=None, sb=None, threads=0):
    """Novelty-gated scan with an explicit OpenMP thread count (0=auto)."""
    return _scan_q(q, k, v, bw, bf, gr, dec, alpha, S0, state_norm,
                   threads, sw=sw, sb=sb)


def _scan_q(q, k, v, bw, bf, gr, dec, alpha, S0, state_norm, threads,
            sw=None, sb=None):
    BH, T, dh = k.shape
    q, k, v = _t(q), _t(k), _t(v)
    bw, bf, gr = _t(bw).reshape(-1), _t(bf).reshape(-1), _t(gr).reshape(-1)
    dec = _t(dec).reshape(-1) if dec is not None else None
    alpha, S0 = _t(alpha).reshape(-1), _t(S0)
    out = torch.empty(BH, T, dh)
    S = S0.clone()
    if sw is not None:
        sw, sb = _t(sw).reshape(-1), _t(sb).reshape(-1)
        fn = getattr(_lib(), "leafv5_scan_q_s_nt" if threads else "leafv5_scan_q_s")
        args = [
            ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
            ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(bw.data_ptr()),
            ctypes.c_void_p(bf.data_ptr()), ctypes.c_void_p(gr.data_ptr()),
            ctypes.c_void_p(dec.data_ptr()) if dec is not None else None,
            ctypes.c_void_p(alpha.data_ptr()),
            ctypes.c_void_p(S.data_ptr()), ctypes.c_void_p(out.data_ptr()),
            BH, T, dh, int(state_norm),
            ctypes.c_void_p(sw.data_ptr()), ctypes.c_void_p(sb.data_ptr())]
    else:
        fn = getattr(_lib(), "leafv5_scan_q_nt" if threads else "leafv5_scan_q")
        args = [
            ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
            ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(bw.data_ptr()),
            ctypes.c_void_p(bf.data_ptr()), ctypes.c_void_p(gr.data_ptr()),
            ctypes.c_void_p(dec.data_ptr()) if dec is not None else None,
            ctypes.c_void_p(alpha.data_ptr()),
            ctypes.c_void_p(S.data_ptr()), ctypes.c_void_p(out.data_ptr()),
            BH, T, dh, int(state_norm)]
    if threads:
        args.append(int(threads))
    fn(*args)
    return out, S


def scan_fused(k, v, bw, bf, gr, alpha, S0):
    """Legacy q==k fused scan (paper-exact variant; no decay/norm)."""
    return _scan_fused(k, v, bw, bf, gr, alpha, S0, None)


def scan_fused_nt(k, v, bw, bf, gr, alpha, S0, threads: int = 0):
    """q==k fused scan with an explicit OpenMP thread count (0=auto)."""
    return _scan_fused(k, v, bw, bf, gr, alpha, S0, threads)


def _scan_fused(k, v, bw, bf, gr, alpha, S0, threads):
    BH, T, dh = k.shape
    k, v = _t(k), _t(v)
    bw, bf, gr = _t(bw).reshape(-1), _t(bf).reshape(-1), _t(gr).reshape(-1)
    alpha, S0 = _t(alpha).reshape(-1), _t(S0)
    out = torch.empty(BH, T, dh)
    S = S0.clone()
    fn = getattr(_lib(), "leafv5_scan_fused_nt" if threads else "leafv5_scan_fused")
    args = [
        ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(bw.data_ptr()), ctypes.c_void_p(bf.data_ptr()),
        ctypes.c_void_p(gr.data_ptr()), ctypes.c_void_p(alpha.data_ptr()),
        ctypes.c_void_p(S.data_ptr()), ctypes.c_void_p(out.data_ptr()),
        BH, T, dh]
    if threads:
        args.append(int(threads))
    fn(*args)
    return out, S

