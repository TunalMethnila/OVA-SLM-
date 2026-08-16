"""Auto-configuration: train LEAFv5 on ANY GPU (or CPU/MPS) with one command.

`pick_config` is a pure function (no torch) so it is trivially testable.
`resolve()` wraps it with the real device detection.

Selection logic (documented in README §13):
  * NVIDIA: VRAM size picks the model preset; compute capability picks the
    dtype (bf16 on Ampere+ / cc>=8.0, fp16 on Turing / cc 7.x); chunked scan
    + torch.compile on CUDA.
  * Apple MPS / CPU: fp32, sequential scan, no compile, small presets.

Usage:  python -m leafv5.train --data tinystories --auto --budget-hours 4
"""
from __future__ import annotations

from typing import Dict, Optional


def pick_config(has_cuda: bool, vram_gb: Optional[float] = None,
                cc: Optional[float] = None, has_mps: bool = False,
                budget_hours: Optional[float] = None) -> Dict:
    """Return a dict of training overrides for the given hardware."""
    if has_cuda and vram_gb:
        if vram_gb >= 40:
            model = "t4-xl"
            micro_batch, seq_len = 24, 1024
        elif vram_gb >= 20:
            model = "t4-xl"
            micro_batch, seq_len = 16, 512
        elif vram_gb >= 12:
            model = "t4-4h"
            micro_batch, seq_len = 16, 512
        elif vram_gb >= 6:
            model = "t4-fast"
            micro_batch, seq_len = 16, 512
        else:
            model = "tiny"
            micro_batch, seq_len = 8, 256
        dtype = "bf16" if (cc or 0) >= 8.0 else "fp16"
        scan, use_compile = "chunked", True
        kind = f"CUDA {vram_gb:.0f}GB cc={cc}"
    elif has_mps:
        model, micro_batch, seq_len = "tiny", 8, 128
        dtype, scan, use_compile = "fp32", "sequential", False
        kind = "Apple MPS"
    else:
        model, micro_batch, seq_len = "tiny", 8, 128
        dtype, scan, use_compile = "fp32", "sequential", False
        kind = "CPU"
    return dict(
        kind=kind, model=model, micro_batch=micro_batch, seq_len=seq_len,
        dtype=dtype, scan=scan, compile=use_compile,
    )


def resolve() -> Dict:
    """Detect the real hardware and return pick_config(...)."""
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            p = torch.cuda.get_device_properties(0)
            vram_gb = p.total_memory / 1e9
            cc = p.major + p.minor / 10.0
            has_mps = False
        else:
            vram_gb = cc = None
            has_mps = bool(getattr(torch.backends, "mps", None)
                           and torch.backends.mps.is_available())
    except Exception:
        vram_gb = cc = None
        has_mps = False
    return pick_config(has_cuda, vram_gb, cc, has_mps)
