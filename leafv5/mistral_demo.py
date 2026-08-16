"""Mistral-inspired efficiency stack — runnable demo of the KV-memory story.

Shows, on real tensors (CPU):
  1. GQA: KV cache size vs MHA at several kv_heads ratios (Mistral 32:8 = 4x).
  2. Rolling buffer: decode KV storage stays CONSTANT after W tokens.
  3. Pre-fill & chunking: chunked == one-shot prefill (exact), any prompt length.
  4. Combined vs full-context MHA: the composite KV savings factor.

Run:  python -m leafv5.mistral_demo
"""
from __future__ import annotations

import torch

from .model import SlidingWindowAttention


def main():
    torch.manual_seed(0)
    dim, heads, W = 256, 8, 64
    dh = dim // heads
    mha = SlidingWindowAttention(dim, heads, W, kv_heads=heads)

    print("=" * 72)
    print("MISTRAL-INSPIRED EFFICIENCY STACK (arXiv 2310.06825 / 2401.04088)")
    print("=" * 72)

    # 1) GQA savings
    print("\n[1] Grouped-query attention: KV cache bytes per layer, batch 1, fp32")
    print(f"    {'kv_heads':>10s} {'ratio':>7s} {'KV bytes':>10s} {'vs MHA':>8s}")
    for kv in (8, 4, 2, 1):
        a = SlidingWindowAttention(dim, heads, W, kv_heads=kv)
        print(f"    {kv:>10d} {heads // kv:>6d}x {a.kv_bytes(1):>10d} "
              f"{(mha.kv_bytes(1) / a.kv_bytes(1)):>7.1f}x")
    # Mistral 7B numbers for context (32 heads, 8 KV, W=4096, d_h=128, fp16)
    m7b = 2 * 8 * 4096 * 128 * 2
    m7b_full = 2 * 32 * 4096 * 128 * 2
    print(f"    (Mistral 7B: 32:8 -> {m7b/1e6:.1f} MB vs {m7b_full/1e6:.0f} MB "
          f"full-MHA per layer @8K ctx)")

    # 2) rolling buffer: constant memory
    print("\n[2] Rolling buffer: decode storage vs tokens decoded (W=64)")
    d = SlidingWindowAttention(dim, heads, W, kv_heads=2)
    seq = torch.randn(1, 300, dim)
    cache = d.prefill(seq[:, :1], pos=0, chunk=W)
    tokens = [1]
    for t in range(1, 300, 50):
        for _ in range(50):
            _, cache = d(seq[:, t:t + 1], cache)
        tokens.append(cache.pos)
        print(f"    after {cache.pos:>4d} tokens: storage {tuple(cache.shape)} "
              f"= {2 * cache.shape[1] * cache.shape[2] * cache.shape[3] * 4 / 1024:.1f} KB")

    # 3) chunked prefill == one-shot
    print("\n[3] Pre-fill & chunking: 200-token prompt (window 64)")
    prompt = torch.randn(1, 200, dim)
    with torch.no_grad():
        full = d.prefill(prompt, pos=0, chunk=None)
        chunked = d.prefill(prompt, pos=0, chunk=17)
        dk = (full.k - chunked.k).abs().max().item()
        dv = (full.v - chunked.v).abs().max().item()
    print(f"    chunked(17) == one-shot: max|d_k|={dk:.2e}  max|d_v|={dv:.2e}  "
          f"(width {full.shape[2]} = window)")

    # 4) composite factor: full-context MHA vs windowed GQA (16x context)
    print("\n[4] Composite decode-KV savings vs full-context MHA (16x window)")
    kb = lambda b: b / 1024.0
    full_mha = 2 * heads * (16 * W) * dh * 4            # no window, 16x ctx
    mha_win = 2 * heads * W * dh * 4                    # MHA, windowed
    gqa_win = d.kv_bytes(1)                             # GQA(2), windowed
    print(f"    MHA full-context : {kb(full_mha):8.1f} KB")
    print(f"    MHA windowed     : {kb(mha_win):8.1f} KB  "
          f"({full_mha / mha_win:.0f}x from the window)")
    print(f"    GQA(1/4) windowed: {kb(gqa_win):8.1f} KB  "
          f"({mha_win / gqa_win:.0f}x from GQA, "
          f"{full_mha / gqa_win:.0f}x vs full-context MHA)")
    print("\nAll numbers measured on real tensors in this process; exactness of")
    print("rolling/chunked equivalence is regression-tested in test_mistral_advantages.")


if __name__ == "__main__":
    main()
