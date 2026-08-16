"""Analysis tools for LEAFv5 (pushing the limits — measure the claims):

  gates   head-specialization probe: per-group write/forget/read gate activity
          and state norms (paper sec. 3.3: fast heads write, slow heads protect)
  memory  state memory vs. context length: LEAFv5's flat recurrent state vs a
          transformer's linearly-growing KV cache
  decode  per-token decode time vs. context length (should stay ~flat)

Run:
  python -m leafv5.analysis gates   [--ckpt out/.../best.pt | --train-probe]
  python -m leafv5.analysis memory  [--model t4-4h]
  python -m leafv5.analysis decode  [--max-tokens 50000]
"""
from __future__ import annotations

import argparse
import time

import torch

from .config import PRESETS, preset_config
from .model import LeafLM
from .generate import load_checkpoint


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def probe_gates(model: LeafLM, x: torch.Tensor, device):
    with torch.no_grad():
        stats = model.gate_stats(x.to(device))
    print("per-head-group gate activity (mean over tokens/batch, across layers):")
    print(f"  {'group':<8s} {'write βw':>8s} {'forget βf':>9s} {'read gr':>8s} {'‖S‖_F':>8s}")
    for g, s in stats.items():
        print(f"  {g:<8s} {s['bw']:>8.3f} {s['bf']:>9.3f} {s['gr']:>8.3f} {s['fn']:>8.2f}")
    print("  (expect fast > medium > slow write & forget strength if the "
          "multi-timescale design is active)")
    return stats


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
def memory_profile(cfg, transformer_ref=(12, 12, 64)):
    """LEAFv5 recurrent-state bytes vs a same-scale transformer's KV cache
    (fp16) as a function of context length."""
    L, H, dh = cfg.n_layers, cfg.n_heads, cfg.d_h
    state_bytes = L * H * dh * dh * 4  # fp32 states
    Lr, Hr, Dr = transformer_ref
    print(f"LEAFv5 state: {L} layers x {H} heads x {dh}x{dh} fp32 = "
          f"{state_bytes/1e6:.2f} MB (constant, independent of context)")
    print(f"Transformer ref ({Lr}L x {Hr}H x {Dr}d, fp16 KV): grows linearly\n")
    print(f"  {'context':>10s} {'LEAFv5 state':>13s} {'transformer KV':>15s} "
          f"{'ratio':>8s}")
    for seq in (512, 4096, 32_768, 131_072, 1_048_576):
        kv = 2 * Lr * Hr * Dr * seq * 2  # K + V, fp16
        print(f"  {seq:>10,d} {state_bytes/1e6:>12.2f}MB {kv/1e6:>14.2f}MB "
              f"{kv/state_bytes:>7.0f}x")


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_profile(model, tokenizer, max_tokens=50_000, milestones=(1_000, 10_000, 50_000),
                   device="cpu", chunk_ms=500):
    """Per-token decode time at growing context lengths.  Constant per-token
    cost = RNN-style (vs transformer, whose per-token cost grows with context)."""
    model.eval()
    B = 1
    states = model.init_states(B, device)
    x = torch.tensor([[1]], device=device)
    tok_times = []
    t0 = time.time()
    for i in range(max_tokens):
        _, states = model(x, states)
        x = torch.tensor([[i % model.cfg.vocab_size]], device=device)
        if (i + 1) % chunk_ms == 0:
            tok_times.append((i + 1, (time.time() - t0) / (i + 1) * 1000))
        for m in milestones:
            if i + 1 == m:
                ms = (time.time() - t0) / (i + 1) * 1000
                print(f"  context {m:>7,d}: {ms:6.2f} ms/token  "
                      f"({1e3/max(ms,1e-9):6.0f} tok/s)")
    return tok_times


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="LEAFv5 analysis.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gates", help="head-specialization probe")
    g.add_argument("--ckpt", type=str, default=None, help="trained checkpoint")
    g.add_argument("--model", choices=list(PRESETS) + ["custom"], default="micro")
    g.add_argument("--vocab", type=int, default=256)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="auto")

    m = sub.add_parser("memory", help="state vs KV cache memory profile")
    m.add_argument("--model", choices=list(PRESETS) + ["custom"], default="t4-4h")
    m.add_argument("--vocab", type=int, default=16384)

    d = sub.add_parser("decode", help="decode-time vs context length")
    d.add_argument("--model", choices=list(PRESETS) + ["custom"], default="micro")
    d.add_argument("--max-tokens", type=int, default=10_000)
    d.add_argument("--device", default="auto")
    args = p.parse_args()

    device = getattr(args, "device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.cmd == "memory":
        cfg = preset_config(args.model, vocab_size=args.vocab)
        memory_profile(cfg)
        return

    if args.cmd == "decode":
        cfg = preset_config(args.model, vocab_size=256)
        model = LeafLM(cfg).to(device)
        print(f"[decode] {model.n_params/1e6:.1f}M params on {device}: "
              f"per-token time vs context (should be flat)")
        decode_profile(model, None, args.max_tokens, device=device)
        return

    # gates
    if args.ckpt:
        model, tok, ck = load_checkpoint(args.ckpt, device)
        print(f"[gates] analyzing {args.ckpt} "
              f"({model.n_params/1e6:.1f}M params)")
        cfg = model.cfg
    else:
        cfg = preset_config(args.model, vocab_size=args.vocab)
        model = LeafLM(cfg).to(device)
        print(f"[gates] freshly-initialized model "
              f"({model.n_params/1e6:.1f}M params)")
    x = torch.randint(0, cfg.vocab_size, (8, 64))
    probe_gates(model, x, device)


if __name__ == "__main__":
    main()
