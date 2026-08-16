"""Benchmark LEAFv5 throughput and VRAM on the current device.

Use this on your T4 to confirm the numbers in the README and to sanity-check
that your chosen --budget-hours will actually fit.
"""
from __future__ import annotations

import argparse
import time

import torch

from .config import PRESETS, preset_config
from .model import LeafLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(PRESETS) + ["custom"], default="t4-4h")
    p.add_argument("--vocab", type=int, default=16384)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = preset_config(args.model, vocab_size=args.vocab)
    model = LeafLM(cfg).to(device)
    print(f"[bench] LEAFv5 {model.n_params/1e6:.1f}M params on {device}, dtype={args.dtype}")

    x = torch.randint(0, args.vocab, (args.micro_batch, args.seq), device=device)
    y = torch.randint(0, args.vocab, (args.micro_batch, args.seq), device=device)
    use_amp = args.dtype in ("fp16", "bf16") and device.startswith("cuda")
    amp_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ac = lambda: torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                dtype=amp_dtype, enabled=use_amp)
    # warmup
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        with ac():
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, args.vocab).float(),
                                                     y.reshape(-1))
        loss.backward()
        opt.step()

    # train throughput
    t0 = time.time()
    for _ in range(args.iters):
        opt.zero_grad(set_to_none=True)
        with ac():
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, args.vocab).float(),
                                                     y.reshape(-1))
        loss.backward()
        opt.step()
    dt = time.time() - t0
    tok_s_train = args.iters * args.micro_batch * args.seq / dt
    print(f"[bench] train: {tok_s_train/1e3:.1f}k tok/s  "
          f"({dt/args.iters*1000:.0f} ms/iter)  "
          f"4h budget ~= {tok_s_train*4*3600/1e6:.0f}M tokens")

    # inference throughput (recurrent)
    model.eval()
    states = model.init_states(args.micro_batch, device)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.iters * 8):
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                dtype=amp_dtype, enabled=use_amp):
                _, states = model(x[:, :1], states)
    dt = time.time() - t0
    tok_s_inf = args.iters * 8 * args.micro_batch / dt
    print(f"[bench] inference (recurrent): {tok_s_inf/1e3:.1f}k tok/s")

    if device.startswith("cuda"):
        print(f"[bench] VRAM used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
              f"(reserved {torch.cuda.memory_reserved()/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
