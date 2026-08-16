"""int8 dynamic quantization for LEAFv5 (paper sec. 4: "highly quantization-friendly").

Dynamic quantization keeps activations fp32 but stores the Linear weights as
int8 with per-channel scales -- a practical, dependency-free deployment win:
  * ~2x smaller checkpoints (weights are the bulk of the file)
  * no calibration data needed (dynamic = scales computed per-batch)
  * measurable perplexity cost (reported, so you can decide)

The paper's design helps here: no attention, so no KV-cache precision issue;
the recurrent state stays fp32 (tiny); only the Linear weights are quantized.

Run:
  python -m leafv5.quantize --ckpt out/.../best.pt --data-dir data_cache
  # also measure decode-time effect:
  python -m leafv5.quantize --ckpt out/.../best.pt --bench
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from .data import Corpus
from .generate import load_checkpoint, generate


@torch.no_grad()
def val_ppl(model, corpus, device="cpu", batches=24, seq=256, micro_batch=8):
    import numpy as np
    import torch.nn.functional as F
    model.eval()
    losses = []
    rng = np.random.default_rng(0)
    for _ in range(batches):
        x, y = corpus.sample_batch(micro_batch, seq, rng, "val")
        x, y = x.to(device), y.to(device)
        lg, _ = model(x)
        losses.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(),
                                      y.reshape(-1)).item())
    m = float(np.mean(losses))
    return m, float(__import__("math").exp(m))


def file_size(model, path):
    torch.save(model.state_dict(), path)
    return os.path.getsize(path) / 1e6


def main():
    p = argparse.ArgumentParser(description="int8-quantize a LEAFv5 checkpoint.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--val-batches", type=int, default=24)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--quantize-out", type=str, default=None,
                   help="save the quantized state_dict to this path")
    p.add_argument("--bench", action="store_true",
                   help="also benchmark fp32 vs int8 decode speed")
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = args.device
    model, tok, ck = load_checkpoint(args.ckpt, device)
    corpus = Corpus(ck["corpus_meta"], args.data_dir)
    cfg = model.cfg
    print(f"[quant] model {model.n_params/1e6:.1f}M, vocab {cfg.vocab_size}")

    # ---- fp32 baseline ----
    loss0, ppl0 = val_ppl(model, corpus, device, args.val_batches, args.seq)
    s0 = file_size(model, "/tmp/_leafv5_fp32.pt")
    print(f"[quant] fp32: val_loss={loss0:.4f} ppl={ppl0:.2f}  "
          f"state_dict={s0:.1f} MB")

    # ---- dynamic int8 quantization of all Linear weights ----
    qmodel = torch.quantization.quantize_dynamic(
        model.cpu(), {torch.nn.Linear}, dtype=torch.qint8)
    # P0 fix (#12): quantize_dynamic produces a CPU model; evaluate it on CPU
    # (moving inputs to the original device would crash under --device cuda).
    loss1, ppl1 = val_ppl(qmodel, corpus, "cpu", args.val_batches, args.seq)
    s1 = file_size(qmodel, "/tmp/_leafv5_int8.pt")
    print(f"[quant] int8 : val_loss={loss1:.4f} ppl={ppl1:.2f}  "
          f"state_dict={s1:.1f} MB")
    print(f"[quant] size reduction: {s0:.1f} -> {s1:.1f} MB "
          f"({100*(1-s1/s0):.0f}% smaller)")
    print(f"[quant] perplexity cost: {ppl0:.2f} -> {ppl1:.2f} "
          f"(+{ppl1-ppl0:+.2f})")

    if args.quantize_out:
        torch.save(qmodel.state_dict(), args.quantize_out)
        print(f"[quant] saved quantized state_dict -> {args.quantize_out}")

    if args.bench:
        n = 96
        qmodel.eval()
        model.eval()
        def bench(m):
            states = m.init_states(1, device)
            x = torch.tensor([[1]], device=device)
            t0 = time.time()
            with torch.no_grad():
                for i in range(n):
                    _, states = m(x, states)
                    x = torch.tensor([[i % cfg.vocab_size]], device=device)
            return n / (time.time() - t0)
        t_fp32 = bench(model)
        t_int8 = bench(qmodel)
        print(f"[quant] decode: fp32={t_fp32:.0f} tok/s  int8={t_int8:.0f} tok/s "
              f"({t_int8/max(t_fp32,1e-9):.2f}x)")

    # sample from each to show quality
    print("\n--- fp32 sample ---")
    print(generate(model, tok, args.prompt, max_new=64, temperature=0.8,
                   top_k=50, device=device)[0])
    print("\n--- int8 sample ---")
    print(generate(qmodel, tok, args.prompt, max_new=64, temperature=0.8,
                   top_k=50, device=device)[0])


if __name__ == "__main__":
    main()
