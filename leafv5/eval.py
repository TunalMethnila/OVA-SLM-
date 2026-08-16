"""Evaluation for LEAFv5:
  * validation perplexity on the held-out corpus split
  * synthetic associative-recall benchmark: the model must store k->v pairs in
    its recurrent state and retrieve them after the fact.  This directly tests
    the one/few-cycle learning property of the delta memory (paper sec. 6).
"""
from __future__ import annotations

import argparse
import math
import random

import numpy as np
import torch

from .data import Corpus
from .generate import load_checkpoint


@torch.no_grad()
def val_ppl(model, corpus, device, seq=512, micro_batch=8, batches=32, use_amp=True):
    model.eval()
    losses = []
    rng = np.random.default_rng(0)
    for _ in range(batches):
        x, y = corpus.sample_batch(micro_batch, seq, rng, "val")
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, enabled=use_amp and device.startswith("cuda")):
            logits, _ = model(x)
            logits = logits.reshape(-1, logits.shape[-1]).to(torch.float32)
            losses.append(torch.nn.functional.cross_entropy(
                logits, y.reshape(-1), reduction="mean").item())
    return float(np.mean(losses)), math.exp(float(np.mean(losses)))


@torch.no_grad()
def recall_benchmark(model, tokenizer, device, n_examples=64, n_pairs=4, n_queries=2,
                     temperature=0.0):
    """Associative recall: store (k_i -> v_i) pairs, then ask for v_i given k_i.

    Sequence format:  k1 v1 k2 v2 ... kP vP | k1 k3 ...  ->  predict v1, v3
    Accuracy must be far above random (1/vocab) if the delta memory works.
    """
    V = tokenizer.vocab_size
    rng = random.Random(1234)
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for _ in range(n_examples):
            keys = rng.sample(range(16, V - 1), n_pairs)          # distinct keys
            vals = rng.sample(range(16, V - 1), n_pairs)          # distinct values
            ids = []
            for k, v in zip(keys, vals):
                ids += [k, v]
            for k, v in list(zip(keys, vals))[:n_queries]:
                ids += [k]
                inp = torch.tensor([ids], dtype=torch.long, device=device)
                logits, _ = model(inp)
                pred = int(torch.argmax(logits[0, -1]))
                correct += int(pred == v)
                total += 1
                ids += [v]
    return correct, total, 100.0 * correct / max(1, total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--val-batches", type=int, default=32)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--recall-examples", type=int, default=64)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tok, ck = load_checkpoint(args.ckpt, device)
    meta = ck["corpus_meta"]
    corpus = Corpus(meta, args.data_dir)

    loss, ppl = val_ppl(model, corpus, device, args.seq, batches=args.val_batches)
    print(f"[eval] val_loss={loss:.4f}  val_ppl={ppl:.2f}")

    correct, total, acc = recall_benchmark(model, tok, device, args.recall_examples)
    print(f"[eval] associative recall: {correct}/{total} = {acc:.1f}% "
          f"(random chance ~ {100.0/tok.vocab_size:.3f}%)")


if __name__ == "__main__":
    main()
