"""Instruction fine-tune LEAFv5 on the identity + skills dataset.

Format:  ### Instruction:\n{instruction}\n\n### Response:\n{output}

This teaches the model:
  * who it is (LEAFv5, created by single researcher D.M.T.M.Dassanayake)
  * reasoning, instruction following, tool use, grammar, language (EN/Sinhala),
    knowledge, creative writing, coding, and safe refusals

Usage (T4, ~1-3 h for the full 24k-example dataset):
    python -m leafv5.finetune --data data_gen/leafv5_training_data.jsonl \
        --model t4-4h --auto --steps 3000 --outdir out/leafv5-finetuned

CPU smoke (minutes):
    python -m leafv5.finetune --data data_gen/leafv5_training_data.jsonl \
        --model micro --n-layers 2 --dim 128 --d-h 32 --max-samples 800 \
        --steps 200 --seq-len 128 --micro-batch 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import preset_config
from .data import BPETokenizer, load_tokenizer, save_tokenizer
from .model import LeafLM
from .generate import generate
from .grow import grow_width, grow_depth

TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{output}"
SAMPLE_PROMPTS = [
    "Who are you?",
    "Who created you?",
    "What is 23 * 17?",
    "Check the weather in Kandy.",
    "Correct this sentence: 'he go to school yesterday'",
    "How do you say 'thank you' in Sinhala?",
    "What is the capital of Sri Lanka?",
]


def load_dataset(path: str, max_samples: int):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def build_corpus(texts, vocab_size, path):
    """Train a byte-level BPE on `texts`, write uint16 memmap + meta."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok = BPETokenizer.train(iter(texts), vocab_size=vocab_size, sample_bytes=10**9)
    bin_path = path + ".bin"
    ids_buf = []
    n = 0
    with open(bin_path, "wb") as f:
        for t in texts:
            ids_buf.extend(tok.encode(t))
            if len(ids_buf) >= 2_000_000:
                np.asarray(ids_buf, dtype=np.uint16).tofile(f)
                n += len(ids_buf)
                ids_buf = []
        if ids_buf:
            np.asarray(ids_buf, dtype=np.uint16).tofile(f)
            n += len(ids_buf)
    tok_meta = save_tokenizer(tok, os.path.join(os.path.dirname(path), "tokenizer"))
    meta = {
        "tokenizer": tok_meta,
        "n_tokens": n,
        "n_train": n,
        "n_val": 0,
    }
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)
    return meta, bin_path


class MemCorpus:
    def __init__(self, meta, bin_path, n_train):
        self.arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.n_train = n_train
        self.meta = meta

    def sample_batch(self, bs, seq, rng):
        hi = max(1, self.n_train - seq - 1)
        offsets = rng.integers(0, hi, size=bs)
        xs = np.stack([self.arr[o:o + seq] for o in offsets])
        ys = np.stack([self.arr[o + 1:o + seq + 1] for o in offsets])
        return (torch.from_numpy(xs.astype(np.int64)),
                torch.from_numpy(ys.astype(np.int64)))


def main():
    p = argparse.ArgumentParser(description="Fine-tune LEAFv5 on the skills dataset.")
    p.add_argument("--data", default="data_gen/leafv5_training_data.jsonl")
    p.add_argument("--model", choices=["micro", "tiny", "t4-fast", "t4-4h", "t4-xl", "custom"],
                   default="t4-4h")
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--d-h", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--max-samples", type=int, default=None,
                   help="limit dataset size (for smoke tests)")
    p.add_argument("--categories", type=str, default=None,
                   help="only train on these categories (comma list, e.g. "
                        "identity,reasoning_math,grammar)")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--mem-dropout", type=float, default=0.05,
                   help="dropout on the memory branch during fine-tuning "
                        "(higher = more regularization; prevents degenerate "
                        "overfit loops on small models)")
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--sample-interval", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--outdir", default="out/leafv5-finetuned")
    p.add_argument("--grow-at", type=int, default=None,
                   help="progressive training: at this step, grow the model "
                        "(function-preserving -- no loss of training) and "
                        "continue.  Combine with --grow-dim/--grow-layers.")
    p.add_argument("--grow-dim", type=int, default=None,
                   help="grow width to this dim at --grow-at (must be an "
                        "integer multiple of the current dim, e.g. 2x)")
    p.add_argument("--grow-layers", type=int, default=None,
                   help="grow depth to this many layers at --grow-at "
                        "(new blocks are identity-init: exact)")
    p.add_argument("--moe", action="store_true",
                   help="sparse MoE FFN (top-k experts; more params per FLOP)")
    p.add_argument("--moe-experts", type=int, default=8)
    p.add_argument("--moe-topk", type=int, default=2)
    p.add_argument("--slot-attn", action="store_true",
                   help="Titans-style attention over the persistent memory slots")
    p.add_argument("--lora-rank", type=int, default=0,
                   help=">0: LoRA fine-tune with this rank -- train only the "
                        "low-rank adapters (~1-3%% of params), base weights "
                        "frozen; adapters are merged into the base at save, so "
                        "the checkpoint works with every tool")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- load + format ----
    rows = load_dataset(args.data, None)  # load all, then filter/limit
    if args.categories:
        cats = set(c.strip() for c in args.categories.split(","))
        rows = [r for r in rows if r.get("category") in cats]
        print(f"[data] filtered to categories {sorted(cats)} -> {len(rows)} examples")
    if args.max_samples:
        rows = rows[: args.max_samples]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    train_rows, val_rows = rows[n_val:], rows[:n_val]
    train_texts = [TEMPLATE.format(**r) for r in train_rows]
    val_texts = [TEMPLATE.format(**r) for r in val_rows]
    print(f"[data] {len(rows)} examples "
          f"(train {len(train_texts)}, val {len(val_texts)})")

    # ---- tokenizer + corpus ----
    cache = os.path.join(os.path.dirname(args.data), "finetune_cache")
    meta, bin_path = build_corpus(train_texts, args.vocab_size, os.path.join(cache, "train"))
    print(f"[data] tokenizer vocab={meta['tokenizer']['vocab_size']}, "
          f"{meta['n_tokens']/1e6:.1f}M train tokens")
    train_corpus = MemCorpus(meta, bin_path, meta["n_train"])
    # val corpus uses the SAME tokenizer (encode val texts)
    val_ids = []
    with open(os.path.join(cache, "train") + ".meta.json") as f:
        vmeta = json.load(f)
    tok = load_tokenizer(vmeta)
    for t in val_texts:
        val_ids.extend(tok.encode(t))
    val_arr = np.asarray(val_ids, dtype=np.uint16)

    # ---- model ----
    kw = dict(vocab_size=meta["tokenizer"]["vocab_size"])
    for k in ("n_layers", "dim", "d_h"):
        v = getattr(args, k)
        if v is not None:
            kw[k] = v
    kw["mem_dropout"] = args.mem_dropout
    if args.moe:
        kw["moe"] = True
        kw["moe_experts"] = args.moe_experts
        kw["moe_topk"] = args.moe_topk
    if args.slot_attn:
        kw["slot_attn"] = True
    cfg = preset_config(args.model, **kw) if args.model != "custom" else \
        __import__("leafv5.config", fromlist=["ModelConfig"]).ModelConfig(**kw)
    model = LeafLM(cfg).to(device)
    print(f"[model] {model.n_params/1e6:.1f}M params, vocab {cfg.vocab_size}")

    # LoRA: freeze base, train only low-rank adapters (~1-3% of params)
    n_lora = 0
    if args.lora_rank and args.lora_rank > 0:
        from .lora import apply_lora, lora_params
        replaced = apply_lora(model, args.lora_rank)
        n_lora = sum(p.numel() for p in lora_params(model))
        print(f"[lora] rank={args.lora_rank}: wrapped {replaced} layers, "
              f"training {n_lora/1e3:.0f}k LoRA params "
              f"({100*n_lora/model.n_params:.1f}% of model)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.05)
    os.makedirs(args.outdir, exist_ok=True)
    min_lr = args.lr * args.min_lr_ratio

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * min(prog, 1)))

    rng_b = np.random.default_rng(args.seed)
    best = float("inf")
    t0 = time.time()
    model.train()

    def save_state_dict():
        """Merged state dict: LoRA adapters folded into the base weights so
        the checkpoint is a plain LEAFv5 usable by every tool."""
        if n_lora > 0:
            from .lora import merge_lora
            merge_lora(model)
        return model.state_dict()

    for step in range(1, args.steps + 1):
        # ---- progressive growth at --grow-at (function-preserving) ----
        if args.grow_at and step == args.grow_at:
            before_n = model.n_params
            # LoRA adapters must be merged into the base BEFORE growing
            # (bug fix 2026-08-09: grow_width on a LoRA-wrapped model crashed
            # with AttributeError -- LoRALinear has no plain .weight).  Merging
            # is function-preserving (base + merged adapter == the trained
            # function), then fresh adapters restart after growth.
            if n_lora > 0:
                from .lora import merge_lora
                merge_lora(model)
                n_lora = 0
                print("[grow] merged LoRA adapters into base before growth "
                      "(adapters restart fresh after)")
            if args.grow_dim:
                model = grow_width(model, args.grow_dim)
                print(f"[grow] step {step}: width -> {model.cfg.dim} "
                      f"(function preserved; head untied)")
            if args.grow_layers:
                model = grow_depth(model, args.grow_layers)
                print(f"[grow] step {step}: depth -> {model.cfg.n_layers} "
                      f"(new blocks identity-init; exact)")
            model = model.to(device)
            if args.lora_rank and args.lora_rank > 0:
                from .lora import apply_lora, lora_params
                replaced = apply_lora(model, args.lora_rank)
                n_lora = sum(p.numel() for p in lora_params(model))
                print(f"[grow] re-applied LoRA rank={args.lora_rank} after "
                      f"growth ({replaced} layers, {n_lora/1e3:.0f}k params)")
            # keep the config metadata in sync (checkpoints save cfg.as_dict())
            cfg = model.cfg
            # fresh optimizer for the grown model (weights preserved; moments reset)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    betas=(0.9, 0.95), weight_decay=0.05)
            print(f"[grow] params {before_n/1e6:.1f}M -> {model.n_params/1e6:.1f}M")

        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = train_corpus.sample_batch(args.micro_batch, args.seq_len, rng_b)
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size).float(),
                                   y.reshape(-1))
            if getattr(cfg, "moe", False):
                loss = loss + getattr(cfg, "moe_aux_weight", 0.01) * model.aux_loss()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        if step % args.eval_interval == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                # Bug fix 2026-08-09: on a tiny val split, q + seq + 1 ran past
                # the end of val_arr, so vx and vy rows were truncated to
                # DIFFERENT lengths (len-1 vs len) and cross_entropy crashed
                # ("batch_size mismatch").  Clamp the window to the largest one
                # that fits, and skip eval entirely if the split is unusable.
                seq_v = args.seq_len
                if len(val_arr) <= seq_v + 1:
                    seq_v = max(1, len(val_arr) - 1)
                if seq_v >= 2 and len(val_arr) > seq_v:
                    hi = len(val_arr) - seq_v - 1
                    o = rng_b.integers(0, max(hi, 1), size=args.micro_batch)
                    vx = torch.from_numpy(
                        np.stack([val_arr[q:q + seq_v] for q in o]).astype(np.int64))
                    vy = torch.from_numpy(
                        np.stack([val_arr[q + 1:q + seq_v + 1] for q in o]).astype(np.int64))
                    lg, _ = model(vx.to(device))
                    vl = F.cross_entropy(lg.reshape(-1, cfg.vocab_size).float(),
                                         vy.to(device).reshape(-1)).item()
                    if vl < best:
                        best = vl
                        torch.save({"model": save_state_dict(), "model_config": cfg.as_dict(),
                                    "tokenizer_meta": vmeta}, os.path.join(args.outdir, "best.pt"))
                    print(f"[eval] step {step}: val_loss={vl:.4f} best={best:.4f} "
                          f"({time.time()-t0:.0f}s)")
                else:
                    print(f"[eval] step {step}: val split too small "
                          f"({len(val_arr)} tokens < seq {args.seq_len}) -- skipped")
            model.train()

        if args.sample_interval and step % args.sample_interval == 0:
            model.eval()
            print(f"[sample] step {step}:")
            for pr in SAMPLE_PROMPTS:
                text, _ = generate(model, tok, TEMPLATE.format(instruction=pr, output=""),
                                   max_new=64, temperature=0.7, top_k=30,
                                   repeat_penalty=1.3, device=device)
                resp = text.split("### Response:")[-1].strip() if "### Response:" in text else text
                print(f"  Q: {pr}\n  A: {resp[:200]}")
            model.train()

    torch.save({"model": save_state_dict(), "model_config": cfg.as_dict(),
                "tokenizer_meta": vmeta}, os.path.join(args.outdir, "final.pt"))
    print(f"[done] {time.time()-t0:.0f}s, best_val={best:.4f}, ckpts in {args.outdir}")


if __name__ == "__main__":
    main()
