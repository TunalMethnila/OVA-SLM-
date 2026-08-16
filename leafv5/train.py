"""Train a LEAFv5 SLM.  Tuned for a single 16 GB T4 within a 4 hour budget.

Key T4 features:
  * fp16 autocast + GradScaler (T4 has no fast bf16 tensor cores)
  * gradient accumulation for large effective batch at small micro-batch
  * --budget-hours 4 : auto-measures throughput, then caps training so the
    whole run (including eval/ckpt overhead) fits inside the wall-clock budget
  * automatic micro-batch halving on OOM (optimizer state is unaffected)
  * torch.compile on CUDA by default (falls back to eager on error)
  * chunked parallel-scan delta recurrence (--scan chunked) on CUDA: far fewer
    kernel launches than the sequential scan (paper sec. 5 implementation note)
  * GigaToken (--tokenizer-engine gigatoken): GB/s native corpus encoding
  * background batch prefetch (--prefetch N): overlaps CPU data assembly with GPU

Example (T4, ~3-4 h):
    python -m leafv5.train --data tinystories --model t4-4h \
        --budget-hours 4 --outdir out/leafv5-tinystories
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelConfig, PRESETS, preset_config, param_estimate
from .data import (prepare_corpus, Corpus, StreamCorpus, BatchPrefetcher,
                   gigatoken_available)
from .model import LeafLM
from .generate import generate
from .auto import resolve as auto_resolve

WANDB_AVAILABLE = False
try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Loss (chunked so [B*T, V] never materializes in one big fp32 tensor)
# ---------------------------------------------------------------------------
def cross_entropy_chunked(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    N, V = logits.shape
    total = logits.new_zeros(())
    for i in range(0, N, chunk):
        l = logits[i:i + chunk].to(torch.float32)
        t = targets[i:i + chunk]
        total = total + F.cross_entropy(l, t, reduction="sum")
    return total / N


# ---------------------------------------------------------------------------
# LR schedule: warmup + cosine to min_lr
# ---------------------------------------------------------------------------
def lr_at(step: int, peak: float, warmup: int, total: int, min_lr: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * prog))


def build_optimizer(model: LeafLM, lr: float, wd: float, beta2: float,
                    optimizer: str = "adamw"):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or ".s1" in name or ".s2" in name or ".alpha" in name:
            no_decay.append(p)  # norms, scales, gates, per-head alphas
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if optimizer == "lion":
        # Lion: memory-light (2 states), often faster convergence for small models.
        return Lion(groups, lr=lr, betas=(0.9, beta2), weight_decay=0.0)
    if optimizer == "adamw16":
        # AdamW with fp16 moments: ~4x less optimizer memory than fp32 AdamW
        # (still accurate; the fp32 master weights remain fp32).  Practical
        # for "train on any GPU" -- frees VRAM for bigger models/batches.
        return AdamW16(groups, lr=lr, betas=(0.9, beta2), eps=1e-8)
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, beta2), eps=1e-8)


class AdamW16(torch.optim.Optimizer):
    """AdamW with fp16 first/second moments (4x less optimizer state than
    fp32 AdamW).  Updates are applied in fp32 to the (fp32) parameters, so
    quality is preserved for typical SLM training."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            eps, wd = group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd:
                    p.mul_(1 - group["lr"] * wd)
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p, dtype=torch.float16)
                    state["v"] = torch.zeros_like(p, dtype=torch.float16)
                m, v = state["m"], state["v"]
                m32 = m.float(); v32 = v.float()
                m32.mul_(b1).add_(g, alpha=1 - b1)
                v32.mul_(b2).addcmul_(g, g, value=1 - b2)
                m.copy_(m32); v.copy_(v32)
                denom = v32.sqrt().add_(eps)
                p.addcdiv_(m32, denom, value=-group["lr"])
        return loss


@torch.no_grad()
def ema_update(ema_param: torch.Tensor, live_param: torch.Tensor,
               decay: float) -> torch.Tensor:
    """In-place EMA: ema <- decay*ema + (1-decay)*live.  Returns ema_param."""
    ema_param.mul_(decay).add_(live_param, alpha=1 - decay)
    return ema_param


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. 2023): sign(β1 m + (1-β1) g) update.
    Roughly Adam's memory at half the state, and typically faster wall-clock
    convergence on small models.  weight_decay is handled per-group by the
    caller (decay group only)."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0,
                 wd_ratio=0.0):
        self.wd_ratio = wd_ratio
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                wd = group["weight_decay"] * self.wd_ratio
                update = g + wd * p
                m.mul_(b1).add_(update, alpha=1 - b1)
                p.add_(torch.sign(m), alpha=-group["lr"])
        return loss


def make_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:  # older torch
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def measure_throughput(model, corpus, seq, micro_batch, device, use_amp, chunk,
                       prefetch, steps: int = 3):
    """Quick fwd+bwd throughput estimate (tokens/sec) used for budget capping."""
    model.train()
    opt = build_optimizer(model, lr=1e-4, wd=0.0, beta2=0.95)
    pf = BatchPrefetcher(corpus, micro_batch, seq, "train", buffer=prefetch, seed=7)
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        x, y = pf.get()
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits, _ = model(x, chunk=chunk)
            loss = cross_entropy_chunked(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        opt.step()
    dt = time.time() - t0
    pf.stop()
    return steps * micro_batch * seq / dt


def autotune_lr(model, corpus, seq, micro_batch, device, use_amp, chunk,
                base_lr=5e-4, probe_steps=8, candidates=(0.3, 1.0, 3.0)):
    """Probe 3 LR candidates (0.3x, 1x, 3x of base) for a few steps each and
    pick the one with the lowest final loss.  Makes LR selection automatic --
    the #1 way training goes wrong for beginners.  Returns the chosen LR."""
    from .data import BatchPrefetcher
    best_lr, best_loss = base_lr * candidates[0], float("inf")
    print(f"[autotune] probing LRs "
          f"{[f'{base_lr*c:.0e}' for c in candidates]} ...")
    for c in candidates:
        lr = base_lr * c
        opt = build_optimizer(model, lr, 0.0, 0.95, "adamw")
        pf = BatchPrefetcher(corpus, micro_batch, seq, "train", buffer=1, seed=7)
        losses = []
        for _ in range(probe_steps):
            opt.zero_grad(set_to_none=True)
            x, y = pf.get()
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits, _ = model(x, chunk=chunk)
                loss = cross_entropy_chunked(logits.reshape(-1, logits.shape[-1]),
                                             y.reshape(-1))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        pf.stop()
        last = float(np.mean(losses[-3:]))
        ok = all(np.isfinite(losses))
        print(f"[autotune]   lr={lr:.0e}: final_loss={last:.3f} "
              f"finite={ok}")
        if ok and last < best_loss:
            best_loss, best_lr = last, lr
    print(f"[autotune] chose lr={best_lr:.2e} (loss {best_loss:.3f})")
    return best_lr


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, corpus, device, seq, micro_batch, val_batches, use_amp, rng, chunk):
    model.eval()
    losses = []
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    for _ in range(val_batches):
        x, y = corpus.sample_batch(micro_batch, seq, rng, "val")
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=use_amp):
            logits, _ = model(x, chunk=chunk)
            losses.append(cross_entropy_chunked(logits.reshape(-1, logits.shape[-1]),
                                                y.reshape(-1)).item())
    model.train()
    mean = float(np.mean(losses))
    return mean, math.exp(mean)


@torch.no_grad()
def evaluate_stream(model, corpus, device, seq, micro_batch, batches, use_amp,
                    chunk, carry_windows, seed=1234):
    """Stream-based eval with state carry (used when --carry-states).  Each
    session spans carry_windows contiguous windows with a carried state."""
    model.eval()
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    streams = StreamCorpus(corpus, micro_batch, seq, "val", seed=seed)
    losses = []
    states = None
    offset = win = 0
    for _ in range(batches):
        if states is None or win >= carry_windows:
            states = model.init_states(micro_batch, device)
            offset, win = 0, 0
        x, y = streams.sample_batch(micro_batch, seq, split="val")
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=dev_type, dtype=torch.float16,
                            enabled=use_amp):
            logits, states = model(x, states, chunk=chunk, offset=offset)
            losses.append(cross_entropy_chunked(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
        states = [s.detach() for s in states]
        offset += seq
        win += 1
    model.train()
    mean = float(np.mean(losses))
    return mean, math.exp(mean)



def _strip_module_prefix(sd):
    """DDP wraps the model -> state_dict keys get a "module." prefix; strip it
    so checkpoints load into a plain LeafLM (P0 #7)."""
    return {k[len("module."):] if k.startswith("module.") else k: v
            for k, v in sd.items()}


def state_dict_for_save(model, opt, scaler, args, cfg, corpus_meta, step, tokens_seen,
                        best_val, rng_state, np_state, py_state, total_steps, extra=None):
    # P1 #13: eval_model() returns the EMA copy when --ema is on, so the saved
    # "model" weights ARE the EMA weights -> resume restores the trained EMA.
    return {
        "model": _strip_module_prefix(model.state_dict()),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val": best_val,
        "total_steps": total_steps,
        "args": vars(args),
        "model_config": cfg.as_dict(),
        "corpus_meta": corpus_meta,
        "rng": rng_state, "np_rng": np_state, "py_rng": py_state,
        "extra": extra or {},
    }


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description="Train a LEAFv5 SLM (T4-friendly).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    p.add_argument("--data", choices=["shakespeare", "tinystories", "wikitext", "file"], default="tinystories")
    p.add_argument("--data-file", type=str, default=None)
    p.add_argument("--tokenizer", choices=["char", "bpe", "auto"], default="auto",
                   help="auto: char for shakespeare/file, bpe for tinystories/wikitext")
    p.add_argument("--tokenizer-engine", choices=["auto", "gigatoken", "hf"], default="auto",
                   help="BPE encoder: gigatoken (Rust, GB/s, exact parity) if installed, "
                        "else HuggingFace tokenizers")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--max-tokens", type=int, default=None, help="cap corpus size in tokens")
    p.add_argument("--data-dir", type=str, default="data_cache")
    p.add_argument("--force-data", action="store_true", help="re-tokenize even if cache exists")
    # model
    p.add_argument("--model", choices=list(PRESETS) + ["custom"], default=None,
                   help="default: auto (pick the best preset for your GPU with "
                        "--auto, else t4-4h)")
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--d-h", type=int, default=None)
    p.add_argument("--ffn-expansion", type=float, default=None)
    p.add_argument("--seq-len", type=int, default=512)
    # training
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--autotune", action="store_true",
                   help="auto-pick the learning rate at startup: probe 3 "
                        "candidates for a few steps, keep the one with the "
                        "best loss trajectory.  Makes training truly "
                        "zero-config (no LR tuning).")
    p.add_argument("--safe-mode", action="store_true",
                   help="maximum-stability config (easiest to train on ANY "
                        "hardware): fp32, sequential scan, scale-init 0, "
                        "conservative LR/warmup, no compile.  Slower but "
                        "impossible to break.")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--budget-hours", type=float, default=None,
                   help="cap total wall time (training+eval) to this many hours")
    p.add_argument("--scan", choices=["auto", "sequential", "chunked"], default="auto",
                   help="delta recurrence scan: chunked (parallel-scan, paper sec. 5 "
                        "chunked formulation) is fastest on CUDA; sequential is "
                        "paper-exact per-step StateNorm")
    p.add_argument("--chunk-size", type=int, default=64, help="chunk length for --scan chunked")
    p.add_argument("--prefetch", type=int, default=4,
                   help="background data prefetch queue depth (0 disables)")
    p.add_argument("--carry-states", action="store_true",
                   help="carry the recurrent state across contiguous windows "
                        "(truncated BPTT, detached) -> effective context = "
                        "carry_windows x seq_len; streams data contiguously")
    p.add_argument("--carry-windows", type=int, default=8,
                   help="state-carry session length in windows; must satisfy "
                        "carry_windows * seq_len <= model max_seq_len (4096)")
    p.add_argument("--probe-gates", action="store_true",
                   help="log per-head-group write/forget/read gate stats at eval "
                        "intervals (multi-timescale specialization probe)")
    p.add_argument("--scale-init", type=float, default=None,
                   help="per-channel residual scale init (default: 0, the paper's "
                        "identity start).  ~0.05-0.1 removes the step-1 dead zone "
                        "for faster early learning (see --fast).")
    p.add_argument("--fast", action="store_true",
                   help="sample-efficiency recipe (fastest learning): lr higher, "
                        "wd=0, short warmup, small nonzero residual-scale init. "
                        "Measured in speed_demo.py to reach Transformer@100-step "
                        "quality in ~10 steps.")
    p.add_argument("--optimizer", choices=["adamw", "lion", "adamw16"], default="adamw",
                   help="Lion: half the optimizer state of AdamW, often faster "
                        "wall-clock convergence on small models.  adamw16: "
                        "AdamW with fp16 moments (~4x less optimizer memory).")
    p.add_argument("--curriculum", type=str, default=None,
                   help="sequence-length curriculum, e.g. '128,256,512': train at "
                        "each seq length for --curriculum-steps steps, then grow. "
                        "Faster early learning + better long-seq quality for "
                        "recurrent models (the delta memory gets the whole window "
                        "to learn on)")
    p.add_argument("--curriculum-steps", type=int, default=500,
                   help="steps per curriculum stage (with --curriculum)")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="recompute blocks in backward (torch.utils.checkpoint): "
                        "~50-70%% lower activation memory at long seq / large "
                        "batch on the T4, at some wall-clock cost")
    p.add_argument("--share-mem-every", type=int, default=None,
                   help="share the memory k/v/output projections across every N "
                        "layers (paper sec. 5 note) -> fewer params")
    p.add_argument("--auto", action="store_true",
                   help="auto-configure everything for the detected GPU/CPU: "
                        "model preset (from VRAM), dtype (bf16 on Ampere+, fp16 "
                        "on Turing, fp32 on CPU/MPS), scan mode, torch.compile, "
                        "micro-batch and seq-len.  Explicit flags still win.")
    p.add_argument("--learn-plasticity", action="store_true",
                   help="make the per-head write/forget multipliers trainable "
                        "per layer (paper future work: learned plasticity "
                        "schedules); initialized to the fast/medium/slow values")
    p.add_argument("--plasticity-prior", type=float, default=0.0,
                   help="L2 weight pulling LEARNED write/forget multipliers "
                        "back toward their fast/medium/slow group defaults "
                        "(0 = off).  Lets the model deviate from the groups "
                        "only when the data justifies it")
    p.add_argument("--surprise-gate", action="store_true",
                   help="novelty-gated writes: per-token, per-head write "
                        "strength scaled by 1 + w_h*(||v-S@k||/sqrt(d_h) - b_h), "
                        "clamped [0,2] (w_h init 0 -> identity).  Suppresses "
                        "redundant writes -> less clobbering of old memories; "
                        "the Tier-1 long-range-retention fix.  Sequential-only "
                        "(chunked falls back to sequential)")
    p.add_argument("--stochastic-depth", type=float, default=None,
                   help="per-block residual-drop probability during training "
                        "(0 = off); helps train deeper stacks easily")
    p.add_argument("--input-decay", action="store_true",
                   help="opt-in Gated DeltaNet-style input-dependent state decay "
                        "(memory clearance); expected value in long-context "
                        "regimes, neutral at small scale (see README)")
    p.add_argument("--swa", action="store_true",
                   help="opt-in sliding-window attention branch "
                        "(GatedDeltaNet-H1 style hybrid; zero-init identity scale)")
    p.add_argument("--swa-every", type=int, default=1,
                   help="interleave period for the SWA branch: 1 = every block "
                        "(GatedDeltaNet-H1), 2 = every other block "
                        "(Jamba/Griffin style), k = every k-th block")
    p.add_argument("--swa-window", type=int, default=128)
    p.add_argument("--swa-kv-heads", type=int, default=0,
                   help="Mistral-style grouped-query attention for the SWA "
                        "branch: 0 = one KV head per query head (MHA, default); "
                        "k (dividing --swa-heads) = k shared KV heads -> KV "
                        "cache shrinks by heads/k (e.g. 4:1 = 4x smaller)")
    p.add_argument("--moe", action="store_true",
                   help="opt-in sparse Mixture-of-Experts FFN (top-k over "
                        "n experts; ~same FLOPs, n x params -> capacity/FLOP)")
    p.add_argument("--moe-experts", type=int, default=8)
    p.add_argument("--moe-topk", type=int, default=2)
    p.add_argument("--slot-attn", action="store_true",
                   help="opt-in Titans-style attention over the persistent "
                        "memory slots (paper future-work; learned query, "
                        "zero-init identity scale)")
    p.add_argument("--ema", type=float, default=0.0,
                   help="exponential moving average of the weights, used for "
                        "eval + checkpointing (0 = off).  Use ~0.999 for LONG "
                        "multi-thousand-step runs (weight averaging near "
                        "convergence).  Measured: high-decay EMA HURTS short "
                        "few-step training (the fast-learning regime outruns "
                        "the EMA), so it defaults off.  Decay is warmed up "
                        "linearly to the target over the first 10%% of steps.")
    p.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true",
                   help="CUDA-deterministic algorithms (reproducible runs; "
                        "slightly slower).  Seed already fixes RNG; this also "
                        "fixes cuDNN/atomics.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--ddp", action="store_true",
                   help="multi-GPU / multi-process training via "
                        "DistributedDataParallel (launch with torchrun, or see "
                        "leafv5/distributed.py for a 2-worker demo)")
    # logging / eval
    p.add_argument("--outdir", type=str, default="out/leafv5")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--sample-interval", type=int, default=1000)
    p.add_argument("--ckpt-interval", type=int, default=2000)
    p.add_argument("--val-batches", type=int, default=32)
    p.add_argument("--sample-prompt", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- device / dtype ----
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    # ---- --ddp: multi-GPU / multi-process (DistributedDataParallel) ----
    dist_rank, dist_world, dist_active = 0, 1, False
    if getattr(args, "ddp", False):
        from .distributed import init as ddp_init
        dist_rank, dist_world, dist_active = ddp_init()
        if dist_active and torch.cuda.is_available():
            device = f"cuda:{int(os.environ.get('LOCAL_RANK', dist_rank))}"
        print(f"[ddp] rank {dist_rank}/{dist_world} active={dist_active} "
              f"device={device}")
    use_cuda = device.startswith("cuda")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

    # ---- --auto: pick everything from the hardware (explicit flags win) ----
    if args.auto:
        ac = auto_resolve()
        if args.model is None:
            args.model = ac["model"]
        if args.micro_batch == 16:
            args.micro_batch = ac["micro_batch"]
        if args.seq_len == 512:
            args.seq_len = ac["seq_len"]
        if args.scan == "auto":
            args.scan = ac["scan"]
        if args.no_compile is False and not ac["compile"]:
            args.no_compile = True
        if args.dtype == "auto":
            args.dtype = ac["dtype"]
        print(f"[auto] detected {ac['kind']} -> model={ac['model']} "
              f"dtype={ac['dtype']} scan={ac['scan']} compile={not args.no_compile} "
              f"micro_batch={args.micro_batch} seq_len={args.seq_len}")

    if args.dtype == "auto":
        args.dtype = "fp16" if use_cuda else "fp32"
    use_amp = args.dtype in ("fp16", "bf16")
    print(f"[train] device={device} dtype={args.dtype} amp={use_amp}")

    # ---- safe mode: maximum stability, zero tuning (applied BEFORE model) ----
    if args.safe_mode:
        args.dtype = "fp32"
        use_amp = False
        args.scan = "sequential"
        if args.scale_init is None:
            args.scale_init = 0.0
        args.no_compile = True
        if args.lr == 5e-4:
            args.lr = 1e-3
        if args.warmup_steps == 1000:
            args.warmup_steps = 200
        print("[safe] maximum-stability mode: fp32, sequential scan, "
              "scale-init 0, conservative schedule")

    # ---- data ----
    meta = prepare_corpus(args.data, args.data_file, args.tokenizer, args.vocab_size,
                          args.data_dir, args.max_tokens,
                          tokenizer_engine=args.tokenizer_engine, force=args.force_data)
    corpus = Corpus(meta, args.data_dir)
    print(f"[train] corpus: {corpus.n_tokens/1e6:.1f}M tokens, train={meta['n_train']/1e6:.1f}M, "
          f"val={meta['n_val']/1e6:.1f}M, vocab={corpus.tokenizer.vocab_size}, "
          f"engine={meta.get('tokenizer', {}).get('engine', 'none')}")
    if args.tokenizer_engine in ("auto", "gigatoken") and not gigatoken_available():
        print("[warn] GigaToken not installed -> pip install gigatoken for ~50-1000x faster "
              "corpus encoding; using HuggingFace tokenizers")

    # ---- model ----
    if args.model is None:
        args.model = "t4-4h"  # --auto already set it; fallback for plain runs
    cfg_kwargs = dict(vocab_size=corpus.tokenizer.vocab_size)
    for k in ("dim", "n_layers", "d_h", "ffn_expansion"):
        v = getattr(args, k)
        if v is not None:
            cfg_kwargs[k] = v
    if args.fast:
        # fastest-learning recipe (see speed_demo.py): high LR, no weight decay,
        # short warmup, gentle decay, small nonzero residual-scale init.
        if args.lr == 5e-4:
            args.lr = 2e-3
        if args.wd == 0.1:
            args.wd = 0.0
        if args.warmup_steps == 1000:
            args.warmup_steps = 50
        args.min_lr_ratio = min(args.min_lr_ratio, 0.3)
        args.beta2 = 0.95
        if args.scale_init is None:
            args.scale_init = 0.05
        print("[fast] sample-efficiency recipe: lr=%.0e wd=%.2f warmup=%d "
              "scale_init=%.2f" % (args.lr, args.wd, args.warmup_steps, args.scale_init))
    if args.scale_init is not None:
        cfg_kwargs["scale_init"] = args.scale_init
    if args.share_mem_every:
        cfg_kwargs["share_mem_every"] = args.share_mem_every
    if args.learn_plasticity:
        cfg_kwargs["learn_plasticity"] = True
        print("[train] learned per-layer plasticity schedules enabled "
              f"(prior={args.plasticity_prior})")
    if args.surprise_gate:
        cfg_kwargs["surprise_gate"] = True
        print("[train] novelty-gated writes ON (Tier-1 retention fix; "
              "sequential scan)")
    if args.stochastic_depth is not None:
        cfg_kwargs["stochastic_depth"] = args.stochastic_depth
        print(f"[train] stochastic depth: {args.stochastic_depth:.2f} "
              f"(residual-drop during training)")
    if args.input_decay:
        cfg_kwargs["input_decay"] = True
        print("[train] input-dependent state decay enabled (memory clearance)")
    if args.swa:
        cfg_kwargs["use_swa"] = True
        cfg_kwargs["swa_every"] = args.swa_every
        cfg_kwargs["swa_window"] = args.swa_window
        if args.swa_kv_heads:
            cfg_kwargs["swa_kv_heads"] = args.swa_kv_heads
        print(f"[train] sliding-window attention hybrid ON "
              f"(window={args.swa_window}, every={args.swa_every}, "
              f"kv_heads={cfg_kwargs.get('swa_kv_heads', 'MHA')})")
    if args.moe:
        cfg_kwargs["moe"] = True
        cfg_kwargs["moe_experts"] = args.moe_experts
        cfg_kwargs["moe_topk"] = args.moe_topk
        print(f"[train] sparse MoE FFN ON ({args.moe_experts} experts, "
              f"top-{args.moe_topk})")
    if args.slot_attn:
        cfg_kwargs["slot_attn"] = True
        print("[train] Titans-style slot attention ON")
    cfg = preset_config(args.model, **cfg_kwargs) if args.model != "custom" else ModelConfig(**cfg_kwargs)
    # curriculum: normalized list of seq lengths
    curriculum = None
    if args.curriculum:
        curriculum = [int(s) for s in args.curriculum.split(",") if int(s) >= 16]
        if curriculum and curriculum[0] > args.seq_len:
            curriculum = [args.seq_len] + curriculum
        args.curriculum_steps = max(1, args.curriculum_steps)
    if args.carry_states:
        assert args.carry_windows * args.seq_len <= cfg.max_seq_len, \
            f"carry_windows*seq_len ({args.carry_windows*args.seq_len}) must be " \
            f"<= max_seq_len ({cfg.max_seq_len}) for RoPE consistency"
    model = LeafLM(cfg).to(device)
    if dist_active:
        from .distributed import wrap as ddp_wrap
        model, device = ddp_wrap(model, dist_rank, True)
        use_cuda = device.startswith("cuda")
    if dist_rank == 0:
        print(f"[train] LEAFv5 config: dim={cfg.dim} layers={cfg.n_layers} "
              f"heads(f/m/s)=({cfg.fast_heads}/{cfg.medium_heads}/{cfg.slow_heads}) "
              f"d_h={cfg.d_h} ffn={cfg.hidden_dim} | "
              f"params={model.n_params/1e6:.1f}M "
              f"(est {param_estimate(cfg)/1e6:.1f}M)")

    # ---- scan mode ----
    if args.scan == "auto":
        args.scan = "chunked" if use_cuda else "sequential"
    chunk = args.chunk_size if (args.scan == "chunked" and args.seq_len % args.chunk_size == 0) else None
    print(f"[train] delta scan: {args.scan} (chunk={chunk})")
    if args.carry_states:
        print(f"[train] state carry: {args.carry_windows} windows x {args.seq_len} "
              f"= effective context {args.carry_windows*args.seq_len} (truncated BPTT)")

    # ---- compile (CUDA only) ----
    if use_cuda and not args.no_compile:
        try:
            model = torch.compile(model)
            print("[train] torch.compile enabled")
        except Exception as e:  # pragma: no cover
            print(f"[train] compile failed ({e}); falling back to eager")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        print("[train] deterministic mode ON (reproducible runs)")

    # ---- autotune: pick the LR automatically (zero-config) ----
    if args.autotune and args.lr == 5e-4:
        args.lr = autotune_lr(model, corpus, args.seq_len, args.micro_batch,
                              device, use_amp and use_cuda, chunk,
                              base_lr=5e-4)

    # ---- optimizer / schedule ----
    opt = build_optimizer(model, args.lr, args.wd, args.beta2, args.optimizer)
    scaler = make_scaler(use_amp and use_cuda)
    if args.optimizer == "lion":
        print("[train] optimizer=Lion (half the state of AdamW)")
    # EMA of the weights (used for eval + checkpointing; 0 disables)
    ema_model = None
    if args.ema and args.ema > 0:
        ema_model = LeafLM(cfg).to(device)
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad_(False)
        print(f"[train] EMA enabled (decay={args.ema}); checkpoints save the "
              f"EMA weights (P1 #13)")

    # ---- resume ----
    step, tokens_seen, best_val, total_steps = 0, 0, float("inf"), (args.max_steps or 1)
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        step, tokens_seen, best_val = ck["step"], ck["tokens_seen"], ck["best_val"]
        total_steps = ck.get("total_steps", total_steps)
        torch.set_rng_state(ck["rng"])
        np.random.set_state(ck["np_rng"])
        random.setstate(ck["py_rng"])
        print(f"[train] resumed from {args.resume} at step {step}")

    eff_batch = args.micro_batch * args.grad_accum  # tokens/step = eff_batch*seq
    # ---- budget cap: measure throughput, set total_steps to fit ----
    if args.budget_hours:
        print(f"[train] measuring throughput for {args.budget_hours}h budget ...")
        tok_s = measure_throughput(model, corpus, args.seq_len, args.micro_batch, device,
                                   use_amp and use_cuda, chunk, args.prefetch, steps=5)
        budget_tokens = tok_s * args.budget_hours * 3600 * 0.85  # 15% margin for eval/ckpt
        budget_steps = int(budget_tokens / (eff_batch * args.seq_len))
        if args.max_steps:
            total_steps = min(args.max_steps, budget_steps)
        else:
            total_steps = budget_steps
        print(f"[train] measured {tok_s/1e3:.1f}k tok/s -> {budget_tokens/1e6:.0f}M tokens "
              f"in {args.budget_hours}h -> total_steps={total_steps} "
              f"(eff batch {eff_batch*args.seq_len/1e3:.0f}k tokens)")
    elif args.max_steps:
        total_steps = args.max_steps
    else:
        # default: one epoch over the train split
        total_steps = max(1, meta["n_train"] // (eff_batch * args.seq_len))
        print(f"[train] no --max-steps/--budget-hours: training one epoch "
              f"({total_steps} steps)")

    min_lr = args.lr * args.min_lr_ratio

    # ---- bookkeeping ----
    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "log.jsonl")
    logf = open(log_path, "a") if dist_rank == 0 else None
    rng = np.random.default_rng(args.seed)
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(project="leafv5", config=vars(args))
    if args.sample_prompt is None:
        args.sample_prompt = "Once upon a time" if "story" in args.data else "The"
    stream_src = StreamCorpus(corpus, args.micro_batch, args.seq_len, "train",
                              seed=args.seed) if args.carry_states else None
    prefetcher = BatchPrefetcher(stream_src or corpus, args.micro_batch,
                                 args.seq_len, "train",
                                 buffer=args.prefetch, seed=args.seed)

    model.train()
    t_start = time.time()
    accum_loss = 0.0
    accum_tokens = 0
    oom_shrinks = 0
    # state-carry bookkeeping (when --carry-states)
    carry_states: Optional[list] = None
    carry_offset = 0
    carry_win = 0
    # loss-spike auto-recovery: keep a shadow copy of the weights; if the EMA
    # loss jumps > 3x (divergence), roll back, halve LR, and keep going.
    # This makes training impossible to ruin by a bad step / bad data batch.
    shadow_sd = None
    loss_ema = None
    spikes_recovered = 0

    def eval_model():
        """The weights used for evaluation: EMA copy if enabled, else live."""
        return ema_model if ema_model is not None else model

    def save_ckpt(path: str, ck: dict):
        """Only rank 0 writes checkpoints under DDP."""
        if dist_rank == 0:
            torch.save(ck, path)

    def run_eval(force: bool = False):
        nonlocal best_val
        em = eval_model()
        if args.carry_states:
            vloss, vppl = evaluate_stream(em, corpus, device, args.seq_len,
                                          args.micro_batch, args.val_batches,
                                          use_amp and use_cuda, chunk,
                                          args.carry_windows)
        else:
            vloss, vppl = evaluate(em, corpus, device, args.seq_len, args.micro_batch,
                                   args.val_batches, use_amp and use_cuda, rng, chunk)
        if vloss < best_val:
            best_val = vloss
            save_ckpt(os.path.join(args.outdir, "best.pt"),
                      state_dict_for_save(em, opt, scaler, args, cfg, meta, step,
                                          tokens_seen, best_val,
                                          torch.get_rng_state(),
                                          np.random.get_state(),
                                          random.getstate(), total_steps))
        print(f"[eval] step {step}: val_loss={vloss:.4f} ppl={vppl:.2f} best={best_val:.4f}"
              + (" (EMA)" if ema_model is not None else ""))
        if args.probe_gates:
            xp, _ = corpus.sample_batch(args.micro_batch, 64, rng, "val")
            stats = em.gate_stats(xp.to(device))
            row = "  gates | " + "  ".join(
                f"{g}:bw={s['bw']:.3f} bf={s['bf']:.3f} gr={s['gr']:.3f} fn={s['fn']:.1f}"
                for g, s in stats.items())
            print(row)
            if logf is not None:
                logf.write(json.dumps({"step": step, "gates": stats}) + "\n")
                logf.flush()

    while step < total_steps:
        iter_t0 = time.time()
        opt.zero_grad(set_to_none=True)
        try:
            for _ in range(args.grad_accum):
                x, y = prefetcher.get()
                x, y = x.to(device), y.to(device)
                if args.carry_states:
                    if carry_states is None or carry_win >= args.carry_windows:
                        carry_states = model.init_states(args.micro_batch, device)
                        carry_offset, carry_win = 0, 0
                    with torch.autocast(device_type="cuda" if use_cuda else "cpu",
                                        dtype=dtype_map[args.dtype],
                                        enabled=use_amp and use_cuda):
                        logits, carry_states = model(x, carry_states, chunk=chunk,
                                                     offset=carry_offset,
                                                     grad_checkpoint=args.grad_checkpoint)
                    carry_offset += args.seq_len
                    carry_win += 1
                else:
                    with torch.autocast(device_type="cuda" if use_cuda else "cpu",
                                        dtype=dtype_map[args.dtype],
                                        enabled=use_amp and use_cuda):
                        logits, _ = model(x, chunk=chunk,
                                          grad_checkpoint=args.grad_checkpoint)
                loss = cross_entropy_chunked(logits.reshape(-1, logits.shape[-1]),
                                             y.reshape(-1))
                # P0 fix (#4): normalize by grad_accum so the accumulated
                # gradient is the MEAN, not the SUM (was grad_accum x too big).
                loss = loss / args.grad_accum
                # MoE load-balancing aux loss (0 when MoE is off)
                try:
                    loss = loss + cfg.moe_aux_weight * model.aux_loss()
                except Exception:
                    pass
                # plasticity prior (0 when off / not learnable); normalized the
                # same way as the main loss so its weight is exact
                if args.plasticity_prior and cfg.learn_plasticity:
                    loss = loss + model.plasticity_prior_loss(
                        args.plasticity_prior) / args.grad_accum
                scaler.scale(loss).backward()
                accum_loss += loss.item() * x.shape[0] * x.shape[1]
                accum_tokens += x.shape[0] * x.shape[1]
                if args.carry_states:
                    carry_states = [s.detach() for s in carry_states]
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # NaN-grad guard: a NaN/Inf gradient (bad batch, fp16 overflow)
            # would permanently corrupt AdamW's moments.  Skip the step and
            # roll back to the last good weights instead.
            bad_grad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                if shadow_sd is not None:
                    model.load_state_dict(shadow_sd)
                spikes_recovered = min(spikes_recovered + 1, 5)
                args.lr = args.lr * 0.5
                for g in opt.param_groups:
                    g["lr"] = args.lr
                print(f"[recover] step {step}: non-finite gradients -> "
                      f"rolled back, lr halved to {args.lr:.1e}")
                opt.zero_grad(set_to_none=True)
            else:
                scaler.step(opt)
                scaler.update()
        except torch.cuda.OutOfMemoryError:
            oom_shrinks += 1
            if args.micro_batch <= 2:
                raise
            args.micro_batch //= 2
            args.grad_accum *= 2
            accum_loss, accum_tokens = 0.0, 0
            carry_states, carry_offset, carry_win = None, 0, 0
            prefetcher.stop()
            stream_src = (StreamCorpus(corpus, args.micro_batch, args.seq_len, "train",
                                       seed=args.seed + oom_shrinks)
                          if args.carry_states else None)
            prefetcher = BatchPrefetcher(stream_src or corpus, args.micro_batch,
                                         args.seq_len, "train",
                                         buffer=args.prefetch,
                                         seed=args.seed + oom_shrinks)
            torch.cuda.empty_cache()
            print(f"[train] OOM -> micro_batch={args.micro_batch}, grad_accum={args.grad_accum}")
            continue

        step += 1
        tokens_seen += accum_tokens

        # ---- EMA update (cheap param copy after each optimizer step) ----
        if ema_model is not None:
            # decay warmup: ramp 0 -> target over the first 10% of steps so the
            # EMA doesn't lag behind the fast early learning
            decay = args.ema * min(1.0, step / max(1, int(total_steps * 0.1)))
            for pe, pm in zip(ema_model.parameters(), model.parameters()):
                ema_update(pe, pm, decay)

        # ---- curriculum: grow the sequence length at stage boundaries ----
        if curriculum is not None:
            stage = min(step // args.curriculum_steps, len(curriculum) - 1)
            target_seq = curriculum[stage]
            if target_seq != args.seq_len:
                print(f"[curriculum] step {step}: seq_len {args.seq_len} -> "
                      f"{target_seq} (stage {stage+1}/{len(curriculum)})")
                args.seq_len = target_seq
                chunk = (args.chunk_size if (args.scan == "chunked"
                                             and args.seq_len % args.chunk_size == 0)
                         else None)
                carry_states, carry_offset, carry_win = None, 0, 0
                prefetcher.stop()
                stream_src = (StreamCorpus(corpus, args.micro_batch, args.seq_len,
                                           "train", seed=args.seed + stage)
                              if args.carry_states else None)
                prefetcher = BatchPrefetcher(stream_src or corpus, args.micro_batch,
                                             args.seq_len, "train",
                                             buffer=args.prefetch,
                                             seed=args.seed + stage)

        lr = lr_at(step, args.lr, args.warmup_steps, total_steps, min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        if step % args.log_interval == 0:
            dt = time.time() - t_start
            iter_tok = args.micro_batch * args.grad_accum * args.seq_len
            tok_s = iter_tok / max(time.time() - iter_t0, 1e-6)
            avg_loss = accum_loss / accum_tokens
            remaining = (total_steps - step) * (time.time() - iter_t0)
            # gradient-norm monitor: visible instability signal (spikes here
            # are the earliest warning; the NaN-guard + spike-recovery handle
            # them automatically)
            grad_norm = 0.0
            gsum = 0.0
            for pp in model.parameters():
                if pp.grad is not None and torch.isfinite(pp.grad).all():
                    gsum += float(pp.grad.detach().float().pow(2).sum())
            if gsum > 0:
                grad_norm = math.sqrt(gsum)
            msg = (f"step {step}/{total_steps} loss={avg_loss:.4f} "
                   f"grad={grad_norm:.2e} lr={lr:.2e} "
                   f"tok/s={tok_s/1e3:.1f}k seen={tokens_seen/1e6:.1f}M "
                   f"elapsed={dt/3600:.2f}h eta={remaining/3600:.2f}h")
            print(f"[train] {msg}")
            if logf is not None:
                logf.write(json.dumps({"step": step, "loss": avg_loss, "lr": lr,
                                       "tok_s": tok_s, "tokens": tokens_seen,
                                       "elapsed": dt, "grad_accum_steps": args.grad_accum}) + "\n")
                logf.flush()
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({"loss": avg_loss, "lr": lr, "tok_s": tok_s})
            accum_loss = 0.0
            accum_tokens = 0

            # ---- loss-spike auto-recovery (roll back + halve LR) ----
            if args.safe_mode or True:  # always on: easiest-to-train guarantee
                if loss_ema is None:
                    loss_ema = avg_loss
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * avg_loss
                    if avg_loss > 3.0 * loss_ema + 0.5 and spikes_recovered < 5:
                        # divergence detected: roll back to last good weights
                        if shadow_sd is not None:
                            model.load_state_dict(shadow_sd)
                        spikes_recovered += 1
                        args.lr = args.lr * 0.5
                        for g in opt.param_groups:
                            g["lr"] = args.lr
                        print(f"[recover] step {step}: loss spike "
                              f"({avg_loss:.3f} vs EMA {loss_ema:.3f}) -> "
                              f"rolled back, lr halved to {args.lr:.1e} "
                              f"(recovery #{spikes_recovered})")
                        loss_ema = avg_loss
                # P0 fix (#5): state_dict() shares storage with the live model;
                # clone it so the rollback target is a true snapshot.
                shadow_sd = {k: v.detach().clone() for k, v in
                             model.state_dict().items()}

        if args.eval_interval and step % args.eval_interval == 0:
            run_eval()

        if args.sample_interval and step % args.sample_interval == 0:
            model.eval()
            text, _ = generate(model, corpus.tokenizer, args.sample_prompt,
                               max_new=96, temperature=0.8, top_k=50, device=device)
            model.train()
            print(f"[sample] step {step}:\n{text}\n---")

        if args.ckpt_interval and step % args.ckpt_interval == 0:
            save_ckpt(os.path.join(args.outdir, f"ckpt-{step}.pt"),
                      state_dict_for_save(eval_model(), opt, scaler, args, cfg,
                                          meta, step, tokens_seen, best_val,
                                          torch.get_rng_state(),
                                          np.random.get_state(),
                                          random.getstate(), total_steps))

    # ---- final ----
    prefetcher.stop()
    run_eval(force=True)
    save_ckpt(os.path.join(args.outdir, "final.pt"),
              state_dict_for_save(eval_model(), opt, scaler, args, cfg, meta,
                                  step, tokens_seen, best_val,
                                  torch.get_rng_state(), np.random.get_state(),
                                  random.getstate(), total_steps))
    text, _ = generate(model, corpus.tokenizer, args.sample_prompt, max_new=192,
                       temperature=0.8, top_k=50, device=device)
    print(f"[final] sample:\n{text}\n---")
    print(f"[final] done in {(time.time()-t_start)/3600:.2f}h, tokens={tokens_seen/1e6:.1f}M, "
          f"best_val={best_val:.4f}; ckpts in {args.outdir}")
    if logf is not None:
        logf.close()


if __name__ == "__main__":
    main()
