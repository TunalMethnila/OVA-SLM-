"""Multi-GPU / multi-process training for LEAFv5 (DDP).

Two pieces:
  1. train.py integration: `--ddp` wraps the model in
     torch.nn.parallel.DistributedDataParallel (gradient all-reduce), sets the
     device to cuda:<local_rank>, and guards saves/logs to rank 0.  Launch with
     torchrun (or the spawn demo below).
  2. Self-contained demo proving the path works:
        python -m leafv5.distributed --world-size 2     # spawns 2 workers (gloo/CPU)
        torchrun --nproc_per_node=2 -m leafv5.distributed  # or via torchrun

Run on GPUs:
    torchrun --nproc_per_node=4 -m leafv5.train --data tinystories \
        --ddp --model t4-4h --auto --budget-hours 4
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import torch


def init(backend: Optional[str] = None, timeout_s: int = 600):
    """Initialize the process group.  Backend auto: nccl on CUDA, else gloo.
    Safe no-op when world_size == 1 (single-GPU / CPU)."""
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world <= 1:
        return rank, world, False
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(
        backend=backend, init_method="env://",
        timeout=torch.distributed.default_pg_timeout if False else
        __import__("datetime").timedelta(seconds=timeout_s))
    return rank, world, True


def wrap(model: torch.nn.Module, rank: int, distributed: bool):
    """DDP-wrap when distributed; return (model, device)."""
    if distributed:
        local = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            model = model.cuda(local)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local])
            return model, f"cuda:{local}"
        # CPU / MPS with multiple processes: DDP works on CPU with gloo
        model = torch.nn.parallel.DistributedDataParallel(model)
        return model, "cpu"
    return model, "cuda" if torch.cuda.is_available() else "cpu"


def rank0(distributed: bool, rank: int) -> bool:
    return (not distributed) or rank == 0


# ---------------------------------------------------------------------------
# Self-contained spawn demo (proves multi-worker training works)
# ---------------------------------------------------------------------------
def _worker(rank: int, world: int, steps: int, device: str):
    import random

    import numpy as np
    import torch.nn.functional as F

    from .config import preset_config
    from .model import LeafLM

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # each spawned worker must join the process group first.
    # mp.spawn doesn't set RANK/WORLD_SIZE env vars (torchrun does), so set
    # them here for the self-contained demo.
    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        torch.distributed.init_process_group(
            backend="gloo", init_method="env://",
            timeout=__import__("datetime").timedelta(seconds=300))
    from .data import CharTokenizer
    import string
    voc = {c: i for i, c in enumerate(string.ascii_lowercase + " .,?!")}
    tok = CharTokenizer(voc)
    V = tok.vocab_size
    # per-rank data: different random windows (rank-sharded RNG)
    rng = np.random.default_rng(rank)
    cfg = preset_config("micro", vocab_size=V, n_layers=2, dim=96, d_h=32,
                        scale_init=0.1)
    model, dev = wrap(LeafLM(cfg), rank, world > 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    losses = []
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        x = torch.tensor([[rng.integers(0, V) for _ in range(32)] for _ in range(4)],
                         device=dev)
        y = x[:, 1:].clone()
        y = torch.cat([y, torch.zeros(4, 1, dtype=torch.long, device=dev)], 1)
        lg = model(x)[0]
        loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    if world > 1:
        torch.distributed.barrier()
    print(f"  [worker {rank}/{world}] final loss={losses[-1]:.4f} "
          f"({losses[0]:.4f} -> {losses[-1]:.4f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()

    if args.world_size <= 1:
        print("single-process demo:")
        _worker(0, 1, args.steps, "cpu")
        print("OK")
        return

    import torch.multiprocessing as mp
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["WORLD_SIZE"] = str(args.world_size)
    print(f"spawning {args.world_size} workers (gloo/CPU)...")
    mp.spawn(_worker, args=(args.world_size, args.steps, "cpu"),
             nprocs=args.world_size, join=True)
    print("multi-worker training OK")


if __name__ == "__main__":
    main()
