"""Long-range memory: retention vs. distance, and the write gate's role.

What this measures (all reproducible, CPU-friendly):

  1. Train the write/read *policy* at SHORT distance with TWO distinguishable
     token pools: memory tokens (the pairs) and distractor tokens (fillers).
     The content-dependent write gate learns "write memory tokens strongly,
     gate distractors down" -- strong, learnable signal at short distance.
  2. Freeze.  At inference, feed P pairs then D distractor tokens then Q
     queries, carrying the tiny constant-size state.  Retention well beyond
     the training window demonstrates the recurrent memory; a reset-state
     baseline (windowed inference) collapses to chance.

Measured limits (documented in the README):
  * Retention is bounded by write-crosstalk: with W writes the readout noise
    grows ~sqrt(W)/sqrt(d_h), so capacity is ~d_h associations (matches the
    DeltaNet literature).  This is why the write gate matters -- suppressing
    distractor writes is what makes long-range retention possible.
  * With uniform random distractors the model cannot retain at distance (it
    cannot tell memory tokens from distractors, so every token writes).
  * Long-range BPTT (queries ~90 tokens after pairs, full gradient) is too
    weak for a tiny model in a few hundred steps; training the gate at short
    distance is the practical route.

Run:  python -m leafv5.longrange_demo [--distances 64,256,1024,4096]
      # flatter retention: python -m leafv5.longrange_demo --fillers 32 --d-h 48
"""
from __future__ import annotations

import argparse
import random
import time

import torch
import torch.nn.functional as F

from .config import preset_config
from .model import LeafLM


def make_train_batch(bs, V, P, Q, K, rng, device):
    """[pairs (pool M)] [K distractor tokens (pool F)] [queries].  Loss masked
    to query positions so the model must read the state."""
    M = list(range(2, V // 2))
    F = list(range(V // 2, V))
    T = 2 * P + K + Q
    xs, ys, masks = [], [], []
    for _ in range(bs):
        keys = rng.sample(M, P)
        vals = rng.sample(M, P)
        qi = rng.sample(range(P), Q)
        ids = []
        for k, v in zip(keys, vals):
            ids += [k, v]
        ids += [rng.choice(F) for _ in range(K)]
        for j in qi:
            ids.append(keys[j])
        y = ids[1:] + [1]
        mask = torch.zeros(T, dtype=torch.bool)
        for j, pos in enumerate(range(2 * P + K, T)):
            mask[pos] = True
            y[pos] = vals[qi[j]]
        xs.append(torch.tensor(ids))
        ys.append(torch.tensor(y))
        masks.append(mask)
    return (torch.stack(xs).to(device), torch.stack(ys).to(device),
            torch.stack(masks).to(device))


@torch.no_grad()
def eval_train(model, x, y, mask, device):
    model.eval()
    logits, _ = model(x, model.init_states(x.shape[0], device))
    pred = logits.argmax(-1)
    hits = ((pred == y) & mask).sum().item()
    tot = mask.sum().item()
    model.train()
    return hits / max(1, tot)


@torch.no_grad()
def eval_at_distance(model, V, P, Q, D, rng, device, n=16, carry=True):
    """Pairs (pool M) -> D distractor tokens (pool F) -> Q queries.
    carry=False scores queries with a fresh state (windowed baseline)."""
    model.eval()
    M = list(range(2, V // 2))
    F = list(range(V // 2, V))
    correct = total = 0
    for _ in range(n):
        keys = rng.sample(M, P)
        vals = rng.sample(M, P)
        qi = rng.sample(range(P), Q)
        ids = []
        for k, v in zip(keys, vals):
            ids += [k, v]
        ids += [rng.choice(F) for _ in range(D - 2 * P)]
        qids = [keys[j] for j in qi]
        targets = [vals[j] for j in qi]
        if carry:
            x = torch.tensor([ids + qids], device=device)
            logits, _ = model(x, model.init_states(1, device))
            q_logits = logits[0, -Q:]
        else:
            xq = torch.tensor([qids], device=device)
            logits, _ = model(xq, model.init_states(1, device))
            q_logits = logits[0]
        for lg, vt in zip(q_logits, targets):
            correct += int(torch.argmax(lg) == vt)
            total += 1
    model.train()
    return correct / max(1, total)


def main():
    p = argparse.ArgumentParser(description="LEAFv5 long-range memory test.")
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--pairs", type=int, default=2)
    p.add_argument("--queries", type=int, default=1)
    p.add_argument("--fillers", type=int, default=16,
                   help="distractor tokens between pairs and queries during training. "
                        "HIGHER pressure (e.g. --fillers 32 --d-h 48) teaches stronger "
                        "distractor suppression -> retention flatter in distance")
    p.add_argument("--vocab", type=int, default=128)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--d-h", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--distances", type=str, default="64,256,1024,4096")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = preset_config("micro", vocab_size=args.vocab, n_layers=args.layers,
                        dim=args.dim, d_h=args.d_h, rope_dim=0)
    model = LeafLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"[train] pairs(mem pool) -> {args.fillers} distractors -> queries; "
          f"loss masked to queries; vocab={args.vocab}")
    best = 0.0
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        x, y, mask = make_train_batch(args.batch, args.vocab, args.pairs,
                                      args.queries, args.fillers, rng, device)
        logits, _ = model(x, model.init_states(args.batch, device))
        lg = logits.reshape(-1, args.vocab)[mask.reshape(-1)]
        tg = y.reshape(-1)[mask.reshape(-1)]
        loss = F.cross_entropy(lg, tg)
        loss.backward()
        opt.step()
        if step % max(1, args.steps // 5) == 0 or step == args.steps:
            acc = eval_train(model, x, y, mask, device)
            best = max(best, acc)
            print(f"  step {step:4d}  loss={loss.item():.4f}  "
                  f"train recall={100*acc:.1f}%  ({time.time()-t0:.0f}s)")

    dists = [int(d) for d in args.distances.split(",")]
    print(f"\n[retention] recall vs distance (state carried / reset baseline), "
          f"chance={100.0/args.vocab:.1f}%:")
    print(f"  {'D':>6s} {'carry':>8s} {'reset':>8s}")
    rng = random.Random(999)
    for D in dists:
        ac = eval_at_distance(model, args.vocab, args.pairs, args.queries, D,
                              rng, device, carry=True)
        ar = eval_at_distance(model, args.vocab, args.pairs, args.queries, D,
                              rng, device, carry=False)
        print(f"  {D:>6d} {100*ac:>7.1f}% {100*ar:>7.1f}%")

    # gate sanity: write-gate strength on memory tokens vs distractor tokens.
    # If the gate learned to suppress distractors, bw(memory) > bw(distractor).
    M_POOL = list(range(2, args.vocab // 2))
    F_POOL = list(range(args.vocab // 2, args.vocab))
    rng2 = random.Random(7)
    mem_batch = torch.tensor([[rng2.choice(M_POOL) for _ in range(16)] for _ in range(8)])
    fill_batch = torch.tensor([[rng2.choice(F_POOL) for _ in range(16)] for _ in range(8)])
    sm = model.gate_stats(mem_batch.to(device))
    sf = model.gate_stats(fill_batch.to(device))
    print("\n[gate ] mean write-gate βw on memory tokens vs distractor tokens:")
    for g in ("fast", "medium", "slow"):
        d = sm[g]["bw"] - sf[g]["bw"]
        print(f"  {g:>6s} group: mem={sm[g]['bw']:.3f}  dist={sf[g]['bw']:.3f}  "
              f"Δ={d:+.3f}{'  (suppresses distractors)' if d > 0.02 else ''}")


if __name__ == "__main__":
    main()
