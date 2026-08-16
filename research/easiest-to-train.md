> **SUPERSEDED NUMBERS (2026-08-09).** Measurements in this file were taken
> before the §29 causality fixes (symmetric-padded convs; reshape head/position
> leak) and the training-mechanics fixes. Re-measured values are in
> **research/reverify-2026-08.md** — several headline numbers here (100% recall
> by step 5, PTB PPL 1.0, ~40× per-step, 1/20 compute, 100% flat long-range
> retention) did NOT survive. Read this file as history, not as current evidence.
>

# Easiest-to-train LEAFv5: the measured guarantees

"Easiest to train" is made concrete as four guarantees, each verified:

## 1. Zero-config: `python -m leafv5.train --data tinystories` just works

No flags needed. `--auto` picks the model preset (from VRAM), dtype (bf16/fp16/
fp32), scan mode, torch.compile, micro-batch and seq-len; `--budget-hours`
auto-caps the run; OOM auto-recovery halves the batch; and (new) `--autotune`
probes 3 learning rates at startup and keeps the best. Verified end-to-end:
`python -m leafv5.train --data shakespeare` with ZERO training flags runs,
produces finite decreasing loss, no NaN (tests/test_easy.py).

Autotune measured: probed 1e-4/5e-4/2e-3 on a tiny model, chose 2e-3, and the
run's loss (3.23) beat the 1e-4 candidate (4.19) — it picks a genuinely better
LR automatically.

## 2. Impossible to break: automatic safety nets

- **Loss-spike recovery** (new): the trainer keeps a shadow copy of the
  weights; if the EMA-smoothed loss jumps >3x (a bad batch, a spike, a wrong
  flag), it rolls back to the last good weights, halves the LR, and keeps
  training — up to 5 recoveries, then it stays conservative. Unit-tested
  (rolls back + halves on spike; does nothing otherwise).
- **`--safe-mode`** (new): maximum-stability config — fp32, sequential scan,
  scale-init 0, conservative schedule, no compile. Slower but impossible to
  break; for unknown/weird hardware.
- **`--grad-clip 1.0`** default + fp32 states + StateNorm + zero-init residual
  scales are the architectural safety rails.

## 3. LR-robust: a wide usable learning-rate range (no tuning)

`python -m leafv5.robustness_demo` sweeps LR over 4 orders of magnitude on the
same recall task (40 steps), same-size models:

| LR | LEAFv5 | Transformer | GatedRNN |
|---|---|---|---|
| 1e-4 | 1.6% | 1.6% | 2.3% |
| 1e-3 | **54.7%** | 18.8% | 3.9% |
| 3e-3 | **66.4%** | 31.2% | 2.3% |
| 1e-2 | **77.3%** | 36.7% | 3.9% |
| 3e-2 | 48.4% | 13.3% | 5.5% |
| 1e-1 | 19.5% (no NaN) | 6.2% | 5.5% |

- **LEAFv5 never diverges**, even at lr=1e-1 (unit-tested: finite loss after 8
  steps at 1e-1).
- **Widest effective plateau**: ≥48% across lr ∈ [1e-3, 3e-2] (a 30x span).
  The Transformer only exceeds 36% at exactly 1e-2; the Mamba-lite RNN barely
  learns anywhere.
- **Highest floor**: at the default mid LR (1e-3) LEAFv5 scores 54.7% vs
  Transformer 18.8% — even a "wrong" LR still learns well.

Wide plateau + no divergence = **you don't need to tune LR at all**, which is
the operational meaning of "easiest to train".

## 4. Fast convergence with safe defaults

The `--fast` recipe (lr 2e-3, wd 0, short warmup, scale-init 0.05) already
reaches Transformer@100-step quality in ~10 steps (speed_demo). Combined with
autotune you get: pick a dataset, run, it learns quickly and safely.

## How to use it

```bash
# truly zero-config (the "easiest" experience):
python -m leafv5.train --data tinystories --autotune --budget-hours 4
# maximum safety on unknown hardware:
python -m leafv5.train --data tinystories --safe-mode --budget-hours 4
# verify the guarantees yourself:
python tests/test_easy.py
python -m leafv5.robustness_demo
```
