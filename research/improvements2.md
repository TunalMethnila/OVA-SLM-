> **SUPERSEDED NUMBERS (2026-08-09).** Measurements in this file were taken
> before the §29 causality fixes (symmetric-padded convs; reshape head/position
> leak) and the training-mechanics fixes. Re-measured values are in
> **research/reverify-2026-08.md** — several headline numbers here (100% recall
> by step 5, PTB PPL 1.0, ~40× per-step, 1/20 compute, 100% flat long-range
> retention) did NOT survive. Read this file as history, not as current evidence.
>

# Round 3: new practical improvements (and what the measurements say)

Continuation of research/comparison.md. Goal: find NEW ways to make LEAFv5
better while staying practical and easy to train. Everything here is a
config-gated drop-in flag; measurements are CPU-sandbox-scale (mechanisms
transfer to GPU).

## 1. What we tried

| improvement | idea / source | measured effect (small scale) | verdict |
|---|---|---|---|
| **Input-dependent global decay** (`--input-decay`) | Gated DeltaNet's memory-clearance: `S <- a_t·S − bf(Sk)kᵀ + bw·v kᵀ`, `a_t∈(0,1)` from a learned projection (bias-init 4.6 ⇒ a≈0.99 initially = paper behavior) | LM held-out @40 steps: 0.629 (off) vs 0.706 (on) — slightly worse; retention @D=64 identical (6%) | **opt-in, default OFF.** Theoretical value only under long-context memory-collision pressure, which the sandbox can't reach. The chunked≡sequential scan stays EXACT with decay (unit-tested). |
| **Memory-branch dropout** (`mem_dropout=0.05` default) | variational-style dropout on the memory output (like attention dropout) | no regression on the recall race (still 100% by step 5); standard small-data regularizer | **keep default 0.05** |
| **Stochastic depth** (`--stochastic-depth`) | per-block residual-drop during training (scale survivors by 1/(1-p)) | plumbing verified; standard for deeper stacks | **opt-in** (0 = off) |
| **EMA of weights** (`--ema`) | standard weight averaging for eval/ckpt | **HURT badly in the few-step regime**: 60 steps of fast learning outrun a 0.99 EMA → held-out 0.069 (live) vs 1.908 (EMA) | **default OFF.** Use `--ema 0.999` only for long multi-thousand-step convergence runs; added linear decay warmup over the first 10% of steps so it can't lag early. |

## 2. The key design lesson

"Standard best practices" are not automatically wins for a *fast-learning
recurrent* model:
- **EMA** assumes weights are near convergence; LEAFv5's whole selling point is
  that weights move a lot early. High-decay EMA is actively harmful there.
- **Input decay** adds a learned mechanism that needs its own gradient budget;
  at small scale the budget is better spent elsewhere. Its benefit is regime-
  specific (long context, memory pressure).
- **Dropout** on the memory branch is the exception: cheap, standard, no
  regression.

This is exactly the "practical and easiest to train" filter: each feature is
measured before it ships as a default, and defaults stay conservative.

## 3. Architecture state (all composable, defaults conservative)

```
memory block (default): read query W_q + short conv-3 + SiLU output gate
                        + persistent slots (64) + mem_dropout 0.05
optional: --input-decay, --stochastic-depth, --ema 0.999 (long runs),
          --learn-plasticity, --share-mem-every 2
scan: sequential (paper-exact) or chunked parallel-scan (CUDA) — exact-equal
      with decay enabled too
```

## 4. Validation (this round)

- `tests/test_sota_upgrade.py` +7 tests: input-decay chunked≡sequential (exact),
  mem-dropout+stochastic-depth train/eval, EMA math. All pass.
- Full suite green: test_model (7), test_speed, test_auto.
- Recall race unchanged: 100% held-out by step 5, 93% by step 3.
- train.py smoke with EMA+stochastic-depth+input-decay+fast: runs, eval marks "(EMA)".
