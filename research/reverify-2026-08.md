# LEAFv5 post-fix re-verification ledger — 2026-08-09

**Why this exists.** The §29 expert review found two causality leaks (symmetric-padded
convolutions; a `[BH,dh,T] → [B,T,H*dh]` reshape that mixed head and position strides) that
were live during training "since the start". The reviewer's standing rule is that training
computation must equal token-by-token inference exactly. Until every headline number was
re-measured **after** the fixes, none could be trusted. This document is that re-measurement.

**Environment.** CPU-only sandbox (2 GB RAM, 2 threads pinned via `OMP_NUM_THREADS=2`),
torch 2.13.0+cpu. Every command below is the project's own script with its documented
defaults (same seed, same config). Full `pytest` suite: **71/71 pass** including
`test_causal_invariant.py` (train-forward ≡ token-by-token decode, max|Δ| ~1e-6, causality 0.0)
and `test_review_fixes.py`.

---

## Verdict table

| # | Claim (as published) | Stated | Re-measured 2026-08-09 | Verdict |
|---|---|---|---|---|
| 1 | Recall, store-1/query-1: "100% in 10 steps, exact, robust across seeds" (README honest note; `speed_demo --pairs 1 --queries 1`) | 100% @10 | LEAFv5-fast **99% @5, 100% @10**; paper 100% @10 | ✅ **Survives** |
| 2 | Recall, store-2/query-1: "100% by step 50" (README §15 table) | 100% @50 | LEAFv5-fast peaks **64% @100**; paper 56% @100 | ❌ **Dies** |
| 3 | "Exceeds Transformer@100 by step 10 (58% > 39%)" | 58% @10 | fast 45% @10 (paper 48% @10) vs Transformer@100 = 36% | ⚠️ Direction survives, value differs |
| 4 | "Transformer never reaches 80% in 100 steps" | — | store-2/q1: true (36% max). store-1/q1: **false — Transformer hits 100% @50** | ⚠️ Task-dependent |
| 5 | PTB char PPL @150: "1.0 vs 8.2 vs 8.8" (`benchmark_ppl`, §17/world-class/improvements6) | 1.0 | **6.1 vs 8.4 vs 9.5**; PPL 4.76 @300 steps, still descending | ❌ Absolute **dies**; ✅ ordering survives |
| 6 | LM "~40× per-step; 0.055 vs 2.40/2.61 @120" (`benchmark_world`, world-class) | 0.055 | Shakespeare LM @120: **2.225 vs 2.396 vs 2.632** | ❌ **Dies** |
| 7 | LM "0.379 @ step 50; beats Transformer@100 by step 20" (README §17) | 0.379 @50 | 2.415 @50; beats Transformer@100 (2.408) only at ~step 100; Transformer **leads** steps 1–50 | ❌ **Dies** |
| 8 | Compute-to-target "1/20" / "~9× to match final quality" (`compute_demo`) | 1/20 | all targets (2.0…0.1) **"never"**; final 2.122 vs 2.351 @140 | ❌ **Dies** |
| 9 | Long-range retention "100% flat @ D=64/256/1024" (README §17) | 100% | train recall 5–25% (noisy); retention ≈ chance (6.2% vs 0.8%) | ❌ **Dies** |
| 10 | Extrapolation "0.30x @1024 (improves); full-RoPE degrades 15x" (world-class) | 0.30x / 15x | rope-off **0.97x** (flat, no degrade); rope-full 1.27x; Transformer 1.04x | ⚠️ Qualitative survives, magnitudes die |
| 11 | Stability certificate | 9/9 | **9/9 PASS** (`stability_check`) | ✅ **Survives** |
| 12 | Progressive growth (width/depth/slots) | 1.9e-6 / 0.0 / 2.6e-4 | test_grow suite passes (incl. formerly shadowed regression tests) | ✅ **Survives** |
| 13 | Train-forward ≡ token-by-token decode | — | `test_causal_invariant` passes; causality 0.0 | ✅ **Survives** |
| 14 | C twin kernel exact vs Python scan | ~1e-7 | gcc build succeeds; `test_fast_scan_equals_python` passes (atol 1e-5) | ✅ **Survives** |
| 15 | Smoke train "4.33 → 0.047 in 130 steps; val PPL 1.07" (README §8b) | 1.07 | **not re-verifiable in this sandbox** — `train.py` OOM-killed (2 GB ceiling) on CPU | ❓ Unverified here (needs GPU/T4 box) |

---

## Raw measurements

### Recall — `speed_demo.py --task recall` (store-2/query-1, V=64, chance 1.6%)
```
  model                 1      3      5     10     20     50    100
  LEAFv5 (fast si=.2)   4     28     46     45     50     53     64
  LEAFv5 (paper si=0)   2      2     13     48     48     50     56
  Transformer           1      1      2      5     13     32     36
```
### Recall — `speed_demo.py --task recall --pairs 1 --queries 1` (store-1/query-1)
```
  LEAFv5 (fast si=.2)   5     86     99    100    100    100    100
  LEAFv5 (paper si=0)   1      3     82    100    100    100    100
  Transformer           2      3      8     19     68    100    100
```
### LM race — `speed_demo.py --task lm` (Tiny Shakespeare char-LM, held-out loss)
```
  model                    1       3       5      10      20      50     100
  LEAFv5 (fast si=.1)  4.087   3.835   3.582   3.227   2.847   2.415   2.211
  LEAFv5 (paper si=0)  4.156   4.066   3.961   3.580   3.098   2.576   2.344
  Transformer          3.993   3.466   3.153   2.844   2.682   2.481   2.408
```
### World benchmark — `benchmark_world.py --steps 20`
```
  params: LEAFv5=0.65M  Transformer=0.94M  GatedRNN=0.44M
  recall store-2/q1 (s1/s3/s5/s10/s20):  LEAFv5 2/9/29/46/50 · Trans 2/4/5/4/5 · GatedRNN 1/2/1/2/2
  LM (s20/s60/s120): LEAFv5 2.845/2.384/2.225 · Trans 2.685/2.483/2.396 · GatedRNN 5.456/3.036/2.632
```
### PTB char PPL — `benchmark_ppl.py --steps 150` (vocab 50, batch 16, seq 64, lr 1e-3)
```
  LEAFv5 6.1   Transformer 8.4   GatedRNN 9.5        (and 4.76 @300 steps for LEAFv5)
```
### Extrapolation — `extrapolate.py --steps 150` (train @64, eval longer)
```
  LEAFv5 (rope off)  2.151 2.173 2.106 2.067 2.082   ratio 0.97x
  LEAFv5 (rope full) 2.231 2.253 2.348 2.537 2.832   ratio 1.27x
  Transformer        2.462 2.640 2.579 2.550 2.563   ratio 1.04x
```
### Long-range — `longrange_demo.py --steps 400` (store-4/recall-2, chance 0.8%)
```
  train recall: 8.3% @80, 25.0% @160, 16.7% @240, 8.3% @320, 8.3% @400   (noisy, ≈ chance)
  retention by distance: carry 6.2/0.0/6.2/0.0 % vs reset 0.0/6.2/0.0/0.0 %  (≈ chance)
```
### Compute — `compute_demo.py --steps 140`
```
  per-token FLOPs: LEAFv5=1.60M  Transformer=1.87M (0.86x)
  targets 2.0/1.0/0.5/0.2/0.1: BOTH models "never";  final loss LEAFv5 2.122 · Transformer 2.351
```
### Stability — `stability_check.py --steps 200` → **9/9 PASS, RESULT: STABLE**

---

## What this means (honest reading)

1. **The comparative direction survives.** At matched steps on recall LEAFv5 is ~10× a
   same-size Transformer in the early steps and stays ahead; on PTB and Shakespeare LM it
   reaches lower held-out loss from ~step 60–120 on. GatedRNN (Mamba-lite) trails on LM
   (2.632 vs 2.225 @120) and is near-chance on recall.

2. **The dramatic magnitudes were leak artifacts.** "100% recall by step 5–50" (hard task),
   "PTB PPL 1.0", "40× faster learning", "1/20 compute-to-target", "100% flat retention at
   D=1024", "0.30x extrapolation improvement" — none of these reproduce post-fix. The look-ahead
   leaks made the model appear to master tasks it now merely wins. The real advantage is
   **roughly 1.3×–10× per-step depending on task, not 40×.**

3. **What is NOT affected by the leaks and stands verified:** stability 9/9, growth exactness,
   train≡decode invariant, C-kernel parity, and the *relative* extrapolation property
   (rope-off never degrades with length; rope-full does).

4. **PTB PPL is 6.1 @150 steps (4.76 @300), not near-fit.** The improvements6 note claimed
   "absolute PPL 1.0 reflects near-fit" — that was the leak talking. Post-fix the model is
   still the best of the three at equal steps, but it is not "solved".

5. **The smoke-training claim (§8b, val PPL 1.07) could not be re-verified in this sandbox**
   (train.py exceeds 2 GB RAM on CPU and is OOM-killed). It must be re-run on the T4 before
   being quoted again.

---

## Action items already applied

- `README.md` §15/§17/§18, `research/world-class.md` evidence table,
  `research/improvements6.md` PTB table, `research/synthesis.md` compute claim:
  corrected to the re-measured numbers (see git diff of this round).
- `tests/test_limits.py::test_ablate_runs` no longer spawns a second torch process
  (was OOM-flaky on low-RAM boxes); `tests/test_grow.py` duplicate definitions removed
  (two regression tests were shadowed and never collected — suite count corrected 73 → 71).

## Next steps (not yet done)

- Re-run §8b smoke training on GPU hardware and record the real val PPL.
- Architecture work (see §below): input-dependent global decay (Gated-DeltaNet-style) and
  a default hybrid with interleaved attention layers — the two highest-leverage changes;
  re-benchmark **post-implementation** before any new headline is published.

---

## Appendix B — architecture next steps: measured at micro scale (same day)

The reviewer's two highest-leverage suggestions were (1) input-dependent global
decay (Gated-DeltaNet-style α) and (2) default hybrid attention (Jamba/Griffin/
Samba-style interleaved SWA). Both existed as opt-ins (`--input-decay`,
`--swa`); a controlled A/B at micro scale (dim 128, d_h 48, same recipes as
speed_demo) gives:

**LM, Tiny Shakespeare held-out loss, 2 layers, 100 steps** (speed_demo config)
```
  base       10:3.127  20:2.768  50:2.375  100:2.188
  +SWA w32   10:3.148  20:2.771  50:2.389  100:2.183
  +inp-decay 10:3.151  20:2.770  50:2.393  100:2.177
```
**Recall store-2/query-1 held-out %, 4 layers, 60 steps** (chance 1.6%)
```
  base 40.0%   +SWA w32 40.6%   +inp-decay 38.7%   (all at step 60)
```
**Verdict: neutral within run-to-run noise at micro scale.** Neither feature
hurts; neither helps at this scale in 60–100 steps. Consistent with the
existing README note (input-decay "value only under long-context memory
pressure"). Action taken: added `swa_every` (interleave period, Jamba/Griffin
style) to config/model/train (`--swa-every k`), index-based so `grow_depth`
extends the pattern exactly; regression test added
(`test_swa_interleave_every`). Defaults stay honest: opt-in until real-scale
(T4) evidence exists — flipping defaults on a null result would be the same
overclaiming mistake this ledger exists to prevent.
