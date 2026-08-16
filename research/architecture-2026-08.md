# Architecture gap analysis — 2026-08-16

What LEAFv5 has, what the real remaining gap is, and the one mechanism that
directly attacks it.  This file is the reasoning; the implementation and the
measured A/B are in `leafv5/model.py` (config `dp_norm`) and
`tests/test_arch_dpnorm.py`.

---

## 1. Where LEAFv5 stands (measured, current)

| capability | status | evidence |
|---|---|---|
| train == decode invariant | exact (~1e-6), causality 0.0 | test_causal_invariant |
| stability | 9/9 + 10/10 STABLE | stability_check* |
| exact growth (width/depth) | 1e-6 / 0.0 | test_grow, test_grow_vs_scratch |
| recall store-1/q1 | 100% in 10 steps | speed_demo |
| micro-scale LM edge | ~1.3-10x per-step, wins at every size | speed_demo, scaling_study |
| long-range retention @ micro | **≈ chance** (training-limited) | retention_study |
| C scan engine | 15-30x faster, exact | scan_engine |

The **one documented architectural weakness** (from the reviewer's Tier-1 and
from `reverify-2026-08.md`): the delta-memory readout `o = S@q` is an
**un-normalized sum** of rank-1 contributions.  Its scale grows with the
number of keys the query matches and drifts with context length — this is the
"crosstalk ∝ 1/√d_h" / scale-drift failure mode behind the retention
collapse and part of why the architecture needs bigger d_h to hold more.

## 2. The 2025 SOTA response: normalized readouts

Three 2025 linear-recurrent lines converge on the same fix:

- **Gated DeltaNet** (ICLR 2025): gates the state decay AND uses a
  normalized readout so the query-key match becomes attention-like.
- **Delta Product** (Samsung, May 2025): keeps TWO states under the same
  delta-rule recurrence — the numerator `S` (value-key outer products) and a
  denominator `D` (ones-key outer products) — and reads
  `o = (S@q)/(D·q)`.  The denominator counts the total weight the query puts
  on past keys, turning the unbounded sum into a bounded weighted average.
- **Mamba-2 / SSD**: state carries a normalization term for the same reason.

The DP trick is exact, linear-cost, and needs **no new learnable machinery**
beyond a per-head bias for numerical stability.  It directly targets the
documented #1 weakness.

## 3. The chosen upgrade: DP-normalized readout (opt-in `dp_norm`)

Per head, per token, in addition to the existing state `S ∈ R^{d_h×d_h}`,
carry a denominator vector `D ∈ R^{d_h}`:

```
# write (same gates/decay as S):
S ← a·S − bf·(S@k)·kᵀ + bw·v·kᵀ
D ← a·D − bf·(Dᵀk)·k   + bw·k          # v replaced by the ones-vector
# read (normalized):
o = (S@q) / (Dᵀq + b_h)                 # b_h: per-head bias, init 1.0
```

Why it should help:
1. **Bounded readout** — `o` becomes a weighted average over past values, so
   it cannot grow with context length (the scale-drift failure mode).
2. **Crosstalk suppression** — a query that matches only its own key gets
   that key's contribution *normalized out* of the total, reducing the
   "shared d_h dimensions" collision penalty.
3. **Attention-like** — `Dᵀq` plays the role of the softmax normalizer at
   linear cost; the delta-rule write/erase is kept (the rapid-adaptation
   property LEAFv5's whole story rests on).

## 4. Measured A/B (2026-08-16) — honest result

Implemented exactly (state carry via `LeafStates.dp`, per-head bias,
train==decode 3.7e-7, growth exact 3e-4, readout boundedness verified), then
measured against the baseline on the established micro-scale tasks:

| task | baseline | dp_norm | verdict |
|---|---|---|---|
| recall store-1/q1 (held-out % @ step 10) | 96.1% | 90.6% | slightly worse |
| LM held-out loss @60-100 steps | 2.346 | 2.362 | tie / slightly worse |
| extrapolation train@64 → eval@1024 (loss ratio) | 1.00x | 1.07x | slightly worse |
| readout scale @64/256/1024 (max|logit|) | 2.72/2.83/2.85 | 2.72/2.83/2.84 | both flat |
| train==decode | — | 3.7e-7 | exact ✓ |
| width+depth growth | — | 3e-4 | exact ✓ |

**Verdict: correctly implemented, but NO micro-scale benefit — a small
regression.**  The mechanism works as designed (bounded readout), but at
micro scale the baseline is already scale-stable (StateNorm bounds S; the
residual identity-start keeps logits bounded), so the failure mode DP
normalization targets is not the binding constraint here — consistent with
the established pattern that every opt-in lever (input-decay, SWA,
surprise-gate) is neutral at micro scale because the tasks are
training-limited.

**Decision:** `dp_norm` stays opt-in (default OFF).  It is regression-tested
(`tests/test_arch_dpnorm.py`), growth-compatible, and train==decode exact —
ready for the T4 scale run, where a real long-context regime (millions of
tokens, thousands of writes per head) is where an un-normalized sum would
actually drift.  No C/Mojo `_dp` kernel was written because the A/B does not
justify the investment (documented, not silently dropped).
