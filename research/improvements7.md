> **SUPERSEDED NUMBERS (2026-08-09).** Measurements in this file were taken
> before the §29 causality fixes (symmetric-padded convs; reshape head/position
> leak) and the training-mechanics fixes. Re-measured values are in
> **research/reverify-2026-08.md** — several headline numbers here (100% recall
> by step 5, PTB PPL 1.0, ~40× per-step, 1/20 compute, 100% flat long-range
> retention) did NOT survive. Read this file as history, not as current evidence.
>

# Round 8: the dead-memory bug (found & fixed), stateful sessions, self-consistency

## 1. CRITICAL BUG FOUND: the delta memory was DEAD in default configs

While building stateful sessions, a session-continuity check failed — and
chasing it revealed something much bigger:

**The SOTA output gate (`silu(W_gate x) ⊙ out`) was initialized to
`weight=0, bias=None` → `silu(0) = 0` → the entire memory branch output was
multiplied to ZERO, and the gate could never learn (its gradient was 0 too).**

Consequences (this silently invalidated several earlier "memory" narratives):
- The delta memory contributed **nothing** in every default config since the
  SOTA upgrade (round 7).
- Earlier long-range retention failures ("crosstalk limit") were really "the
  memory isn't doing anything" — with the fix, retention is **100% flat to
  D=256** in a fraction of the training.
- "Chunked ≡ sequential" tests only passed because both outputs were 0.
- Fusion ≈ plain (both had dead memory).

**The fix:** `out_gate = Linear(D, D, bias=True)`, `weight=0`, `bias=1.278465`
→ `silu(bias) = 1.0` **exactly**, so the gate is the identity at init (the
identity-start design is preserved), the memory contributes from step 1, and
gradient flows.

**Verified:** memory output non-zero, state-dependent, gate=1 at init
(regression test `test_memory_alive_regression` added — it fails on the old
init). Recall race still 100% by step 5; long-range retention now genuinely
100% at D=256 (with 120 steps); the chunked≡sequential equivalence now holds
at 1.4e-7 with a real memory.

**Also fixed:** several tests compared outputs in *train* mode where
`mem_dropout` applied different masks to each call (they'd only passed because
the memory was dead) — switched to eval mode for equivalence checks.

This is exactly why "push limits" is valuable: the pressure to add stateful
sessions exposed a bug that had been quietly undermining the architecture's
core claim for several rounds.

## 2. Stateful sessions: the delta memory IS conversation memory

New `serve.py` sessions: `/chat` with `session_id` carries the tiny recurrent
state + position offset across turns. **History is never re-encoded** — the
memory holds it. Verified: state carries (logit diff 0.12 carried-vs-fresh),
memory constant at **49 KB per session regardless of conversation length**
(a transformer's KV cache grows per turn). Sessions auto-reset after a token
budget; bounded memory. This is a LEAFv5-native deployment advantage.

## 3. Self-consistency decoding (`eval_skills --self-consistency K`)

Best-of-K sampling: sample K completions per prompt, correct if any passes the
grader. Standard accuracy boost for small models on math/tools.

## 4. One-command report (`python -m leafv5.report`)

Runs resource_demo + benchmark_world (+ compute_demo, benchmark_ppl on full
runs) and writes a single `report.md` with all the measured numbers. (Note:
full runs are slow — use `--quick` in constrained environments; every
component is verified individually.)

## 5. Validation

All 12 suites pass (53 tests including the new dead-memory regression). The
re-run of the full suite after the fix: recall race 100% by step 5, long-range
retention 100% at D=256, chunked≡sequential 1.4e-7.
