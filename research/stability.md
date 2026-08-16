> **SUPERSEDED NUMBERS (2026-08-09).** Measurements in this file were taken
> before the §29 causality fixes (symmetric-padded convs; reshape head/position
> leak) and the training-mechanics fixes. Re-measured values are in
> **research/reverify-2026-08.md** — several headline numbers here (100% recall
> by step 5, PTB PPL 1.0, ~40× per-step, 1/20 compute, 100% flat long-range
> retention) did NOT survive. Read this file as history, not as current evidence.
>

# Stability round: the formal "very stable" guarantee

"Very stable" is now a **measured certificate**, not a claim. This round
hardened the remaining real gaps and added a formal battery.

## 1. Real bugs found & fixed (hardening)

- **`generate()` crashed on an empty prompt** (`IndexError: list index out of
  range`) and **`max_new=0` still emitted a token** (the prompt-pass sampled
  unconditionally). Fixed: empty prompts seed a start token; the first token
  is only sampled when `max_new > 0`; plus a **defensive finite-logits
  fallback** (if logits are ever NaN, they're zeroed — the model can never
  emit NaN).
- Verified edge cases now: empty prompt, max_new=0, temperature<=0, negative
  temperature, top_k=10^9 — all produce valid output, never crash.

## 2. Training-time instability visibility

- **Gradient-norm monitor** in `train.py`: every log interval now prints
  `grad=` (the global grad L2 norm, post-clip). A gradient spike is the
  earliest instability signal; the NaN-guard and loss-spike recovery already
  handle them automatically — now you can *see* it happening.
- **`--deterministic`**: CUDA-deterministic algorithms (cuDNN/atomics) for
  fully reproducible runs, on top of the existing seed.

## 3. Stability certification (`python -m leafv5.stability_check`)

A 9-check battery that stress-tests the model and prints a PASS/FAIL
certificate:

```
[PASS] edge inputs (empty/0/new/NaN-ish) never crash
[PASS] determinism (same input -> same output)
[PASS] weight perturbation +-1% stays proportional     (rel 0.011)
[PASS] input perturbation (1 token) bounded            (rel 1.048)
[PASS] state perturbation bounded                      (rel 0.114)
[PASS] NaN-grad guard fires on injected NaN
[PASS] training RECOVERS after NaN (200 steps, LR 3e-2)
[PASS] states bounded after stress
[PASS] 24-layer stack forward+backward finite
STABILITY CERTIFICATE: 9/9 passed  ->  RESULT: STABLE
```

Notable: the certification itself caught a test-logic bug on its first run
(the intentionally-injected NaN was counted as a training failure) — fixed to
measure the real property (recovery). The architecture passes everything:
perturbations stay proportional (no blow-up), NaN can't corrupt training, 24
layers train with finite grads, states stay bounded under 300 stress steps.

## 4. The stability stack (all layers, all verified)

| layer | mechanism | verified by |
|---|---|---|
| architecture | StateNorm (‖S‖_F ≈ √d_h), L2-norm q/k/v, fp32 states, zero-init residual scales | stability_cert + stability suites |
| optimizer | grad-clip 1.0 default, fp32 params, NaN-grad guard, loss-spike recovery (roll back + halve LR), autotune LR | test_stability, test_easy |
| training | `--safe-mode` (fp32/sequential/conservative), `--deterministic`, grad-norm monitor, OOM auto-recovery | test_easy |
| inference | edge-input handling, finite-logits fallback, repetition guard, max-consecutive | stability_cert |
| deployment | int8 quantization (loss-preserving), weight-store pack/unpack (loss-preserving) | test_stability, test_round6 |

## 5. Validation

All 13 suites / 59 tests pass. `python -m leafv5.stability_check` =
**9/9 STABLE**.
