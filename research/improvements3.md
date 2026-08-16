> **SUPERSEDED NUMBERS (2026-08-09).** Measurements in this file were taken
> before the §29 causality fixes (symmetric-padded convs; reshape head/position
> leak) and the training-mechanics fixes. Re-measured values are in
> **research/reverify-2026-08.md** — several headline numbers here (100% recall
> by step 5, PTB PPL 1.0, ~40× per-step, 1/20 compute, 100% flat long-range
> retention) did NOT survive. Read this file as history, not as current evidence.
>

# Round 4: pushing the remaining limits (measured, honest)

## 1. Fixed the stale native kernels (correctness + a real fast path)

The C/Mojo kernels were validated for the PRE-SOTA architecture (read `o=S@k`).
The SOTA upgrade added a separate **read query** (`o=S@q`, erase along k), so
the kernels no longer matched the model. Rewrote `mojo/c_ref/leafv5_scan.c`:

- `leafv5_scan_q` — the current architecture: query-read, delta erase along k,
  optional input decay `a·S` and StateNorm.
- `leafv5_scan_fused` — kept for the q==k (paper-exact) variant, documented as
  such.

**Validated:** `max|Δ| = 1.1e-7` vs the torch sequential scan (both out and
state) — the kernel now matches the CURRENT model exactly.

**Wired into the model:** `MultiTimescaleDeltaV2.forward(fast=True)` calls the
C kernel in eval/no-grad (graceful Python fallback if the .so isn't built);
`generate()` auto-enables it when available. Outputs are **bit-identical** to
the Python path.

**Measured:** prompt/scan pass (512 tokens): Python 506 ms → fast **176 ms
(2.9×)**. Single-token decode is only ~1.1× faster — the per-step torch
overhead (projections, gates, FFN, head) dominates at 1 token/step, an honest
caveat. The C scan's real win is batched/prompt processing and long sequences.

## 2. Sliding-window attention hybrid (opt-in, `--swa`)

The last documented SOTA gap (GatedDeltaNet-H1 style): a causal local-window
attention branch per block, with its **own zero-init residual scale** (identity
at init → safe to add anywhere, and **exact under width growth**: max|Δlogit|
= 0.0 on a trained model). The paper's no-attention default is preserved.

**Measured (micro, Shakespeare LM, held-out):**
| step | no-SWA | SWA w=32 |
|---|---|---|
| 40 | 0.295 | **0.275** |
| 120 | 0.050 | 0.049 |

Mild early speedup, negligible at convergence — at this scale the delta memory
already covers local patterns. Kept **opt-in**; expect the win to grow at
larger scale / shorter windows where exact local mixing matters more.

## 3. Skill eval harness (`leafv5/eval_skills.py`) — does fine-tuning actually teach?

Automated, transparent graders per skill (identity → "Dassanayake"/"LEAFv5";
math → recompute the answer from the question; grammar → "Corrected"/"No.";
tools → valid JSON with a tool name; Sinhala → any Sinhala unicode; social →
non-empty; safety → refusal markers). Held-out fresh prompts from the data_gen
banks (no memorization credit). Plus `generate(max_consecutive=N)`: stops on
pathological repetition loops.

**Honest results in the sandbox (micro models, CPU):**
- The harness discriminates: an identity-only fine-tune scores **identity 30%,
  social 100%** vs **0% on every untrained category** — it measures real
  learning, not noise.
- BUT the sandbox's ~1-2M-param models cannot do fluent open-ended generation:
  they collapse into repetition loops ("TheTheThe…", "I am am am…"), capping
  scores. The models genuinely know the content (direct probes show "LEAFv5…",
  "single researcher who created…") but degenerate under sampling.
- On a real T4 with `t4-4h`, the same pipeline is the meaningful test:
  `python -m leafv5.finetune --data ... --model t4-4h --auto --steps 3000` then
  `python -m leafv5.eval_skills --ckpt out/.../best.pt`.

## 4. Capacity / scale verification

- `t4-xl` preset builds: **271.5M params** (the ~250M class).
- **seq=2048** forward+backward works on a small model (5.7 s on CPU) — the
  architecture is long-context-ready.
- **d_h=128** (Gated DeltaNet's optimal head dim) config works; state is only
  0.07 MB/layer, and memory capacity scales ~√d_h (directly attacks the
  crosstalk limit measured earlier).

## 5. TL;DR

Fixed the native kernels to match the current architecture (exact) and wired a
2.9× prompt-pass fast path; added the opt-in SWA hybrid (identity-init, exact
growth, mild early win at micro scale); built an automated skill-eval harness
with honest tiny-model caveats; verified the 271M preset, 2048 context, and
d_h=128. All 26 tests pass (6 suites).
