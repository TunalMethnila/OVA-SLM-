# Round 7: reasoning data, PEFT, deterministic decoding, standard benchmark

## 1. Chain-of-thought math data (verified reasoning)

`data_gen` now emits ~40% of the 3,000 math examples as **chain-of-thought**
("show your work") with step-by-step answers — all answers computed by the
generator and **verified: 0 wrong out of 600 spot-recomputed examples** (the
verifier caught and fixed a real bug: CoT-add firing on three-number additions).
Training on CoT teaches the model to reason step by step, which is the
practical path to better arithmetic on small models.

## 2. LoRA parameter-efficient fine-tuning (`--lora-rank R`)

New `leafv5/lora.py`: wraps the memory/FFN/gate Linears in low-rank adapters
(B=0 → **output identical to the base at init**), freezes all base weights, and
trains only the adapters. Measured on micro: **18 layers wrapped, ~6-12% of
params trainable** (on t4-4h that's ~1-2%); adapters learn, base stays frozen;
**merge is faithful** (|Δ| ~1e-4 vs the LoRA output, pure float associativity).
At save time the adapters are **merged into the base** so the checkpoint is a
plain LEAFv5 usable by every tool (generate/serve/grow/quantize). This is the
practical "adapt on any GPU" story: a fine-tune that trains ~1-2% of the
weights and merges back.

## 3. Beam search (`beam_search`, `eval_skills --beam`)

Deterministic decoding for accuracy-critical tasks (math, tool JSON): keeps
the top-`beam_size` sequences by log-prob, returns the best complete one.
Wired into `eval_skills --beam N`.

## 4. Standard-corpus perplexity: Penn Treebank (PTB)

New `benchmark_ppl.py` — the classic small-LM benchmark (char-level PTB, 5.1M
train chars, 50-char vocab), matched steps:

| model | valid PPL @150 steps |
|---|---|
| **LEAFv5** | **6.1** |
| Transformer | 8.4 |
| GatedRNN (Mamba-lite) | 9.5 |

**LEAFv5's valid perplexity is ~1.4–1.6× lower than both baselines on the
canonical small-model corpus.** (Re-measured 2026-08-09 after the §29 causality
fixes: the previously published 1.0 was a look-ahead artifact — the leaks made
the model appear to near-fit PTB. Post-fix, PPL is 6.1 @150 steps and still
descending — 4.76 @300 — so the *relative* gap at equal steps is the claim, and
it matches the Shakespeare and synthetic results: a real but modest per-step
learning edge at micro scale, not 40×.)

## 5. Validation

`tests/test_round7.py` (4 tests: LoRA identity/train/merge, CoT math verified,
beam search, PTB plumbing). Full suite = **52 tests / 12 suites**, all pass.
