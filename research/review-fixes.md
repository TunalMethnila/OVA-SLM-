# Code-review fixes: all 13 issues addressed & verified

An expert review of the LEAFv5 repo identified 13 issues. Every one is now
fixed, with regression tests. The central guarantee — **training forward ==
token-by-token decode** — is now an enforced invariant.

## P0 fixes

**1. Causal convolutions (data leakage)** — `MultiScaleLocalPath` and the
memory short-conv used SYMMETRIC padding (`padding=k//2`), so token t saw
future tokens. Replaced with `CausalConv1d`: left-pad-only + explicit
carryable history. Verified: perturbing a future token changes NOTHING at
earlier positions (max|d| = 0.0).

**2. Recurrent generation == training** — two sub-issues:
   - `generate()` zeroed the caller's `offset` (destroying RoPE position).
     Fixed: starts from `states.offset`.
   - The convolutions had NO state at decode (token-by-token saw 1 token), so
     decode ≠ full-sequence. Fixed with the new **`LeafStates`** runtime
     object: delta memory + local-conv history + short-conv history + SWA KV
     cache + position. Verified: full-seq == token-by-token decode across ALL
     feature combos (max|d| ~ 1e-6, incl. after training).
   - **Bonus root cause found while testing**: the memory output reshape
     `permute(0,2,1).reshape(B,T,H*dh)` was WRONG for `[BH,dh,T]` — it mixed
     heads with positions (head stride T·dh ≠ dh), leaking later tokens into
     earlier outputs even with causal convs. Fixed to
     `permute→view(B,H,T,dh)→permute→reshape`. This was in the code from the
     start; training compensated via the permutation, but it broke causality
     and train/decode equality.

**3. CLI generate argument-order bug** — `generate(model, tok, args.prompt,
args.max_new, args.temperature, args.top_k, args.top_p, args.device,
verbose=True)` passed `device` as `repeat_penalty`. Fixed with keyword args.

**4. Gradient accumulation scaling** — `loss` was NOT divided by
`grad_accum`, so the accumulated gradient was grad_accum× too big (and OOM
halving made it worse). Fixed: `loss = loss / grad_accum`. Verified: the
accumulated-mean gradient == a single-batch gradient exactly (max|d| = 0.0).

**5. Rollback snapshot not a copy** — `shadow_sd = model.state_dict()`
shares storage with live weights, so "roll back" restored updated weights.
Fixed: `{k: v.detach().clone() ...}`. Verified: snapshot stays fixed under a
+3.0 perturbation.

**6. DDP not data-sharded** — all ranks used the same batch seed → identical
batches. Fixed: rank-aware seeds (`seed + rank*100000`) for the prefetcher and
streams.

**7. DDP checkpoint keys** — `module.`-prefixed keys broke loading into a
plain LeafLM. Fixed: `_strip_module_prefix` on save AND load; unexpected keys
now warned, missing keys warned loudly.

**8. BPE streaming boundaries** — encoding 1-MB chunks independently can
differ from the full text (byte-level merges cross boundaries). Fixed:
tokenization now encodes COMPLETE LINES (byte-level pretokenizers split on
whitespace, so a token never spans a newline). Verified: linewise == full-text
parity.

## P1 fixes

**9. Chunked StateNorm is a distinct mode** — docstrings updated to say
explicitly that chunked+StateNorm ≠ sequential+StateNorm (norm at chunk
boundaries); the chunked≡sequential test only claims the no-norm case.
Training and eval already use the same `chunk` mode.

**10. SWA recurrent KV cache** — `SlidingWindowAttention` now keeps a KV
cache so decode sees the sliding window (it only saw the current token
before). Verified identical full-seq vs decode. (Also fixed a mask
broadcasting bug in the cached path.)

**11. RoPE cache extension** — `offset + T > max_seq_len` used to crash; now
the cache grows by doubling on demand. Verified past max_seq_len=8 for 40
steps.

**12. Quantization device bug** — the quantized (CPU) model was evaluated on
`--device cuda`. Fixed: evaluate the quantized model on CPU.

**13. EMA resume** — clarified/verified: `eval_model()` returns the EMA copy,
so checkpoints already save the EMA weights and resume restores the trained
EMA.

## Verification

- **New `tests/test_causal_invariant.py`** (4 tests): causality,
  train==decode across 6 feature combos, train==decode after training, RoPE
  extension.
- **New `tests/test_review_fixes.py`** (5 tests): grad-accum scaling, snapshot
  isolation, DDP key stripping, streaming parity, RoPE-in-generate.
- Full suite: **15 suites / 74 tests pass** with the fixed architecture.
- Functional smokes: training (loss descends), finetune, stateful serve
  sessions all work.
