# LEAFv5 bug hunt — 2026-08-09

A full audit pass: static analysis (pyflakes), module-by-module code reading of
every leafv5/*.py, and targeted boundary probes.  **8 real bugs found and
fixed**, each with a regression test in `tests/test_bugfix_aug09.py` (7 tests;
plus the `max_new=0` semantic fix folded into the beam test).

---

## 1. `generate()` ignored the caller's offset for plain-list states  (model/generate)

**Code path.** `serve.py` tracks a session's absolute position and passes
`states` (a plain list of delta states) + `offset` to `generate()`.  But
`generate()` did `offset = getattr(states, "offset", 0)` — always 0 for a list
— and the first forward `model(inp, states, fast=fast)` didn't pass `offset`
at all.  So every stateful turn restarted RoPE positions at 0.

**Impact.** With `rope_dim=0` (default) invisible; with RoPE on, the second and
later turns of a stateful session had wrong positional encodings.

**Fix.** Use the caller's `offset` unless the state carries a nonzero one, and
pass `offset=` on the prompt pass.

**Verified.** Stateful turn-2 (both carried `LeafStates` and plain-list +
offset) == one-shot full-sequence generation with `rope_dim=96`, exactly.

## 2. `beam_search()` never conditioned on the full prompt  (generate)

**Code path.** The old loop started beams from `model.init_states` and fed only
`toks[-1:]` with `offset=len(toks)` — the first iteration fed the *last prompt
token* from a *fresh* state (offset off by one).  The model never saw tokens
0..len(ids)-2, so beam decoding was effectively unconditional.

**Fix.** Prefill the full prompt in one pass; the first expansion comes from the
prompt pass's last logits (same as `generate()`); subsequent steps feed the
latest token with `offset=len(toks)-1`.

**Verified.** Beam's first token == greedy's first token; `max_new=0` returns
`""` (semantic fix #8, matching `generate()`).

## 3. `Corpus.sample_batch` val split read past the end on tiny corpora  (data)

**Code path.** The val branch inflated `n_val` when the split was smaller than
one window, then picked offsets such that the y slice `[o+1 : o+seq+1]` ran
past `arr[n_tokens]` → `IndexError` / x-y length mismatch (the "val-batch
crash on tiny val splits" from the §29 review was only partially fixed).

**Fix.** Clamp the window to `n_tokens - 1` and use the same safe fallback as
the train branch; identical fix in `StreamCorpus.sample_batch`.

**Verified.** 200-token corpus, seq 64/128/256: train and val batches match in
shape, in bounds.

## 4. `weights.py` salted `hash()` broke cross-process unpack  (weights)

**Code path.** `h = hash(w.cpu().numpy().tobytes())` — Python's built-in
`hash()` on bytes is salted per process (`PYTHONHASHSEED`).  In-process
pack/unpack worked (same seed), but a packed model saved to disk then unpacked
in a new process hit `KeyError` on the shared refs — despite the README's
"smaller checkpoints" claim implying persistence.

**Fix.** `hashlib.sha256(...).hexdigest()`.

**Verified.** Pack → save → load in a **subprocess** → unpack succeeds with the
right key count.

## 5. finetune `--lora-rank` + `--grow-at` crashed  (finetune)

**Code path.** `grow_width` reads `.weight` on memory/FFN/gate Linears; a
LoRA-wrapped model has `LoRALinear` (no plain `.weight`) → `AttributeError`.

**Fix.** Before growing: `merge_lora(model)` (function-preserving), grow, then
re-apply fresh LoRA adapters with a printed note.

**Verified.** Full finetune smoke (LoRA rank 8 → grow dim 64→128 at step 5)
runs end to end; unit test asserts function preservation (max|d| ~ 1e-6).

## 6. finetune eval crashed on a tiny val split  (finetune)

**Code path.** The eval block sliced `val_arr[q : q+seq]` / `[q+1 : q+seq+1]`;
on a tiny val array both slices truncated to *different* lengths (len vs len-1),
`np.stack` succeeded (equal within each stack), and `cross_entropy` crashed
with "batch_size 3168 != 3152".

**Fix.** Clamp the eval window to the largest that fits; skip eval with a
warning if the split is unusable.

**Verified.** The earlier smoke run that crashed now prints clean eval lines.

## 7. `serve.py` session offset drifted  (serve)

**Code path.** The session offset was recomputed from
`len(prompt) + len(out)`; when the repetition guard stopped generation before
`max_new`, the recomputed offset disagreed with the actual carried state
position.

**Fix.** Store `new_states.offset` (the exact absolute position returned by
`generate`).

**Verified.** In-process `session_chat` unit test: offsets are ints, monotonic
across turns.

## 8. `beam_search(max_new=0)` emitted a token  (generate)

**Code path.** The initial prompt-pass expansion always produced one token even
for `max_new=0` (inconsistent with `generate()`'s `max_new=0 → ""`).

**Fix.** Early return `""` for `max_new <= 0`.

---

## Verification status after the fixes

- Full suite: **94 tests / 18 suites, all passing** (89 + 5 in the two
  memory-split runs on the 2 GB CPU sandbox).
- Base stability certificate: **9/9 STABLE**; Mistral-stack certificate:
  **10/10 STABLE** (re-run after the changes).
- `test_causal_invariant.py` (train == decode, causality 0.0) still green —
  the data/generate/finetune changes don't touch the model's forward math.
- pyflakes: only benign "imported but unused" notes remain (deliberate
  re-exports / test imports).

## Notes on what was checked and found NOT broken

- Delta-memory scan math (`_sequential`, `_chunked_scan`, `_maps`, StateNorm):
  verified against the documented recurrence; chunked==sequential exactness is
  already regression-tested.
- RoPE caching/extension, `CausalConv1d` state carry, `MultiScaleLocalPath`,
  GQA/rolling-buffer/prefill (already certified in the Mistral round).
- Resume RNG restore (`set_rng_state`, not `manual_seed`), DDP seed sharding,
  grad-accum division, EMA checkpointing — all from §29, still correct.
- `uint16` corpus ids: guarded by a `vocab_size <= 65535` check in
  `prepare_corpus`.
