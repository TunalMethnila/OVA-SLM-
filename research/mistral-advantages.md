# Mistral's architectural advantages — adopted into LEAFv5

Survey date: 2026-08-09. Sources: Mistral 7B (arXiv:2310.06825), Mixtral 8x7B
(arXiv:2401.04088), Mistral's own documentation of the rolling-buffer cache and
pre-fill & chunking.

Mistral's efficiency story is three attention-side ideas (GQA, SWA, rolling
buffer + pre-fill-and-chunking) plus one scaling idea (Mixtral's top-2 MoE).
This note maps each onto LEAFv5: what already existed, what was added today
(2026-08-09), and what is honestly measured.

---

## 1. Sliding-window attention (SWA) — already present, now interleavable

**Mistral 7B:** each token attends to at most the previous W=4096 tokens;
stacked layers extend the effective receptive field to ≈ W·L (the paper's
figure: after 4 layers a 3-token window reaches 12 tokens). Cost drops from
O(T²) to O(W·T).

**LEAFv5:** `SlidingWindowAttention` (opt-in `--swa`, zero-init residual scale
so the no-attention default is preserved). **Added 2026-08-09:**
`--swa-every k` — Mistral puts SWA in *every* layer; Jamba/Griffin/Samba-style
hybrids interleave attention periodically. LEAFv5 now supports both:
`--swa-every 1` (Mistral) or `k > 1` (periodic). The pattern is index-based so
`grow_depth` extends it exactly (regression-tested).

## 2. Grouped-query attention (GQA) — NEW 2026-08-09

**Mistral 7B:** 32 query heads share 8 KV heads (ratio 4:1) → KV cache 4×
smaller, less memory bandwidth at decode, higher batch sizes.

**LEAFv5 (added):** `--swa-kv-heads k` (config `swa_kv_heads`, default 0 =
MHA, fully backward compatible). KV projections now output `kv_heads·d_h`
channels; each KV head serves a contiguous group of query heads
(`repeat_interleave` grouping, the standard GQA layout).

**Measured (this sandbox, 2026-08-09):**
- `kv_heads == heads` ⇒ bit-identical to the pre-GQA MHA (max|Δ| < 1e-6).
- `kv_heads = 1` with 4 heads ⇒ KV cache width 4× smaller; `kv_bytes`
  scales by exactly `heads/kv_heads` (4096 vs 8192 bytes/layer/batch at
  heads=4, kv=2 vs 4, window 16, d_h 16).
- Train-forward ≡ token-by-token decode still holds with GQA + interleave
  (added to `test_causal_invariant.py`, max|Δ| < 1e-4).

## 3. Rolling-buffer KV cache — NEW 2026-08-09

**Mistral 7B:** "The cache has a fixed size of W, and the keys and values for
the timestep i are stored in position i mod W of the cache. When the position
i is larger than W, past values are overwritten, and the size of the cache
stops increasing." → decode memory is **constant** in sequence length.

**LEAFv5 (added):** `RollingKVCache` — fixed `[B, Hkv, W, d_h]` storage with
`(pos + j) mod W` writes (`index_copy_` into preallocated buffers, no per-step
reallocation) and a slice-and-cat `window()` that returns the most recent W
tokens in chronological order. Pass a `RollingKVCache` as the SWA `kv_cache`
to decode with it; `SlidingWindowAttention.prefill` returns one.

**Measured (this sandbox, 2026-08-09):**
- Rolling decode == tuple-cache decode **exactly** (max|Δ| = 0.0 over 40
  tokens) — same math, different storage.
- After 40 tokens into a W=16 cache: storage shape still `[2, Hkv, 16, d_h]`
  (constant), 40 tokens stored in 16 slots.
- Combined with GQA(1/4) the decode KV memory is **16× smaller** than
  full-context MHA (4× from GQA, 4× from the window at 4× window length) —
  the same composite factor Mistral advertises.

## 4. Pre-fill & chunking — NEW 2026-08-09

**Mistral:** instead of one quadratic pre-fill over the whole prompt, split it
into chunks of size ≤ W and process them sequentially with the rolling cache.
"Bounding memory usage during pre-fill to the same level as during generation"
and enabling arbitrary-length prompts.

**LEAFv5 (added):** `SlidingWindowAttention.prefill(x, pos, chunk)` — chunked
rolling pre-fill for any prompt length; returns a `RollingKVCache` ready for
decode.

**Measured (this sandbox, 2026-08-09):**
- Chunked (7-token chunks) == one-shot pre-fill of a 50-token prompt:
  max|Δ| = 6e-8 (pure float associativity), identical cache width (= W).
- Decode continuation through both caches gives identical next-token logits
  (atol 1e-6).

## 5. Mixtral-style top-2 MoE — already present, now verified

**Mixtral 8x7B:** 8 SwiGLU experts per layer, router picks top-2 per token,
load-balancing aux loss (ST-MoE: `n_e · Σ f_i p_i`) trained with, not used at
inference; 46.7B total / 12.9B active.

**LEAFv5:** `--moe` (`moe_experts=8`, `moe_topk=2`, `--moe-aux-weight 0.01`)
— identical formulation (top-2 of 8 full-size SwiGLU experts, router, aux
loss added by the trainer). **Verified 2026-08-09** in
`test_mixtral_moe_aux_loss`: aux loss equals the hand-computed
`n_e · Σ f_i p_i`, model-level `aux_loss()` wires through, training stable.

---

## Honest notes

- **Everything above is functional/efficiency plumbing with exact-equivalence
  tests — not a claim of quality gains.** GQA trades a little representational
  capacity for a 4× smaller KV cache; the rolling buffer and chunked pre-fill
  change *how* memory is allocated, not what the model computes (proven exact).
  At micro scale (dim 128, ≤ 100 steps) SWA and input-decay were measured
  *neutral within noise* on recall/LM (see `reverify-2026-08.md` Appendix B) —
  so SWA/GQA stay **opt-in**, pending evidence at real (T4) scale.
- **Defaults are untouched:** `swa_kv_heads=0` (MHA), SWA off, MoE off. Every
  existing checkpoint and config loads unchanged; the new features are strictly
  additive flags.
- The one real-scale question this sandbox cannot answer: whether GQA's
  capacity reduction hurts quality at T4 scale, and whether interleaved SWA
  lifts the pure-recurrence recall ceiling (the reviewer's highest-leverage
  hypothesis). `--swa --swa-every k --swa-kv-heads k` is the flag set to test
  it on the T4 run.
- Mistral's *other* ingredients (RoPE, RMSNorm, SiLU, KV-cache quantization in
  newer models) — LEAFv5 already has RoPE/RMSNorm/SiLU; KV-cache quantization
  (KVQ) is listed as future work, not implemented.

## Files

- `leafv5/model.py`: `RollingKVCache`, `SlidingWindowAttention` (GQA +
  rolling + `prefill`), `MoEFFN` (verified).
- `leafv5/config.py`: `swa_kv_heads`; `leafv5/train.py`: `--swa-kv-heads`.
- `tests/test_mistral_advantages.py` (6 tests), plus a GQA+interleave variant
  added to `tests/test_causal_invariant.py`.
- `leafv5/mistral_demo.py`: runnable KV-memory comparison.

---

## Appendix — stability certification of the stack (2026-08-09)

The Mistral stack gets its own stability certificate, mirroring
`stability_check.py`: `python -m leafv5.stability_check_mistral` →
**MISTRAL-STACK STABILITY CERTIFICATE: 10/10 passed, RESULT: STABLE**.
The 10 measured properties:

1. **Boundary exactness** — rolling-buffer decode == tuple-cache decode at
   *every* step across 3+ window wraps (W-1, W, W+1, 2W-1, 2W, 2W+1, and
   pos%W==0): max|Δ| = **0.0**; storage stays `[B, Hkv, W, dh]`.
2. **Position-offset prefill** — `prefill(hist, pos=7)` + decode == one-shot
   full-seq windowed forward over the same tokens: max|Δ| = **8.4e-9**.
3. **Determinism** — rolling decode, prefill, MoE forward, and the full
   SWA+GQA model are all **bit-identical** on repeated same-input runs.
4. **Long decode** — 3000-token rolling decode: all outputs finite, max|out|
   0.04, storage constant, position counter exact.
5. **Edge cases + guards** — W=1, heads=1, kv=1, T==W, T==W+1, batch 1/7 all
   run finite; GQA divisibility (`heads % kv_heads != 0`) and `chunk > W`
   raise cleanly; empty prefill returns None.
6. **MoE stability** — 100 training steps: loss finite, aux loss peaks at
   2.02 (< n_e=8), **all 8 experts utilized** (per-expert counts 1448–1774,
   near-balanced), router logits bounded (max 0.88).
7. **Deep stack** — 12 layers + SWA/GQA(kv=1)/MoE forward+backward: grads
   finite.
8. **Chunked prefill** — chunk ∈ {1, 2, 3, 7, W} == one-shot prefill:
   max|Δ| = 1.8e-7, width == W.
9. **Low precision** — bf16 rolling == tuple exact (Δ=0.0), fp16 forward
   finite.
10. **Train == decode with GQA after training** — full-seq == rolling decode
    on a trained SWA+GQA model: max|Δ| = **4.8e-7** (reviewer's invariant).

**Two latent bugs the certification found and fixed (2026-08-09):**
- `RollingKVCache.window()` assumed the first written slot was 0; prefilled
  at a nonzero offset (stateful resume) it returned unwritten zero slots as
  history. Fixed by tracking the first valid absolute position; the partial
  fill now returns the correct circular range. Regression: check #2.
- The full-sequence attention path built its causal mask as float32, which
  promoted bf16/fp16 scores to fp32 and crashed the low-precision matmul.
  Fixed by building the mask in `scores.dtype`. Regression: check #9.

All 10 checks are guarded by `tests/test_stability_mistral.py` (9 tests);
the base 9/9 certificate and the 87-test suite remain green.
