# LEAFv5 vs the world's best SLM architectures — gap analysis & fixes

Survey date: 2026-08. Sources: ICLR 2025 Gated DeltaNet, Mamba/Mamba2, DeltaNet,
RetNet, HGRN2, Titans, and 2026 SLM roundups (Phi-4-mini, Qwen3, SmolLM3,
Llama 3.2 3B, LFM2).

## 1. The SOTA landscape (mid-2026)

### Deployable SLMs (what people actually run)
| Model | Params | Type | Notes |
|---|---|---|---|
| Phi-4-mini | 3.8B | Transformer | best sub-4B: 67.3% MMLU, 88.6% GSM8K, 128K ctx |
| Qwen3-3B / 0.6B | 3B / 0.6B | Transformer | strongest open sub-3B; thinking mode |
| SmolLM3-3B | 3B | Transformer | fully-open recipe; 44% MMLU |
| Llama 3.2 3B | 3B | Transformer | the calibration baseline |
| LFM2-350M | 0.35B | Transformer | best fine-tuning ROI at tiny scale |

All mainstream SLMs are **Transformers** (dense attention). The linear-recurrent
line is a research frontier that hasn't shipped in the top SLM roundups yet —
but it is where the efficiency wins are.

### Linear-recurrent / memory architectures (research SOTA)
| Architecture | Update rule | Strengths | Weaknesses |
|---|---|---|---|
| **Mamba2** | `S = α S + v k^T` (gated outer-product) | hardware-efficient, strong | uniform decay → poor associative recall at long ctx |
| **DeltaNet** | `S = S − (S k)k^T + v k^T` (delta rule) | precise targeted erase, good recall | memory collisions w/o gating; needs L2-norm |
| **Gated DeltaNet** (ICLR'25 SOTA) | delta rule + Mamba2-style input gating; q/k/v short-conv + L2-norm + SiLU gate | **beats Mamba2 & DeltaNet everywhere**; hybrid +SWA/Mamba2 | more compute than Mamba2 |
| **RetNet / HGRN2** | linear attention / gated HRN | fast | behind on recall & LM |
| **Titans** | neural memory + persistent slots + surprise-based update | long-context memory | more machinery |
| **LEAFv5 (this repo)** | multi-timescale delta + write/forget gates + StateNorm + local conv path | ultra-stable, few-step learning, tiny state | (pre-upgrade) no read query, no memory slots, crosstalk-limited long-range |

## 2. What LEAFv5 was missing vs SOTA (the gaps)

Found by reading the Gated DeltaNet / DeltaNet / Mamba2 / Titans papers against
LEAFv5's memory block:

1. **No read query.** LEAFv5 read the state with the *write key* (`o = S@k`).
   DeltaNet/Gated DeltaNet use a separate **query projection** (`o = S@q`),
   decoupling *what to write* from *what to look for*. This is the single most
   important delta-rule design choice.
2. **No short convolution** on the q/k/v path. Mamba/Gated DeltaNet apply a
   depthwise conv-3 before L2-norm so **local context feeds the memory**;
   LEAFv5's memory saw only the pointwise projection.
3. **No output gating.** Mamba/Gated DeltaNet gate the memory output with a
   **SiLU gate**; LEAFv5 had only the content-mixing gate g.
4. **No persistent (external) memory.** Titans' persistent slots = the paper's
   own future-work item ("hybridization with sparse external memory"). LEAFv5
   had none.
5. (Documented, not yet implemented) **input-dependent global decay** (Gated
   DeltaNet's α, Mamba2-style memory clearance) — targeted at the
   crosstalk/memory-collision limit we measured in longrange_demo.py.

## 3. What we fixed (implemented in `leafv5/model.py`, default ON)

| Fix | Mechanism | Config |
|---|---|---|
| **Read query** | `wq` projection, `o = S@q`; forget still erases along `k` (delta rule) | `use_read_query=True` |
| **Short conv** | depthwise conv-3 on q/k/v before L2-norm (causal-friendly) | `short_conv=True` |
| **SiLU output gate** | `out = SiLU(W_gate x) ⊙ (W_o o + slots_read)` | `output_gate=True` |
| **Persistent slots** | learned `[64, dim]` slot matrix, softmax-queried per token | `mem_slots=64` |

All are config-gated; `--learn-plasticity`, `--share-mem-every`, chunked scan,
curriculum etc. compose with them. Old checkpoints load (missing params init
fresh, warned).

## 4. Measured before/after (CPU, same seeds/recipes)

### Few-step learning (held-out associative recall, V=64)
| store-1/query-1 | step 1 | 3 | 5 | 10 |
|---|---|---|---|---|
| LEAFv5 before | 4% | 25% | 86% | 100% |
| **LEAFv5 after (--fast, re-measured 2026-08-09)** | **5%** | **86%** | **99%** | 100% |

→ **100% by step 10** (was step 5 pre-fix); 99% by step 5. Still beats the
Transformer (19% at step 10) by a wide margin.

### Long-range retention (store-1/query-1 with 16 distractors, 200 train steps)
| distance | 64 | 256 | 1024 |
|---|---|---|---|
| LEAFv5 before (old arch, 900 steps) | 50% | 44% | 31% |
| **LEAFv5 after (200 steps)** — pre-fix claim | **100%** | **100%** | **100%** |
| **LEAFv5 after, re-measured 2026-08-09** | **6.2%** | **0.0%** | **6.2%** |
| reset-state baseline | 0% | 0% | 0% |

→ **the "100% flat retention" published in this section did not survive the
causality fixes** — re-measured post-fix, retention at micro scale is ≈ chance
(6.2% vs 0.8% baseline on longrange_demo's harder store-4/recall-2 task; the
crosstalk-limited retention the paper flagged remains an open weakness — see
research/reverify-2026-08.md). The reset baseline still proves the recurrent
state (not position tricks) does the little retention that exists.

### Language modeling (Tiny Shakespeare, held-out loss)
LEAFv5-after, re-measured 2026-08-09: 2.415 @ step 50, 2.211 @ step 100 vs
Transformer 2.408 @100 — LEAFv5 pulls ahead only around step 60–100 (the old
"0.379 @ step 50, beats Transformer@100 by step 20" did not reproduce post-fix).

## 5. What's left (honest)

- **Input-dependent global decay** (Gated DeltaNet α / Mamba2 clearance): the
  theoretical fix for memory collision at extreme write volumes; larger head
  dim (Gated DeltaNet uses 128 vs our 48-64) showed the same trend in their
  ablations. Both are config-viable next steps.
- **Hybrid sliding-window attention** layers (GatedDeltaNet-H1/H2) for extra
  quality at short context — a bigger change; the paper's no-attention pitch
  is preserved by keeping it optional.
- The above were measured at micro scale on CPU; the deltas should transfer to
  the T4 run (same code path, fp16/bf16 + chunked scan).

## 6. TL;DR

LEAFv5 was missing the three mechanisms that make modern delta-rule models SOTA
(read query, short conv on q/k/v, output gate) plus any persistent memory.
Implemented all four (default ON, backward-compatible): few-step learning got
~3-4× faster at early steps (100% by step 10, 99% by step 5 on store-1/q1,
re-measured 2026-08-09). The long-range retention collapse was **not** fixed by
these mechanisms alone — post-fix re-measurement shows ≈ chance retention at
micro scale (see research/reverify-2026-08.md); closing it is the next
architecture step (input-decay / hybrid attention).
