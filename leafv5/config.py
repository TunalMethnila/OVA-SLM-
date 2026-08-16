"""Configuration dataclasses and presets for LEAFv5 models."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    """LEAFv5 model hyperparameters (see paper section 3 and table of recommended configs)."""

    vocab_size: int = 16384
    # --- core dimensions ---
    dim: int = 768                # model width D
    n_layers: int = 14            # number of LEAFv5 blocks
    fast_heads: int = 4           # fast (highest-plasticity) memory heads
    medium_heads: int = 4         # medium memory heads
    slow_heads: int = 4           # slow (most-protected) memory heads
    d_h: int = 48                 # memory head state size: S in R^{d_h x d_h}
    ffn_expansion: float = 2.5    # compact SwiGLU FFN expansion (paper: 2.0-2.5x)
    # --- positional / tying ---
    max_seq_len: int = 4096       # RoPE cache length
    tie_weights: bool = True      # tie LM head to token embedding
    rope_base: float = 10000.0
    rope_dim: Optional[int] = None  # None -> rotate the full width
    # --- memory stability knobs (paper sec. 3.3) ---
    write_strength: Tuple[float, float, float] = (1.0, 0.6, 0.3)   # fast/med/slow
    forget_strength: Tuple[float, float, float] = (1.0, 0.8, 0.5)  # fast/med/slow
    alpha_init: float = 0.5       # residual readout scale from previous state
    scale_init: float = 0.0       # per-channel residual scale init.  0 = the
    #   paper's identity-start (max stability).  A small nonzero value (0.05-0.1)
    #   removes the step-1 gradient dead-zone and speeds early learning
    #   (see speed_demo.py / the --fast recipe) at a tiny stability cost.
    share_mem_every: int = 0      # >1: share the memory k/v/output projections
    #   across every N layers (paper sec. 5 note) -> fewer params
    state_norm: bool = True         # per-step StateNorm (soft spectral bound,
    #   ||S||_F <= sqrt(d_h)); the core training-stability guarantee.  OFF is
    #   for ablations only (unbounded states can drift).
    learn_plasticity: bool = False  # paper "future work": make the per-head
    #   write/forget multipliers TRAINABLE per layer (initialized to the
    #   fast/medium/slow group values).  Lets the model learn its own plasticity
    #   schedule (e.g. fast early layers, slow late layers).
    plasticity_prior: float = 0.0   # L2 regularization pulling the LEARNED
    #   write/forget multipliers back toward their fast/medium/slow group
    #   defaults (0 = off).  A "good prior" for learnable plasticity: the model
    #   may deviate from the timescale groups, but only when the data justifies
    #   it.  Used by train.py --plasticity-prior.
    surprise_gate: bool = False     # OPT-IN novelty-gated writes (Tier-1 fix for
    #   long-range retention / memory collisions): per-token, per-head write
    #   strength is scaled by 1 + w_h*(s_h - b_h), clamped to [0,2], where
    #   s_h = ||v_t - S@k_t||/sqrt(d_h) is the NORMALIZED novelty of the value
    #   being written vs what the current state already predicts.  High-surprise
    #   (novel) writes are boosted; redundant writes (the state already knows
    #   the value) are suppressed -> less clobbering of old memories.
    #   w_h inits to 0 (identity at init -> backward compatible), b_h to 0.5.
    #   Sequential-scan only (chunked falls back to sequential; C/Mojo twins
    #   updated in the same pass).
    dp_norm: bool = False           # OPT-IN Delta-Product-style NORMALIZED
    #   readout (Samsung 2025; the 2025 linear-recurrent SOTA response to
    #   crosstalk/scale-drift): alongside S, carry a denominator vector
    #   D in R^{d_h} obeying the SAME delta-rule recurrence with the value
    #   vector replaced by ones:
    #       D <- a*D - bf*(D^T k) k + bw*k
    #   and read  o = (S@q) / (D^T q + b_h)  (b_h per-head bias, init 1.0).
    #   The readout becomes a bounded weighted average (attention-like at
    #   linear cost) instead of an unbounded sum — directly attacking the
    #   documented crosstalk/scale-drift failure mode.  Sequential-scan only
    #   (chunked falls back to sequential).  MEASURED (2026-08-16): exact
    #   implementation, but neutral-to-slightly-worse at micro scale on
    #   recall/LM/extrapolation — the baseline is already scale-stable there.
    #   Opt-in pending the T4 scale run (see research/architecture-2026-08.md).
    # --- SOTA-inspired upgrades (default ON; see research/comparison.md) ---
    use_read_query: bool = True   # DeltaNet/Gated-DeltaNet: read with a SEPARATE
    #   query projection q (o = S@q) instead of the write key k.  Decouples
    #   writing from reading; standard in all modern delta-rule models.
    short_conv: bool = True       # Mamba/Gated-DeltaNet: 1-d depthwise conv
    #   (kernel 3) on the q/k/v projections before L2-norm -> local context
    #   feeds the memory.
    output_gate: bool = True      # Mamba-style SiLU output gate on the memory
    #   output (per-channel), improving expressiveness.
    mem_slots: int = 64           # Titans-style PERSISTENT memory slots
    #   (paper future work: "hybridization with sparse external memory"): a
    #   fixed learned matrix [mem_slots, dim] queried per token and added to
    #   the memory output.  Extra capacity at ~zero inference cost.
    # --- round-3 practical upgrades (research/improvements2.md) ---
    input_decay: bool = False     # Gated DeltaNet-style input-dependent global
    #   decay a_t in (0,1) on the state before each update:
    #   S <- a_t*S - bf*(S@k) k^T + bw*v k^T.  Gives the memory "clearance" so
    #   old associations fade when the input says so (theoretical fix for memory
    #   collision at extreme write volumes).  OPT-IN (--input-decay): measured
    #   neutral-to-slightly-worse at small scale; expected benefit only in
    #   long-context/memory-pressure regimes (Gated DeltaNet).  a starts ~0.99
    #   (bias-initialized) so behavior matches the paper's until learned.
    mem_dropout: float = 0.05     # dropout on the memory branch output
    #   (variational-style, like attention dropout): small-data regularization.
    stochastic_depth: float = 0.0  # 0 = off; >0 = per-block residual-drop
    #   probability during training (drop the whole branch with prob p, scale
    #   survivors by 1/(1-p)).  Makes deeper stacks easier to train.
    use_swa: bool = False         # OPT-IN hybrid (GatedDeltaNet-H1 style):
    #   add a causal sliding-window attention branch to blocks, with its own
    #   ZERO-INIT residual scale (identity at init -> exact growth, and the
    #   paper's no-attention default is preserved).  Best for short-context
    #   quality; the delta memory keeps the long-range/linear story.
    swa_every: int = 1            # interleave period for the SWA branch:
    #   1 = every block (GatedDeltaNet-H1), 2 = every other block (Jamba/Griffin
    #   style), k = every k-th block.  Index-based, so grow_depth extends the
    #   pattern exactly.  Measured at micro scale (2026-08-09): neutral on
    #   recall/LM within noise — plumbing for scale tests, not a default win.
    swa_window: int = 128         # sliding window size for the SWA branch
    swa_heads: int = 4
    # Mistral-style grouped-query attention for the SWA branch (arXiv
    # 2310.06825): swa_kv_heads=0 -> one KV head per query head (MHA, the
    # default, fully backward compatible); swa_kv_heads=k (k divides
    # swa_heads) -> k shared KV heads.  KV cache shrinks by swa_heads/k
    # (Mistral 7B uses 32 query / 8 KV = 4x smaller cache) at a small
    # representational cost; combined with the rolling buffer the decode KV
    # memory is constant in sequence length.
    swa_kv_heads: int = 0
    # --- world-class upgrades (research/world-class.md) ---
    moe: bool = False             # OPT-IN sparse Mixture-of-Experts FFN
    #   (Qwen3/DeepSeek-style): top-k routing over n_experts SwiGLU experts at
    #   ~the same FLOPs as the dense FFN -> far more params per FLOP.
    moe_experts: int = 8
    moe_topk: int = 2
    moe_aux_weight: float = 0.01  # load-balancing auxiliary loss weight
    slot_attn: bool = False       # OPT-IN Titans-style attention over the
    #   persistent memory slots (paper future-work "hybridization with sparse
    #   external memory"): learned query projection, slots as keys/values,
    #   zero-init output scale (identity at init -> growth-safe).

    @property
    def n_heads(self) -> int:
        return self.fast_heads + self.medium_heads + self.slow_heads

    @property
    def groups(self) -> Tuple[int, int, int]:
        return (self.fast_heads, self.medium_heads, self.slow_heads)

    @property
    def hidden_dim(self) -> int:
        """SwiGLU hidden width, rounded to a multiple of 64 for tensor-core friendliness."""
        return int(round(self.dim * self.ffn_expansion / 64.0)) * 64

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Presets.  "t4-4h" is sized to fit a 16 GB T4 comfortably with >4x headroom
# in VRAM while keeping wall-clock training inside a 4 hour budget at fp16.
# (Paper's ~150M recommendation, trimmed embedding by tying + a 16k BPE vocab.)
# ---------------------------------------------------------------------------
PRESETS: dict = {
    "t4-4h": dict(
        dim=768, n_layers=14,
        fast_heads=4, medium_heads=4, slow_heads=4,
        d_h=48, ffn_expansion=2.5,
    ),
    "t4-fast": dict(   # ~40M: max tokens in the 4h budget (data-limited regime)
        dim=512, n_layers=12,
        fast_heads=4, medium_heads=4, slow_heads=2,
        d_h=48, ffn_expansion=2.5,
    ),
    "t4-xl": dict(     # ~250M: the paper's ~350M-class config, trimmed for T4/4h
        dim=1024, n_layers=20,
        fast_heads=6, medium_heads=6, slow_heads=4,
        d_h=64, ffn_expansion=2.25,
    ),
    "tiny": dict(
        dim=512, n_layers=10,
        fast_heads=3, medium_heads=3, slow_heads=2,
        d_h=48, ffn_expansion=2.5,
    ),
    "micro": dict(
        dim=256, n_layers=8,
        fast_heads=2, medium_heads=2, slow_heads=2,
        d_h=32, ffn_expansion=2.5,
    ),
}


def preset_config(name: str, **overrides) -> ModelConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)} or pass --dim/--n-layers/etc.")
    base = dict(PRESETS[name])
    base.update(overrides)
    return ModelConfig(**base)


def param_estimate(cfg: ModelConfig) -> int:
    """Rough parameter count (embedding included, head tied)."""
    mem = 2 * cfg.dim * (cfg.n_heads * cfg.d_h) + cfg.dim * (cfg.n_heads * cfg.d_h)  # wk,wv,wo
    mem += 3 * cfg.dim * cfg.n_heads                                                    # write/forget/read gates
    local = cfg.dim * (3 + 5 + 9 + 15)
    ffn = 3 * cfg.dim * cfg.hidden_dim
    per_block = mem + local + ffn + 2 * cfg.dim  # norms + scales
    emb = cfg.vocab_size * cfg.dim
    return per_block * cfg.n_layers + emb
