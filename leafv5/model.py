"""LEAFv5 architecture (paper sec. 3).

Structure:  Token Embedding + RoPE  ->  N x LEAFv5 Block  ->  Final RMSNorm  ->  LM Head

LEAFv5 Block:
    x_n = RMSNorm(x)
    local = DWConv3 + DWConv5 + DWConv9 + DWConv15 (x_n)          # multi-scale local path
    mem   = MultiTimescaleDeltaV2(x_n)                             # delta memory
    g     = sigmoid(W_g x_n);  mixed = g*mem + (1-g)*local        # content-dependent mixing
    x     = x + s1 . mixed                                         # per-channel scale, init 0
    x     = x + s2 . SwiGLU(RMSNorm(x))                            # init 0

All ops are linear layers, elementwise ops and depthwise convs (paper sec. 5).
Memory states are always kept in fp32 for stability; the chunked parallel-scan
path implements the paper's "chunked formulation" implementation note (sec. 5).
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ---------------------------------------------------------------------------
# Explicit recurrent state (reviewer-recommended LayerState): every part of the
# computation that needs history is carried here, so full-sequence training
# forward == single-token decode EXACTLY.
# ---------------------------------------------------------------------------
class LeafStates:
    """Per-model recurrent state:
      delta : list[layer]  [B, H, d_h, d_h]        -- delta memory S
      local : list[layer]  list[conv] [B, D, k-1]  -- local-path conv history
      short : list[layer]  [B, H*d_h, 2]           -- memory short-conv history
      swa_kv: list[layer]  (k_hist, v_hist) or None-- SWA KV cache
      dp    : list[layer]  [B, H, d_h] or None     -- DP-norm denominator D
      offset: int                                  -- absolute position
    """

    __slots__ = ("delta", "local", "short", "swa_kv", "dp", "offset")

    def __init__(self, delta, local=None, short=None, swa_kv=None, offset=0,
                 dp=None):
        self.delta = list(delta)
        self.local = list(local) if local is not None else None
        self.short = list(short) if short is not None else None
        self.swa_kv = list(swa_kv) if swa_kv is not None else None
        self.dp = list(dp) if dp is not None else None
        self.offset = int(offset)

    # ---- backward-compat shims: old code treats states as a list of delta S ----
    def __len__(self):
        return len(self.delta)

    def __getitem__(self, i):
        return self.delta[i]

    def __iter__(self):
        return iter(self.delta)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """RMSNorm computed in fp32 for stability, output cast back to input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms * self.weight.to(torch.float32)).to(dtype)


def statenorm(S: torch.Tensor, d_h: int, eps: float = 1e-6) -> torch.Tensor:
    """StateNorm: scale each head's S in R^{d_h x d_h} so its Frobenius norm
    stays bounded at sqrt(d_h).  Soft spectral bounding -> the recurrent state
    can never blow up, which is the core training-stability guarantee of LEAFv5
    alongside L2-normalized keys/values and zero-init residual scales.
    """
    Sf = S.to(torch.float32)
    norm = Sf.norm(dim=(-1, -2), keepdim=True)
    scale = math.sqrt(d_h) / (norm + eps)
    return (Sf * scale).to(S.dtype)


# ---------------------------------------------------------------------------
# Rotary positional embeddings (applied to the input embedding, paper sec. 3.1)
# ---------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """RoPE applied to the first `dim` channels (dim < width rotates a subset;
    the unrotated channels keep the representation position-invariant so the
    delta memory stays content-addressable)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "rope_dim must be even"
        self.dim = dim
        self.base = base
        if dim == 0:  # no positional rotation (content-addressable memory mode)
            return
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [T, half]
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def _extend(self, need: int):
        """Dynamically extend the RoPE cache (P1: offset+T > max_seq_len used
        to crash; now it grows by doubling)."""
        cur = self.cos_cached.shape[0]
        if need <= cur:
            return
        new_len = max(need, cur * 2)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2,
                                                     dtype=torch.float32) / self.dim))
        t = torch.arange(cur, new_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = torch.cat([self.cos_cached, freqs.cos()])
        sin = torch.cat([self.sin_cached, freqs.sin()])
        self.cos_cached = cos
        self.sin_cached = sin

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        if self.dim == 0:
            return x
        B, T, D = x.shape
        rd = self.dim
        self._extend(offset + T)  # dynamic cache (P1: no more shape crashes)
        cos = self.cos_cached[offset: offset + T].to(x.dtype)
        sin = self.sin_cached[offset: offset + T].to(x.dtype)
        x_rot = x[..., :rd]
        x1 = x_rot[..., 0::2]
        x2 = x_rot[..., 1::2]
        rot = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
        if rd == D:
            return rot
        return torch.cat([rot, x[..., rd:]], dim=-1)


# ---------------------------------------------------------------------------
# Causal depthwise conv (P0 fix: the old convs used SYMMETRIC padding, so
# token t saw future tokens -- data leakage.  This one is left-padded only and
# carries explicit recurrent state so token-by-token decode == full-sequence
# training exactly.)
# ---------------------------------------------------------------------------
class CausalConv1d(nn.Module):
    """Left-padded (causal) depthwise conv with a carryable history state.

    full-sequence:  out = conv(cat(zeros(k-1), x))      (left-pad = causal)
    token-by-token: out = conv(cat(state, x_t)); state <- last k-1 values
    Both give identical outputs for the same input stream.
    """

    def __init__(self, dim: int, kernel: int, groups: int = 1, bias: bool = False,
                 std: float = 0.02):
        super().__init__()
        self.kernel = kernel
        # no padding in the conv itself; history is prepended explicitly
        self.conv = nn.Conv1d(dim, dim, kernel, groups=groups, bias=bias)
        nn.init.normal_(self.conv.weight, std=std)

    @property
    def weight(self) -> torch.Tensor:
        """Proxy so external code (grow.py, tests) can use .weight directly."""
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """x: [B, C, T]; state: [B, C, k-1] (or None -> zero-init history).
        Returns (out [B, C, T], new_state [B, C, k-1])."""
        k = self.kernel
        B, C, T = x.shape
        if k <= 1:
            return self.conv(x), x.new_zeros(B, C, 0)
        if state is None:
            state = x.new_zeros(B, C, k - 1)
        full = torch.cat([state, x], dim=-1)               # [B, C, (k-1)+T]
        out = self.conv(full)                              # valid -> [B, C, T]
        new_state = full[..., -(k - 1):].contiguous()
        return out, new_state


# ---------------------------------------------------------------------------
# Multi-scale depthwise local path  (paper sec. 3.2, item 2)
# ---------------------------------------------------------------------------
class MultiScaleLocalPath(nn.Module):
    """local = DWConv_3(x) + DWConv_5(x) + DWConv_9(x) + DWConv_15(x).
    All causal + stateful: full-seq forward == recurrent decode."""

    def __init__(self, dim: int, kernels=(3, 5, 9, 15), std: float = 0.02):
        super().__init__()
        self.convs = nn.ModuleList(
            [CausalConv1d(dim, k, groups=dim, std=std) for k in kernels])

    def forward(self, x: torch.Tensor,
                states: Optional[List[torch.Tensor]] = None):
        """x: [B, T, D]; states: list of [B, D, k-1] histories (or None).
        Returns (out [B, T, D], new_states list of [B, D, k-1])."""
        xt = x.transpose(1, 2)  # [B, D, T]
        out = None
        new_states: List[torch.Tensor] = []
        for i, c in enumerate(self.convs):
            st = states[i] if states is not None else None
            o, ns = c(xt, st)
            out = o if out is None else out + o
            new_states.append(ns)
        return out.transpose(1, 2), new_states


# ---------------------------------------------------------------------------
# Stabilized Multi-Timescale Delta Memory v2  (paper sec. 3.3, the core)
# ---------------------------------------------------------------------------
class MultiTimescaleDeltaV2(nn.Module):
    """Stabilized Multi-Timescale Delta Memory v2 (paper sec. 3.3), upgraded
    with the mechanisms that made DeltaNet / Gated DeltaNet / Mamba SOTA:

      * READ QUERY (DeltaNet/Gated-DeltaNet): o = S@q with its own projection
        W_q, decoupling reading from the write key k.
      * SHORT CONV (Mamba/Gated-DeltaNet): depthwise conv-3 on q/k/v before
        L2-norm, so local context feeds the memory.
      * SiLU OUTPUT GATE (Mamba): per-channel gate on the memory output.
      * PERSISTENT SLOTS (Titans; paper future work "hybridization with sparse
        external memory"): a fixed learned slot matrix queried per token.

    Core update (per head, per token), with L2-normalized q/k/v:
        k = L2Norm(W_k x)   v = L2Norm(W_v x)   q = L2Norm(W_q x)
        bw = sigmoid(W_write x) * w_mult    bf = sigmoid(W_forget x) * f_mult
        S <- S - bf*(S@k) k^T + bw*v k^T        # stabilized delta write/forget
        S <- StateNorm(S)                        # soft spectral bound
        o = g . (S@q) + alpha . (S_prev@q)       # residual readout (query-based)
        out = SiLU(W_gate x) . (W_o o + slots_read)

    Slow heads carry lower write/forget multipliers -> stronger protection
    against overwriting (paper sec. 3.3).  States are always fp32.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        D, H, dh = cfg.dim, cfg.n_heads, cfg.d_h
        self.d_h = dh
        self.n_heads = H
        hd = H * dh
        self.wk = nn.Linear(D, hd, bias=False)      # keys
        self.wv = nn.Linear(D, hd, bias=False)      # values
        self.wq = nn.Linear(D, hd, bias=False) if cfg.use_read_query else None
        self.w_write = nn.Linear(D, H, bias=False)  # per-head write gate
        self.w_forget = nn.Linear(D, H, bias=False) # per-head forget gate
        self.w_read = nn.Linear(D, H, bias=False)   # per-head readout gate g
        self.wo = nn.Linear(hd, D, bias=False)      # memory output projection
        self.alpha = nn.Parameter(torch.full((H,), cfg.alpha_init))
        # Gated DeltaNet-style input-dependent decay a_t in (0,1).  Bias 4.6 ->
        # sigmoid(4.6) ~ 0.99 at init (≈ paper behavior until learned).
        self.w_decay = (nn.Linear(D, H, bias=True) if cfg.input_decay else None)
        if self.w_decay is not None:
            nn.init.zeros_(self.w_decay.weight)
            nn.init.constant_(self.w_decay.bias, 4.6)
        # variational-style dropout on the memory branch output
        self.mem_dropout = nn.Dropout(cfg.mem_dropout) if cfg.mem_dropout > 0 else None
        # Mamba-style short conv on q/k/v (depthwise, per channel)
        self.short_conv = None
        if cfg.short_conv:
            # P0 fix: causal + stateful (was symmetric-padded -> data leakage
            # and train/decode mismatch)
            self.short_conv = CausalConv1d(hd, 3, groups=hd, std=0.02)
        # Mamba-style SiLU output gate (per-channel).
        # BUG FIX: silu(0) == 0, so a zero-init gate with NO bias multiplies the
        # whole memory branch to zero and can never learn (its gradient is 0
        # too) -- the delta memory was silently DEAD in default configs.  Init
        # weight=0, bias=1.27846 -> silu(bias) == 1.0 EXACTLY, so the gate is
        # the identity at init (preserves the identity-start design), the
        # memory branch contributes from step 1, and gradient flows.
        self.out_gate = nn.Linear(D, D, bias=True) if cfg.output_gate else None
        if self.out_gate is not None:
            nn.init.zeros_(self.out_gate.weight)
            nn.init.constant_(self.out_gate.bias, 1.278465)
        # Titans-style persistent memory slots (paper future-work: hybridization
        # with sparse external memory).  Simple readout by default; --slot-attn
        # upgrades it to a proper attention over the slots.
        self.slots = None
        self.slot_q = None
        self.slot_scale = None
        if cfg.mem_slots > 0:
            self.slots = nn.Parameter(torch.randn(cfg.mem_slots, D) * 0.02)
            if cfg.slot_attn:
                self.slot_q = nn.Linear(D, D, bias=False)
                nn.init.normal_(self.slot_q.weight, std=0.02)
                self.slot_scale = nn.Parameter(torch.zeros(D))  # identity at init

        # per-head multipliers derived from the plasticity groups.  Fixed
        # buffers by default (paper sec. 3.3); with cfg.learn_plasticity they
        # become TRAINABLE per-layer parameters ("learned per-layer plasticity
        # schedules" -- paper future-work list), initialized to the group values.
        gid = torch.cat([torch.full((n,), i) for i, n in enumerate(cfg.groups)]).long()
        base_w = torch.empty(H)
        base_f = torch.empty(H)
        for i, (w, f) in enumerate(zip(cfg.write_strength, cfg.forget_strength)):
            base_w[gid == i] = w
            base_f[gid == i] = f
        # keep the group defaults around for the plasticity prior (and for
        # gate_stats) whether or not the multipliers are trainable
        self.register_buffer("_base_w", base_w.clone(), persistent=False)
        self.register_buffer("_base_f", base_f.clone(), persistent=False)
        if cfg.learn_plasticity:
            self.write_mult = nn.Parameter(base_w.clone())
            self.forget_mult = nn.Parameter(base_f.clone())
        else:
            self.register_buffer("write_mult", base_w.clone(), persistent=False)
            self.register_buffer("forget_mult", base_f.clone(), persistent=False)
        # NOVELTY-GATED WRITES (Tier-1 retention fix, opt-in): per-head
        # w_h (init 0 -> identity) and b_h.  bw_eff = bw *
        # clamp(1 + w_h*(s - b_h), 0, 2), s = ||v - S@k||/sqrt(d_h).
        # b_h inits to 1/sqrt(d_h): a unit write against a ZERO state has
        # surprise exactly 1/sqrt(d_h), so first writes start neutral; writes
        # the state already predicts (s << b) are suppressed, genuinely novel
        # ones (s >> b) are boosted.  w_h=0 -> factor 1 (backward compatible).
        self.surprise_gate = cfg.surprise_gate
        self.surprise_w = None
        self.surprise_b = None
        if cfg.surprise_gate:
            self.surprise_w = nn.Parameter(torch.zeros(H))
            self.surprise_b = nn.Parameter(torch.full((H,), 1.0 / math.sqrt(dh)))
        # DP-normalized readout (Samsung Delta Product, 2025): a denominator
        # state D in R^{d_h} per head + a per-head bias for stability.
        # o = (S@q) / (D^T q + b_h).  b_h init 1.0 (safe: no writes -> 0/1).
        self.dp_norm = cfg.dp_norm
        self.d_bias = None
        if cfg.dp_norm:
            self.d_bias = nn.Parameter(torch.ones(H))
        self._init_weights()

    def _init_weights(self):
        for m in (self.wk, self.wv, self.wo):
            nn.init.normal_(m.weight, std=0.02)
        if self.wq is not None:
            nn.init.normal_(self.wq.weight, std=0.02)
        for m in (self.w_write, self.w_forget, self.w_read):
            nn.init.zeros_(m.weight)  # gates start at sigmoid(0) = 0.5

    def _proj(self, w, x, B, T, H, dh, short_state: Optional[torch.Tensor] = None):
        """Project x -> [B, T, H, dh], optionally short-conv (causal+stateful),
        then L2-norm.  Returns (out, new_short_state)."""
        hd = H * dh
        x = w(x).view(B, T, H, dh)
        ns = None
        if self.short_conv is not None:
            x, ns = self.short_conv(x.permute(0, 2, 3, 1).reshape(B, hd, T),
                                    short_state)
            x = x.reshape(B, H, dh, T).permute(0, 3, 1, 2).contiguous()
        return torch.nn.functional.normalize(x, dim=-1), ns

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None,
                short_state: Optional[torch.Tensor] = None,
                chunk: Optional[int] = None, state_norm: Optional[bool] = None,
                fast: bool = False, dp_state: Optional[torch.Tensor] = None):
        """x: [B, T, D] -> (out [B, T, D], new_delta [B,H,dh,dh],
        new_short_state [B, H*dh, 2] or None, new_dp [B,H,dh] or None).
        dp_state: DP-norm denominator D [B,H,dh] (or None -> zeros).
        state_norm defaults to cfg.state_norm (ablation-configurable).

        chunk=None  -> sequential scan (per-step StateNorm).
        chunk>0     -> chunked parallel-scan formulation (paper sec. 5): the
                       linear delta recurrence is composed with a Hillis-Steele
                       scan over per-token affine maps.  NOTE (P1): with
                       StateNorm ON this is a DIFFERENT recurrence from the
                       sequential scan (norm lands at chunk boundaries, not
                       every token) -- keep the same mode for train and eval.
        fast        -> use the validated C kernel (mojo/c_ref) for the scan in
                       eval/no-grad (fallback to Python if unavailable).
        """
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_h

        # q/k/v are SEPARATE conv streams sharing weights; each keeps its own
        # history (short_state: [3, B, H*dh, 2] or None for fresh zeros).
        proj_attrs = ["wk", "wv"] if self.wq is None else ["wk", "wv", "wq"]
        proj_out = {}
        new_short = []
        for idx, attr in enumerate(proj_attrs):
            st = short_state[idx] if short_state is not None else None
            p, ns = self._proj(getattr(self, attr), x, B, T, H, dh, st)
            proj_out[attr] = p
            new_short.append(ns)
        k, v = proj_out["wk"], proj_out["wv"]
        q = proj_out.get("wq", k)
        new_short_state = (torch.stack(new_short) if self.short_conv is not None
                           else None)
        bw = torch.sigmoid(self.w_write(x)).view(B, T, H, 1) * self.write_mult.view(1, 1, H, 1)
        bf = torch.sigmoid(self.w_forget(x)).view(B, T, H, 1) * self.forget_mult.view(1, 1, H, 1)
        gr = torch.sigmoid(self.w_read(x)).view(B, T, H, 1)
        # input-dependent global decay a_t in (0,1), per head (Gated DeltaNet)
        dec = None
        if self.w_decay is not None:
            dec = torch.sigmoid(self.w_decay(x)).view(B, T, H, 1)  # [B,T,H,1]

        if state is None:
            S = torch.zeros(B, H, dh, dh, device=x.device, dtype=torch.float32)
        else:
            S = state.to(torch.float32)

        # novelty-gated writes are sequential-only (the factor depends on the
        # live state); the chunked parallel scan falls back to sequential.
        if state_norm is None:
            state_norm = self.cfg.state_norm
        # DP-norm denominator state (None -> fresh zeros; only meaningful when
        # self.dp_norm; carried across token-by-token decode via LeafStates.dp)
        BH = B * H
        Df = None
        if self.dp_norm:
            Df = (dp_state.to(torch.float32).reshape(BH, dh)
                  if dp_state is not None
                  else S.new_zeros(BH, dh))
        use_chunked = chunk is not None and 1 < chunk <= T and T % chunk == 0
        if self.surprise_gate or self.dp_norm:
            use_chunked = False     # DP readout is sequential by nature
        if fast and not torch.is_grad_enabled() and not self.dp_norm:
            out, S = self._sequential_fast(k, v, q, bw, bf, gr, dec, S, state_norm)
        elif use_chunked:
            out, S = self._chunked_scan(k, v, q, bw, bf, gr, dec, S, chunk, state_norm)
        else:
            out, S, Df = self._sequential(k, v, q, bw, bf, gr, dec, S,
                                          state_norm, Df)
        if self.dp_norm:
            D_new = Df.reshape(B, H, dh)
        else:
            D_new = None
        # P0 reshape fix: [BH, dh, T] -> permute -> [BH, T, dh] ->
        # view(B,H,T,dh) -> permute -> [B, T, H, dh] -> [B, T, H*dh].
        # (The old .permute(0,2,1).reshape(B,T,H*dh) and view(B,H,T,dh) both
        # scrambled heads with positions -- later tokens leaked into earlier
        # outputs and head channels were permuted.)
        out = out.permute(0, 2, 1).view(B, H, T, dh).permute(0, 2, 1, 3) \
            .reshape(B, T, H * dh)
        out = self.wo(out)
        if self.mem_dropout is not None:
            out = self.mem_dropout(out)
        # persistent memory slots (Titans): softmax query over the slot matrix
        if self.slots is not None:
            if self.slot_q is not None:
                # attention over memory with a learned query + zero-init scale
                q = self.slot_q(x)
                xq = q.float() @ self.slots.float().t() * (self.cfg.mem_slots ** -0.5)
                attn = torch.softmax(xq, dim=-1).to(x.dtype)
                out = out + (self.slot_scale * (attn @ self.slots)).to(x.dtype)
            else:
                xq = x.float() @ self.slots.float().t() * (self.cfg.mem_slots ** -0.5)
                attn = torch.softmax(xq, dim=-1).to(x.dtype)
                out = out + attn @ self.slots
        # Mamba-style SiLU output gate
        if self.out_gate is not None:
            out = torch.nn.functional.silu(self.out_gate(x)) * out
        return out, S.reshape(B, H, dh, dh), new_short_state, D_new

    # -- per-token affine maps: S_t = S_{t-1} M_t + N_t -----------------------
    @staticmethod
    def _maps(kf, vf, bwf, bff, dh, decf=None):
        """kf/vf [BH, C, dh] (fp32), bwf/bff [BH, C, 1] -> M, N [BH, C, dh, dh].
        M = a*I - bf k k^T  (a = input decay, 1 when disabled)."""
        kk = torch.matmul(kf.unsqueeze(-1), kf.unsqueeze(-2))            # [BH,C,dh,dh]
        eye = torch.eye(dh, device=kf.device, dtype=torch.float32).view(1, 1, dh, dh)
        a = decf.unsqueeze(-1) if decf is not None else 1.0
        M = a * eye - bff.unsqueeze(-1) * kk
        N = bwf.unsqueeze(-1) * torch.matmul(vf.unsqueeze(-1), kf.unsqueeze(-2))
        return M, N, eye

    def _sequential(self, k, v, q, bw, bf, gr, dec, S, state_norm, Df=None):
        """Paper-exact sequential scan with per-step StateNorm (when enabled).

        Df: [BH, dh] DP-norm denominator (None -> DP readout off).  When on:
            D <- a*D - bf*(D^T k) k + bw*k      (value vector -> ones vector)
            o = (S@q) / (D^T q + b_h)           (b_h per-head bias, init 1)
        Returns (out [BH, dh, T], Sf, Df)."""
        B, T, H, dh = k.shape
        BH = B * H
        kf = k.permute(0, 2, 1, 3).reshape(BH, T, dh).float()
        vf = v.permute(0, 2, 1, 3).reshape(BH, T, dh).float()
        qf = q.permute(0, 2, 1, 3).reshape(BH, T, dh).float()
        bwf = bw.permute(0, 2, 1, 3).reshape(BH, T, 1).float()
        bff = bf.permute(0, 2, 1, 3).reshape(BH, T, 1).float()
        grf = gr.permute(0, 2, 1, 3).reshape(BH, T, 1).float()
        decf = dec.permute(0, 2, 1, 3).reshape(BH, T, 1).float() if dec is not None else None
        alpha = self.alpha.float().repeat(B).view(BH, 1, 1).contiguous()
        db = None
        if Df is not None:
            db = self.d_bias.float().repeat(B).view(BH, 1)       # [BH,1]
        Sf = S.reshape(BH, dh, dh)
        outs = []
        for t in range(T):
            kt1 = kf[:, t].unsqueeze(-1)   # [BH, dh, 1]
            kt2 = kf[:, t].unsqueeze(1)
            vt = vf[:, t].unsqueeze(-1)
            qt1 = qf[:, t].unsqueeze(-1)
            bwt = bwf[:, t:t + 1]          # [BH, 1, 1] (keep singleton dims)
            bft = bff[:, t:t + 1]
            grt = grf[:, t:t + 1]
            at = decf[:, t:t + 1] if decf is not None else 1.0
            ok = torch.bmm(Sf, kt1)                              # erase along k (delta rule)
            if Df is not None:
                # DP-normalized reads: denom = D^T q + b_h (clamped > 0)
                dq_prev = torch.bmm(Df.unsqueeze(1), qt1).squeeze(1)   # [BH,1]
                den_prev = torch.clamp(dq_prev + db, min=1e-3)
                o_prev = torch.bmm(Sf, qt1) / den_prev.unsqueeze(-1)
            else:
                o_prev = torch.bmm(Sf, qt1)                      # read PRE-update (query)
            if self.surprise_gate:
                # novelty-gated write (Tier-1): boost writes whose value the
                # state doesn't already predict; suppress redundant writes
                d = (vt - ok)                                    # [BH, dh, 1]
                s = d.norm(dim=1) / math.sqrt(dh)                # [BH, 1] in [0,2]
                w = self.surprise_w.float().repeat(B).view(BH, 1, 1)
                b = self.surprise_b.float().repeat(B).view(BH, 1, 1)
                fac = (1.0 + w * (s.unsqueeze(-1) - b)).clamp(0.0, 2.0)
                bwt = bwt * fac
            Sf = at * Sf - bft * torch.bmm(ok, kt2) + bwt * torch.bmm(vt, kt2)
            if Df is not None:
                # denominator follows the SAME recurrence with v -> ones-vector
                # (gates must be [BH,1] here: [BH,1,1] * [BH,dh] broadcasts to
                # [BH,BH,dh] — a real shape bug caught by the DP scan test)
                dk = torch.bmm(Df.unsqueeze(1), kt1).squeeze(1)  # [BH,1] = D^T k
                Df = at * Df - bft.squeeze(-1) * (dk * kt1.squeeze(-1)) \
                    + bwt.squeeze(-1) * kt1.squeeze(-1)
            if state_norm:
                Sf = statenorm(Sf, dh)
            if Df is not None:
                dq_new = torch.bmm(Df.unsqueeze(1), qt1).squeeze(1)
                den_new = torch.clamp(dq_new + db, min=1e-3)
                o_new = torch.bmm(Sf, qt1) / den_new.unsqueeze(-1)
            else:
                o_new = torch.bmm(Sf, qt1)                       # read POST-update
            outs.append((grt * o_new + alpha * o_prev).squeeze(-1))
        return torch.stack(outs, dim=-1), Sf, Df               # out, S, D

    def _sequential_fast(self, k, v, q, bw, bf, gr, dec, S, state_norm):
        """C-kernel scan (mojo/c_ref/leafv5_scan.so): exact twin of
        _sequential, ~250x faster on CPU decode.  Falls back to the Python
        scan if the kernel is not built."""
        B, T, H, dh = k.shape
        BH = B * H
        kf = k.permute(0, 2, 1, 3).reshape(BH, T, dh).float().contiguous()
        vf = v.permute(0, 2, 1, 3).reshape(BH, T, dh).float().contiguous()
        qf = q.permute(0, 2, 1, 3).reshape(BH, T, dh).float().contiguous()
        bwf = bw.permute(0, 2, 1, 3).reshape(BH, T).float().contiguous()
        bff = bf.permute(0, 2, 1, 3).reshape(BH, T).float().contiguous()
        grf = gr.permute(0, 2, 1, 3).reshape(BH, T).float().contiguous()
        decf = (dec.permute(0, 2, 1, 3).reshape(BH, T).float().contiguous()
                if dec is not None else None)
        alpha = self.alpha.detach().float().repeat(B).contiguous()
        S0 = S.reshape(BH, dh, dh).float().contiguous()
        try:
            from mojo.c_ref import scan_q, scan_q_s
            if self.surprise_gate:
                sw = self.surprise_w.detach().float().repeat(B).contiguous()
                sb = self.surprise_b.detach().float().repeat(B).contiguous()
                out, Snew = scan_q_s(qf, kf, vf, bwf, bff, grf, decf, alpha, S0,
                                     state_norm, sw, sb)
            else:
                out, Snew = scan_q(qf, kf, vf, bwf, bff, grf, decf, alpha, S0,
                                   state_norm)
        except Exception:
            # kernel unavailable -> exact Python fallback
            outp, Snew, _ = self._sequential(k, v, q, bw, bf, gr, dec, S,
                                               state_norm, None)
            return outp, Snew
        # [BH, T, dh] -> [B, T, H*dh] via [B,H,T,dh]
        out = out.view(B, H, T, dh).permute(0, 2, 1, 3).reshape(B, T, H * dh)
        return out.permute(0, 2, 1).reshape(BH, dh, T), Snew

    def _chunked_scan(self, k, v, q, bw, bf, gr, dec, S, chunk, state_norm):
        """Parallel-scan (Hillis-Steele) over the affine delta recurrence.

        Compose(A, B) = (M_A M_B, N_A M_B + N_B).  After the scan, op at t is
        the composition of ops 0..t, so S_t = S_in M_pref_t + N_pref_t and
        o_t = gr(S_t q_t) + alpha (S_{t-1} q_t)  (query-based reads).
        M_t = a_t I - bf_t k_t k_t^T with input-dependent decay a_t.
        """
        B, T, H, dh = k.shape
        BH = B * H
        kf = k.permute(0, 2, 1, 3).reshape(BH, T, dh).float()
        vf = v.permute(0, 2, 1, 3).reshape(BH, T, dh).float()
        qf = q.permute(0, 2, 1, 3).reshape(BH, T, dh).float()
        bwf = bw.permute(0, 2, 1, 3).reshape(BH, T, 1).float()
        bff = bf.permute(0, 2, 1, 3).reshape(BH, T, 1).float()
        grf = gr.permute(0, 2, 1, 3).reshape(BH, T, 1).float()
        decf = dec.permute(0, 2, 1, 3).reshape(BH, T, 1).float() if dec is not None else None
        alpha = self.alpha.float().repeat(B).view(BH, 1, 1).contiguous()
        Sf = S.reshape(BH, 1, dh, dh)
        outs = []
        for c in range(0, T, chunk):
            C = min(chunk, T - c)
            M, N, eye = self._maps(kf[:, c:c + C], vf[:, c:c + C],
                                   bwf[:, c:c + C], bff[:, c:c + C], dh,
                                   decf[:, c:c + C] if decf is not None else None)
            # Hillis-Steele prefix scan: compose(pad_op, cur_op)
            step = 1
            while step < C:
                Mp = torch.cat([eye.expand(BH, step, dh, dh), M[:, :-step]], dim=1)
                Np = torch.cat([N.new_zeros(BH, step, dh, dh), N[:, :-step]], dim=1)
                M_new = torch.matmul(Mp, M)        # (M_pad)(M_cur)
                N_new = torch.matmul(Np, M) + N    # (N_pad)(M_cur) + N_cur
                M, N = M_new, N_new
                step *= 2
            # S_t for every t in the chunk, then query-based reads
            S_all = torch.matmul(Sf, M) + N                        # [BH,C,dh,dh]
            o_new = torch.matmul(S_all, qf[:, c:c + C].unsqueeze(-1)).squeeze(-1)
            M_prev = torch.cat([eye.expand(BH, 1, dh, dh), M[:, :-1]], dim=1)
            N_prev = torch.cat([N.new_zeros(BH, 1, dh, dh), N[:, :-1]], dim=1)
            S_prev_all = torch.matmul(Sf, M_prev) + N_prev
            o_prev = torch.matmul(S_prev_all, qf[:, c:c + C].unsqueeze(-1)).squeeze(-1)
            outs.append(grf[:, c:c + C] * o_new + alpha * o_prev)  # [BH,C,dh]
            # chunk-final state (StateNorm at the boundary, per the chunked formulation)
            Sf = torch.matmul(Sf, M[:, -1:]) + N[:, -1:]
            if state_norm:
                Sf = statenorm(Sf, dh)
        return torch.cat(outs, dim=1).permute(0, 2, 1), Sf.reshape(BH, dh, dh)


# ---------------------------------------------------------------------------
# Sliding-window attention (OPT-IN hybrid, GatedDeltaNet-H1 style).
# Mistral-inspired efficiency stack (arXiv 2310.06825): sliding window +
# grouped-query attention + rolling-buffer KV cache + pre-fill & chunking.
# ---------------------------------------------------------------------------
class RollingKVCache:
    """Mistral's rolling-buffer KV cache: FIXED [B, Hkv, W, dh] storage where
    the key/value for absolute position i is written to slot (i mod W).  After
    W tokens the buffer stops growing — decode KV memory is constant in
    sequence length, and the per-step allocation of a naive cat/truncate cache
    is avoided.  `window()` returns the most recent W keys/values in
    chronological order (the slice-and-cat trick), so attention is identical
    to the non-rolling cache up to float rounding.

    Inference-only (no autograd): used by the token-by-token decode path and
    by `SlidingWindowAttention.prefill` (pre-fill & chunking).
    """

    def __init__(self, k, v, pos=0, window=None):
        # k, v: [B, Hkv, T, dh] initial contents at absolute positions
        # [pos, pos+T).  Storage is preallocated to exactly `window` slots
        # (default: T), so the buffer never grows past the window.
        B, H, T, dh = k.shape
        self.W = window if (window and window > 0) else T
        assert T <= self.W, "initial fill larger than the window"
        self.start = pos                 # first valid absolute position
        self.pos = pos
        self.k = k.new_zeros(B, H, self.W, dh)
        self.v = v.new_zeros(B, H, self.W, dh)
        if T:
            slots = torch.arange(pos, pos + T, device=k.device) % self.W
            self.k.index_copy_(2, slots, k)
            self.v.index_copy_(2, slots, v)
        self.pos = pos + T

    @property
    def shape(self):
        return self.k.shape

    def append(self, k, v):
        """Append T new tokens (T <= W); writes them at (pos+j) mod W."""
        B, H, T, dh = k.shape
        assert T <= self.W, "append larger than the window"
        slots = torch.arange(self.pos, self.pos + T, device=k.device) % self.W
        self.k.index_copy_(2, slots, k)
        self.v.index_copy_(2, slots, v)
        self.pos += T

    def window(self):
        """Most recent min(pos-start, W) keys/values in chronological order:
        [B, Hkv, m, dh].  Correct for ANY start position (not just 0):
        before the buffer fills, the valid tokens occupy the circular range
        [start % W, start % W + n_valid); after it fills, all W slots are
        valid and the oldest is at pos % W.  (Stability fix 2026-08-09: the
        previous version assumed start=0 and returned unwritten zero slots
        as history when prefilled at a nonzero offset.)"""
        n_valid = self.pos - self.start
        if n_valid <= 0:
            return self.k[:, :, :0].contiguous(), self.v[:, :, :0].contiguous()
        if n_valid >= self.W:
            p = self.pos % self.W
            k = torch.cat([self.k[:, :, p:], self.k[:, :, :p]], dim=2)
            v = torch.cat([self.v[:, :, p:], self.v[:, :, :p]], dim=2)
            return k.contiguous(), v.contiguous()
        s = self.start % self.W
        e = s + n_valid
        if e <= self.W:
            return self.k[:, :, s:e].contiguous(), self.v[:, :, s:e].contiguous()
        k = torch.cat([self.k[:, :, s:], self.k[:, :, :e - self.W]], dim=2)
        v = torch.cat([self.v[:, :, s:], self.v[:, :, :e - self.W]], dim=2)
        return k.contiguous(), v.contiguous()


class SlidingWindowAttention(nn.Module):
    """Causal attention over a local window, with its own ZERO-INIT residual
    scale (identity at init).  Adds local exact-token-mixing capacity on top of
    the delta memory; the paper's no-attention default is preserved (off).

    Mistral-inspired additions (2026-08-09):
      * grouped-query attention (kv_heads): KV cache shrinks by heads/kv_heads
        (0 -> MHA, fully backward compatible);
      * rolling-buffer KV cache (RollingKVCache): constant decode memory;
      * `prefill()`: Mistral pre-fill & chunking — long prompts are processed
        in chunks of <= W with the rolling cache, bounding pre-fill memory to
        the same level as generation.
    """

    def __init__(self, dim: int, heads: int, window: int, kv_heads: int = 0):
        super().__init__()
        self.heads = heads
        self.dh = dim // heads
        self.kv_heads = kv_heads if (kv_heads and kv_heads > 0) else heads
        assert self.heads % self.kv_heads == 0, \
            f"swa_heads ({self.heads}) must be divisible by swa_kv_heads " \
            f"({self.kv_heads})"
        self.groups = self.heads // self.kv_heads
        self.window = window
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, self.kv_heads * self.dh, bias=False)
        self.wv = nn.Linear(dim, self.kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        for w in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(w.weight, std=0.02)
        self.scale = nn.Parameter(torch.zeros(dim))  # zero -> identity at init

    def _expand_kv(self, k, v):
        """[B, Hkv, T, dh] -> [B, H, T, dh]; each KV head serves a contiguous
        group of `groups` query heads (standard GQA grouping)."""
        if self.kv_heads == self.heads:
            return k, v
        return (k.repeat_interleave(self.groups, dim=1),
                v.repeat_interleave(self.groups, dim=1))

    def _proj(self, x: torch.Tensor):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.heads, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv_heads, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv_heads, self.dh).transpose(1, 2)
        return q, k, v

    def forward(self, x: torch.Tensor, kv_cache=None, pos: int = None):
        """x: [B, T, D].  kv_cache:
          * None        -> full-sequence path (windowed causal mask);
          * (k, v)      -> decode path, tuple cache of the most recent W tokens
                           (backward-compatible cat/truncate);
          * RollingKVCache -> decode path, Mistral rolling buffer.
        Returns (out, new_cache)."""
        B, T, D = x.shape
        W = self.window
        q, k_raw, v_raw = self._proj(x)    # k_raw/v_raw are [B, Hkv, T, dh]
        if kv_cache is not None:
            # decode path: cache holds only keys <= current position -> NO mask.
            # The cache keeps Hkv heads (unexpanded); expansion happens only
            # for the attention itself, so the cache width never grows.
            if isinstance(kv_cache, RollingKVCache):
                cache = kv_cache
                cache.append(k_raw, v_raw)
                kw, vw = self._expand_kv(*cache.window())
                new_cache = cache
            else:
                kh, vh = kv_cache
                kcat = torch.cat([kh, k_raw], dim=2)[:, :, -W:]
                vcat = torch.cat([vh, v_raw], dim=2)[:, :, -W:]
                kw, vw = self._expand_kv(kcat, vcat)
                new_cache = (kcat.contiguous(), vcat.contiguous())
            scores = torch.matmul(q, kw.transpose(-1, -2)) / (self.dh ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, vw)
        else:
            # full-sequence path: causal + window band over the T x T scores
            k, v = self._expand_kv(k_raw, v_raw)
            scores = torch.matmul(q, k.transpose(-1, -2)) / (self.dh ** 0.5)
            mask = torch.full((T, T), float("-inf"), device=x.device,
                              dtype=scores.dtype)
            mask = torch.triu(mask, diagonal=1)
            for i in range(T):
                lo = i - W + 1
                if lo > 0:
                    mask[i, :lo] = float("-inf")
            attn = torch.softmax(scores + mask, dim=-1)
            out = torch.matmul(attn, v)
            # cache stores Hkv heads (unexpanded) so the next decode step's
            # cat matches the freshly-projected keys' width
            new_cache = (k_raw[:, :, -W:].contiguous(),
                         v_raw[:, :, -W:].contiguous())
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.wo(out) * self.scale, new_cache

    def prefill(self, x: torch.Tensor, pos: int = 0, chunk: int = None):
        """Mistral pre-fill & chunking: process a prompt of ANY length in
        chunks of `chunk` (default: the window W) tokens, rolling the KV
        buffer between chunks.  Bounds pre-fill memory to O(W) regardless of
        prompt length and returns a RollingKVCache ready for decode.

        Chunked pre-fill is EXACTLY equivalent to one-shot pre-fill
        (same keys/values in the same slots) up to float rounding."""
        B, T, D = x.shape
        chunk = chunk or self.window
        assert chunk <= self.window, "prefill chunk > window defeats the purpose"
        cache = None
        for s in range(0, T, chunk):
            xc = x[:, s:s + chunk]
            qc, kc, vc = self._proj(xc)
            if cache is None:
                cache = RollingKVCache(kc, vc, pos=pos + s, window=self.window)
            else:
                cache.append(kc, vc)
        return cache

    def kv_bytes(self, batch: int, dtype=torch.float32):
        """KV cache bytes for one layer at `batch` (both K and V)."""
        return (2 * batch * self.kv_heads * self.window * self.dh *
                torch.tensor([], dtype=dtype).element_size())


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # up
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)  # gate
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)  # down
        nn.init.normal_(self.w3.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w2(x)) * self.w1(x))


# ---------------------------------------------------------------------------
# Sparse Mixture-of-Experts FFN (OPT-IN; Qwen3/DeepSeek-style param scaling)
# ---------------------------------------------------------------------------
class MoEFFN(nn.Module):
    """Top-k MoE over `n_experts` SwiGLU experts, each full-size (hidden_dim).
    ~same FLOPs as the dense FFN but n_experts x the params -> much more
    capacity per FLOP.  Standard load-balancing aux loss exposed via
    `aux_loss()` (added by the trainer with --moe-aux-weight).  Router is
    small-init so the residual identity start (zero s2) is preserved."""

    def __init__(self, dim: int, hidden_dim: int, n_experts: int, top_k: int,
                 bias: bool = False):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.experts = nn.ModuleList(
            [SwiGLUFFN(dim, hidden_dim, bias=bias) for _ in range(n_experts)])
        self._aux = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        logits = self.router(flat)                       # [N, n_experts]
        top_logits, top_idx = torch.topk(logits, self.top_k, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        # load-balancing aux loss: n_e * sum(f_i * p_i)
        f = torch.zeros(self.n_experts, device=x.device)
        f.scatter_add_(0, top_idx.reshape(-1),
                       torch.ones(top_idx.numel(), device=x.device) / top_idx.numel())
        p = probs.mean(0)
        self._aux = self.n_experts * (f * p).sum()
        # gather expert outputs
        weights = torch.softmax(top_logits.float(), dim=-1).to(x.dtype)  # [N, k]
        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            idx = top_idx[:, k]                          # [N]
            w = weights[:, k:k + 1]
            for e in range(self.n_experts):
                mask = idx == e
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](flat[mask])
        return out.view(B, T, D)

    def aux_loss(self) -> torch.Tensor:
        return self._aux


# ---------------------------------------------------------------------------
# LEAFv5 block (paper sec. 3.2)
# ---------------------------------------------------------------------------
class LeafBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, use_swa: Optional[bool] = None):
        super().__init__()
        self.cfg = cfg
        self.norm1 = RMSNorm(cfg.dim)
        self.local_path = MultiScaleLocalPath(cfg.dim)
        self.memory = MultiTimescaleDeltaV2(cfg)
        self.mix_gate = nn.Linear(cfg.dim, cfg.dim, bias=False)  # content-dependent mixing g
        nn.init.zeros_(self.mix_gate.weight)
        # per-channel residual scales, initialized to ZERO by default
        # (paper sec. 3.2, 4: identity-start highways).  --fast / scale_init>0
        # uses a small nonzero init for faster early learning.
        self.s1 = nn.Parameter(torch.full((cfg.dim,), cfg.scale_init))
        self.norm2 = RMSNorm(cfg.dim)
        if cfg.moe:
            self.ffn = MoEFFN(cfg.dim, cfg.hidden_dim, cfg.moe_experts,
                              cfg.moe_topk)
        else:
            self.ffn = SwiGLUFFN(cfg.dim, cfg.hidden_dim)
        self.s2 = nn.Parameter(torch.full((cfg.dim,), cfg.scale_init))
        # OPT-IN sliding-window attention branch (own zero-init scale -> identity)
        # use_swa=None -> follow cfg.use_swa; LeafLM passes the index-based
        # (cfg.swa_every) decision so hybrid interleave works per block.
        self.swa = None
        if (cfg.use_swa if use_swa is None else use_swa):
            self.swa = SlidingWindowAttention(cfg.dim, cfg.swa_heads,
                                              cfg.swa_window,
                                              kv_heads=cfg.swa_kv_heads)

    def forward(self, x: torch.Tensor, st: Optional[tuple] = None,
                chunk: Optional[int] = None, fast: bool = False):
        """st: (delta_state, local_states, short_state, swa_kv) or None.
        Returns (x, new_st: same tuple shape, or None if st was None)."""
        # stochastic depth: with prob p drop each residual branch during training
        # (scale survivors by 1/(1-p)); at eval everything runs.
        p = self.cfg.stochastic_depth
        sd1 = sd2 = False
        scale1 = scale2 = 1.0
        if p > 0 and self.training:
            sd1 = torch.rand(1).item() < p
            sd2 = torch.rand(1).item() < p
            s = 1.0 / (1.0 - p)
            scale1 = scale2 = s
        if st is None:
            d_st = l_st = sh_st = swa = dp_st = None
        else:
            # back-compat: a 4-tuple (old state format) has no dp slot
            d_st, l_st, sh_st, swa = st[:4]
            dp_st = st[4] if len(st) > 4 else None
        xn = self.norm1(x)
        local, new_local = self.local_path(xn, l_st)
        mem, d_st, sh_st, dp_st = self.memory(xn, d_st, sh_st, chunk=chunk,
                                              fast=fast, dp_state=dp_st)
        g = torch.sigmoid(self.mix_gate(xn))
        mixed = g * mem + (1.0 - g) * local
        # per-channel residual scales are fp32 params; keep the stream in x.dtype
        # (otherwise the residual would silently upcast fp16 training to fp32)
        if not sd1:
            x = x + (scale1 * self.s1 * mixed).to(x.dtype)
        # OPT-IN sliding-window attention branch (own zero-init scale -> identity)
        new_swa = None
        if self.swa is not None:
            swa_out, new_swa = self.swa(x, swa)
            x = x + swa_out.to(x.dtype)
        h = self.ffn(self.norm2(x))
        if not sd2:
            x = x + (scale2 * self.s2 * h).to(x.dtype)
        if st is None:
            return x, None
        return x, (d_st, new_local, sh_st, new_swa, dp_st)


# ---------------------------------------------------------------------------
# Full LM
# ---------------------------------------------------------------------------
class LeafLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        # rope_dim None/0 -> no positional rotation (content-addressable memory)
        self.rope = RotaryEmbedding(cfg.rope_dim or 0, cfg.max_seq_len, cfg.rope_base)
        # hybrid interleave: block i gets the SWA branch when
        # use_swa and (i % swa_every == 0) — Jamba/Griffin-style periodic
        # attention, exact under grow_depth (index-based).
        self.blocks = nn.ModuleList([
            LeafBlock(cfg, use_swa=(cfg.use_swa and (i % cfg.swa_every == 0)))
            for i in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self._apply_shared_projections()

    def _apply_shared_projections(self):
        """Paper sec. 5 implementation note: "slow-path projections may be
        shared every 2 layers".  When cfg.share_mem_every > 1, every
        share_mem_every-th block shares its memory k/v/output projections with
        the previous block (fewer params, small quality cost)."""
        ev = self.cfg.share_mem_every
        if ev and ev > 1:
            n = 0
            for i in range(ev, self.cfg.n_layers, ev):
                for attr in ("wk", "wv", "wo"):
                    setattr(self.blocks[i].memory, attr,
                            self.blocks[i - 1].memory.__getattr__(attr))
                n += 1
            print(f"[model] shared memory projections across {n} layer pairs "
                  f"(share_mem_every={ev})")

    def forward(self, idx: torch.Tensor,
                states: Optional["LeafStates"] = None,
                offset: int = 0, chunk: Optional[int] = None,
                grad_checkpoint: bool = False, fast: bool = False):
        """idx: [B, T] long.  Returns (logits [B, T, V], new_states or None).

        states: a LeafStates (delta + local-conv history + short-conv history
        + SWA KV + offset).  When None (training), every stateful component
        uses fresh zero history -> STRICTLY CAUSAL, and this forward is
        EXACTLY reproducible by token-by-token decode with carried LeafStates
        (the central train==decode invariant).
        offset: absolute position of the first token (also carried in states).
        fast: use the validated C scan kernel in eval/no-grad."""
        if isinstance(states, (list, tuple)):
            # back-compat: a plain list of delta states is treated as delta-only
            # (conv/SWA histories start fresh)
            states = LeafStates(states, None, None, None, 0)
        if states is not None and offset == 0:
            offset = states.offset  # explicit offset=0 means "use carried"
        x = self.tok_emb(idx)
        x = self.rope(x, offset=offset)
        carry = states is not None
        new_delta = [] if carry else None
        new_local = [] if carry else None
        new_short = [] if carry else None
        new_swa = [] if carry else None
        new_dp = [] if carry else None
        for i, blk in enumerate(self.blocks):
            if carry:
                blk_st = (
                    states.delta[i],
                    states.local[i] if states.local is not None else None,
                    states.short[i] if states.short is not None else None,
                    states.swa_kv[i] if states.swa_kv is not None else None,
                    states.dp[i] if states.dp is not None else None,
                )
            else:
                blk_st = None
            if grad_checkpoint:
                x, ns = torch.utils.checkpoint.checkpoint(
                    blk, x, blk_st, chunk, use_reentrant=False)
            else:
                x, ns = blk(x, blk_st, chunk=chunk, fast=fast)
            if carry:
                new_delta.append(ns[0])
                new_local.append(ns[1])
                new_short.append(ns[2])
                new_swa.append(ns[3])
                new_dp.append(ns[4])
        x = self.norm_f(x)
        logits = self.head(x)
        new_states = (LeafStates(new_delta, new_local, new_short, new_swa,
                                 offset + idx.shape[1], dp=new_dp)
                      if carry else None)
        return logits, new_states

    def init_states(self, batch: int, device) -> "LeafStates":
        """Fresh recurrent state: zero delta memory, zero conv histories,
        empty SWA cache, offset 0."""
        L, H, dh, D = self.cfg.n_layers, self.cfg.n_heads, self.cfg.d_h, self.cfg.dim
        delta = [
            torch.zeros(batch, H, dh, dh, device=device, dtype=torch.float32)
            for _ in range(L)
        ]
        local = [
            [torch.zeros(batch, D, c.kernel - 1, device=device,
                         dtype=torch.float32) for c in blk.local_path.convs]
            for blk in self.blocks
        ]
        short = [
            torch.zeros(3, batch, H * dh, 2, device=device,
                        dtype=torch.float32)
            if blk.memory.short_conv is not None else None
            for blk in self.blocks
        ]
        swa_kv = [None for _ in range(L)]
        dp = ([torch.zeros(batch, H, dh, device=device, dtype=torch.float32)
               for _ in range(L)] if self.cfg.dp_norm else None)
        return LeafStates(delta, local, short, swa_kv, 0, dp=dp)

    @torch.no_grad()
    def gate_stats(self, x: torch.Tensor, states: Optional[List[torch.Tensor]] = None):
        """Head-specialization probe: per-head mean write/forget/read gate
        activity and final state norms over a batch, aggregated per plasticity
        group (fast/medium/slow).  Validates the paper's multi-timescale design
        and can guide plasticity tuning.  Returns dict[group -> dict[str,float]].
        """
        self.eval()
        B, T = x.shape
        h = self.tok_emb(x)
        h = self.rope(h)
        cat = {"bw": [], "bf": [], "gr": [], "fn": []}
        for i, blk in enumerate(self.blocks):
            xn = blk.norm1(h)
            mem = blk.memory
            H = mem.n_heads
            bw = torch.sigmoid(mem.w_write(xn)).view(B, T, H) * mem.write_mult
            bf = torch.sigmoid(mem.w_forget(xn)).view(B, T, H) * mem.forget_mult
            gr = torch.sigmoid(mem.w_read(xn)).view(B, T, H)
            cat["bw"].append(bw.mean(dim=(0, 1)))   # [H]
            cat["bf"].append(bf.mean(dim=(0, 1)))
            cat["gr"].append(gr.mean(dim=(0, 1)))
            st = states[i] if states is not None else None
            h, ns = blk(h, st, chunk=None)
            cat["fn"].append(ns.norm(dim=(-1, -2)).mean(dim=0))  # [H] mean over B
        cat = {k: torch.stack(v) for k, v in cat.items()}  # [L, H]
        gid = torch.cat([torch.full((n,), g) for g, n in enumerate(self.cfg.groups)]).long()
        names = ["fast", "medium", "slow"]
        out = {}
        for g in range(len(names)):
            m = gid == g
            out[names[g]] = {k: float(cat[k][:, m].mean()) for k in cat}
        return out

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def aux_loss(self) -> torch.Tensor:
        """Sum of MoE load-balancing aux losses (0 when MoE is off)."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for blk in self.blocks:
            if isinstance(blk.ffn, MoEFFN):
                total = total + blk.ffn.aux_loss()
        return total

    def plasticity_prior_loss(self, lam: float) -> torch.Tensor:
        """L2 prior pulling LEARNED write/forget multipliers back toward their
        fast/medium/slow group defaults (lam=0 -> 0).  Tier-1: gives the model
        the freedom to discover better timescales, but only when the data
        justifies deviating from the paper's groups."""
        if not lam or not self.cfg.learn_plasticity:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        tot = torch.tensor(0.0, device=next(self.parameters()).device)
        for blk in self.blocks:
            mem = blk.memory
            if isinstance(mem.write_mult, nn.Parameter):
                dw = mem.write_mult - mem._base_w
                df = mem.forget_mult - mem._base_f
                tot = tot + (dw * dw).sum() + (df * df).sum()
        return lam * tot
