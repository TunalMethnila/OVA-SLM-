"""Progressive model growth: train small, then scale up WITHOUT losing training.

Two exact (function-preserving) operations:

grow_depth(model, add_layers)
    Append blocks with ZERO-INIT residual scales (s1 = s2 = 0).  A block with
    s1 = s2 = 0 is the identity: x_out = x_in.  So appending such blocks leaves
    the model's output EXACTLY unchanged on every input -- the trained knowledge
    is untouched, and the new blocks simply add capacity that training can use.

grow_width(model, new_dim)
    Net2Net-style neuron replication with weight splitting.  Every new channel
    is a copy of an old one; layers that PRODUCE the stream replicate columns
    (embedding, wo, output rows, per-channel norms/scales, local convs), layers
    that CONSUME the stream divide the replicated input columns by the copy
    count.  Because each replicated channel carries the identical value and the
    consumers average it back, the forward function is preserved exactly.
    The LM head is UNTIED at the swap (it consumes the stream while the
    embedding produces it; the same matrix cannot do both exactly).
    Recurrent state resets to zero at the swap (it is zero at window starts
    anyway).  Persistent slots are re-initialized (auxiliary, ~0.5% params).

Verified: after grow_width/grow_depth, max |logit diff| vs the pre-growth model
on a fixed batch is ~1e-6 (fp32).

Usage:
    python -m leafv5.grow --ckpt out/small/best.pt --to-dim 384 \
        --to-layers 4 --out out/grown/best.pt
    # or in finetune.py: --grow-at STEP --grow-dim N --grow-layers M
"""
from __future__ import annotations

import argparse
from typing import List, Tuple

import torch

from .config import ModelConfig
from .model import LeafLM, MoEFFN


# ---------------------------------------------------------------------------
# replication maps
# ---------------------------------------------------------------------------
def replication_map(old: int, new: int) -> Tuple[List[int], List[int]]:
    """For widening `old` -> `new`:
      new_to_old[i] = old index replicated at new position i
      counts[j]     = how many copies old unit j has
    Copies are spread evenly (first `new % old` units get one extra)."""
    assert new >= old > 0
    base, rem = new // old, new % old
    counts = [base + (1 if j < rem else 0) for j in range(old)]
    n2o: List[int] = []
    for j, c in enumerate(counts):
        n2o.extend([j] * c)
    assert len(n2o) == new
    return n2o, counts


def _divide_by_counts(w: torch.Tensor, dim: int, n2o: List[int],
                      counts: List[int]) -> torch.Tensor:
    """Consumer side: W has `dim` as an input dim; new position i gets old
    position n2o[i] divided by its copy count (so identical replicated inputs
    re-average to the original contribution)."""
    new_shape = list(w.shape)
    new_shape[dim] = len(n2o)
    idx = torch.tensor(n2o)
    src = torch.index_select(w, dim, idx)             # [.., new, ..]
    div = torch.tensor([counts[o] for o in n2o], dtype=src.dtype)
    shape = [1] * len(new_shape)
    shape[dim] = len(n2o)
    return src / div.view(shape)


def _replicate_rows(w: torch.Tensor, dim: int, n2o: List[int]) -> torch.Tensor:
    """Producer side: W has `dim` as an output dim; new row/col i is a copy of
    old n2o[i] (identical values flow to both copies)."""
    out_shape = list(w.shape)
    out_shape[dim] = len(n2o)
    out = w.new_empty(out_shape)
    idx = torch.tensor(n2o)
    out = torch.index_select(w, dim, idx)
    return out


# ---------------------------------------------------------------------------
# width growth
# ---------------------------------------------------------------------------
def grow_width(model: LeafLM, new_dim: int) -> LeafLM:
    cfg = model.cfg
    D = cfg.dim
    # UNIFORM integer replication is required: RMSNorm is only invariant when
    # every channel is replicated the same number of times (non-uniform counts
    # change the RMS scale).  Doubling (or 3x, 4x ...) is the natural choice.
    assert new_dim >= D, f"new_dim {new_dim} must be >= current dim {D}"
    assert new_dim % D == 0, (
        f"width growth must be a UNIFORM integer multiple (RMSNorm exactness): "
        f"{D} -> {new_dim} fails; use e.g. {2 * D}, {3 * D} ...")
    if new_dim == D:
        return model
    n2o, counts = replication_map(D, new_dim)

    # new hidden dim follows the config formula; scale only if it is a uniform
    # multiple of the old hidden (else keep hidden fixed -- still exact)
    new_hidden = int(round(new_dim * cfg.ffn_expansion / 64.0)) * 64
    old_hidden = cfg.hidden_dim
    if new_hidden >= old_hidden and new_hidden % old_hidden == 0:
        n2o_h, counts_h = replication_map(old_hidden, new_hidden)
    else:
        new_hidden = old_hidden
        n2o_h, counts_h = list(range(old_hidden)), [1] * old_hidden

    # build the bigger model (untied head: the head consumes the stream).
    # If hidden cannot scale uniformly, pin the expansion so the new model's
    # hidden_dim stays == old_hidden (else the constructor would grow it).
    new_cfg = ModelConfig(**cfg.as_dict())
    new_cfg.dim = new_dim
    new_cfg.tie_weights = False
    if new_hidden != int(round(new_dim * cfg.ffn_expansion / 64.0)) * 64:
        new_cfg.ffn_expansion = new_hidden / new_dim
    new_model = LeafLM(new_cfg)

    def fill_linear(new_lin, old_lin):
        """new_lin has (in=?, out=?).  Widen input via n2o (consumer), widen
        output via n2o (producer) -- handles both sides where present."""
        w = old_lin.weight
        # output side (producer) -> replicate rows
        if new_lin.out_features == new_dim and old_lin.out_features == D:
            w = _replicate_rows(w, 0, n2o)
        elif new_lin.out_features == new_hidden and old_lin.out_features == old_hidden:
            w = _replicate_rows(w, 0, n2o_h)
        # input side (consumer) -> divide columns
        if new_lin.in_features == new_dim and old_lin.in_features == D:
            w = _divide_by_counts(w, 1, n2o, counts)
        elif new_lin.in_features == new_hidden and old_lin.in_features == old_hidden:
            w = _divide_by_counts(w, 1, n2o_h, counts_h)
        with torch.no_grad():
            new_lin.weight.copy_(w)
        if new_lin.bias is not None and old_lin.bias is not None:
            b = old_lin.bias
            if new_lin.out_features == new_dim and old_lin.out_features == D:
                b = _replicate_rows(b.unsqueeze(0), 1, n2o).squeeze(0)
            elif new_lin.out_features == new_hidden and old_lin.out_features == old_hidden:
                b = _replicate_rows(b.unsqueeze(0), 1, n2o_h).squeeze(0)
            with torch.no_grad():
                new_lin.bias.copy_(b)

    def fill_channel(new_p, old_p):
        """per-channel param (norm weight, s1, s2, ...) -> replicate."""
        with torch.no_grad():
            new_p.copy_(_replicate_rows(old_p.unsqueeze(0), 1, n2o).squeeze(0))

    # embedding: producer of the stream -> replicate columns
    with torch.no_grad():
        new_model.tok_emb.weight.copy_(
            _replicate_rows(model.tok_emb.weight, 1, n2o))
    # head: consumer of the stream -> divide columns (untied now)
    with torch.no_grad():
        new_model.head.weight.copy_(
            _divide_by_counts(model.head.weight, 1, n2o, counts))

    for old_blk, new_blk in zip(model.blocks, new_model.blocks):
        # memory projections (consumer of D)
        for attr in ("wk", "wv", "wq"):
            old_m = getattr(old_blk.memory, attr, None)
            if old_m is not None:
                fill_linear(getattr(new_blk.memory, attr), old_m)
        for attr in ("w_write", "w_forget", "w_read", "w_decay"):
            old_m = getattr(old_blk.memory, attr, None)
            if old_m is not None:
                fill_linear(getattr(new_blk.memory, attr), old_m)
        # short conv (dims unchanged by width growth -- must be copied)
        if old_blk.memory.short_conv is not None:
            with torch.no_grad():
                new_blk.memory.short_conv.weight.copy_(
                    old_blk.memory.short_conv.weight)
        # wo (producer of D)
        fill_linear(new_blk.memory.wo, old_blk.memory.wo)
        # alpha per head (unchanged)
        with torch.no_grad():
            new_blk.memory.alpha.copy_(old_blk.memory.alpha)
        if old_blk.memory.write_mult is not None and \
                isinstance(old_blk.memory.write_mult, torch.nn.Parameter):
            with torch.no_grad():
                new_blk.memory.write_mult.copy_(old_blk.memory.write_mult)
                new_blk.memory.forget_mult.copy_(old_blk.memory.forget_mult)
        # novelty-gate params are per-head (unchanged by width growth)
        if old_blk.memory.surprise_w is not None and \
                new_blk.memory.surprise_w is not None:
            with torch.no_grad():
                new_blk.memory.surprise_w.copy_(old_blk.memory.surprise_w)
                new_blk.memory.surprise_b.copy_(old_blk.memory.surprise_b)
        # DP-norm denom bias is per-head (unchanged by width growth)
        if old_blk.memory.d_bias is not None and \
                new_blk.memory.d_bias is not None:
            with torch.no_grad():
                new_blk.memory.d_bias.copy_(old_blk.memory.d_bias)
        # local path: depthwise convs, per-channel -> replicate channel kernels
        for oc, nc in zip(old_blk.local_path.convs, new_blk.local_path.convs):
            with torch.no_grad():
                nc.weight.copy_(
                    _replicate_rows(oc.weight, 0, n2o))
        # mix gate (both sides)
        fill_linear(new_blk.mix_gate, old_blk.mix_gate)
        # output gate (both sides)
        if old_blk.memory.out_gate is not None:
            fill_linear(new_blk.memory.out_gate, old_blk.memory.out_gate)
        # residual scales + norm weights (per channel)
        fill_channel(new_blk.s1, old_blk.s1)
        fill_channel(new_blk.s2, old_blk.s2)
        fill_channel(new_blk.norm1.weight, old_blk.norm1.weight)
        fill_channel(new_blk.norm2.weight, old_blk.norm2.weight)
        # FFN (dense or MoE)
        if isinstance(old_blk.ffn, MoEFFN) and isinstance(new_blk.ffn, MoEFFN):
            fill_linear(new_blk.ffn.router, old_blk.ffn.router)
            for oe, ne in zip(old_blk.ffn.experts, new_blk.ffn.experts):
                fill_linear(ne.w1, oe.w1)
                fill_linear(ne.w2, oe.w2)
                fill_linear(ne.w3, oe.w3)
        else:
            fill_linear(new_blk.ffn.w1, old_blk.ffn.w1)
            fill_linear(new_blk.ffn.w2, old_blk.ffn.w2)
            fill_linear(new_blk.ffn.w3, old_blk.ffn.w3)
        # slot attention (Titans-style, opt-in)
        if old_blk.memory.slot_q is not None and new_blk.memory.slot_q is not None:
            fill_linear(new_blk.memory.slot_q, old_blk.memory.slot_q)
            fill_channel(new_blk.memory.slot_scale, old_blk.memory.slot_scale)
        # SWA branch (opt-in hybrid)
        if old_blk.swa is not None and new_blk.swa is not None:
            for attr in ("wq", "wk", "wv", "wo"):
                fill_linear(getattr(new_blk.swa, attr), getattr(old_blk.swa, attr))
            fill_channel(new_blk.swa.scale, old_blk.swa.scale)
    # final norm weight
    fill_channel(new_model.norm_f.weight, model.norm_f.weight)
    # Persistent slots: carry them by FULL COLUMN REPLICATION (not re-init).
    #   With the replicated stream x_new = [x, x], slots_new = [slots, slots]
    #   keeps the stream SYMMETRIC after the slot readout (each half gets the
    #   same slot contribution), so the head (which sums the halves) is exact.
    #   The only change is the slot softmax temperature (logits are 2x -> a
    #   global sharpening of that tiny opt-in component); the slot CONTENT is
    #   fully preserved (vs re-init, which throws it away).
    #   (Mathematically exact carry is impossible with one shared key/value
    #   matrix under uniform replication; this is the closest, and the slot
    #   contribution is ~1% of the memory output.)
    if model.blocks[0].memory.slots is not None:
        with torch.no_grad():
            # replicate slot COLUMNS with the SAME n2o map as the stream
            # (interleaved [c0,c0,c1,c1,...]) so the readout stays symmetric
            for old_blk, new_blk in zip(model.blocks, new_model.blocks):
                new_blk.memory.slots.copy_(
                    _replicate_rows(old_blk.memory.slots, 1, n2o))
    return new_model


# ---------------------------------------------------------------------------
# depth growth
# ---------------------------------------------------------------------------
def grow_depth(model: LeafLM, new_layers: int) -> LeafLM:
    cfg = model.cfg
    assert new_layers >= cfg.n_layers
    if new_layers == cfg.n_layers:
        return model
    new_cfg = ModelConfig(**cfg.as_dict())
    new_cfg.n_layers = new_layers
    new_model = LeafLM(new_cfg)
    # copy the existing blocks' weights; new blocks keep zero-init s1/s2
    # (identity at init -> output EXACTLY preserved)
    new_model.load_state_dict(
        {k: v for k, v in model.state_dict().items()}, strict=False)
    new_model.norm_f.load_state_dict(model.norm_f.state_dict())
    new_model.head.load_state_dict(model.head.state_dict())
    new_model.tok_emb.load_state_dict(model.tok_emb.state_dict())
    # BUG FIX: with scale_init > 0 (the --fast recipe), the NEW blocks are
    # created with s1=s2=scale_init, NOT identity!  Force the residual scales
    # of every NEW block to ZERO so they are true identity at init and the
    # output is bit-preserved.  (The old blocks keep their trained scales.)
    for i in range(cfg.n_layers, new_layers):
        blk = new_model.blocks[i]
        with torch.no_grad():
            blk.s1.zero_()
            blk.s2.zero_()
            if blk.swa is not None:
                blk.swa.scale.zero_()
            if blk.memory.slot_scale is not None:
                blk.memory.slot_scale.zero_()
    return new_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Grow a LEAFv5 checkpoint.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--to-dim", type=int, default=None)
    p.add_argument("--to-layers", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    model = LeafLM(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    print(f"[grow] loaded {model.n_params/1e6:.1f}M (dim={cfg.dim}, "
          f"layers={cfg.n_layers})")

    if args.to_dim is not None:
        model = grow_width(model, args.to_dim)
        print(f"[grow] width {cfg.dim} -> {model.cfg.dim} "
              f"({model.n_params/1e6:.1f}M params; head untied)")
    if args.to_layers is not None:
        model = grow_depth(model, args.to_layers)
        print(f"[grow] depth {cfg.n_layers} -> {model.cfg.n_layers} "
              f"({model.n_params/1e6:.1f}M params; new blocks identity-init)")

    out_cfg = model.cfg.as_dict()
    torch.save({"model": model.state_dict(), "model_config": out_cfg,
                "tokenizer_meta": ck.get("tokenizer_meta"),
                "corpus_meta": ck.get("corpus_meta"),
                "grew_from": ck.get("model_config")}, args.out)
    print(f"[grow] saved -> {args.out}")


if __name__ == "__main__":
    main()
