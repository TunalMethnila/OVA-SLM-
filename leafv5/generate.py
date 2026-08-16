"""Recurrent generation with LEAFv5.

Inference is purely recurrent: a tiny [H, d_h, d_h] state per layer is carried
forward, so memory is constant in sequence length (paper sec. 5).
"""
from __future__ import annotations

import argparse
import time
from typing import List, Optional, Tuple

import torch

from .config import ModelConfig
from .data import load_tokenizer
from .model import LeafLM


def load_checkpoint(path: str, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    model = LeafLM(cfg).to(device)
    sd = ck["model"]
    # P0 #7: normalize DDP "module."-prefixed keys on the LOAD side too, so a
    # checkpoint saved from a wrapped model restores correctly.
    sd = {k[len("module."):] if k.startswith("module.") else k: v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load] WARN {len(missing)} params initialized fresh "
              f"(missing from checkpoint): {missing[:4]}"
              f"{'...' if len(missing) > 4 else ''}")
    if unexpected:
        print(f"[load] WARN {len(unexpected)} unexpected checkpoint keys ignored")
    model.eval()
    meta = ck.get("corpus_meta") or ck.get("tokenizer_meta")
    if meta is None:
        raise KeyError("checkpoint has no tokenizer metadata (corpus_meta/"
                       "tokenizer_meta)")
    tok = load_tokenizer(meta)
    return model, tok, ck


@torch.no_grad()
def generate(model: LeafLM, tokenizer, prompt: str, max_new: int = 200,
             temperature: float = 0.8, top_k: int = 50, top_p: Optional[float] = None,
             repeat_penalty: float = 1.0, max_consecutive: int = 0,
             device="auto",
             states: Optional[List[torch.Tensor]] = None,
             offset: int = 0, verbose: bool = False) -> Tuple[str, List[torch.Tensor]]:
    """Recurrent sampling.  Returns (text, final_states).
    offset: absolute position of the prompt's first token (for stateful
    sessions -- carry `states` + `offset` across turns so the delta memory IS
    the conversation context and history is never re-encoded)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = tokenizer.encode(prompt)
    B = 1
    t0 = time.time()
    model.eval()
    # use the validated C scan kernel when available (decode speedup; exact)
    fast = False
    try:
        import os as _os
        from mojo.c_ref import _lib  # noqa: F401  (raises if .so missing)
        _so = _os.path.join(_os.path.dirname(_os.path.dirname(
            _os.path.abspath(__file__))), "mojo", "c_ref", "leafv5_scan.so")
        fast = _os.path.exists(_so)
    except Exception:
        fast = False
    def _sample(logit):
        logit = logit / max(temperature, 1e-4)
        if repeat_penalty and repeat_penalty != 1.0:
            for t_ in set(ids[-64:]):  # penalize recently seen tokens
                logit[t_] = logit[t_] / repeat_penalty if logit[t_] > 0 \
                    else logit[t_] * repeat_penalty
        if top_k and top_k > 0:
            v, _ = torch.topk(logit, min(top_k, logit.shape[-1]))
            logit[logit < v[-1]] = -float("inf")
        if top_p:
            sorted_l, idx = torch.sort(logit, descending=True)
            cum = torch.cumsum(torch.softmax(sorted_l, -1), -1)
            mask = cum > top_p
            mask[1:] = mask[:-1].clone()
            mask[0] = False
            logit[idx[mask]] = -float("inf")
        if temperature <= 0:
            return int(torch.argmax(logit))
        return int(torch.multinomial(torch.softmax(logit, -1), 1).item())

    with torch.no_grad():
        # The model's output at position t predicts token t+1 (the state already
        # includes token t), so the first generated token comes from the LAST
        # logit of the one-shot prompt pass -- re-feeding ids[-1] would double-
        # write it into the delta memory and corrupt the state (an off-by-one).
        if states is None:
            states = model.init_states(B, device)
        if not ids:                       # empty prompt: seed with a start token
            ids = [0]
        # Absolute position for RoPE: prefer the state's carried offset when it
        # is nonzero (LeafStates from a previous turn); otherwise use the
        # caller's offset (serve.py passes its own session offset for plain-list
        # states, which carry no position -- bug fix 2026-08-09: this was
        # overwritten to 0, so stateful RoPE positions restarted every turn).
        carried = getattr(states, "offset", 0)
        if carried:
            offset = carried
        new_ids: List[int] = []
        if max_new > 0:
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            logits, states = model(inp, states, offset=offset, fast=fast)
            offset = offset + len(ids)
            nxt = _sample(logits[0, -1].clone())
            new_ids.append(nxt)
            ids.append(nxt)
        consec = 0
        while len(new_ids) < max_new:
            inp = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
            logits, states = model(inp, states, offset=offset, fast=fast)
            if not torch.isfinite(logits).all():   # defensive: never emit NaN
                logits = torch.zeros_like(logits)
            offset += 1
            nxt = _sample(logits[0, -1].clone())
            # stop early on pathological repetition (tiny-model safeguard)
            consec = consec + 1 if (len(ids) and nxt == ids[-1]) else 0
            if max_consecutive and consec >= max_consecutive:
                break
            new_ids.append(nxt)
            ids.append(nxt)
    dt = time.time() - t0
    if verbose:
        print(f"[generate] {len(new_ids)} tokens in {dt:.2f}s "
              f"({len(new_ids)/max(dt,1e-6):.0f} tok/s)")
    return tokenizer.decode(new_ids), states


def main():
    p = argparse.ArgumentParser(description="Generate text with a trained LEAFv5.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    model, tok, ck = load_checkpoint(args.ckpt, args.device)
    print(f"[generate] model: {model.n_params/1e6:.1f}M params | "
          f"state memory per token: {model.cfg.n_heads*model.cfg.d_h*model.cfg.d_h*4/1e6:.3f} MB/layer")
    text, _ = generate(model, tok, prompt=args.prompt, max_new=args.max_new,
                       temperature=args.temperature, top_k=args.top_k,
                       top_p=args.top_p, device=args.device, verbose=True)
    print(f"--- prompt: {args.prompt!r} ---")
    print(text)


if __name__ == "__main__":
    main()


@torch.no_grad()
def beam_search(model: LeafLM, tokenizer, prompt: str, max_new: int = 64,
                beam_size: int = 4, device="auto") -> str:
    """Greedy-equivalent deterministic decoding for tasks where accuracy
    matters (math, tool JSON): beam search over the recurrent states.
    Returns the best (highest log-prob) complete sequence."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids:                       # empty prompt: seed with a start token
        ids = [0]
    if max_new <= 0:                  # match generate()'s max_new=0 semantics
        return ""
    # Condition on the FULL prompt in one pass (bug fix 2026-08-09: the old
    # loop fed only the last prompt token from a fresh state, so beams were
    # essentially unconditional; the first expansion must come from the prompt
    # pass's last logits, exactly like generate()).  The prompt pass also gives
    # the state that the first generated token continues from.
    with torch.no_grad():
        prompt_t = torch.tensor([ids], dtype=torch.long, device=device)
        logits0, pstate = model(prompt_t, model.init_states(1, device))
        logp0 = torch.log_softmax(logits0[0, -1].float(), -1)
        top0 = torch.topk(logp0, min(beam_size, logp0.shape[-1]))
        beams = [(ids + [int(i)], pstate, float(v))
                 for v, i in zip(top0.values, top0.indices)]
    for _ in range(1, max_new):       # already generated 1 token above
        new_beams = []
        for toks, states, lp in beams:
            # feed the most recent token (absolute position len(toks)-1) with
            # the state that precedes it -- no re-feeding of earlier tokens
            inp = torch.tensor([[toks[-1]]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, states = model(inp, states, offset=len(toks) - 1)
            logp = torch.log_softmax(logits[0, -1].float(), -1)
            top = torch.topk(logp, min(beam_size, logp.shape[-1]))
            for v, i in zip(top.values, top.indices):
                new_beams.append((toks + [int(i)], states, lp + float(v)))
        beams = sorted(new_beams, key=lambda b: -b[2])[:beam_size]
    if not beams:                     # max_new <= 0
        return ""
    best = max(beams, key=lambda b: b[2] / max(1, len(b[0]) - len(ids)))
    return tokenizer.decode(best[0][len(ids):])
