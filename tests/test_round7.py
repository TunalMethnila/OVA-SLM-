"""Tests for round-7 pushes: LoRA PEFT, chain-of-thought math data, beam
search, PTB benchmark plumbing.
Run:  python tests/test_round7.py
"""
import os
import random
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_gen"))

from leafv5.config import preset_config
from leafv5.model import LeafLM
from leafv5.lora import apply_lora, lora_params, merge_lora


def test_lora_peft():
    """LoRA: identity at init, ~small % trainable, learns, merges to plain."""
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=2, dim=128, d_h=32,
                        scale_init=0.1)
    m = LeafLM(cfg).eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        before, _ = m(x, m.init_states(2, torch.device("cpu")))
    replaced = apply_lora(m, 8)
    n_lora = sum(p.numel() for p in lora_params(m))
    assert replaced > 10
    assert 0 < n_lora < m.n_params
    with torch.no_grad():
        after_wrap, _ = m(x, m.init_states(2, torch.device("cpu")))
    assert torch.allclose(after_wrap, before, atol=1e-6)  # identity at init
    opt = torch.optim.AdamW(lora_params(m), lr=1e-3)
    for _ in range(10):
        opt.zero_grad()
        y = torch.randint(0, 256, (2, 16))
        lg, _ = m(x, m.init_states(2, torch.device("cpu")))
        F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).backward()
        opt.step()
    with torch.no_grad():
        after_train, _ = m(x, m.init_states(2, torch.device("cpu")))
    # learning happened: at least one B adapter is nonzero after training
    b_moved = any((p != 0).any().item()
                  for n, p in m.named_parameters() if n.endswith(".B"))
    assert b_moved, "LoRA adapters did not move!"
    merge_lora(m)
    with torch.no_grad():
        final, _ = m(x, m.init_states(2, torch.device("cpu")))
    assert (final - after_train).abs().max().item() < 1e-3  # merge faithful
    # merged model is a plain LeafLM (works with load_state_dict)
    m2 = LeafLM(cfg)
    m2.load_state_dict(m.state_dict())
    print(f"  LoRA PEFT OK (wrapped={replaced}, trainable={n_lora}, "
          f"merge faithful)")


def test_cot_math_verified():
    import make_dataset as md
    rng = random.Random(0)
    ex = md.make_arithmetic(rng, 300)
    bad = 0
    for e in ex:
        q, a = e["instruction"], e["output"]
        nums = [int(x) for x in re.findall(r"-?\d+", q)]
        if len(nums) < 2:
            continue
        if "apples" in q:
            exp = nums[0] + nums[1]
        elif "(" in q:
            exp = (nums[0] + nums[1]) * nums[2]
        elif "Calculate" in q:
            exp = nums[0] + nums[1] * nums[2]
        elif "*" in q and "-" in q:
            exp = nums[0] * nums[1] - nums[2]
        elif "*" in q and "+" in q:
            exp = nums[0] * nums[1] + nums[2]
        elif "-" in q and "+" in q:
            exp = nums[0] - nums[1] + nums[2]
        elif "*" in q:
            exp = nums[0] * nums[1]
        elif "+" in q and len(nums) >= 3:
            exp = nums[0] + nums[1] + nums[2]
        elif "+" in q:
            exp = nums[0] + nums[1]
        elif "-" in q:
            exp = nums[0] - nums[1]
        else:
            continue
        if f"Answer: {exp}" not in a and f"answer is {exp}" not in a:
            bad += 1
    assert bad == 0, f"{bad} wrong answers"
    cot = sum(1 for e in ex if "Step" in e["output"])
    assert cot > 0
    print(f"  CoT math verified: {len(ex)} examples, 0 wrong, {cot} CoT")


def test_beam_search():
    from leafv5.data import CharTokenizer
    from leafv5.generate import beam_search
    import string
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=2, dim=96, d_h=32)
    m = LeafLM(cfg).eval()
    tok = CharTokenizer({c: i for i, c in enumerate(string.ascii_lowercase)})
    out = beam_search(m, tok, "abc", max_new=8, beam_size=3, device="cpu")
    assert isinstance(out, str) and len(out) <= 8
    print(f"  beam search OK (returned {len(out)} tokens)")


def test_ptb_benchmark_imports():
    from leafv5.benchmark_ppl import fetch, get_batch
    import numpy as np
    assert callable(fetch) and callable(get_batch)
    x, y = get_batch(np.arange(1000), 4, 32, np.random.default_rng(0))
    assert x.shape == (4, 32)
    print("  PTB benchmark plumbing OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_lora_peft()
    test_cot_math_verified()
    test_beam_search()
    test_ptb_benchmark_imports()
    print("\nRound-7 tests passed.")
