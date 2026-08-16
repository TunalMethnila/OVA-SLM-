"""The central correctness invariant (reviewer's #1/#2):
  1. CAUSALITY: changing a FUTURE token must not change earlier outputs.
  2. TRAIN == DECODE: full-sequence forward equals token-by-token recurrent
     decode with carried LeafStates, exactly (fp tolerance), for all feature
     combinations (delta memory, causal local convs, short conv, SWA).
Run:  python tests/test_causal_invariant.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.config import preset_config
from leafv5.model import LeafLM


def test_causality():
    """Local path + memory must be causal: perturb a future token, earlier
    logits unchanged."""
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=2, dim=128, d_h=32,
                        scale_init=0.1)
    m = LeafLM(cfg).eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        a, _ = m(x, m.init_states(2, torch.device("cpu")))
        xp = x.clone(); xp[:, 10] = (xp[:, 10] + 5) % 256   # perturb FUTURE pos
        b, _ = m(xp, m.init_states(2, torch.device("cpu")))
    # positions 0..9 must be identical (only pos >= 10 may change)
    d = (a[:, :10] - b[:, :10]).abs().max().item()
    print(f"  causality: max|d| over positions < 10 = {d:.2e}")
    assert d < 1e-5, "FUTURE TOKEN LEAKED INTO EARLIER OUTPUTS!"
    print("  future token does not affect earlier outputs OK")


def _full_vs_decode(m, x):
    """Full-sequence forward vs token-by-token decode (carried LeafStates)."""
    m.eval()
    with torch.no_grad():
        lg_full, _ = m(x, m.init_states(1, torch.device("cpu")))
    st = m.init_states(1, torch.device("cpu"))
    outs = []
    with torch.no_grad():
        for t in range(x.shape[1]):
            lg, st = m(x[:, t:t + 1], st)   # offset carried inside states
            outs.append(lg)                 # [1,1,V]
    lg_dec = torch.cat(outs, 1)             # [1,T,V]
    return lg_full, lg_dec


def _test_config(name, cfg_kw):
    torch.manual_seed(0)
    cfg = preset_config("micro", **cfg_kw)
    m = LeafLM(cfg).eval()
    x = torch.randint(0, 256, (1, 24))
    lg_full, lg_dec = _full_vs_decode(m, x)
    d = (lg_full - lg_dec).abs().max().item()
    rel = d / (lg_full.abs().max().item() + 1e-9)
    print(f"  [{name:28s}] max|d|={d:.2e}  rel={rel:.2e}")
    assert d < 1e-4, (name, d)
    print("    -> full-seq == token-by-token OK")


def test_train_equals_decode():
    print("train == decode invariant (all feature combos):")
    _test_config("default", dict(vocab_size=256, n_layers=2, dim=128, d_h=32))
    _test_config("scale_init 0.1", dict(vocab_size=256, n_layers=2, dim=128,
                                        d_h=32, scale_init=0.1))
    _test_config("short_conv off", dict(vocab_size=256, n_layers=2, dim=128,
                                        d_h=32, short_conv=False))
    _test_config("slot_attn", dict(vocab_size=256, n_layers=2, dim=128, d_h=32,
                                   slot_attn=True))
    _test_config("SWA hybrid", dict(vocab_size=256, n_layers=2, dim=128, d_h=32,
                                    use_swa=True, swa_window=8))
    _test_config("SWA + slots + conv", dict(vocab_size=256, n_layers=2, dim=128,
                                            d_h=32, use_swa=True, swa_window=8,
                                            slot_attn=True, scale_init=0.1))
    # Mistral-style: GQA (1 KV head), interleaved SWA, rolling-buffer-safe
    _test_config("SWA GQA kv=1 every=2", dict(vocab_size=256, n_layers=4,
                                              dim=128, d_h=32, use_swa=True,
                                              swa_window=8, swa_kv_heads=1,
                                              swa_every=2, scale_init=0.1))
    print("  ALL configs: full-sequence == recurrent decode OK")


def test_train_equals_decode_after_training():
    """The invariant must hold with LEARNED weights too."""
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=2, dim=128, d_h=32,
                        use_swa=True, swa_window=8, scale_init=0.1)
    m = LeafLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    xb = torch.randint(0, 256, (4, 16)); yb = torch.randint(0, 256, (4, 16))
    for _ in range(6):
        opt.zero_grad()
        lg, _ = m(xb, m.init_states(4, torch.device("cpu")))
        F.cross_entropy(lg.reshape(-1, 256), yb.reshape(-1)).backward()
        opt.step()
    x = torch.randint(0, 256, (1, 24))
    lg_full, lg_dec = _full_vs_decode(m, x)
    d = (lg_full - lg_dec).abs().max().item()
    print(f"  after training: max|d| = {d:.2e}")
    assert d < 1e-4, d
    print("  trained model: full-seq == recurrent decode OK")


def test_rope_extension():
    """offset+T > max_seq_len must extend the cache, not crash (P1 #11)."""
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=1, dim=96, d_h=32,
                        max_seq_len=16, rope_dim=96)
    m = LeafLM(cfg).eval()
    st = m.init_states(1, torch.device("cpu"))
    with torch.no_grad():
        for t in range(40):  # way past max_seq_len=16
            lg, st = m(torch.tensor([[t % 256]]), st)
    assert torch.isfinite(lg).all()
    print("  RoPE dynamic cache extension OK (40 steps past max_seq_len=16)")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_causality()
    test_train_equals_decode()
    test_train_equals_decode_after_training()
    test_rope_extension()
    print("\nCausal-invariant tests passed.")
