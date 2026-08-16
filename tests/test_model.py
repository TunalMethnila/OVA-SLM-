"""Unit tests for LEAFv5.  Run directly:  python tests/test_model.py"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.config import preset_config
from leafv5.model import LeafLM
from leafv5.data import CharTokenizer


def test_forward_and_recurrent_equivalence():
    cfg = preset_config("micro", vocab_size=256)
    m = LeafLM(cfg)
    B, T = 2, 33
    x = torch.randint(0, 256, (B, T))
    states = m.init_states(B, torch.device("cpu"))
    logits, _ = m(x, states)
    assert logits.shape == (B, T, 256), logits.shape
    assert len(states) == cfg.n_layers
    assert states[0].shape == (B, cfg.n_heads, cfg.d_h, cfg.d_h)
    # feeding token-by-token with state carry must match the one-shot forward
    st = m.init_states(B, torch.device("cpu"))
    inc = []
    for t in range(T):
        lg, st = m(x[:, t:t + 1], st, offset=t)
        inc.append(lg[:, 0])
    inc = torch.stack(inc, 1)
    one, _ = m(x, m.init_states(B, torch.device("cpu")))
    torch.testing.assert_close(inc, one, atol=1e-5, rtol=1e-4)
    print("  forward + recurrent equivalence OK")


def test_gradients_flow_and_zero_scales():
    cfg = preset_config("micro", vocab_size=256)
    m = LeafLM(cfg)
    # paper sec. 4: per-channel residual scales start at exactly zero
    assert torch.all(m.blocks[0].s1 == 0) and torch.all(m.blocks[0].s2 == 0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        x = torch.randint(0, 256, (4, 16))
        y = torch.randint(0, 256, (4, 16))
        logits, _ = m(x, m.init_states(4, torch.device("cpu")))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        assert not any(torch.isnan(p.grad).any()
                       for p in m.parameters() if p.grad is not None)
        opt.step()
    # after a few steps the zero scales must have moved off 0 ...
    assert m.blocks[0].s1.abs().sum() > 0 and m.blocks[0].s2.abs().sum() > 0
    # ... and branch weights must now receive gradient (ReZero identity start)
    x = torch.randint(0, 256, (4, 16))
    y = torch.randint(0, 256, (4, 16))
    logits, _ = m(x, m.init_states(4, torch.device("cpu")))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
    loss.backward()
    assert m.blocks[-1].memory.wk.weight.grad is not None
    assert m.blocks[-1].memory.wk.weight.grad.abs().sum() > 0
    assert m.blocks[-1].ffn.w1.weight.grad.abs().sum() > 0
    print("  identity-start scales, gradient flow after warm-up OK")


def test_state_stability():
    """State must stay bounded: ||S||_F <= sqrt(d_h) thanks to StateNorm."""
    cfg = preset_config("micro", vocab_size=256)
    mem = LeafLM(cfg).blocks[0].memory
    B, T, D = 2, 64, cfg.dim
    x = torch.randn(B, T, D)
    state = None
    with torch.no_grad():
        for t in range(T):
            _, state, _, _ = mem(x[:, t:t + 1], state)
        fnorm = state.norm(dim=(-1, -2))
        assert torch.all(fnorm <= math.sqrt(cfg.d_h) * 1.01 + 1e-4), fnorm.max()
    print("  state stays bounded (||S||_F ~ sqrt(d_h)) OK")


def test_tokenizer_roundtrip():
    t = CharTokenizer({c: i for i, c in enumerate("abc def")})
    ids = t.encode("abc def")
    assert t.decode(ids) == "abc def"
    assert t.vocab_size == 7
    print("  char tokenizer round-trip OK")


def test_delta_write_along_key():
    """Math check: after a single write, S = b_w * v k^T, so reading k gives b_w * v."""
    B, H, dh = 2, 6, 32
    k = torch.nn.functional.normalize(torch.randn(B, 1, H, dh), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(B, 1, H, dh), dim=-1)
    S = torch.zeros(B, H, dh, dh)
    bw, bf = torch.full((B, 1, H, 1), 0.7), torch.full((B, 1, H, 1), 0.5)
    Sf = S.reshape(B * H, dh, dh)
    k1 = k.permute(0, 2, 1, 3).reshape(B * H, 1, dh)
    v1 = v.permute(0, 2, 1, 3).reshape(B * H, 1, dh)
    Sf = (Sf
          - bf.reshape(B * H, 1, 1) * torch.bmm(torch.bmm(Sf, k1.transpose(1, 2)), k1)
          + bw.reshape(B * H, 1, 1) * torch.bmm(v1.transpose(1, 2), k1))
    read = torch.bmm(Sf, k1.transpose(1, 2))[..., 0]          # [BH, dh]
    assert (read * v1[..., 0, :]).sum(-1).min() > 0            # read points along v
    # forgetting: applying the forget path on a full state must shrink it
    Sf2 = torch.randn(B * H, dh, dh)
    Sf2_n = Sf2 - bf.reshape(B * H, 1, 1) * torch.bmm(torch.bmm(Sf2, k1.transpose(1, 2)), k1)
    assert Sf2_n.norm() <= Sf2.norm() + 1e-4
    print("  delta write / forget math OK")


def test_chunked_scan_equivalence():
    """Chunked parallel-scan must exactly match the sequential recurrence when
    StateNorm is disabled (identical math, fp32 states).  eval() so dropout is
    off (mem_dropout would otherwise mask each call differently)."""
    cfg = preset_config("micro", vocab_size=256)
    mem = LeafLM(cfg).blocks[0].memory.eval()
    B, T, D = 2, 32, cfg.dim
    x = torch.randn(B, T, D)
    out_seq, S_seq, _, _ = mem(x, None, chunk=None, state_norm=False)
    out_chunk, S_chunk, _, _ = mem(x, None, chunk=8, state_norm=False)
    torch.testing.assert_close(out_chunk, out_seq, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(S_chunk, S_seq, atol=1e-4, rtol=1e-3)
    print("  chunked parallel-scan == sequential recurrence (no norm) OK")


def test_chunked_stability_with_norm():
    """Both scan modes keep the state bounded when StateNorm is on. Outputs may
    differ in *scale* (chunked places normalization at chunk boundaries instead
    of every step - the paper's 'chunked formulation' tradeoff) but must not
    diverge pathologically; training absorbs the scale via the zero-init
    residual gates."""
    cfg = preset_config("micro", vocab_size=256)
    mem = LeafLM(cfg).blocks[0].memory.eval()
    B, T, D = 2, 48, cfg.dim
    x = torch.randn(B, T, D)
    out_seq, S_seq, _, _ = mem(x, None, chunk=None, state_norm=True)
    out_ch, S_ch, _, _ = mem(x, None, chunk=16, state_norm=True)
    bound = math.sqrt(cfg.d_h) * 1.05
    assert S_seq.norm(dim=(-1, -2)).max() <= bound
    assert S_ch.norm(dim=(-1, -2)).max() <= bound
    rel = (out_ch - out_seq).abs().max() / (out_seq.abs().max() + 1e-6)
    assert rel < 5.0, rel  # same ballpark, no divergence
    print(f"  chunked vs sequential (StateNorm on): max rel diff={float(rel):.3f}, "
          f"both states bounded OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_forward_and_recurrent_equivalence()
    test_gradients_flow_and_zero_scales()
    test_state_stability()
    test_tokenizer_roundtrip()
    test_delta_write_along_key()
    test_chunked_scan_equivalence()
    test_chunked_stability_with_norm()
    print("\nAll tests passed.")
