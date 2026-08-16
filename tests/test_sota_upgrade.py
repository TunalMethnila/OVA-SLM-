"""Tests for the SOTA-upgrade memory (read query, short conv, output gate,
persistent slots), input decay, mem dropout, stochastic depth, EMA, and
backward compatibility.
Run:  python tests/test_sota_upgrade.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.config import preset_config
from leafv5.model import LeafLM
from leafv5.train import ema_update


def test_upgrade_features_present():
    cfg = preset_config("micro", vocab_size=256)  # defaults: all ON
    m = LeafLM(cfg)
    mem = m.blocks[0].memory
    assert mem.wq is not None, "read query missing"
    assert mem.short_conv is not None, "short conv missing"
    assert mem.out_gate is not None, "output gate missing"
    assert mem.slots is not None and mem.slots.shape[0] == cfg.mem_slots
    x = torch.randn(2, 8, cfg.dim)
    k = mem._proj(mem.wk, x, 2, 8, cfg.n_heads, cfg.d_h)[0]
    q = mem._proj(mem.wq, x, 2, 8, cfg.n_heads, cfg.d_h)[0]
    assert not torch.allclose(k, q)
    print("  read-query / short-conv / output-gate / slots present OK")


def test_backward_compat_off():
    """Turning all upgrades off reproduces the original paper memory."""
    cfg = preset_config("micro", vocab_size=256, use_read_query=False,
                        short_conv=False, output_gate=False, mem_slots=0)
    m = LeafLM(cfg)
    mem = m.blocks[0].memory
    assert mem.wq is None and mem.short_conv is None
    assert mem.out_gate is None and mem.slots is None
    x = torch.randint(0, 256, (2, 16))
    lg, states = m(x, m.init_states(2, torch.device("cpu")))
    assert lg.shape == (2, 16, 256)
    print("  upgrade-off backward-compatible forward OK")


def test_upgrade_forward_backward():
    cfg = preset_config("micro", vocab_size=256, scale_init=0.1)
    m = LeafLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        lg, _ = m(x, m.init_states(2, torch.device("cpu")))
        loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        assert not any(torch.isnan(p.grad).any()
                       for p in m.parameters() if p.grad is not None)
        opt.step()
    mem = m.blocks[0].memory
    targets = {
        "wq": mem.wq.weight, "slots": mem.slots,
        "out_gate": mem.out_gate.weight, "short_conv": mem.short_conv.weight,
    }
    for pname, p in targets.items():
        assert p.grad is not None, f"{pname} got no gradient"
        assert p.grad.abs().sum() > 0, f"{pname} gradient is zero"
    print("  upgraded memory trains (gradients flow to wq/slots/out_gate/conv) OK")


def test_old_checkpoint_loads():
    """A checkpoint saved WITHOUT the new params must still load."""
    cfg_old = preset_config("micro", vocab_size=256, use_read_query=False,
                            short_conv=False, output_gate=False, mem_slots=0)
    m_old = LeafLM(cfg_old)
    sd_old = m_old.state_dict()
    cfg_new = preset_config("micro", vocab_size=256)
    m_new = LeafLM(cfg_new)
    missing, _ = m_new.load_state_dict(sd_old, strict=False)
    assert len(missing) > 0
    x = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        lg, _ = m_new(x, m_new.init_states(2, torch.device("cpu")))
    assert torch.isfinite(lg).all()
    print(f"  old checkpoint loads into new arch ({len(missing)} fresh params) OK")


def test_input_decay_chunked_exact():
    """With input decay ON, the chunked scan must still equal the sequential
    recurrence (M = a*I - bf k k^T composes correctly)."""
    cfg = preset_config("micro", vocab_size=256, input_decay=True)
    mem = LeafLM(cfg).blocks[0].memory.eval()
    B, T, D = 2, 32, cfg.dim
    x = torch.randn(B, T, D)
    out_seq, S_seq, _, _ = mem(x, None, chunk=None, state_norm=False)
    out_chunk, S_chunk, _, _ = mem(x, None, chunk=8, state_norm=False)
    torch.testing.assert_close(out_chunk, out_seq, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(S_chunk, S_seq, atol=1e-4, rtol=1e-3)
    print("  input-decay chunked == sequential (exact) OK")


def test_mem_dropout_and_stochastic_depth():
    cfg = preset_config("micro", vocab_size=256, mem_dropout=0.1,
                        stochastic_depth=0.2, scale_init=0.1)
    m = LeafLM(cfg)
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    m.train()
    lg, _ = m(x, m.init_states(2, torch.device("cpu")))
    loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
    loss.backward()
    assert not any(torch.isnan(p.grad).any()
                   for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        lg1, _ = m(x, m.init_states(2, torch.device("cpu")))
        lg2, _ = m(x, m.init_states(2, torch.device("cpu")))
    torch.testing.assert_close(lg1, lg2)
    print("  mem-dropout + stochastic-depth train/eval OK")


def test_ema_math():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    ema_update(a, b, 0.9)
    torch.testing.assert_close(a, torch.tensor([1.2, 2.2]))
    print("  EMA update math OK")


def test_swa_interleave_every():
    """--swa-every k: SWA on blocks i%k==0 only; identity at init; trains;
    grow_depth extends the same index pattern exactly."""
    from leafv5.grow import grow_depth
    cfg = preset_config("micro", vocab_size=256, n_layers=4, use_swa=True,
                        swa_every=2, swa_window=16, mem_slots=0, scale_init=0.1)
    m = LeafLM(cfg)
    pat = [blk.swa is not None for blk in m.blocks]
    assert pat == [True, False, True, False], pat
    assert all(b.swa.scale.abs().max().item() < 1e-9 for b in m.blocks if b.swa)
    # trains: block-0 SWA scale becomes nonzero
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        xi = torch.randint(0, 256, (2, 16))
        yi = torch.randint(0, 256, (2, 16))
        lg, _ = m(xi, m.init_states(2, torch.device("cpu")))
        F.cross_entropy(lg.reshape(-1, 256), yi.reshape(-1)).backward()
        opt.step()
    assert m.blocks[0].swa.scale.abs().sum() > 0
    # growth extends the pattern exactly and preserves behavior
    g = grow_depth(m, 6)
    pat_g = [blk.swa is not None for blk in g.blocks]
    assert pat_g == [True, False, True, False, True, False], pat_g
    m.eval(); g.eval()
    xt = torch.randint(0, 256, (2, 20))
    with torch.no_grad():
        a = m(xt, m.init_states(2, torch.device("cpu")))[0]
        b = g(xt, g.init_states(2, torch.device("cpu")))[0]
    assert (b - a).abs().max().item() < 5e-4
    print("  SWA interleave (every=2): pattern, identity, train, growth exact OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_upgrade_features_present()
    test_backward_compat_off()
    test_upgrade_forward_backward()
    test_old_checkpoint_loads()
    test_input_decay_chunked_exact()
    test_mem_dropout_and_stochastic_depth()
    test_ema_math()
    test_swa_interleave_every()
    print("\nSOTA-upgrade tests passed.")
