"""Tests for the world-class upgrades: MoE FFN, slot attention, benchmark
baselines.
Run:  python tests/test_world.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.config import preset_config
from leafv5.grow import grow_width
from leafv5.model import LeafLM


def test_moe_builds_trains_and_aux():
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, moe=True, moe_experts=6,
                        moe_topk=2, scale_init=0.1, mem_slots=0)
    m = LeafLM(cfg)
    dense = LeafLM(preset_config("micro", vocab_size=256, mem_slots=0,
                                 scale_init=0.1))
    assert m.n_params > 2 * dense.n_params  # MoE = many more params
    assert isinstance(m.blocks[0].ffn, __import__(
        "leafv5.model", fromlist=["MoEFFN"]).MoEFFN)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        x = torch.randint(0, 256, (4, 16))
        y = torch.randint(0, 256, (4, 16))
        lg, _ = m(x, m.init_states(4, torch.device("cpu")))
        loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)) \
            + 0.01 * m.aux_loss()
        loss.backward()
        opt.step()
    assert m.aux_loss().item() > 0
    assert not any(torch.isnan(p.grad).any()
                   for p in m.parameters() if p.grad is not None)
    print("  MoE: many params, trains with aux loss OK")


def test_moe_growth_exact():
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, moe=True, moe_experts=4,
                        moe_topk=2, scale_init=0.1, mem_slots=0)
    m = LeafLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad()
        x = torch.randint(0, 256, (4, 16))
        y = torch.randint(0, 256, (4, 16))
        lg, _ = m(x, m.init_states(4, torch.device("cpu")))
        F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).backward()
        opt.step()
    g = grow_width(m, 256)
    m.eval(); g.eval()
    xt = torch.randint(0, 256, (4, 20))
    with torch.no_grad():
        lo = m(xt, m.init_states(4, torch.device("cpu")))[0]
        ln = g(xt, g.init_states(4, torch.device("cpu")))[0]
    assert (ln - lo).abs().max().item() < 5e-4
    print("  MoE growth exact OK")


def test_slot_attn_identity_trains_grows():
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, slot_attn=True, scale_init=0.1)
    m = LeafLM(cfg)
    mem = m.blocks[0].memory
    assert mem.slot_q is not None and torch.all(mem.slot_scale == 0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        x = torch.randint(0, 256, (4, 16))
        y = torch.randint(0, 256, (4, 16))
        lg, _ = m(x, m.init_states(4, torch.device("cpu")))
        F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).backward()
        opt.step()
    assert mem.slot_scale.abs().sum() > 0
    assert mem.slot_q.weight.grad.abs().sum() > 0
    g = grow_width(m, 256)
    m.eval(); g.eval()
    xt = torch.randint(0, 256, (4, 20))
    with torch.no_grad():
        lo = m(xt, m.init_states(4, torch.device("cpu")))[0]
        ln = g(xt, g.init_states(4, torch.device("cpu")))[0]
    assert (ln - lo).abs().max().item() < 5e-4
    print("  slot-attn: identity-init, trains, growth exact OK")


def test_benchmark_baselines_forward():
    from leafv5.benchmark_world import GatedRNN, TinyTransformer
    torch.manual_seed(0)
    V = 64
    t = TinyTransformer(V, dim=64, layers=2)
    r = GatedRNN(V, dim=64, layers=2)
    x = torch.randint(0, V, (2, 12))
    assert t(x).shape == (2, 12, V)
    assert r(x).shape == (2, 12, V)
    print("  Transformer + GatedRNN baselines forward OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_moe_builds_trains_and_aux()
    test_moe_growth_exact()
    test_slot_attn_identity_trains_grows()
    test_benchmark_baselines_forward()
    print("\nWorld-class tests passed.")
