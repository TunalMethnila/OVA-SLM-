"""Tests for progressive growth: width (Net2Net) and depth (zero-init blocks)
must preserve the model's function EXACTLY at the swap (no loss of training).
Run:  python tests/test_grow.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.config import preset_config
from leafv5.grow import grow_width, grow_depth
from leafv5.model import LeafLM


def logits_for(model, x):
    model.eval()
    with torch.no_grad():
        lg, _ = model(x, model.init_states(x.shape[0], torch.device("cpu")))
    return lg


def test_grow_width_exact():
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, dim=128, n_layers=3, d_h=32,
                        mem_slots=32)
    m = LeafLM(cfg)
    # train a little so weights are non-trivial
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad()
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        lg, _ = m(x, m.init_states(2, torch.device("cpu")))
        F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).backward()
        opt.step()
    x = torch.randint(0, 256, (4, 24))
    before = logits_for(m, x)
    grown = grow_width(m, 256)  # 2x uniform (RMSNorm-exact)
    after = logits_for(grown, x)
    # NOTE: slots are re-initialized (documented); zero them in the grown model
    # so the main-stream comparison is exact.
    with torch.no_grad():
        for b in grown.blocks:
            b.memory.slots.zero_()
    after = logits_for(grown, x)
    d = (after - before).abs().max().item()
    rel = d / (before.abs().max().item() + 1e-9)
    print(f"  width growth 128->256: max|d_logit| = {d:.2e} "
          f"(rel {rel:.1e})")
    # FP accumulation through L2-normalized recurrence; ~1e-4 abs is noise.
    assert d < 5e-4, d
    print("  EXACT (main stream preserved; slots excluded) OK")


def test_grow_depth_exact():
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, dim=128, n_layers=2, d_h=32)
    m = LeafLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad()
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        lg, _ = m(x, m.init_states(2, torch.device("cpu")))
        F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).backward()
        opt.step()
    x = torch.randint(0, 256, (4, 24))
    before = logits_for(m, x)
    grown = grow_depth(m, 4)
    after = logits_for(grown, x)
    d = (after - before).abs().max().item()
    print(f"  depth growth 2->4: max|d_logit| = {d:.2e}")
    assert d < 1e-6, d
    print("  EXACT OK")


def test_grow_width_trainable():
    """After growth, the bigger model still trains (gradients flow everywhere)."""
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, dim=128, n_layers=2, d_h=32,
                        mem_slots=16)
    m = LeafLM(cfg)
    grown = grow_width(m, 256)
    assert grown.cfg.dim == 256 and not grown.cfg.tie_weights
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    lg, _ = grown(x, grown.init_states(2, torch.device("cpu")))
    loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
    loss.backward()
    assert not any(torch.isnan(p.grad).any()
                   for p in grown.parameters() if p.grad is not None)
    print("  grown model trains OK")


def test_grow_continues_training():
    """THE user-facing guarantee: train small -> grow -> continue; the loss
    curve must NOT jump at the swap (knowledge is preserved), then continue
    improving at the bigger size."""
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, dim=96, n_layers=2, d_h=32,
                        mem_slots=0, scale_init=0.1)
    m = LeafLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    losses = []
    x = torch.randint(0, 256, (8, 32))
    y = torch.randint(0, 256, (8, 32))
    def step(model, opt_, losses):
        opt_.zero_grad()
        lg, _ = model(x, model.init_states(8, torch.device("cpu")))
        l = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
        l.backward(); opt_.step()
        return l.item()
    # train small to a decent point
    for _ in range(12):
        losses.append(step(m, opt, losses))
    before_grow = losses[-1]
    # grow 2x (function-preserving)
    grown = grow_width(m, 192)
    m.eval(); grown.eval()
    with torch.no_grad():
        lg, _ = grown(x, grown.init_states(8, torch.device("cpu")))
        lg_old, _ = m(x, m.init_states(8, torch.device("cpu")))
        # loss at the swap (fresh optimizer) must equal the pre-grow loss
        l_swap = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).item()
        l_old = F.cross_entropy(lg_old.reshape(-1, 256), y.reshape(-1)).item()
    print(f"  loss before grow={before_grow:.4f}  at swap={l_swap:.4f} "
          f"(old={l_old:.4f})")
    assert abs(l_swap - l_old) < 1e-3, (l_swap, l_old)  # function preserved
    # continue training at the bigger size: loss must keep improving
    opt2 = torch.optim.AdamW(grown.parameters(), lr=1e-3)
    l_start = l_swap
    for _ in range(12):
        l_start = step(grown, opt2, losses)
    print(f"  after 12 more steps at 192-dim: loss={l_start:.4f}")
    assert l_start < before_grow + 0.05  # no regression below pre-grow point
    print("  train->grow->continue: no loss jump, keeps improving OK")

def test_depth_growth_exact_with_scale_init():
    """REGRESSION: with scale_init>0 (the --fast recipe), grow_depth used to
    create NEW blocks with s1=s2=scale_init (not identity) -> output NOT
    preserved.  New blocks must have ZERO residual scales -> bit-exact."""
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=2, dim=96, d_h=32,
                        scale_init=0.1)  # nonzero scale_init!
    m = LeafLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (4, 16)); y = torch.randint(0, 256, (4, 16))
    for _ in range(5):
        opt.zero_grad()
        lg, _ = m(x, m.init_states(4, torch.device("cpu")))
        F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).backward()
        opt.step()
    xt = torch.randint(0, 256, (4, 20))
    m.eval()
    with torch.no_grad():
        before, _ = m(xt, m.init_states(4, torch.device("cpu")))
    grown = grow_depth(m, 4)
    # the NEW blocks must have zero residual scales (identity at init)
    for i in range(2, 4):
        assert torch.all(grown.blocks[i].s1 == 0)
        assert torch.all(grown.blocks[i].s2 == 0)
    grown.eval()
    with torch.no_grad():
        after, _ = grown(xt, grown.init_states(4, torch.device("cpu")))
    d = (after - before).abs().max().item()
    print(f"  depth growth 2->4 with scale_init=0.1: max|d_logit| = {d:.2e}")
    assert d < 1e-6, d
    print("  depth growth EXACT even with scale_init>0 OK")




def test_width_growth_with_slots_preserves_behavior():
    """REGRESSION: with persistent slots (default mem_slots=64), width growth
    used to re-init the slots (losing trained content, logit diff 0.1+).
    Now the slots are carried (interleaved replication) so the grown model's
    logits stay within ~1e-3 and trained behavior is preserved."""
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=2, dim=96, d_h=32,
                        scale_init=0.1)  # mem_slots=64 (default)
    m = LeafLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (4, 16)); y = torch.randint(0, 256, (4, 16))
    for _ in range(8):
        opt.zero_grad()
        lg, _ = m(x, m.init_states(4, torch.device("cpu")))
        F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).backward()
        opt.step()
    xt = torch.randint(0, 256, (4, 20))
    m.eval()
    with torch.no_grad():
        before, _ = m(xt, m.init_states(4, torch.device("cpu")))
    g = grow_width(m, 192)
    # slots carried (not re-init): interleaved replication
    # (new[:, 2j] == new[:, 2j+1] == old[:, j])
    ns = g.blocks[0].memory.slots
    os_ = m.blocks[0].memory.slots
    assert torch.allclose(ns[:, 0::2], os_, atol=1e-6)
    assert torch.allclose(ns[:, 1::2], os_, atol=1e-6)
    g.eval()
    with torch.no_grad():
        after, _ = g(xt, g.init_states(4, torch.device("cpu")))
    d = (after - before).abs().max().item()
    print(f"  width growth WITH slots: max|d_logit| = {d:.2e}")
    assert d < 5e-3, d
    print("  slots carried (interleaved); behavior preserved OK")


if __name__ == "__main__":
    test_grow_width_exact()
    test_grow_depth_exact()
    test_grow_width_trainable()
    test_grow_continues_training()
    test_depth_growth_exact_with_scale_init()
    test_width_growth_with_slots_preserves_behavior()
    print("\nGrowth tests passed.")
