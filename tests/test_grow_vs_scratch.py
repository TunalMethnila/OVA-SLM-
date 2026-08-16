"""Smoke tests for the pipeline experiment (grow_vs_scratch).

These guard the headline claim in research/paper-draft.md §3: the experiment
runs, growth preserves the function, and the compute ratio is < 1 (growth is
cheaper than scratch by construction).
Run:  python tests/test_grow_vs_scratch.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.config import preset_config
from leafv5.grow_vs_scratch import flops_per_token, run
from leafv5.grow import grow_width, grow_depth
from leafv5.model import LeafLM


def test_flops_ratio_monotonic():
    """Bigger model -> more FLOPs/token; growing is cheaper than scratch."""
    torch.manual_seed(0)
    V = 128
    f0 = preset_config("micro", vocab_size=V, n_layers=2, dim=128, d_h=48,
                       rope_dim=0, scale_init=0.1)
    f1 = preset_config("micro", vocab_size=V, n_layers=4, dim=256, d_h=48,
                       rope_dim=0, scale_init=0.1)
    fl_s, fl_b = flops_per_token(LeafLM(f0)), flops_per_token(LeafLM(f1))
    assert fl_s < fl_b, (fl_s, fl_b)
    assert 0 < fl_s and 0 < fl_b
    print(f"  FLOPs/token: small={fl_s/1e6:.2f}M big={fl_b/1e6:.2f}M "
          f"(ratio {fl_s/fl_b:.2f}) OK")


def test_pipeline_smoke_runs_and_grows_exact():
    """Tiny run of the full pipeline: trains, grows exactly, returns sane
    losses and a compute ratio < 1."""
    torch.manual_seed(0)
    r = run(seed=0, steps=6, bs=8, seq=16)
    assert r["growth_d"] < 5e-3, r["growth_d"]      # function preserved at swap
    assert r["flops_ratio"] < 1.0, r["flops_ratio"] # growth is cheaper
    assert r["A_final_loss"] < r["A_small_loss"] + 0.5  # no quality cliff
    assert torch.isfinite(torch.tensor([r["A_final_loss"], r["B_final_loss"]])).all()
    print(f"  pipeline smoke: growth_d={r['growth_d']:.2e} "
          f"flops_ratio={r['flops_ratio']:.2f} A={r['A_final_loss']:.3f} "
          f"B={r['B_final_loss']:.3f} OK")


def test_width_depth_growth_exact_on_trained_model():
    """Width+depth growth of a TRAINED model preserves the function (the
    honest, post-training number, not just init)."""
    torch.manual_seed(0)
    V = 256
    cfg = preset_config("micro", vocab_size=V, n_layers=2, dim=128, d_h=32,
                        scale_init=0.1)
    m = LeafLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, V, (4, 16)); y = torch.randint(0, V, (4, 16))
    for _ in range(4):
        opt.zero_grad()
        lg, _ = m(x, m.init_states(4, torch.device("cpu")))
        torch.nn.functional.cross_entropy(lg.reshape(-1, V), y.reshape(-1)).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        before = m(x, m.init_states(4, torch.device("cpu")))[0]
    g = grow_depth(grow_width(m, 256), 4)
    g.eval()
    with torch.no_grad():
        after = g(x, g.init_states(4, torch.device("cpu")))[0]
    d = (after - before).abs().max().item()
    assert d < 5e-3, d
    print(f"  trained-model width+depth growth: max|d|={d:.2e} OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_flops_ratio_monotonic()
    test_pipeline_smoke_runs_and_grows_exact()
    test_width_depth_growth_exact_on_trained_model()
    print("\nGrowth-pipeline smoke tests passed.")
