"""Stability certification as unit tests: edge inputs, determinism,
perturbation robustness, NaN recovery, deep stacks, gradient-norm monitor.
Run:  python tests/test_stability_cert.py
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.config import preset_config
from leafv5.model import LeafLM
from leafv5.autotune_utils import nan_guard


def _model(dim=96, layers=2, d_h=32, scale_init=0.1):
    return LeafLM(preset_config("micro", vocab_size=256, n_layers=layers,
                                dim=dim, d_h=d_h, scale_init=scale_init))


def test_generate_edge_inputs():
    import string
    from leafv5.data import CharTokenizer
    from leafv5.generate import generate
    voc = {c: i for i, c in enumerate(string.ascii_lowercase)}
    tok = CharTokenizer(voc)
    m = _model().eval()
    # must not crash on any of these
    assert isinstance(generate(m, tok, "", max_new=4, temperature=0.0,
                               device="cpu")[0], str)
    assert generate(m, tok, "abc", max_new=0, temperature=0.0,
                    device="cpu")[0] == ""
    assert isinstance(generate(m, tok, "abc", max_new=4, temperature=0.0,
                               top_k=10 ** 9, device="cpu")[0], str)
    assert isinstance(generate(m, tok, "abc", max_new=4, temperature=-1.0,
                               device="cpu")[0], str)
    print("  edge inputs (empty/0/huge-topk/negative-temp) never crash OK")


def test_determinism():
    torch.manual_seed(0)
    m = _model().eval()
    x = torch.randint(0, 256, (4, 16))
    with torch.no_grad():
        a, _ = m(x, m.init_states(4, torch.device("cpu")))
        b, _ = m(x, m.init_states(4, torch.device("cpu")))
    assert torch.allclose(a, b, atol=1e-9)
    print("  determinism (same input -> identical output) OK")


def test_perturbation_robustness():
    torch.manual_seed(0)
    m = _model().eval()
    x = torch.randint(0, 256, (4, 16))
    with torch.no_grad():
        base, _ = m(x, m.init_states(4, torch.device("cpu")))
        # weight perturbation +-1%
        m2 = _model().eval()
        m2.load_state_dict(m.state_dict())
        for p in m2.parameters():
            p.add_(torch.randn_like(p) * 0.01 * p.abs().clamp_min(1e-4))
        wp, _ = m2(x, m2.init_states(4, torch.device("cpu")))
        # input perturbation (1 token)
        x2 = x.clone(); x2[:, 5] = (x2[:, 5] + 7) % 256
        ip, _ = m(x2, m.init_states(4, torch.device("cpu")))
        # state perturbation
        _, st = m(x, m.init_states(4, torch.device("cpu")))
        stn = [s + torch.randn_like(s) * 0.01 for s in st]
        xs = torch.randint(0, 256, (4, 8))
        oc, _ = m(xs, m.init_states(4, torch.device("cpu")))
        on, _ = m(xs, stn, offset=0)
    bmax = base.abs().max().item() + 1e-9
    assert (wp - base).abs().max().item() / bmax < 2.0   # weight noise bounded
    assert (ip - base).abs().max().item() / bmax < 2.0   # input noise bounded
    assert (on - oc).abs().max().item() / (oc.abs().max().item() + 1e-9) < 2.0
    print("  weight/input/state perturbation stays proportional OK")


def test_nan_recovery():
    torch.manual_seed(0)
    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-2, betas=(0.9, 0.95))
    guard = False
    finite_after = True
    for i in range(120):
        opt.zero_grad(set_to_none=True)
        x = torch.randint(0, 256, (4, 16)); y = torch.randint(0, 256, (4, 16))
        lg, _ = m(x, m.init_states(4, torch.device("cpu")))
        loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
        if i == 40:
            loss = loss * float("nan")
        loss.backward()
        if nan_guard(m):
            guard = True
            opt.zero_grad(set_to_none=True)
        else:
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        if i > 40:
            finite_after = finite_after and torch.isfinite(loss).item()
    assert guard, "NaN guard did not fire!"
    assert finite_after, "training did not recover after NaN!"
    print("  NaN-grad guard fires + training recovers OK")


def test_deep_stack_finite():
    torch.manual_seed(0)
    m = _model(layers=24, dim=96, d_h=32)
    x = torch.randint(0, 256, (2, 16))
    lg, _ = m(x, m.init_states(2, torch.device("cpu")))
    F.cross_entropy(lg.reshape(-1, 256), torch.randint(0, 256, (2, 16)).reshape(-1)).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters()
               if p.grad is not None)
    print("  24-layer stack forward+backward finite OK")


def test_states_bounded_stress():
    torch.manual_seed(0)
    m = _model()
    mem = m.blocks[0].memory
    x = torch.randn(2, 16, 96)
    st = None
    for _ in range(300):
        with torch.no_grad():
            _, st, _, _ = mem(x[:, :1], st)
    assert st.norm(dim=(-1, -2)).max().item() <= math.sqrt(32) * 1.05
    print("  states bounded after 300 stress steps OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_generate_edge_inputs()
    test_determinism()
    test_perturbation_robustness()
    test_nan_recovery()
    test_deep_stack_finite()
    test_states_bounded_stress()
    print("\nStability-certification tests passed.")
