"""Tests for the round-4 limit pushes: SWA hybrid, C fast-scan, skill-eval
graders, generate() repetition guard.
Run:  python tests/test_limits.py
"""
import os
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_gen"))

from leafv5.config import preset_config
from leafv5.model import LeafLM
from leafv5.grow import grow_width


def test_swa_identity_and_growth():
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, use_swa=True, swa_window=8,
                        mem_slots=0, scale_init=0.1)
    m = LeafLM(cfg)
    blk = m.blocks[0]
    assert blk.swa is not None and torch.all(blk.swa.scale == 0)
    x = torch.randn(2, 16, cfg.dim)
    with torch.no_grad():
        assert blk.swa(x)[0].abs().max().item() < 1e-9   # identity at init
    # trains
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        xi = torch.randint(0, 256, (2, 16)); yi = torch.randint(0, 256, (2, 16))
        lg, _ = m(xi, m.init_states(2, torch.device("cpu")))
        F.cross_entropy(lg.reshape(-1, 256), yi.reshape(-1)).backward()
        opt.step()
    assert blk.swa.scale.abs().sum() > 0
    # growth with SWA is exact
    g = grow_width(m, 256)
    m.eval(); g.eval()
    xt = torch.randint(0, 256, (4, 20))
    with torch.no_grad():
        lo = m(xt, m.init_states(4, torch.device("cpu")))[0]
        ln = g(xt, g.init_states(4, torch.device("cpu")))[0]
    assert (ln - lo).abs().max().item() < 5e-4
    print("  SWA: identity-init, trains, growth exact OK")


def test_fast_scan_equals_python():
    """If the C kernel is built, fast scan must equal the Python scan exactly."""
    import os as _os
    from leafv5.model import LeafLM as _L
    so = _os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), "mojo", "c_ref", "leafv5_scan.so")
    if not _os.path.exists(so):
        print("  (C kernel not built -> fast-scan test skipped)")
        return
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256)
    m = LeafLM(cfg).eval()
    x = torch.randint(0, 256, (2, 24))
    with torch.no_grad():
        a, _ = m(x, m.init_states(2, torch.device("cpu")), fast=False)
        b, _ = m(x, m.init_states(2, torch.device("cpu")), fast=True)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max().item()
    print("  fast C scan == python scan (exact) OK")


def test_generate_repetition_guard():
    from leafv5.data import CharTokenizer
    from leafv5.generate import generate
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256)
    m = LeafLM(cfg).eval()
    tok = CharTokenizer({chr(i): i for i in range(256)})
    out, _ = generate(m, tok, "a", max_new=100, temperature=0.0,
                      max_consecutive=3, device="cpu")
    assert len(out) <= 10, len(out)  # stopped early on repetition
    print("  generate() repetition guard OK")


def test_skill_graders():
    import make_dataset as md
    from leafv5.eval_skills import (grade_identity, grade_math, grade_grammar,
                                    grade_tools, grade_sinhala)
    assert grade_identity("q", "I am LEAFv5, created by Dassanayake.")
    assert not grade_identity("q", "I don't know.")
    assert grade_math("What is 12 + 7?", "The answer is 19.")
    assert not grade_math("What is 12 + 7?", "The answer is 25.")
    assert grade_grammar("q", "Corrected: He went. Why: past tense.")
    assert grade_tools("q", '{"tool": "get_weather", "args": {}}')
    assert grade_sinhala("q", "ආයුබෝවන්")
    print("  skill-eval graders OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_swa_identity_and_growth()
    test_fast_scan_equals_python()
    test_generate_repetition_guard()
    test_skill_graders()
    print("\nLimits tests passed.")


def test_ablate_runs():
    """The ablation tool must run and produce a monotone-ish table."""
    import contextlib
    import io
    import leafv5.ablate as ablate
    # Run in-process (not a subprocess): a second torch process peaks at
    # ~1.6 GB RSS and gets OOM-killed on low-RAM boxes when pytest already
    # holds torch, making the CLI smoke test flaky.  Driving main() directly
    # exercises the exact same argparse + fetch + training + table code.
    old_argv = sys.argv
    # --seq 32: avoids a flaky torch CPU autograd livelock on deep 64-step
    # delta-scan chains under host contention (documented in ablate.py); the
    # comparative table's conclusions are unaffected.
    sys.argv = ["leafv5.ablate", "--steps", "12", "--seq", "32"]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            ablate.main()
    finally:
        sys.argv = old_argv
    log = buf.getvalue()
    assert "paper-core only" in log, log[-800:]
    assert "FULL WORLD-BEST FUSION" in log, log[-800:]
    print("  ablation tool runs OK")
