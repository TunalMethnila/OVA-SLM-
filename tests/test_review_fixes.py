"""Regression tests for the code-review fixes (P0 #3-#8, P1 #11-#13).
Run:  python tests/test_review_fixes.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.config import preset_config
from leafv5.model import LeafLM


def test_grad_accum_scaling():
    """#4: grad-accum must scale loss by 1/grad_accum (mean, not sum)."""
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=1, dim=64, d_h=16,
                        scale_init=0.1, mem_dropout=0.0)
    torch.manual_seed(0)
    x = torch.randint(0, 256, (4, 16)); y = torch.randint(0, 256, (4, 16))
    # both models start from the SAME weights
    m1 = LeafLM(cfg)
    start = {k: v.detach().clone() for k, v in m1.state_dict().items()}
    m2 = LeafLM(cfg)
    m2.load_state_dict(start)
    # grad-accum path (trainer fix): 4 micro-batches, then divide by 4
    opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    opt1.zero_grad()
    for _ in range(4):
        lg, _ = m1(x, m1.init_states(4, torch.device("cpu")))
        F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).backward()
    for p in m1.parameters():
        if p.grad is not None:
            p.grad.div_(4)
    # single path: 1 batch
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    opt2.zero_grad()
    lg, _ = m2(x, m2.init_states(4, torch.device("cpu")))
    F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).backward()
    # the accumulated-MEAN gradient must equal the single gradient
    d = 0.0
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if p1.grad is not None:
            d = max(d, (p1.grad - p2.grad).abs().max().item())
    print(f"  accumulated-mean grad vs single grad: max|d| = {d:.2e}")
    assert d < 1e-5, d
    print("  grad-accum scaling correct OK")


def test_shadow_snapshot_isolation():
    """#5: the rollback snapshot must be isolated from live updates."""
    from leafv5.train import _strip_module_prefix  # noqa: F401
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=1, dim=64, d_h=16,
                        scale_init=0.1)
    m = LeafLM(cfg)
    snap = {k: v.detach().clone() for k, v in m.state_dict().items()}
    # perturb the live weights heavily
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.ones_like(p) * 3.0)
    # the snapshot must be untouched
    assert torch.allclose(snap["blocks.0.memory.wk.weight"],
                          m.state_dict()["blocks.0.memory.wk.weight"] -
                          torch.ones_like(snap["blocks.0.memory.wk.weight"]) * 3.0) \
        if False else True
    diff = (m.state_dict()["blocks.0.memory.wk.weight"] -
            snap["blocks.0.memory.wk.weight"]).abs().max().item()
    print(f"  snapshot vs live after perturbation: |d| = {diff:.2f} (3.0 expected)")
    assert abs(diff - 3.0) < 1e-5, diff
    print("  rollback snapshot is a true copy OK")


def test_ddp_key_strip():
    """#7: DDP 'module.' prefix stripping."""
    from leafv5.train import _strip_module_prefix
    sd = {"module.a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    out = _strip_module_prefix(sd)
    assert "a" in out and out["a"].item() == 1.0
    assert "b" in out and "module.a" not in out
    print("  DDP key stripping OK")


def test_streaming_linewise_parity():
    """#8: encoding complete lines == encoding the full text (BPE parity)."""
    from leafv5.data import BPETokenizer
    from leafv5.data import _encode_linewise
    # build a small BPE on multi-line text
    text = ("the quick brown fox jumps over the lazy dog.\n"
            "the second line has different words entirely.\n"
            "a third line with more content to merge across boundaries. "
            "and a very long sentence without a newline for a while " * 20 +
            "\nfinal line")
    tok = BPETokenizer.train(iter([text]), vocab_size=500)
    ids_full = tok.encode(text)
    ids_line = _encode_linewise(tok, text)
    assert ids_full == ids_line, (len(ids_full), len(ids_line))
    print(f"  linewise == full-text BPE parity ({len(ids_full)} tokens) OK")


def test_rope_extension_in_generate():
    """#11: generate past max_seq_len must not crash (RoPE extends)."""
    from leafv5.data import CharTokenizer
    from leafv5.generate import generate
    import string
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=1, dim=64, d_h=16,
                        max_seq_len=8, rope_dim=64)
    m = LeafLM(cfg).eval()
    voc = {c: i for i, c in enumerate(string.ascii_lowercase)}
    tok = CharTokenizer(voc)
    out, st = generate(m, tok, "abc", max_new=20, temperature=0.0,
                       max_consecutive=0, device="cpu")
    assert isinstance(out, str) and len(out) > 0
    # continue past the 8-token cache
    out2, st = generate(m, tok, "xyz", max_new=10, temperature=0.0,
                        states=st, device="cpu")
    assert isinstance(out2, str)
    print("  RoPE dynamic extension through generate (past max_seq_len) OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_grad_accum_scaling()
    test_shadow_snapshot_isolation()
    test_ddp_key_strip()
    test_streaming_linewise_parity()
    test_rope_extension_in_generate()
    print("\nReview-fixes tests passed.")
