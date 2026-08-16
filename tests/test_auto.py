"""Tests for auto-config (train-on-any-GPU) and learned plasticity."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.auto import pick_config  # noqa: E402
from leafv5.config import preset_config  # noqa: E402
from leafv5.model import LeafLM  # noqa: E402


def test_auto_config_matrix():
    # T4 (16 GB, cc 7.5) -> t4-4h, fp16, chunked, compile
    c = pick_config(True, 16, 7.5)
    assert c["model"] == "t4-4h" and c["dtype"] == "fp16" and c["scan"] == "chunked"
    # A100 (80 GB, cc 8.0) -> t4-xl, bf16
    c = pick_config(True, 80, 8.0)
    assert c["model"] == "t4-xl" and c["dtype"] == "bf16"
    # 3090 (24 GB, cc 8.6) -> t4-xl, bf16
    c = pick_config(True, 24, 8.6)
    assert c["model"] == "t4-xl" and c["dtype"] == "bf16"
    # 8 GB laptop GPU -> t4-fast, fp16
    c = pick_config(True, 8, 7.5)
    assert c["model"] == "t4-fast" and c["dtype"] == "fp16"
    # CPU -> tiny, fp32, sequential, no compile
    c = pick_config(False, None, None)
    assert c["model"] == "tiny" and c["dtype"] == "fp32" \
        and c["scan"] == "sequential" and c["compile"] is False
    # Apple MPS -> tiny, fp32
    c = pick_config(False, None, None, has_mps=True)
    assert c["model"] == "tiny" and c["dtype"] == "fp32"
    print("  auto-config matrix OK:", {k: v for k, v in c.items() if k != "kind"})


def test_learned_plasticity():
    # nonzero scale_init so the memory branch gets gradient immediately
    # (paper's zero-init highways create a step-1 dead zone by design)
    cfg = preset_config("micro", vocab_size=256, learn_plasticity=True,
                        scale_init=0.1)
    m = LeafLM(cfg)
    mem = m.blocks[0].memory
    assert isinstance(mem.write_mult, torch.nn.Parameter)
    assert mem.write_mult.requires_grad
    # initialized to the group values (fast=1.0, medium=0.6, slow=0.3)
    gid = torch.cat([torch.full((n,), i) for i, n in enumerate(cfg.groups)]).long()
    assert torch.allclose(mem.write_mult[gid == 0], torch.tensor(1.0))
    assert torch.allclose(mem.write_mult[gid == 1], torch.tensor(0.6))
    # a few steps: forward + backward must update the plasticity params
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        x = torch.randint(0, 256, (2, 8))
        y = torch.randint(0, 256, (2, 8))
        lg, _ = m(x, m.init_states(2, torch.device("cpu")))
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        opt.step()
    assert mem.write_mult.grad is not None and mem.write_mult.grad.abs().sum() > 0
    # fixed mode stays a non-parameter buffer
    cfg2 = preset_config("micro", vocab_size=256)
    m2 = LeafLM(cfg2)
    assert not isinstance(m2.blocks[0].memory.write_mult, torch.nn.Parameter)
    print("  learned plasticity forward/backward OK")


if __name__ == "__main__":
    test_auto_config_matrix()
    test_learned_plasticity()
    print("\nAuto-config + plasticity tests passed.")
