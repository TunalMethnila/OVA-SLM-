"""Regression tests for the compact pack-file format (save_packed/load_packed)
and the pack/unpack round-trip — added after the 2026-08-13 audit found that
torch.save of the packed dict produced a BIGGER file than fp32.
Run:  python tests/test_weights_pack.py
"""
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.config import preset_config
from leafv5.model import LeafLM
from leafv5.weights import (load_packed, pack_model, save_packed,
                            unpack_model)


def test_roundtrip_in_memory():
    """pack -> unpack must reproduce the model within int8-residual error."""
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=4096, dim=768, n_layers=2,
                        d_h=48, mem_slots=0)
    m = LeafLM(cfg)
    packed = pack_model(m.state_dict(), rank=0, quant_residual=True, shared=True)
    sd = unpack_model(packed)
    m2 = LeafLM(cfg)
    m2.load_state_dict(sd)
    x = torch.randint(0, 4096, (2, 8))
    with torch.no_grad():
        a, _ = m(x, m.init_states(2, torch.device("cpu")))
        b, _ = m2(x, m2.init_states(2, torch.device("cpu")))
    # int8 residual => small logit drift (documented); must not be garbage
    assert (a - b).abs().max().item() < 0.1, (a - b).abs().max().item()
    print("  in-memory pack/unpack round-trip OK")


def test_compact_file_smaller_than_fp32():
    """THE regression: save_packed must produce a SMALLER file than fp32
    (torch.save of the packed dict used to be 1.6x BIGGER)."""
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=4096, dim=768, n_layers=2,
                        d_h=48, mem_slots=0)
    m = LeafLM(cfg)
    packed = pack_model(m.state_dict(), rank=0, quant_residual=True,
                        shared=True)
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "m.pt")
        pk = os.path.join(td, "m.pk")
        torch.save(m.state_dict(), fp)
        save_packed(packed, pk)
        ratio = os.path.getsize(fp) / os.path.getsize(pk)
        assert ratio > 2.0, f"packed file only {ratio:.2f}x smaller"
        print(f"  file: {ratio:.2f}x smaller (fp32 {os.path.getsize(fp)/1e6:.1f} "
              f"MB -> {os.path.getsize(pk)/1e6:.1f} MB) OK")


def test_file_survives_process_boundary():
    """load_packed must work in a NEW process (the format is self-contained)."""
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=1024, dim=256, n_layers=1,
                        d_h=32, mem_slots=0)
    m = LeafLM(cfg)
    packed = pack_model(m.state_dict(), rank=0, quant_residual=True,
                        shared=True)
    with tempfile.TemporaryDirectory() as td:
        pk = os.path.join(td, "m.pk")
        save_packed(packed, pk)
        code = (
            "import torch, sys; sys.path.insert(0, %r); "
            "from leafv5.weights import load_packed, unpack_model; "
            "sd = unpack_model(load_packed(%r)); print(len(sd))"
            % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), pk)
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-500:]
        assert int(r.stdout.strip()) == len(m.state_dict()), r.stdout
    print("  pack file loads in a new process OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_roundtrip_in_memory()
    test_compact_file_smaller_than_fp32()
    test_file_survives_process_boundary()
    print("\nWeights pack-file tests passed.")
