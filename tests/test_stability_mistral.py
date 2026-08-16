"""Mistral-stack stability certification as unit tests.

Guards every property in leafv5/stability_check_mistral.py (GQA, rolling
buffer, pre-fill & chunking, Mixtral MoE):
  1. boundary exactness (rolling == tuple at every step, 3+ window wraps)
  2. position-offset prefill (regression: window() start bug, 2026-08-09)
  3. determinism (rolling/prefill/MoE/full-model bit-identical)
  4. long decode (3000 tokens) finite + bounded + storage constant
  5. edge cases (W=1, heads=1, kv=1, T==W, T==W+1, batch 1/7) + guards
  6. MoE stability (100 steps, aux loss in range, all experts used)
  7. 12-layer stack + SWA/GQA/MoE forward+backward finite
  8. chunked prefill exact (chunk 1/2/3/7/W == one-shot)
  9. low precision (bf16 rolling==tuple exact; fp16 finite)
  10. train==decode with SWA+GQA after training (reviewer's invariant)
Run:  python tests/test_stability_mistral.py
"""
import contextlib
import io
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.stability_check_mistral import (  # noqa: E402
    CHECKS, main, check_boundary_exactness, check_chunked_prefill_exact,
    check_deep_stack_new_features, check_determinism, check_edge_cases_and_guards,
    check_long_decode_bounded, check_low_precision, check_moe_stability,
    check_position_offset_prefill, check_train_equals_decode_gqa)


def _seed():
    torch.manual_seed(0)


def test_boundary_and_offset_exactness():
    _seed()
    assert check_boundary_exactness("cpu")
    assert check_position_offset_prefill("cpu")
    print("  boundary + position-offset exactness OK")


def test_determinism_and_long_decode():
    _seed()
    assert check_determinism("cpu")
    assert check_long_decode_bounded("cpu")
    print("  determinism + 3000-token bounded decode OK")


def test_edge_cases_and_guards():
    _seed()
    assert check_edge_cases_and_guards("cpu")
    print("  edge cases + guards OK")


def test_moe_stability():
    _seed()
    assert check_moe_stability("cpu")
    print("  MoE stability OK")


def test_deep_stack_new_features():
    _seed()
    assert check_deep_stack_new_features("cpu")
    print("  12-layer SWA/GQA/MoE stack finite OK")


def test_chunked_prefill_exact():
    _seed()
    assert check_chunked_prefill_exact("cpu")
    print("  chunked prefill exact OK")


def test_low_precision():
    _seed()
    assert check_low_precision("cpu")
    print("  bf16/fp16 stable OK")


def test_train_equals_decode_gqa():
    _seed()
    assert check_train_equals_decode_gqa("cpu")
    print("  train==decode with GQA after training OK")


def test_certificate_cli():
    """The certificate CLI must run and print STABLE."""
    old_argv = sys.argv
    sys.argv = ["leafv5.stability_check_mistral"]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            ok = main()
    finally:
        sys.argv = old_argv
    out = buf.getvalue()
    assert ok
    assert "MISTRAL-STACK STABILITY CERTIFICATE: 10/10 passed" in out, out[-500:]
    assert "RESULT: STABLE" in out
    print("  certificate CLI runs, 10/10 STABLE OK")


if __name__ == "__main__":
    _seed()
    for fn in (test_boundary_and_offset_exactness, test_determinism_and_long_decode,
               test_edge_cases_and_guards, test_moe_stability,
               test_deep_stack_new_features, test_chunked_prefill_exact,
               test_low_precision, test_train_equals_decode_gqa,
               test_certificate_cli):
        fn()
    print("\nMistral-stack stability tests passed.")
