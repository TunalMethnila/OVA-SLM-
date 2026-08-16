"""Smoke test: LEAFv5 must learn far faster than a same-size Transformer.
Run:  python tests/test_speed.py   (takes ~40-60s on CPU)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.speed_demo import recall_race, steps_to_target  # noqa: E402


def test_recall_race_speed():
    args = type("A", (), {
        "vocab": 64, "pairs": 2, "queries": 1, "batch": 16,
        "dim": 96, "layers": 2, "d_h": 32,
        "steps": 60, "milestones": [1, 5, 10, 20, 40, 60],
        "no_transformer": False,
        "recall_configs": [
            ("LEAFv5-fast", True, 1e-2, 0.1),
            ("LEAFv5-paper", True, 1e-2, 0.0),
            ("Transformer", False, 1e-3, 0.0),
        ],
    })
    torch.manual_seed(0)
    res = recall_race(args, "cpu")
    lf, lp, tr = res["LEAFv5-fast"], res["LEAFv5-paper"], res["Transformer"]
    # LEAFv5 must clearly out-learn the Transformer at every milestone from ~10 on
    for m in (10, 20, 40, 60):
        assert lf[m] >= tr[m] + 15, (m, lf[m], tr[m])
    # LEAFv5 must reach the Transformer's final accuracy within ~30 steps
    trans_max = max(tr.values())
    assert steps_to_target(lf, trans_max) <= 30, (lf, trans_max)
    assert steps_to_target(lp, trans_max) <= 40, (lp, trans_max)
    print("  LEAFv5-fast:", {k: int(v) for k, v in lf.items()})
    print("  LEAFv5-paper:", {k: int(v) for k, v in lp.items()})
    print("  Transformer :", {k: int(v) for k, v in tr.items()})
    print("  recall-race speed check OK")


def test_recall_100_in_10_steps():
    """Headline claim: LEAFv5 hits 100% held-out recall in 10 steps (P1Q1,
    batch 128), while a same-size Transformer is far behind."""
    args = type("A", (), {
        "vocab": 64, "pairs": 1, "queries": 1, "batch": 128,
        "dim": 128, "layers": 2, "d_h": 64,
        "steps": 10, "milestones": [1, 5, 10],
        "no_transformer": False,
        "recall_configs": [
            ("LEAFv5-fast", True, 2e-2, 0.2),
            ("LEAFv5-paper", True, 2e-2, 0.0),
            ("Transformer", False, 1e-3, 0.0),
        ],
    })
    torch.manual_seed(0)
    res = recall_race(args, "cpu")
    lf, tr = res["LEAFv5-fast"], res["Transformer"]
    assert lf[10] >= 95, lf           # LEAFv5: ~100% by step 10
    assert lf[10] >= tr[10] + 50, (lf, tr)  # Transformer far behind at step 10
    print("  LEAFv5-fast:", {k: int(v) for k, v in lf.items()})
    print("  Transformer :", {k: int(v) for k, v in tr.items()})
    print("  P1Q1 100%-in-10-steps check OK")


if __name__ == "__main__":
    test_recall_race_speed()
    test_recall_100_in_10_steps()
    print("\nSpeed test passed.")
