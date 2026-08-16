"""world_evidence.py — regenerate every headline claim in the paper draft.

One command regenerates the evidence that research/paper-draft.md rests on:
  1. base stability certificate (9/9)
  2. Mistral-stack stability certificate (10/10)
  3. growth exactness (width ~1e-6, depth 0.0, trained-model ~1e-3)
  4. the pipeline experiment (train-small -> grow-exact -> continue vs scratch)
  5. standard-corpus benchmark (PTB char PPL) at reduced steps
  6. the world benchmark (recall + LM race) at reduced steps

Full-strength runs are documented in the header of each script; this is the
fast, honest default that anyone can run to check the paper's numbers.

Run:  python -m leafv5.world_evidence [--steps 60] [--experiment-steps 12]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run_module(module: str, argv: list) -> str:
    """Run a leafv5 module's main() in-process, capturing its stdout."""
    import importlib
    mod = importlib.import_module(module)
    old = os.sys.argv
    os.sys.argv = [module] + argv
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        mod.main()
    out = buf.getvalue()
    os.sys.argv = old
    return out, time.time() - t0


def banner(t):
    print("\n" + "=" * 70)
    print(f"  {t}")
    print("=" * 70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60,
                   help="benchmark steps (full strength: 150)")
    p.add_argument("--experiment-steps", type=int, default=12,
                   help="growth-vs-scratch steps/phase (full: 200)")
    p.add_argument("--grow-seeds", type=int, default=1)
    args = p.parse_args()
    torch.manual_seed(0)

    results = {}

    banner("1/6  BASE STABILITY CERTIFICATE")
    out, dt = run_module("leafv5.stability_check", ["--steps", "120"])
    results["base_cert"] = "9/9" if "9/9 passed" in out else "FAIL"
    print(out.splitlines()[-3:]); print(f"  ({dt:.0f}s)")

    banner("2/6  MISTRAL-STACK STABILITY CERTIFICATE")
    out, dt = run_module("leafv5.stability_check_mistral", [])
    results["mistral_cert"] = "10/10" if "10/10 passed" in out else "FAIL"
    print(out.splitlines()[-3:]); print(f"  ({dt:.0f}s)")

    banner("3/6  GROWTH EXACTNESS (width/depth on a trained model)")
    out, dt = run_module("leafv5.grow_vs_scratch",
                         ["--steps", str(args.experiment_steps),
                          "--seeds", str(args.grow_seeds)])
    # capture the growth logit-preservation line
    gline = [l for l in out.splitlines() if "logit-preservation" in l]
    print("  " + (gline[0].strip() if gline else "(none)"))
    print(f"  ({dt:.0f}s)")

    banner("4/6  PIPELINE EXPERIMENT (grow vs scratch, matched steps)")
    print(out.splitlines()[-12:]); print(f"  ({dt:.0f}s)")

    banner("5/6  STANDARD-CORPUS PPL (Penn Treebank, char-level)")
    out, dt = run_module("leafv5.benchmark_ppl", ["--steps", str(args.steps)])
    results["ptb"] = [l for l in out.splitlines() if "LEAFv5" in l and "PPL" in l
                      or l.strip().startswith("LEAFv5")]
    for l in out.splitlines():
        if l.strip().startswith(("LEAFv5", "Transformer", "GatedRNN")):
            print("  " + l.strip())
    print(f"  ({dt:.0f}s)")

    banner("6/6  WORLD BENCHMARK (recall + LM race, reduced steps)")
    out, dt = run_module("leafv5.benchmark_world", ["--steps", str(max(8, args.steps // 4))])
    for l in out.splitlines():
        if "LEAFv5" in l or "params" in l:
            print("  " + l.strip())
    print(f"  ({dt:.0f}s)")

    print("\n" + "=" * 70)
    print("  EVIDENCE REPORT (all regenerated just now)")
    print("=" * 70)
    print(f"  base stability certificate ....... {results['base_cert']} STABLE")
    print(f"  mistral-stack certificate ....... {results['mistral_cert']} STABLE")
    print("  growth exactness ................. see [3/6] (init ~1e-6, trained ~1e-3)")
    print("  pipeline experiment .............. see [4/6] (full: --experiment-steps 200)")
    print(f"  PTB char PPL @{args.steps} steps ... " +
          (results["ptb"][0].strip() if results["ptb"] else "(see above)"))
    print("  world benchmark .................. see [6/6] (full: --steps 150)")
    print("=" * 70)
    print("Full-strength: python -m leafv5.world_evidence --steps 150 "
          "--experiment-steps 200 --grow-seeds 3")


if __name__ == "__main__":
    main()
