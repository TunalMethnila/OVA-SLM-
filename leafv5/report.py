"""One-command project report: runs the key LEAFv5 benchmarks and writes a
Markdown report.md with the measured numbers.

Run:  python -m leafv5.report [--quick]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="tiny steps (fast smoke; real numbers come from full runs)")
    p.add_argument("--out", default="report.md")
    args = p.parse_args()

    steps = 10 if args.quick else 120
    report = []
    report.append("# LEAFv5 project report\n")
    report.append(f"generated: {time.strftime('%Y-%m-%d %H:%M')} "
                  f"({'quick' if args.quick else 'full'})\n")

    def run(mod, extra=None, label=None):
        cmd = [sys.executable, "-m", f"leafv5.{mod}"] + (extra or [])
        print(f"== {label or mod} ==")
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=__import__("os").path.dirname(
                               __import__("os").path.dirname(
                                   __import__("os").path.abspath(__file__))))
        out = (r.stdout + r.stderr).strip()
        print(out[-800:])
        report.append(f"\n## {label or mod}\n\n```\n{out[-1500:]}\n```\n")

    run("resource_demo", ["--model", "micro"], "Resources vs Transformer")
    run("benchmark_world", ["--steps", str(steps)], "World benchmark (recall + LM)")
    if not args.quick:
        run("compute_demo", ["--steps", str(steps)], "Compute-to-target")
        run("benchmark_ppl", ["--steps", str(steps)], "Penn Treebank PPL")

    with open(args.out, "w") as f:
        f.write("\n".join(report))
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
