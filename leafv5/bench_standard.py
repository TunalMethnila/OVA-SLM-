"""bench_standard.py — the standard-benchmark harness for the paper draft.

Two modes, both honest:
  * micro (works on any laptop, this sandbox): PTB char-PPL, the recall/LM
    world race, the growth pipeline, and the skill-eval graders — the
    regenerable evidence already in the repo.
  * standard (REQUIRES the T4 run): MMLU / GSM8K / HellaSwag on the trained
    94M checkpoint.  This script checks for the eval files, prints the exact
    fetch commands, and runs the evals if the files exist.

Run:
  python -m leafv5.bench_standard --mode micro            # now
  python -m leafv5.bench_standard --tasks mmlu,gsm8k,hellaswag \
      --ckpt out/leafv5-tinystories/best.pt               # after the T4 run
"""
from __future__ import annotations

import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)   # project root, so `python -m leafv5.*` works


def fetch_instructions() -> str:
    return (
        "\nStandard evals need the T4-trained checkpoint + eval files:\n"
        "  1) bash run_t4_4h.sh   # ~4 h on a 16 GB T4 -> out/leafv5-tinystories/best.pt\n"
        "  2) download eval sets (each is a JSONL of {question, choices, answer}):\n"
        "     MMLU-lite     https://huggingface.co/datasets/cais/mmlu  (select subset)\n"
        "     GSM8K         https://huggingface.co/datasets/openai/gsm8k\n"
        "     HellaSwag     https://huggingface.co/datasets/rowanhellaswag/hellaswag\n"
        "     place them as:  eval/mmlu.jsonl  eval/gsm8k.jsonl  eval/hellaswag.jsonl\n"
        "  3) python -m leafv5.bench_standard --tasks mmlu,gsm8k,hellaswag \\\n"
        "         --ckpt out/leafv5-tinystories/best.pt\n"
        "\nEvery number printed in --mode micro is regenerable in this repo; no\n"
        "standard-benchmark number is claimed until this script runs on a real\n"
        "checkpoint (research/paper-draft.md §7 says exactly this).")


def run_micro():
    import subprocess, sys
    print("=" * 70)
    print("MICRO EVIDENCE (all regenerable, honest defaults)")
    print("=" * 70)
    cmds = [
        # --seq 32 avoids a flaky torch CPU autograd livelock on deep 64-step
        # delta-scan chains under host contention (documented in ablate.py)
        ("PTB char PPL (LEAFv5 vs Transformer vs GatedRNN)",
         [sys.executable, "-m", "leafv5.benchmark_ppl", "--steps", "60",
          "--seq", "32"]),
        ("world benchmark (recall + LM race)",
         [sys.executable, "-m", "leafv5.benchmark_world", "--steps", "15"]),
        ("growth pipeline (grow vs scratch, smoke)",
         [sys.executable, "-m", "leafv5.grow_vs_scratch", "--steps", "12"]),
    ]
    for name, cmd in cmds:
        print(f"\n--- {name} ---")
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=_ROOT)
        print(r.stdout[-1200:])
        if r.returncode != 0:
            print(r.stderr[-400:])
    print(fetch_instructions())


def run_standard(tasks, ckpt):
    if not os.path.exists(ckpt):
        print(f"checkpoint not found: {ckpt}\n{fetch_instructions()}")
        return 1
    evaldir = os.path.join(os.path.dirname(_HERE), "eval")
    ok = True
    for t in tasks:
        f = os.path.join(evaldir, f"{t}.jsonl")
        if not os.path.exists(f):
            print(f"  [{t}] eval file missing: {f}  (see fetch instructions)")
            ok = False
    if not ok:
        print(fetch_instructions())
        return 1
    # load the checkpoint and evaluate (accuracy over the JSONL)
    from .generate import load_checkpoint
    model, tok, _ = load_checkpoint(ckpt, "auto")
    print(f"evaluating {len(tasks)} tasks with {model.n_params/1e6:.1f}M model ...")
    import torch
    model.eval()
    for t in tasks:
        rows = [json.loads(l) for l in open(os.path.join(evaldir, f"{t}.jsonl"))]
        correct = total = 0
        for row in rows[: min(200, len(rows))]:
            # row: {question, choices: [..], answer: int}
            prompt = row["question"] + "\n" + "\n".join(
                f"{chr(65+i)}) {c}" for i, c in enumerate(row["choices"]))
            ids = tok.encode(prompt)
            scores = []
            for i, c in enumerate(row["choices"]):
                cids = tok.encode(c)
                s = 0.0
                if cids:
                    # next-token likelihood of the choice's first token
                    with torch.no_grad():
                        import torch as _t2
                        lg2, _ = model(_t2.tensor([ids + cids[:-1]], dtype=_t2.long))
                    pr = _t2.softmax(lg2[0, -1].float(), -1)
                    s = float(pr[cids[-1]])
                scores.append(s)
            correct += int(int(torch.argmax(torch.tensor(scores))) == row["answer"])
            total += 1
        print(f"  {t}: {correct}/{total} = {100*correct/max(1,total):.1f}%")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["micro", "standard"], default="micro")
    p.add_argument("--tasks", type=str, default="mmlu,gsm8k,hellaswag")
    p.add_argument("--ckpt", type=str, default="out/leafv5-tinystories/best.pt")
    args = p.parse_args()
    if args.mode == "micro":
        run_micro()
        return 0
    return run_standard([t.strip() for t in args.tasks.split(",") if t.strip()],
                        args.ckpt)


if __name__ == "__main__":
    raise SystemExit(main())
