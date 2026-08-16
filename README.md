# LEAFv5 — Fast, small language models (T4-trainable SLM)

LEAFv5 is a compact, from-scratch PyTorch implementation of the LEAFv5 architecture: a linear-time, memory-efficient small language model (SLM) designed to train from scratch on a 16 GB NVIDIA T4 within a 4-hour budget.

This repository includes training, evaluation, recurrent generation, fast native kernels (Mojo/C), and dataset tooling optimized for sample-efficiency and edge-friendly inference.

Highlights
- Train a ~102M parameter LEAFv5 on a 16 GB T4 in ~4 hours (preset: `t4-4h`).
- Recurrent delta-memory core: constant-size inference state, linear time per token.
- Fast corpus encoding via GigaToken (Rust) for GB/s tokenization.
- Chunked parallel-scan recurrence for high throughput on T4s.
- Mojo + validated C kernels for native-speed inference on CPU.

Quick links
- Quick smoke test (CPU): tests and a short train/generate cycle — see Quickstart below.
- T4 one-command run: `bash run_t4_4h.sh` (sets up, trains with the recommended preset).
- Mojo native kernels: `mojo/README.md`.
- Identity & skills dataset: `data_gen/README.md`.

Quickstart (smoke test, ~5 minutes, CPU)

```bash
pip install -r requirements.txt
python tests/test_model.py
python -m leafv5.train --data shakespeare --model micro \
    --seq-len 64 --micro-batch 8 --grad-accum 2 --lr 1e-3 \
    --max-steps 130 --outdir out/smoke --data-dir data_cache
python -m leafv5.generate --ckpt out/smoke/best.pt --prompt "Romeo" --max-new 200
python -m leafv5.eval --ckpt out/smoke/best.pt --data-dir data_cache
```

One-command T4 run (recommended)

```bash
# Install dependencies (PyTorch with CUDA as appropriate)
pip install torch tokenizers numpy gigatoken
bash run_t4_4h.sh
# or run manually with flags demonstrated in the repository
```

Why LEAFv5 (short)
- Constant-size recurrent state per layer → inference memory independent of context length.
- Delta-memory write/read mechanism that enables very fast one/few-step adaptation (associative recall).
- Highly optimized training pipeline (GigaToken, chunked scan, background prefetch, fp16 + torch.compile support on CUDA/T4).
- Native kernels (Mojo/C) for fast CPU inference.

Repository layout (top-level)
- leafv5/: model implementation, training, eval, generation, utilities.
- mojo/: Mojo kernels, C reference, and benchmarks.
- tests/: unit and integration tests.
- run_t4_4h.sh, leafv5_t4.ipynb: one-command / notebook for the 4-hour T4 run.

Where to look for detail
- The main README used to contain an extensive, fully-documented technical walkthrough. If you need the complete, line-by-line details (math, experiments, measured benchmarks, design decisions), consult the repository's commit history or the detailed docs and files: the long-form explanations are present inside this repo (search for sections, or open files such as `mojo/README.md` and `data_gen/README.md`).

Advanced features & flags
- `--scan {sequential,chunked}`: chunked uses a parallel-scan formulation to reduce kernel launches and improve throughput.
- `--tokenizer-engine {auto,gigatoken,hf}`: use GigaToken (fast Rust encoder) when available.
- `--budget-hours H`: caps total training steps to finish within the allotted wall-clock hours (measures tok/s, computes tokens budget).
- `--fast`: sample-efficiency recipe for very rapid few-step learning.
- `--auto`: hardware-aware auto-configuration (model/dtype/scan/compile tuned to detected GPU).

Contributing
- Issues and PRs welcome. Please include reproducible steps and logs when reporting bugs.
- Tests: run `python -m pytest` and ensure all suites pass locally before submitting a PR.

License & citation
- This project is released under the MIT license. If you use LEAFv5 in research or a project, please cite the repository and any associated paper.

Contact
- Author / maintainer: D.M.T.M. Dassanayake (details in repo).


<!-- End of concise README. The original, very long technical walkthrough is preserved in the repository history and scattered docs (see mojo/README.md, data_gen/README.md, and the code). If you'd like, I can archive the full original README into docs/README_FULL.md and keep this concise top-level README; tell me and I'll commit that as a follow-up. -->