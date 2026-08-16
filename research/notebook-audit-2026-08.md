# Mini research finding — audit of the Colab notebook (leafv5_t4.ipynb)

Date: 2026-08-13.  Scope: a correctness + completeness audit of the 100-cell
T4 notebook, with every claim below **measured** against the actual cells and
the actual code (not assumed).

---

## F1 — CORRECTNESS (high): the recall-demo cell contradicts its own markdown

**Evidence (measured).**  §8b's markdown says LEAFv5 hits **"100% held-out
recall in 10 steps"** — which is true for the *store-1/query-1* task
(`speed_demo --task recall --pairs 1 --queries 1`: 99% @5, 100% @10, verified
again this session).  But the demo cell directly under it runs:

```
!python -m leafv5.recall_demo --steps 300 --dim 192 --layers 4 --rope-dim 0
```

`recall_demo` defaults to **store-4 / recall-2** — the *hard* task that
measures **1.6–2.3% at 40 steps** (chance = 1.6%) at micro scale (re-verified
this session).  A user running the cell sees near-chance output immediately
after a markdown promise of 100%, concludes the model is broken, and stops.

**Why it happened.**  `recall_demo` was written for the pre-fix era when
store-4/recall-2 hit ~96-99%; post-fix (see `research/reverify-2026-08.md`)
that task is training-limited at micro scale and scores ≈ chance.  The
markdown was updated to the honest numbers but the demo cell was not switched
to the task those numbers describe.

**Fix.**  Run `speed_demo --task recall --pairs 1 --queries 1` in that cell
(99%@5, 100%@10 — matches the markdown), and keep `recall_demo` as a clearly
labeled *hard-task* cell with honest expectations.

## F2 — COVERAGE (medium): headline CPU-runnable evidence is missing

The notebook proves the pipeline but omits several of the project's strongest
**CPU-runnable** demonstrations:

| omitted | what it shows | status |
|---|---|---|
| `benchmark_ppl` | standard-corpus evidence: PTB char PPL **6.1 vs 8.4 vs 9.5** (LEAFv5 / Transformer / GatedRNN) | runs on CPU, ~3 min |
| `benchmark_world` | recall + LM race vs Transformer and a Mamba-family GatedRNN | runs on CPU |
| smart weight storage (`weights.py`) | shared + SVD + int8-residual packing → 3.9–4.85× smaller checkpoints | runs on CPU |
| stateful sessions | "the recurrent memory IS conversation context" (a headline feature) | can be shown ckpt-free via `generate` with carried states |
| LoRA | `--lora-rank` PEFT example in the finetune section | runs on CPU (smoke) |
| verdict dashboard | an end-of-notebook cell collating "what this run proved" | — |

**Fix.**  Add cells for each; make the stateful-session demo checkpoint-free
so it always runs.

## F3 — ROBUSTNESS (medium): cells crash confusingly on out-of-order execution

`generate` (cell 71) and `quantize` (cell 74) assume `out/.../best.pt` exists.
If a user runs them before training finishes (or training OOMs), they raise a
raw `FileNotFoundError`.  **Fix:** existence guards with actionable messages
("run the training cell first — best.pt appears after the first eval").

## F4 — REPRODUCIBILITY (low): `--seed` not stated in the train command

`train.py` defaults `--seed 42` (verified), so runs are already reproducible —
but the notebook's main training cell doesn't state it.  **Fix:** add
`--seed 42` explicitly (documentation of intent, zero behavior change).

## F5 — SCIENCE PACKAGING (low): no collation at the end

The notebook proves many things but never summarizes them.  A final
"verdict" cell that prints the key numbers (certificates, growth Δ, speed
race, pipeline compute ratio) turns the notebook from a recipe into evidence.

---

## What the audit did NOT find (checked, clean)

- `%%writefile` embeds: 52 cells, all match the repo byte-for-byte (the only
  diff is the intentional empty-marker for `mojo/__init__.py`).
- Dataset cell: `build_all(n=20000, seed=42)` produces **exactly 24,935**
  examples — identical to the shipped `leafv5_training_data.jsonl`.
- No empty cells; nbformat valid; kernelspec python3.
- No remaining inflated claims in markdown (all re-measured numbers).
- `train.py` does save `best.pt` on first eval (line 788) — the generate/
  quantize cells' target is real (only timing/robustness, not existence).

## Priority

1. Fix F1 (correctness — trust-breaking).
2. Add F2 cells (evidence — the strongest honest demonstrations).
3. F3 guards + F4 explicit seed + F5 verdict (packaging).

---

## Addendum (same session, follow-up findings)

### F6 — NEW (correctness, significant): the "3.9–4.85× smaller checkpoints" claim was tensor-level, not file-level

**Evidence (measured).**  `pack_model` reduces *tensor bytes* ~4× (rank-0 int8
residual: 4.0×, max err 3.8e-4) — but `torch.save` of the packed dict produces
a **bigger file** than the fp32 state_dict (measured 0.67–0.69× on 16M models;
1.6× worse on the single-matrix case).  Pickle adds per-small-tensor overhead
that swamps the savings.  So the README's "checkpoints 3.9–4.85× smaller" was
measuring tensor bytes via `report()`, not real files.

**Fix (shipped).**  New compact binary format in `leafv5/weights.py`:
`save_packed`/`load_packed` — full structure (incl. non-tensor metadata)
pickled with tensor placeholders; all payloads in ONE contiguous byte buffer
with a small index.  Measured on a 16M model: **64.85 MB → 14.67 MB = 4.42×
smaller file**, structure round-trips, loads in a new process.
Regression-tested (`tests/test_weights_pack.py`, 3 tests).  README updated to
the honest file-level claim.

### F7 — NEW (latent bug found while validating the weights demo): `path` shadowing

`save_packed(packed, path)` used `path` as the loop variable over tensors,
then `open(path, ...)` — the parameter was overwritten by the last tensor's
path tuple (TypeError on any real call).  Found by executing the new notebook
cell; fixed (loop var renamed).

### Status

All five original findings fixed and re-validated; two new findings (F6, F7)
found and fixed during implementation.  The notebook is now 113 cells; every
new cell executed successfully in a simulated fresh session (58/58 targets in
the focused run, 0 errors).
