#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# reproduce_all.sh — regenerate EVERY number in research/paper-draft.md.
#
# One command, honest defaults (reduced-strength so it fits a laptop/CI in a
# few minutes).  Full-strength flags are printed at the end.
#
#   bash reproduce_all.sh            # reduced (~5-10 min on CPU)
#   bash reproduce_all.sh --full     # full-strength benchmarks + 3 seeds
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

EVID=out/evidence
mkdir -p "$EVID"

echo "=============================================================="
echo " LEAFv5 evidence regeneration — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="

if [ "${1:-}" = "--full" ]; then
    STEPS=150; EXP=200; SEEDS=3
else
    STEPS=60;  EXP=12;  SEEDS=1
fi

echo "[1/8] stability certificates"
python -m leafv5.stability_check --steps 120           2>&1 | tee "$EVID/cert-base.txt"    | tail -3
python -m leafv5.stability_check_mistral              2>&1 | tee "$EVID/cert-mistral.txt" | tail -3

echo "[2/8] growth-vs-scratch pipeline experiment (steps/phase=$EXP, seeds=$SEEDS)"
python -m leafv5.grow_vs_scratch --steps "$EXP" --seeds "$SEEDS" \
    2>&1 | tee "$EVID/grow-vs-scratch.txt" | tail -12

echo "[3/8] standard-corpus PPL (PTB char, steps=$STEPS)"
python -m leafv5.benchmark_ppl --steps "$STEPS" \
    2>&1 | tee "$EVID/bench-ppl.txt" | tail -6

echo "[4/8] world benchmark (recall + LM race, steps=$((STEPS/4)))"
python -m leafv5.benchmark_world --steps $((STEPS/4)) \
    2>&1 | tee "$EVID/bench-world.txt" | tail -20

echo "[5/8] native scan engine (C twin: build, validate vs torch, benchmark)"
bash mojo/c_ref/build.sh >/dev/null 2>&1 && \
    OMP_NUM_THREADS=2 python mojo/c_ref/bench.py 2>&1 | tee "$EVID/scan-engine.txt" \
    | grep -E "validate|GFLOP/s|faster"

echo "[6/8] test suite (fast subset)"
python -m pytest tests/test_grow.py tests/test_causal_invariant.py \
    tests/test_stability_cert.py tests/test_mistral_advantages.py \
    tests/test_bugfix_aug09.py tests/test_scan_engine.py -q 2>&1 | tail -2

echo "[7/8] Tier-1 evidence (scaling study, ablation, retention levers)"
python -m leafv5.scaling_study --steps 60 2>&1 | tee "$EVID/scaling.txt" | tail -6
python -m leafv5.ablate_suite --steps 60 2>&1 | tee "$EVID/ablation.txt" | tail -10
python -m leafv5.retention_study --steps 60 --distractors 8 --pairs 2 \
    2>&1 | tee "$EVID/retention.txt" | tail -6

echo "[8/8] certificates re-verified"
grep -h "passed" "$EVID"/cert-*.txt | tail -2

echo ""
echo "=============================================================="
echo " EVIDENCE SAVED TO out/evidence/ — full strength:"
echo "   bash reproduce_all.sh --full     # ~30-60 min on a GPU/beefy CPU"
echo "=============================================================="
