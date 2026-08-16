#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One-command LEAFv5 training on a 16 GB T4, sized to finish in < 4 hours.
#
#   ~94M params (dim 768, 14 layers, 4/4/4 fast/med/slow heads, d_h 48, FFN 2.5x)
#   fp16 autocast + GradScaler, torch.compile, gradient accumulation
#   --budget-hours 4  auto-caps total steps so the run fits in the wall clock
#   --scan chunked    parallel-scan delta recurrence (paper sec. 5 chunked
#                     formulation) -> far fewer kernel launches on GPU
#   --tokenizer-engine gigatoken  -> GB/s native corpus encoding (Rust)
#   --prefetch 4      background batch assembly overlaps CPU data with GPU
#   ~3-6 GB VRAM  (well under the 16 GB T4 limit)
#
# Usage:  bash run_t4_4h.sh
# -----------------------------------------------------------------------------
set -euo pipefail

pip install -q torch tokenizers numpy gigatoken   # torch w/ CUDA on T4 machine
python -m leafv5.train \
    --data tinystories \
    --model t4-4h \
    --tokenizer bpe --vocab-size 16384 \
    --tokenizer-engine gigatoken \
    --seq-len 128 \
    --curriculum "256,512" --curriculum-steps 2500 \
    --micro-batch 16 --grad-accum 8 \
    --scan chunked --chunk-size 64 --prefetch 4 \
    --optimizer lion --fast --grad-checkpoint \
    --budget-hours 4 \
    --outdir out/leafv5-tinystories \
    --sample-interval 2000 --eval-interval 2000 --ckpt-interval 5000

echo "--- generating with the trained model ---"
python -m leafv5.generate \
    --ckpt out/leafv5-tinystories/best.pt \
    --prompt "Once upon a time, a little girl named Lily" \
    --max-new 200

echo "--- int8 quantization check (70% smaller, ~zero PPL cost) ---"
python -m leafv5.quantize --ckpt out/leafv5-tinystories/best.pt \
    --data-dir data_cache --quantize-out out/leafv5-int8.pt --bench
