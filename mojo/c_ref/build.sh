#!/usr/bin/env bash
# Build the C reference kernel as a shared library for ctypes.
#   * -O3 -march=native : AVX2/AVX-512 FMA auto-vectorization
#   * -fopenmp          : parallel scan over batch*heads (scales across cores)
#   * -funroll-loops    : unroll the small matvec/update loops
# Falls back to a serial build if OpenMP is unavailable.
set -euo pipefail
cd "$(dirname "$0")"
FLAGS="-O3 -march=native -funroll-loops -fPIC -shared -Wall"
if gcc -fopenmp -E -x c /dev/null >/dev/null 2>&1; then
    FLAGS="$FLAGS -fopenmp"
    echo "building with OpenMP + SIMD (auto-vectorized for this CPU)"
else
    echo "WARNING: OpenMP not available; building serial kernel"
fi
gcc $FLAGS leafv5_scan.c -o leafv5_scan.so -lm
echo "built leafv5_scan.so"
