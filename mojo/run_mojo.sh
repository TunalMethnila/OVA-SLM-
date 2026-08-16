#!/usr/bin/env bash
# Run the Mojo LEAFv5 kernels (needs the Mojo SDK, ~20 GB disk, free account).
#
# One-time install (requires a free account token from developer.modular.com/download):
#   curl https://get.modular.com | MODULAR_AUTH=<your-key> sh -
#   modular install mojo
#   export MODULAR_HOME="$HOME/.modular"
#   export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"
#
set -euo pipefail
cd "$(dirname "$0")/.."
mojo run mojo/leafv5.mojo   # self-check: fused==general, one-shot recall, state bound
mojo run mojo/bench.mojo    # benchmark on T4-like shapes
