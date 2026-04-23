#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Run pipeline with default parameters.
python3 src/main.py \
  --input data/iq_capture.cf32 \
  --fs 20000000 \
  --out results/default_run
