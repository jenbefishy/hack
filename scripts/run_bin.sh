#!/usr/bin/env bash
set -euo pipefail

# Example for int16 interleaved IQ (.bin) input.
python3 src/main.py \
  --input data/iq_capture.bin \
  --fs 20000000 \
  --out results/bin_run
