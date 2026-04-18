#!/usr/bin/env bash
set -euo pipefail

# Process full signal without frame limit.
python3 src/main.py \
  --input data/iq_capture.npy \
  --fs 20000000 \
  --windows-count 8 \
  --window-ms 20 \
  --max-frames 0 \
  --out results/full_signal_run
