#!/usr/bin/env bash
set -euo pipefail

# Example with .npy source and custom visualization settings.
python3 src/main.py \
  --input data/iq_capture.npy \
  --fs 20000000 \
  --windows-count 8 \
  --window-ms 20 \
  --max-frames 5 \
  --out results/npy_run
