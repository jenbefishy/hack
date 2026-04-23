#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Process full signal without frame limit (many PNGs; MP4 skipped by default).
python3 src/main.py \
  --input data/iq_capture.npy \
  --fs 20000000 \
  --windows-count 8 \
  --window-ms 20 \
  --max-frames 0 \
  --video-fps 0 \
  --out results/full_signal_run
