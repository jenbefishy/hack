#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Warning: ffmpeg is not in PATH — MP4 will not be created (PNG frames only)." >&2
fi

OUT="${1:-results/npy_run}"
# Example with .npy source; --video-fps 25 matches ~64 µs PAL line at 20 MS/s.
python3 src/main.py \
  --input data/iq_capture.npy \
  --fs 20000000 \
  --windows-count 8 \
  --window-ms 20 \
  --max-frames 5 \
  --video-fps 25 \
  --out "$OUT"

MP4="$OUT/reconstructed_frames.mp4"
if [[ -f "$MP4" ]]; then
  echo "Video: $(readlink -f "$MP4" 2>/dev/null || echo "$PWD/$MP4")"
else
  echo "No MP4 — expected: $MP4 (cwd: $PWD). Install ffmpeg or pass --video-fps 0 to skip." >&2
fi
