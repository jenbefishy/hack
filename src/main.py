#!/usr/bin/env python3
"""CLI entrypoint for ATV processing pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atv import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(description="ATV tasks: FM demodulation, sync detection and frame reconstruction.")
    parser.add_argument("--input", type=Path, default=Path("data/iq_capture.cf32"), help="Input IQ file (.cf32, .npy, .bin)")
    parser.add_argument("--fs", type=float, default=20e6, help="Sampling rate in Hz")
    parser.add_argument("--out", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--windows-count", type=int, default=5, help="Number of overview windows to save")
    parser.add_argument("--window-ms", type=float, default=30.0, help="Duration of each overview window in milliseconds")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3,
        help="Maximum reconstructed frames to save (<=0 means no limit, process full signal)",
    )
    return parser


def main() -> None:
    """Parse arguments, run pipeline and print summary."""
    args = build_parser().parse_args()
    report = run_pipeline(
        input_path=args.input,
        fs=args.fs,
        out_dir=args.out,
        windows_count=args.windows_count,
        window_ms=args.window_ms,
        max_frames=args.max_frames,
    )

    print("Done.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved outputs in: {args.out.resolve()}")


if __name__ == "__main__":
    main()
