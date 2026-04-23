"""High-level processing pipeline for ATV decode tasks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .reconstruction import reconstruct_frames
from .signal_processing import estimate_broadcast_video_fps, fm_demodulate, load_iq
from .sync_detection import detect_sync_pulses
from .visualization import (
    save_debug_plot,
    save_overview_windows,
    save_reconstructed_frames,
    save_reconstructed_video,
)


def run_pipeline(
    input_path: Path,
    fs: float,
    out_dir: Path,
    windows_count: int,
    window_ms: float,
    max_frames: int,
    video_fps: float = -1.0,
) -> dict:
    """Run complete decode pipeline and save all intermediate artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    iq = load_iq(input_path)
    demod = fm_demodulate(iq)
    sync = detect_sync_pulses(demod, fs=fs)

    np.save(out_dir / "demodulated_signal.npy", demod)
    np.save(out_dir / "hsync_starts.npy", sync["hsync_starts"])
    np.save(out_dir / "vsync_starts.npy", sync["vsync_starts"])
    np.save(out_dir / "vsync_ends.npy", sync["vsync_ends"])
    np.save(out_dir / "vsync_intervals.npy", sync["vsync_intervals"])

    report = {
        "input_file": str(input_path),
        "fs_hz": fs,
        "num_iq_samples": int(iq.size),
        "num_demod_samples": int(demod.size),
        "threshold": sync["threshold"],
        "line_period_samples": sync["line_period_samples"],
        "line_period_us": (sync["line_period_samples"] / fs * 1e6) if sync["line_period_samples"] else None,
        "hsync_width_samples_median": sync["hsync_width_samples_median"],
        "hsync_width_us_median": (sync["hsync_width_samples_median"] / fs * 1e6) if sync["hsync_width_samples_median"] else None,
        "hsync_count": int(sync["hsync_starts"].size),
        "vsync_count": int(sync["vsync_starts"].size),
        "vsync_interval_duration_us_median": (
            float(np.median((sync["vsync_intervals"][:, 1] - sync["vsync_intervals"][:, 0]) / fs * 1e6))
            if sync["vsync_intervals"].size
            else None
        ),
    }

    (out_dir / "sync_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    save_debug_plot(out_dir, demod, sync["hsync_starts"], sync["vsync_intervals"], fs)
    save_overview_windows(
        out_dir,
        demod,
        sync["hsync_starts"],
        sync["vsync_intervals"],
        fs,
        window_ms=window_ms,
        count=windows_count,
    )
    frames, frames_meta = reconstruct_frames(
        demod=demod,
        hsync=sync["hsync_starts"],
        vsync_intervals=sync["vsync_intervals"],
        fs=fs,
        max_frames=max(0, max_frames),
    )
    save_reconstructed_frames(out_dir, frames)
    video_path: Path | None = None
    if video_fps != 0 and frames:
        eff_fps = float(video_fps) if video_fps > 0 else estimate_broadcast_video_fps(frames_meta.get("line_period_us"))
        report["reconstructed_video_fps"] = eff_fps
        video_path = save_reconstructed_video(out_dir, frames, fps=eff_fps)
    (out_dir / "reconstruction_report.json").write_text(
        json.dumps(frames_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if video_path is not None:
        report["reconstructed_video"] = video_path.name
    elif video_fps != 0 and frames:
        report["reconstructed_video"] = None
        if shutil.which("ffmpeg") is None:
            report["reconstructed_video_note"] = "ffmpeg not on PATH; install ffmpeg to build MP4"
        else:
            report["reconstructed_video_note"] = (
                "MP4 encoding failed; see ffmpeg_encoding.stderr.log in the output directory"
            )

    return report
