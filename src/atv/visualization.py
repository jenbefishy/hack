"""Plotting helpers for sync detection and reconstructed frames."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def save_debug_plot(
    out_dir: Path,
    demod: np.ndarray,
    hsync: np.ndarray,
    vsync_intervals: np.ndarray,
    fs: float,
) -> None:
    """Save short preview plot with HSYNC and VSYNC overlays."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    n = min(demod.size, int(0.006 * fs))
    x = np.arange(n) / fs * 1e3
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, demod[:n], lw=0.8, color="steelblue", label="Demodulated")

    h_local = hsync[hsync < n]
    if h_local.size:
        ax.vlines(h_local / fs * 1e3, ymin=np.min(demod[:n]), ymax=np.max(demod[:n]), color="green", alpha=0.3, lw=0.8, label="HSYNC")

    v_local = vsync_intervals[(vsync_intervals[:, 0] < n) & (vsync_intervals[:, 1] > 0)] if vsync_intervals.size else np.empty((0, 2), dtype=np.int64)
    if v_local.size:
        first = True
        for s, e in v_local:
            ss = max(0, int(s))
            ee = min(n, int(e))
            if ee <= ss:
                continue
            ax.axvspan(ss / fs * 1e3, ee / fs * 1e3, color="red", alpha=0.12, label="VSYNC interval" if first else None)
            ax.vlines([ss / fs * 1e3, ee / fs * 1e3], ymin=np.min(demod[:n]), ymax=np.max(demod[:n]), color="red", alpha=0.7, lw=1.0, label="VSYNC bounds" if first else None)
            first = False

    ax.set_title("FM demod + sync detection (first 6 ms)")
    ax.set_xlabel("Time, ms")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "sync_preview.png", dpi=140)
    plt.close(fig)


def save_overview_windows(
    out_dir: Path,
    demod: np.ndarray,
    hsync: np.ndarray,
    vsync_intervals: np.ndarray,
    fs: float,
    window_ms: float,
    count: int,
) -> None:
    """Save multiple evenly-spaced long windows of the signal."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if count <= 0:
        return
    win = int(fs * window_ms / 1000.0)
    if win <= 0:
        return

    n_total = demod.size
    max_start = max(0, n_total - win)
    starts = np.linspace(0, max_start, count, dtype=np.int64)

    for i, start in enumerate(starts, 1):
        end = int(min(n_total, start + win))
        start = int(max(0, end - win))
        x = (np.arange(start, end) - start) / fs * 1e3
        y = demod[start:end]

        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.plot(x, y, lw=0.5, color="steelblue", alpha=0.9, label="Demodulated")

        h_local = hsync[(hsync >= start) & (hsync < end)]
        if h_local.size:
            ax.vlines((h_local - start) / fs * 1e3, ymin=np.min(y), ymax=np.max(y), color="green", alpha=0.12, lw=0.45, label="HSYNC")

        v_local = vsync_intervals[(vsync_intervals[:, 0] < end) & (vsync_intervals[:, 1] > start)] if vsync_intervals.size else np.empty((0, 2), dtype=np.int64)
        first = True
        for s, e in v_local:
            ss = max(start, int(s))
            ee = min(end, int(e))
            if ee <= ss:
                continue
            xs = (ss - start) / fs * 1e3
            xe = (ee - start) / fs * 1e3
            ax.axvspan(xs, xe, color="red", alpha=0.14, label="VSYNC interval" if first else None)
            ax.vlines([xs, xe], ymin=np.min(y), ymax=np.max(y), color="red", alpha=0.95, lw=1.1, label="VSYNC bounds" if first else None)
            first = False

        ax.set_title(f"Window {i}/{count}: {window_ms:g} ms (samples {start}:{end})")
        ax.set_xlabel("Time, ms")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right", ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"window_{int(round(window_ms))}ms_{i}.png", dpi=170)
        plt.close(fig)


def save_reconstructed_frames(out_dir: Path, frames: list[np.ndarray]) -> None:
    """Save reconstructed frames as grayscale PNG images (pixel-accurate layout)."""
    if not frames:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    dpi = 130
    for i, fr in enumerate(frames, 1):
        u = np.asarray(fr, dtype=np.uint8)
        if u.ndim != 2:
            continue
        h_px, w_px = int(u.shape[0]), int(u.shape[1])
        fig_w = max(5.0, min(16.0, w_px / dpi))
        fig_h = max(4.0, min(24.0, h_px / dpi))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.imshow(
            u,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
            aspect="equal",
            origin="upper",
        )
        ax.set_title(f"Reconstructed frame {i}")
        ax.set_xlabel("Active line (resampled pixels)")
        ax.set_ylabel("Line")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.06)
        fig.savefig(out_dir / f"reconstructed_frame_{i}.png", dpi=dpi, pad_inches=0.02)
        plt.close(fig)


def save_reconstructed_video(
    out_dir: Path,
    frames: list[np.ndarray],
    fps: float,
    filename: str = "reconstructed_frames.mp4",
    crf: int = 18,
) -> Path | None:
    """Write reconstructed uint8 grayscale frames as one H.264 MP4 (requires ``ffmpeg`` on PATH).

    Pads all frames to the same height and width (top-left aligned) so codecs accept
    a regular stream. Uses short GOP for few-frame previews to limit temporal pumping.
    """
    if not frames or fps <= 0:
        return None
    if shutil.which("ffmpeg") is None:
        return None
    max_h = max(int(f.shape[0]) for f in frames)
    max_w = max(int(f.shape[1]) for f in frames)
    # yuv420p (libx264 default) requires even width and height.
    max_h = (max_h + 1) // 2 * 2
    max_w = (max_w + 1) // 2 * 2
    if max_h < 2 or max_w < 2:
        return None
    out_path = out_dir / filename
    fps_c = float(np.clip(float(fps), 0.5, 120.0))
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{max_w}x{max_h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps_c),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(int(np.clip(crf, 10, 35))),
        "-preset",
        "medium",
        "-tune",
        "grain",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    # Short clips: all-intra avoids B-frame flicker when only a handful of frames exist.
    if len(frames) <= 64:
        idx = cmd.index(str(out_path))
        cmd[idx:idx] = ["-g", "1", "-keyint_min", "1", "-sc_threshold", "0"]
    stderr_path = out_dir / "ffmpeg_encoding.stderr.log"
    stderr_file = None
    try:
        stderr_file = open(stderr_path, "w", encoding="utf-8")
    except OSError:
        pass
    stderr_arg = stderr_file if stderr_file is not None else subprocess.DEVNULL
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_arg,
        )
    except OSError:
        if stderr_file is not None:
            stderr_file.close()
        return None
    assert proc.stdin is not None
    write_ok = True
    try:
        for fr in frames:
            u = np.asarray(fr, dtype=np.uint8)
            if u.ndim != 2:
                continue
            h, w = int(u.shape[0]), int(u.shape[1])
            plane = np.zeros((max_h, max_w), dtype=np.uint8)
            plane[:h, :w] = u
            rgb = np.stack([plane, plane, plane], axis=-1)
            proc.stdin.write(rgb.tobytes())
    except BrokenPipeError:
        write_ok = False
        proc.kill()
    finally:
        proc.stdin.close()
    ret = proc.wait()
    if stderr_file is not None:
        stderr_file.close()
    if not write_ok or ret != 0 or not out_path.is_file():
        if out_path.is_file():
            out_path.unlink(missing_ok=True)
        return None
    # Success: drop empty noise log if present.
    try:
        if stderr_path.is_file() and stderr_path.stat().st_size == 0:
            stderr_path.unlink()
    except OSError:
        pass
    return out_path
