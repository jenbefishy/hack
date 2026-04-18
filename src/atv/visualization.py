"""Plotting helpers for sync detection and reconstructed frames."""

from __future__ import annotations

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
    """Save reconstructed frames as grayscale PNG images."""
    if not frames:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    for i, fr in enumerate(frames, 1):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fr, cmap="gray", aspect="auto", vmin=0, vmax=255)
        ax.set_title(f"Reconstructed frame {i}")
        ax.set_xlabel("Sample in active line")
        ax.set_ylabel("Line")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_dir / f"reconstructed_frame_{i}.png", dpi=150)
        plt.close(fig)
