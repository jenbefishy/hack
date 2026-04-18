"""Core signal processing and utility functions for ATV decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Return centered moving average with a fixed window size."""
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="same")


def load_iq(path: Path) -> np.ndarray:
    """Load IQ samples from supported formats: .cf32, .npy, .bin."""
    suffix = path.suffix.lower()
    if suffix == ".cf32":
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 2 != 0:
            raw = raw[:-1]
        iq = raw[0::2] + 1j * raw[1::2]
        return iq.astype(np.complex64, copy=False)
    if suffix == ".npy":
        iq = np.load(path, mmap_mode="r")
        return np.asarray(iq, dtype=np.complex64)
    if suffix == ".bin":
        raw = np.fromfile(path, dtype=np.int16)
        if raw.size % 2 != 0:
            raw = raw[:-1]
        i = raw[0::2].astype(np.float32) / 32768.0
        q = raw[1::2].astype(np.float32) / 32768.0
        return (i + 1j * q).astype(np.complex64, copy=False)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def fm_demodulate(iq: np.ndarray) -> np.ndarray:
    """Demodulate FM IQ stream into real-valued baseband."""
    # Standard quadrature FM discriminator: dphi[n] = arg(x[n] * conj(x[n-1])).
    dphi = np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float32)
    dphi -= np.median(dphi)
    demod = moving_average(dphi, window=5).astype(np.float32)
    return demod


def find_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find start and end indices for contiguous True-runs."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return starts, ends


def robust_threshold(signal: np.ndarray) -> float:
    """Compute a robust low-tail threshold for sync-tip detection."""
    p5 = float(np.percentile(signal, 5.0))
    p25 = float(np.percentile(signal, 25.0))
    # Sync tips should be in the lower tail; this keeps threshold data-adaptive.
    return p5 + 0.35 * (p25 - p5)


def estimate_line_period(hsync_starts: np.ndarray, fs: float) -> float:
    """Estimate nominal line period from plausible HSYNC distances."""
    if hsync_starts.size < 8:
        return float("nan")
    d = np.diff(hsync_starts).astype(np.float64)
    plausible = d[(d > fs * 45e-6) & (d < fs * 90e-6)]
    if plausible.size < 6:
        return float("nan")
    return float(np.median(plausible))


def periodicity_score(x: np.ndarray, lag: int) -> float:
    """Normalized autocorrelation-like periodicity score."""
    if lag <= 0:
        return 0.0
    if x.size <= lag + 4:
        return 0.0
    a = x[lag:].astype(np.float64, copy=False)
    b = x[:-lag].astype(np.float64, copy=False)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rms_energy(x: np.ndarray) -> float:
    """Root-mean-square energy of a signal segment."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64, copy=False) ** 2)))


def shift_with_edge_padding(x: np.ndarray, shift: int) -> np.ndarray:
    """Shift 1D array left/right while padding edges with border values."""
    if shift == 0:
        return x
    y = np.empty_like(x)
    if shift > 0:
        y[:shift] = x[0]
        y[shift:] = x[:-shift]
    else:
        k = -shift
        y[-k:] = x[-1]
        y[:-k] = x[k:]
    return y


def best_row_shift(row_lp: np.ndarray, ref_lp: np.ndarray, max_shift: int) -> int:
    """Find best horizontal shift by maximizing normalized correlation."""
    best_shift = 0
    best_score = -1e30
    n = row_lp.size
    for sh in range(-max_shift, max_shift + 1):
        if sh >= 0:
            a = row_lp[sh:]
            b = ref_lp[: n - sh]
        else:
            k = -sh
            a = row_lp[: n - k]
            b = ref_lp[k:]
        if a.size < 16:
            continue
        aa = a - np.mean(a)
        bb = b - np.mean(b)
        den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        if den < 1e-12:
            continue
        score = float(np.dot(aa, bb) / den)
        if score > best_score:
            best_score = score
            best_shift = sh
    return int(best_shift)
