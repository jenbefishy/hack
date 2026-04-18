"""Frame reconstruction from HSYNC/VSYNC intervals."""

from __future__ import annotations

import numpy as np

from .signal_processing import best_row_shift, moving_average, shift_with_edge_padding


def reconstruct_frames(
    demod: np.ndarray,
    hsync: np.ndarray,
    vsync_intervals: np.ndarray,
    fs: float,
    max_frames: int,
) -> tuple[list[np.ndarray], dict]:
    """Build grayscale frame candidates from demodulated ATV signal.

    Args:
        max_frames: Maximum number of frames to keep.
            If max_frames <= 0, no frame limit is applied.
    """
    if hsync.size < 20:
        return [], {"reason": "not_enough_hsync"}

    hsync = np.sort(hsync.astype(np.int64))
    d = np.diff(hsync).astype(np.float64)
    plausible = d[(d > fs * 45e-6) & (d < fs * 90e-6)]
    if plausible.size < 10:
        return [], {"reason": "cannot_estimate_line_period"}
    line_period = int(round(float(np.median(plausible))))
    smooth_demod = moving_average(demod.astype(np.float32, copy=False), window=7).astype(np.float32)

    # Keep only stable line starts close to the nominal period.
    dd = np.diff(hsync).astype(np.float64)
    stable = np.concatenate(([True], np.abs(dd - line_period) <= 0.35 * line_period))
    line_starts = hsync[stable]
    if line_starts.size < 100:
        return [], {"reason": "not_enough_stable_lines"}

    separators: list[tuple[int, int]] = []
    if vsync_intervals.size:
        for s, e in vsync_intervals.astype(np.int64):
            separators.append((int(s), int(e)))
    else:
        gaps = np.diff(line_starts)
        idx = np.flatnonzero(gaps > 1.6 * line_period)
        for i in idx:
            s = int(line_starts[i])
            e = int(line_starts[i + 1])
            separators.append((s, e))

    if not separators:
        return [], {"reason": "no_frame_separators"}

    separators.sort(key=lambda p: p[0])
    frames: list[np.ndarray] = []
    frame_meta: list[dict] = []

    for i, (_, sep_end) in enumerate(separators):
        # Frame starts after this separator and ends before the next one.
        start_idx = np.searchsorted(line_starts, sep_end, side="left")
        if i + 1 < len(separators):
            next_start = separators[i + 1][0]
            end_idx = np.searchsorted(line_starts, next_start, side="left")
        else:
            end_idx = line_starts.size - 1

        if end_idx - start_idx < 120:
            continue

        lines = line_starts[start_idx:end_idx]
        if lines.size < 120:
            continue

        # Adaptive active video window inside each line.
        alpha = 0.24
        beta = 0.92
        width = max(64, int(round((beta - alpha) * line_period)))
        rows: list[np.ndarray] = []
        row_shifts: list[int] = []
        search = max(2, int(round(0.08 * line_period)))
        max_h_shift = max(2, int(round(0.06 * width)))

        # First pass: refine sync-tip for each detected line start.
        tips: list[int] = []
        for ls in lines:
            ls_i = int(ls)
            s0 = max(0, ls_i - search)
            s1 = min(smooth_demod.size, ls_i + search + 1)
            if s1 - s0 < 3:
                continue
            tip = s0 + int(np.argmin(smooth_demod[s0:s1]))
            tips.append(int(tip))

        if len(tips) < 140:
            continue

        # Second pass: extract active video and resample every row.
        raw_rows: list[np.ndarray] = []
        valid_line_lengths: list[int] = []
        for i_tip in range(len(tips) - 1):
            t0 = tips[i_tip]
            t1 = tips[i_tip + 1]
            ll = int(t1 - t0)
            if ll < int(0.65 * line_period) or ll > int(1.45 * line_period):
                continue
            a = int(t0 + alpha * ll)
            b = int(t0 + beta * ll)
            if a < 0 or b > demod.size or b <= a + 8:
                continue
            seg = demod[a:b].astype(np.float32, copy=False)
            x_src = np.linspace(0.0, 1.0, seg.size, endpoint=False, dtype=np.float32)
            x_dst = np.linspace(0.0, 1.0, width, endpoint=False, dtype=np.float32)
            row = np.interp(x_dst, x_src, seg).astype(np.float32, copy=False)
            raw_rows.append(row)
            valid_line_lengths.append(ll)

        if len(raw_rows) < 120:
            continue

        # Third pass: correlate adjacent rows for horizontal alignment.
        ref_lp = moving_average(raw_rows[0], window=11).astype(np.float32)
        rows.append(raw_rows[0])
        row_shifts.append(0)
        for r in raw_rows[1:]:
            r_lp = moving_average(r, window=11).astype(np.float32)
            sh = best_row_shift(r_lp, ref_lp, max_h_shift)
            aligned = shift_with_edge_padding(r, sh).astype(np.float32, copy=False)
            aligned_lp = moving_average(aligned, window=11).astype(np.float32)
            rows.append(aligned)
            row_shifts.append(int(sh))
            ref_lp = (0.80 * ref_lp + 0.20 * aligned_lp).astype(np.float32)

        if len(rows) < 120:
            continue

        frame = np.vstack(rows).astype(np.float32, copy=False)

        # Keep the longest vertically consistent region.
        if frame.shape[0] > 140:
            frame_lp = moving_average(frame.mean(axis=1), window=5).astype(np.float32)
            drow = np.abs(np.diff(frame_lp))
            thr = float(np.median(drow) + 3.0 * (np.median(np.abs(drow - np.median(drow))) + 1e-9))
            bad = drow > thr
            cut_idx = np.flatnonzero(bad)
            if cut_idx.size:
                bounds = [0] + [int(i + 1) for i in cut_idx] + [frame.shape[0]]
                spans = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1) if bounds[k + 1] - bounds[k] >= 120]
                if spans:
                    s_best, e_best = max(spans, key=lambda p: p[1] - p[0])
                    frame = frame[s_best:e_best]
                    row_shifts = row_shifts[s_best:e_best]

        if frame.shape[0] < 120:
            continue

        # Robust normalization to 8-bit grayscale.
        p1, p99 = np.percentile(frame, [1.0, 99.0])
        if p99 <= p1:
            continue
        frame = np.clip((frame - p1) / (p99 - p1), 0.0, 1.0)
        frame_u8 = (frame * 255.0).astype(np.uint8)
        frames.append(frame_u8)
        frame_meta.append(
            {
                "frame_index": len(frames) - 1,
                "source_separator_index": i,
                "line_count": int(frame_u8.shape[0]),
                "width": int(frame_u8.shape[1]),
                "mean_abs_row_shift": float(np.mean(np.abs(np.asarray(row_shifts, dtype=np.float32)))) if row_shifts else 0.0,
                "line_len_mean_samples": float(np.mean(np.asarray(valid_line_lengths, dtype=np.float32))) if valid_line_lengths else 0.0,
                "line_len_std_samples": float(np.std(np.asarray(valid_line_lengths, dtype=np.float32))) if valid_line_lengths else 0.0,
            }
        )

        if max_frames > 0 and len(frames) >= max_frames:
            break

    meta = {
        "line_period_samples": int(line_period),
        "line_period_us": float(line_period / fs * 1e6),
        "frames_built": len(frames),
        "frame_meta": frame_meta,
    }
    return frames, meta
