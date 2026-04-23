"""Frame reconstruction from HSYNC/VSYNC intervals.

Classic composite video (after FM demodulation for ATV) is specified in ITU-R BT.470
and related recommendations: each line has front porch, horizontal sync (~4.7 µs),
back porch, colour burst (PAL/NTSC), then active picture, then repeat. A full
decoder typically applies: FM discriminator with correct deviation scaling, optional
CCIR/CCITT video de-emphasis, band-limiting to video baseband, DC restoration
(black-level clamp on back porch), horizontal PLL locked to line frequency, vertical
state machine for interlaced fields (equalizing / serrated sync), luma–chroma
separation for colour. Amateur FM ATV is often treated as luminance-only; see e.g.
SDRangel demodatv (lines-per-frame, fractional samples/line, sync standard) and
overview articles on PAL/NTSC decoding.

This module builds grayscale rasters from stable HSYNC markers: per-line sync tips are
refined in a small window around each detected line start, then **consecutive tips**
define the true (fractional) line length for that row—avoiding an integer comb that
drifts and causes horizontal banding. Active video is ``alpha``–``beta`` of each
interval, resampled to a common width, then lightly aligned to a fixed template.
"""

from __future__ import annotations

import numpy as np

from .signal_processing import best_row_shift, moving_average, shift_with_edge_padding


def _median_smooth_1d(x: np.ndarray, half_win: int) -> np.ndarray:
    """Running median (odd window), edges mirrored with shorter windows."""
    n = int(x.size)
    if n == 0:
        return x
    hw = max(0, int(half_win))
    if hw == 0:
        return x.astype(np.float64, copy=False)
    out = np.empty(n, dtype=np.float64)
    for k in range(n):
        lo = max(0, k - hw)
        hi = min(n, k + hw + 1)
        out[k] = float(np.median(x[lo:hi]))
    return out


def _sync_tips_from_line_starts(
    line_starts: np.ndarray,
    smooth: np.ndarray,
    search: int,
) -> np.ndarray:
    """Refined sync-tip index per line (minimum of smoothed demod near each HSYNC)."""
    n = int(line_starts.size)
    tips = np.empty(n, dtype=np.int64)
    for k in range(n):
        ls = int(line_starts[k])
        s0 = max(0, ls - search)
        s1 = min(int(smooth.size), ls + search + 1)
        if s1 - s0 < 3:
            tips[k] = ls
        else:
            tips[k] = int(s0 + int(np.argmin(smooth[s0:s1])))
    return tips


def _align_rows_to_fixed_template(
    frame: np.ndarray,
    roi_frac_lo: float = 0.55,
    max_shift_frac: float = 0.18,
    median_half_width: int = 0,
) -> tuple[np.ndarray, list[int]]:
    """Shift each row to maximize correlation with one global template (median ROI).

    Unlike correlating each row to a slowly updated reference, a fixed template keeps
    the right blanking edge vertical without ``walking'' drift.
    """
    h, w = frame.shape
    r0 = max(0, min(w - 32, int(w * roi_frac_lo)))
    r1 = w
    roi_w = r1 - r0
    if roi_w < 32:
        return frame, [0] * h
    q = max(8, h // 4)
    sub = np.vstack([frame[:q], frame[-q:]]).astype(np.float32)
    tmpl = moving_average(
        np.median(sub[:, r0:r1], axis=0),
        window=min(11, roi_w | 1),
    ).astype(np.float32)
    max_shift = max(6, int(round(max_shift_frac * float(roi_w))))
    win = min(11, roi_w | 1)
    shifts_raw: list[int] = []
    for i in range(h):
        piece_lp = moving_average(frame[i, r0:r1].astype(np.float32), window=win).astype(np.float32)
        shifts_raw.append(int(best_row_shift(piece_lp, tmpl, max_shift)))
    sh_arr = np.asarray(shifts_raw, dtype=np.float64)
    if median_half_width > 0:
        rad = int(median_half_width)
    else:
        rad = max(16, min(120, h // 10))
    sh_sm = np.empty(h, dtype=np.float64)
    for k in range(h):
        lo = max(0, k - rad)
        hi = min(h, k + rad + 1)
        sh_sm[k] = float(np.median(sh_arr[lo:hi]))
    shifts_i = np.round(sh_sm).astype(np.int32)
    out = np.empty_like(frame, dtype=np.float32)
    shifts = [int(shifts_i[i]) for i in range(h)]
    for i in range(h):
        out[i] = shift_with_edge_padding(frame[i].astype(np.float32, copy=False), int(shifts_i[i]))
    return out, shifts


def _edge_column_per_row(frame: np.ndarray, j0: int, j1: int) -> np.ndarray:
    """Robust per-row horizontal index of strongest vertical edge (blanking vs luma)."""
    d = np.abs(np.diff(frame[:, j0:j1].astype(np.float64), axis=1))
    h, dw = d.shape
    if dw < 4:
        return np.zeros(h, dtype=np.float64)
    # Top-3 gradient columns averaged → less jitter than plain argmax.
    k = min(3, dw)
    idx = np.argpartition(-d, kth=k - 1, axis=1)[:, :k].astype(np.float64)
    return j0 + 1.0 + np.mean(idx, axis=1)


def _lock_horizontal_bands_overlap(
    frame: np.ndarray,
    strip_h: int | None = None,
    overlap: int | None = None,
    max_shift: int | None = None,
) -> np.ndarray:
    """Align stacked horizontal bands by correlating thin overlapping row windows.

    Integer line clocks leave small phase steps every few dozen lines; template
    alignment can still leave seams. Here each strip below a boundary is shifted
    horizontally to maximize correlation with the strip above (cumulative).
    """
    h, w = frame.shape
    if h < 40 or w < 96:
        return frame
    strip_h = strip_h or max(10, min(26, h // 22))
    overlap = overlap or max(8, min(22, strip_h - 1))
    max_shift = max_shift or max(32, min(220, int(0.25 * float(w))))
    c0, c1 = int(0.10 * w), int(0.90 * w)
    if c1 - c0 < 48:
        c0, c1 = max(0, w // 8), w - max(0, w // 8)

    out = frame.astype(np.float32, copy=True)
    boundaries = list(range(0, h, strip_h))
    if boundaries[-1] != h:
        boundaries.append(h)

    # Avoid weak matches that accumulate horizontal drift (smears / torn frames).
    min_corr = 0.30
    max_step = max(8, min(36, w // 18))
    for bi in range(1, len(boundaries) - 1):
        y_mid = boundaries[bi]
        if y_mid <= 0 or y_mid >= h:
            continue
        y_lo = max(0, y_mid - overlap)
        y_hi = min(h, y_mid + overlap)
        if y_mid - y_lo < 2 or y_hi - y_mid < 2:
            continue
        a_blk = out[y_lo:y_mid, c0:c1].astype(np.float64, copy=False)
        b_blk = out[y_mid:y_hi, c0:c1].astype(np.float64, copy=False)
        nra, nrb = a_blk.shape[0], b_blk.shape[0]
        nr = min(nra, nrb)
        if nr < 2:
            continue
        # Same row count: bottom of strip above vs top of strip below.
        a_blk = a_blk[-nr:, :]
        b_blk = b_blk[:nr, :]
        ncol = min(a_blk.shape[1], b_blk.shape[1])
        if ncol < 24:
            continue
        best_s, best_sc = 0, -1e30
        for sh in range(-max_shift, max_shift + 1):
            if sh >= 0:
                a = a_blk[:, : ncol - sh]
                b = b_blk[:, sh : sh + a.shape[1]]
            else:
                k = -sh
                b = b_blk[:, : ncol - k]
                a = a_blk[:, k : k + b.shape[1]]
            if a.shape[1] < 16 or a.shape[0] < 2 or b.shape[0] < 2:
                continue
            af = a.ravel() - float(np.mean(a))
            bf = b.ravel() - float(np.mean(b))
            den = float(np.linalg.norm(af) * np.linalg.norm(bf))
            if den < 1e-9:
                continue
            sc = float(np.dot(af, bf) / den)
            if sc > best_sc:
                best_sc = sc
                best_s = sh
        if best_sc >= min_corr and best_s != 0:
            bs = int(np.clip(int(best_s), -max_step, max_step))
            if bs == 0:
                continue
            for yy in range(y_mid, h):
                out[yy] = shift_with_edge_padding(out[yy], bs)
    return out


def _unify_vertical_band_offsets(
    frame: np.ndarray,
    roi_lo: float = 0.18,
    roi_hi: float = 0.55,
    depth: int = 0,
) -> np.ndarray:
    """Align stacked vertical bands using the strong edge inside the active-video ROI.

    Per-slice medians of the edge column jump by tens to hundreds of pixels between
    line groups (``bands''). We merge adjacent coarse line blocks with similar edge
    medians, then shift each merged band to match the reference band (strongest
    horizontal structure in the ROI). Recursive passes (up to four) converge after
    large shifts.
    """
    h, w = frame.shape
    if h < 16 or w < 64:
        return frame
    j0 = max(0, min(w - 16, int(w * roi_lo)))
    j1 = max(j0 + 8, min(w, int(w * roi_hi)))
    edges = _edge_column_per_row(frame, j0, j1)
    d = np.abs(np.diff(frame[:, j0:j1].astype(np.float64), axis=1))

    blk = max(16, min(40, h // 12))
    blocks: list[tuple[int, int, float]] = []
    b = 0
    while b * blk < h:
        s, e = int(b * blk), min(int((b + 1) * blk), h)
        b += 1
        if e <= s:
            continue
        blocks.append((s, e, float(np.median(edges[s:e]))))
    if len(blocks) < 2:
        return frame

    merged: list[tuple[int, int, float]] = []
    cs, ce, cm = blocks[0]
    merge_tol = float(max(32, min(120.0, 0.055 * float(w))))
    for s, e, m in blocks[1:]:
        if abs(m - cm) < merge_tol:
            ce = e
            cm = float(np.median(edges[cs:ce]))
        else:
            merged.append((cs, ce, cm))
            cs, ce, cm = s, e, m
    merged.append((cs, ce, cm))
    if len(merged) < 2:
        return frame

    s_ref, e_ref, _ = max(
        merged,
        key=lambda t: float(np.mean(np.max(d[t[0] : t[1], :].astype(np.float64), axis=1))),
    )
    tgt = float(np.median(edges[s_ref:e_ref]))
    max_shift = max(32, w // 2 - 2)
    out = frame.astype(np.float32, copy=True)
    changed = False
    for s, e, m in merged:
        if s == s_ref and e == e_ref:
            continue
        extra = int(np.clip(int(np.round(tgt - m)), -max_shift, max_shift))
        if extra == 0:
            continue
        changed = True
        for ii in range(s, e):
            out[ii] = shift_with_edge_padding(out[ii], extra)
    if changed and depth < 5:
        return _unify_vertical_band_offsets(out, roi_lo, roi_hi, depth + 1)
    return out


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
    frames_float: list[np.ndarray] = []
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

        # Active video as a fraction of each *measured* line (consecutive sync tips).
        alpha = 0.22
        beta = 0.92
        search = max(2, int(round(0.08 * line_period)))

        nlines = int(lines.size)
        if nlines < 140:
            continue

        tips = _sync_tips_from_line_starts(lines, smooth_demod, search)
        dtip = np.diff(tips.astype(np.float64))
        plausible_tip = dtip[(dtip >= 0.82 * float(line_period)) & (dtip <= 1.18 * float(line_period))]
        mlen = float(np.median(plausible_tip)) if plausible_tip.size >= 24 else float(line_period)
        # HSYNC tip spacing jitters tens of samples; raw per-line length smears static scenes (shear).
        dtip_sm = _median_smooth_1d(dtip, half_win=7)
        lo_b = 0.93 * mlen
        hi_b = 1.07 * mlen
        dtip_sm = np.clip(dtip_sm, lo_b, hi_b)
        width = max(64, int(round((beta - alpha) * mlen)))
        # One nominal line length for geometry: same (α,β) slice on every row → no shear from length dither.
        lf_geom = float(mlen)

        raw_rows: list[np.ndarray] = []
        valid_line_lengths: list[int] = []
        demod_sz = int(demod.size)
        for k in range(nlines - 1):
            t0 = int(tips[k])
            t1 = int(tips[k + 1])
            meas = float(dtip_sm[k]) if np.isfinite(dtip_sm[k]) else lf_geom
            # Clip active window end before next sync (spacing still logged in valid_line_lengths).
            t_end = min(t1 - 8, t0 + int(round(lf_geom)), demod_sz - 1)
            if t_end <= t0 + 16:
                continue
            a = int(np.round(t0 + alpha * lf_geom))
            b = int(np.round(t0 + beta * lf_geom))
            b = min(b, t_end, demod_sz - 1)
            a = max(t0, min(a, b - 9))
            if b <= a + 8:
                continue
            seg = demod[a:b].astype(np.float32, copy=False)
            x_src = np.linspace(0.0, 1.0, seg.size, endpoint=False, dtype=np.float32)
            x_dst = np.linspace(0.0, 1.0, width, endpoint=False, dtype=np.float32)
            row = np.interp(x_dst, x_src, seg).astype(np.float32, copy=False)
            raw_rows.append(row)
            valid_line_lengths.append(int(round(meas)))

        if len(raw_rows) < 120:
            continue

        frame = np.vstack(raw_rows).astype(np.float32, copy=False)

        # Drop short runs of inconsistent lines (often mixed half-frames).
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

        if frame.shape[0] < 120:
            continue

        nh = int(frame.shape[0])
        med_w = max(6, min(56, nh // 18))
        frame, row_shifts = _align_rows_to_fixed_template(
            frame,
            max_shift_frac=0.055,
            median_half_width=med_w,
        )

        # Drop VSYNC-field fragments (not a full ~25 ms raster at PAL line rates).
        if frame.shape[0] < 220:
            continue

        fi = len(frames_float)
        frames_float.append(frame.astype(np.float32, copy=False))
        frame_meta.append(
            {
                "frame_index": fi,
                "source_separator_index": i,
                "line_count": int(frame.shape[0]),
                "width": int(frame.shape[1]),
                "mean_abs_row_shift": float(np.mean(np.abs(np.asarray(row_shifts, dtype=np.float64)))) if row_shifts else 0.0,
                "line_len_mean_samples": float(np.mean(np.asarray(valid_line_lengths, dtype=np.float32))) if valid_line_lengths else 0.0,
                "line_len_std_samples": float(np.std(np.asarray(valid_line_lengths, dtype=np.float32))) if valid_line_lengths else 0.0,
            }
        )

        if max_frames > 0 and len(frames_float) >= max_frames:
            break

    # One global contrast scale for all frames avoids brightness flicker in MP4 playback.
    frames: list[np.ndarray] = []
    g1: float | None = None
    g99: float | None = None
    if frames_float:
        # Per-line DC (AM/FM slow gain) vs global contrast: removes vertical fade on static shots.
        centered: list[np.ndarray] = []
        for f in frames_float:
            r = f.astype(np.float32, copy=False)
            row_med = np.median(r, axis=1, keepdims=True)
            centered.append((r - row_med).astype(np.float32, copy=False))
        pool = np.concatenate([f.ravel() for f in centered])
        g1 = float(np.percentile(pool, 1.0))
        g99 = float(np.percentile(pool, 99.0))
        if g99 <= g1:
            g99 = g1 + 1e-6
        scale = 255.0 / (g99 - g1)
        for f in centered:
            frames.append(np.clip((f - g1) * scale, 0.0, 255.0).astype(np.uint8))

    meta = {
        "line_period_samples": int(line_period),
        "line_period_us": float(line_period / fs * 1e6),
        "frames_built": len(frames),
        "frame_meta": frame_meta,
        "grayscale_p1": g1,
        "grayscale_p99": g99,
        "reconstruction": "hsync_tip_intervals",
    }
    return frames, meta
