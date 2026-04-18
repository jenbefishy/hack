"""HSYNC/VSYNC detection logic for demodulated analog TV signals."""

from __future__ import annotations

import numpy as np

from .signal_processing import (
    estimate_line_period,
    find_runs,
    moving_average,
    periodicity_score,
    rms_energy,
    robust_threshold,
)


def detect_sync_pulses(demod: np.ndarray, fs: float) -> dict:
    """Detect HSYNC and VSYNC intervals from demodulated signal."""
    smooth = moving_average(demod, window=9).astype(np.float32)
    thr = robust_threshold(smooth)
    low = smooth < thr

    starts, ends = find_runs(low)
    widths = (ends - starts).astype(np.int32)
    if starts.size == 0:
        return {
            "threshold": thr,
            "line_period_samples": None,
            "hsync_width_samples_median": None,
            "hsync_starts": np.array([], dtype=np.int64),
            "vsync_starts": np.array([], dtype=np.int64),
            "vsync_ends": np.array([], dtype=np.int64),
            "vsync_intervals": np.empty((0, 2), dtype=np.int64),
            "pulse_starts": starts.astype(np.int64),
            "pulse_widths": widths.astype(np.int32),
        }

    min_w = int(fs * 2.0e-6)
    max_w = int(fs * 10.0e-6)
    h_mask = (widths >= min_w) & (widths <= max_w)
    h_starts = starts[h_mask].astype(np.int64)
    h_widths = widths[h_mask]

    if h_starts.size >= 8:
        line_period = estimate_line_period(h_starts, fs)
        if np.isfinite(line_period):
            d = np.diff(h_starts).astype(np.float64)
            good = np.abs(d - line_period) <= 0.35 * line_period
            stable = np.concatenate(([True], good))
            h_starts = h_starts[stable]
            h_widths = h_widths[stable]
    else:
        line_period = float("nan")

    h_med = float(np.median(h_widths)) if h_widths.size else float("nan")
    local_base = moving_average(smooth, window=max(11, int(fs * 0.10e-3))).astype(np.float32)
    upper_comp = np.clip(smooth - local_base, 0.0, None)
    lower_comp = np.clip(local_base - smooth, 0.0, None)

    # VSYNC candidates: significantly longer low-level pulses and/or long HSYNC gaps.
    long_mask = widths >= max(min_w, int(1.8 * h_med if np.isfinite(h_med) else fs * 8e-6))
    vsync_long = starts[long_mask].astype(np.int64)

    if h_starts.size >= 3 and np.isfinite(line_period):
        gaps = np.diff(h_starts).astype(np.float64)
        gap_idx = np.flatnonzero(gaps > 1.6 * line_period)
        vsync_gaps = h_starts[gap_idx + 1]
    else:
        vsync_gaps = np.array([], dtype=np.int64)

    raw_vsync = np.unique(np.concatenate((vsync_long, vsync_gaps))).astype(np.int64)
    if raw_vsync.size:
        # Merge nearby candidates into VSYNC groups.
        cluster_gap = int(fs * 2.5e-3)  # 2.5 ms
        clusters: list[list[int]] = [[int(raw_vsync[0])]]
        for idx in raw_vsync[1:]:
            if int(idx) - clusters[-1][-1] < cluster_gap:
                clusters[-1].append(int(idx))
            else:
                clusters.append([int(idx)])

        # Build a low-frequency trend to capture full VSYNC "valley" boundaries.
        trend_window = max(11, int(fs * 0.20e-3))  # 0.20 ms
        trend = moving_average(smooth, window=trend_window).astype(np.float32)
        t_med = float(np.median(trend))
        t_mad = float(np.median(np.abs(trend - t_med))) + 1e-12
        valley_thr = t_med - 1.8 * 1.4826 * t_mad
        valley_mask = trend < valley_thr
        # Low-level density helps reject sparse periodic HSYNC fragments near edges.
        low_density = moving_average(low.astype(np.float32), window=max(11, int(fs * 0.18e-3)))
        dense_low_thr = 0.16
        valley_starts, valley_ends = find_runs(valley_mask)
        upper_quiet_thr = max(0.05, float(np.percentile(upper_comp, 45.0)))
        lower_active_thr = max(0.02, float(np.percentile(lower_comp, 70.0)))

        min_vsync_w = int(fs * 0.08e-3)  # 0.08 ms
        max_vsync_w = int(fs * 8.0e-3)  # 8 ms
        keep = (valley_ends - valley_starts >= min_vsync_w) & (valley_ends - valley_starts <= max_vsync_w)
        valley_starts = valley_starts[keep]
        valley_ends = valley_ends[keep]

        intervals: list[tuple[int, int]] = []
        for cl in clusters:
            cl_start = cl[0]
            cl_end = cl[-1]
            cl_center = (cl_start + cl_end) // 2

            # Prefer full valley boundaries (start->end of dip) if available.
            iv_start = int(cl_start)
            iv_end = int(cl_end)
            if valley_starts.size:
                hit = np.flatnonzero((valley_starts <= cl_center) & (valley_ends >= cl_center))
                if hit.size:
                    k = int(hit[0])
                    iv_start = int(valley_starts[k])
                    iv_end = int(valley_ends[k])
                else:
                    # If center not inside valley run, snap to nearest one within 1 ms.
                    centers = (valley_starts + valley_ends) // 2
                    nearest = int(np.argmin(np.abs(centers - cl_center)))
                    if abs(int(centers[nearest]) - cl_center) <= int(fs * 1.0e-3):
                        iv_start = int(valley_starts[nearest])
                        iv_end = int(valley_ends[nearest])

            # Trim edges to avoid capturing bordering periodic HSYNC structure.
            left_trim = int(fs * 0.08e-3)  # 0.08 ms
            right_trim = int(fs * 0.06e-3)  # 0.06 ms
            min_keep = int(fs * 0.05e-3)
            if iv_end - iv_start > (left_trim + right_trim + min_keep):
                iv_start += left_trim
                iv_end -= right_trim

            # Refine boundaries by requiring dense low-level occupancy.
            if iv_end > iv_start:
                lo = max(0, iv_start)
                hi = min(demod.size, iv_end)
                if hi > lo:
                    seg = low_density[lo:hi]
                    dense_idx = np.flatnonzero(seg >= dense_low_thr)
                    if dense_idx.size:
                        iv_start = lo + int(dense_idx[0])
                        iv_end = lo + int(dense_idx[-1]) + 1
                        # Asymmetric padding: keep left tight, extend right tail a bit more.
                        left_pad = int(fs * 0.03e-3)  # 0.03 ms
                        right_pad = int(fs * 0.12e-3)  # 0.12 ms
                        iv_start = max(0, iv_start - left_pad)
                        iv_end = min(demod.size, iv_end + right_pad)

                    # If density lock is weak, trim to the valley core by lower-component prominence.
                    core = lower_comp[lo:hi]
                    if core.size:
                        peak = float(np.max(core))
                        if peak > 1e-9:
                            core_thr = max(lower_active_thr, 0.55 * peak)
                            core_idx = np.flatnonzero(core >= core_thr)
                            if core_idx.size:
                                cs = lo + int(core_idx[0])
                                ce = lo + int(core_idx[-1]) + 1
                                core_pad_l = int(fs * 0.03e-3)
                                core_pad_r = int(fs * 0.05e-3)
                                iv_start = max(iv_start, max(0, cs - core_pad_l))
                                iv_end = min(iv_end, min(demod.size, ce + core_pad_r))

            # Shift left boundary rightward to the first stable "broken structure" zone.
            if iv_end - iv_start > int(fs * 0.20e-3):
                scan_w = max(64, int(fs * 0.05e-3))
                scan_step = max(1, scan_w // 4)
                need = 3
                good = 0
                best_start = None
                for k in range(iv_start, max(iv_start + 1, iv_end - scan_w + 1), scan_step):
                    b = min(iv_end, k + scan_w)
                    if b - k < scan_w:
                        break
                    upper_rms = float(np.sqrt(np.mean(upper_comp[k:b] ** 2)))
                    lower_rms = float(np.sqrt(np.mean(lower_comp[k:b] ** 2)))
                    dens = float(np.mean(low_density[k:b]))
                    cond = (
                        (lower_rms / (upper_rms + 1e-9) >= 2.0)
                        and (upper_rms <= 1.15 * upper_quiet_thr)
                        and (dens >= 0.05)
                    )
                    if cond:
                        good += 1
                        if good >= need:
                            best_start = k - (need - 1) * scan_step
                            break
                    else:
                        good = 0
                if best_start is not None and best_start > iv_start:
                    max_shift = int(fs * 0.90e-3)
                    iv_start = min(best_start, iv_start + max_shift)

            # Extend right boundary for broken periodic structure.
            if np.isfinite(line_period) and iv_end > iv_start:
                lag = max(1, int(round(line_period)))
                step = max(1, lag // 4)
                block = max(2 * lag, int(fs * 0.20e-3))
                max_extra = int(fs * 1.0e-3)
                target_end = min(demod.size, iv_end + max_extra)
                probe_end = iv_end
                while probe_end + step < target_end:
                    s = probe_end
                    e = min(target_end, probe_end + block)
                    if e - s <= lag + 4:
                        break
                    low_score = periodicity_score(lower_comp[s:e], lag)
                    high_score = periodicity_score(upper_comp[s:e], lag)
                    density_mean = float(np.mean(low_density[s:e]))
                    upper_rms = float(np.sqrt(np.mean(upper_comp[s:e] ** 2)))
                    lower_rms = float(np.sqrt(np.mean(lower_comp[s:e] ** 2)))
                    broken_by_periodicity = (low_score >= 0.32) and (high_score <= 0.22) and (density_mean >= 0.05)
                    broken_by_asymmetry = (
                        (lower_rms / (upper_rms + 1e-9) >= 2.2)
                        and (upper_rms <= upper_quiet_thr)
                        and (lower_rms >= lower_active_thr)
                        and (density_mean >= 0.04)
                    )
                    if not (broken_by_periodicity or broken_by_asymmetry):
                        break
                    probe_end += step
                iv_end = int(probe_end)

            iv_start = int(iv_start)
            iv_end = int(iv_end)
            if iv_end <= iv_start:
                iv_end = iv_start + 1
            intervals.append((iv_start, iv_end))

        # Fallback: add valleys with broken structure even if no raw seed.
        if np.isfinite(line_period):
            lag = max(1, int(round(line_period)))
            for vs, ve in zip(valley_starts, valley_ends):
                s = int(vs)
                e = int(ve)
                if e - s < int(fs * 0.40e-3):
                    continue
                low_score = periodicity_score(lower_comp[s:e], lag)
                high_score = periodicity_score(upper_comp[s:e], lag)
                upper_rms = float(np.sqrt(np.mean(upper_comp[s:e] ** 2)))
                lower_rms = float(np.sqrt(np.mean(lower_comp[s:e] ** 2)))
                dens = float(np.mean(low_density[s:e]))
                broken_by_periodicity = (low_score >= 0.78) and (high_score <= 0.62)
                broken_by_asymmetry = (lower_rms / (upper_rms + 1e-9) >= 1.6) and (dens >= 0.06)
                if not (broken_by_periodicity or broken_by_asymmetry):
                    continue

                overlaps = False
                for a, b in intervals:
                    if not (e < a or s > b):
                        overlaps = True
                        break
                if overlaps:
                    continue

                add_l = int(fs * 0.02e-3)
                add_r = int(fs * 0.08e-3)
                intervals.append((max(0, s - add_l), min(demod.size, e + add_r)))

        # Merge overlaps/touching intervals.
        intervals.sort(key=lambda p: p[0])
        merged: list[tuple[int, int]] = []
        for s, e in intervals:
            if not merged or s > merged[-1][1]:
                merged.append((s, e))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))

        min_final_w = int(fs * 0.06e-3)
        merged = [(s, e) for (s, e) in merged if (e - s) >= min_final_w]

        # Quality gate: duration and structural break.
        quality: list[tuple[int, int]] = []
        if np.isfinite(line_period):
            lag = max(1, int(round(line_period)))
            ctx = max(3 * lag, int(fs * 0.40e-3))
            for s, e in merged:
                if e - s > int(fs * 0.35e-3):
                    ws = max(64, int(fs * 0.06e-3))
                    step = max(1, ws // 3)
                    runs: list[tuple[int, int]] = []
                    run_s = None
                    run_e = None
                    for k in range(s, max(s + 1, e - ws + 1), step):
                        b = min(e, k + ws)
                        if b - k < ws:
                            break
                        upper_ps_w = periodicity_score(upper_comp[k:b], lag)
                        upper_r_w = rms_energy(upper_comp[k:b])
                        lower_r_w = rms_energy(lower_comp[k:b])
                        ratio_w = lower_r_w / (upper_r_w + 1e-9)
                        dens_w = float(np.mean(low_density[k:b]))
                        broken_w = (
                            (ratio_w >= 1.45) and (dens_w >= 0.06) and (upper_r_w <= 1.35 * upper_quiet_thr)
                        ) or ((ratio_w >= 1.20) and (dens_w >= 0.06) and (upper_ps_w <= 0.70))
                        if broken_w:
                            if run_s is None:
                                run_s, run_e = k, b
                            else:
                                run_e = b
                        else:
                            if run_s is not None:
                                runs.append((run_s, run_e))
                                run_s = None
                                run_e = None
                    if run_s is not None:
                        runs.append((run_s, run_e))

                    if runs:
                        peak_idx = s + int(np.argmax(lower_comp[s:e]))
                        chosen = None
                        for rs, re in runs:
                            if rs <= peak_idx <= re:
                                chosen = (rs, re)
                                break
                        if chosen is None:
                            chosen = max(runs, key=lambda p: p[1] - p[0])
                        trim_l = int(fs * 0.04e-3)
                        trim_r = int(fs * 0.06e-3)
                        s = max(s, chosen[0] - trim_l)
                        e = min(e, chosen[1] + trim_r)

                dur_us = (e - s) / fs * 1e6
                la = max(0, s - ctx)
                lb = s
                ra = e
                rb = min(demod.size, e + ctx)
                ups = periodicity_score(upper_comp[s:e], lag)
                up_i = rms_energy(upper_comp[s:e])
                up_pre = rms_energy(upper_comp[la:lb])
                up_post = rms_energy(upper_comp[ra:rb])
                updrop_pre = (up_pre - up_i) / (up_pre + 1e-9) if up_pre > 0 else 0.0
                updrop_post = (up_post - up_i) / (up_post + 1e-9) if up_post > 0 else 0.0
                too_short = dur_us < 190.0
                no_structure_break = (ups > 0.94) and (updrop_pre < 0.18) and (updrop_post < 0.18)
                if too_short or no_structure_break:
                    continue
                quality.append((int(s), int(e)))
        else:
            quality = merged

        # Augment/expand with valley-backed candidates.
        augmented = list(quality)
        lag = max(1, int(round(line_period))) if np.isfinite(line_period) else 0
        for vs, ve in zip(valley_starts, valley_ends):
            s = int(vs)
            e = int(ve)
            if e - s < int(fs * 0.30e-3):
                continue
            dur_us = (e - s) / fs * 1e6
            upper_rms = rms_energy(upper_comp[s:e])
            lower_rms = rms_energy(lower_comp[s:e])
            ratio = lower_rms / (upper_rms + 1e-9)
            ups = periodicity_score(upper_comp[s:e], lag) if np.isfinite(line_period) else 0.0
            lps = periodicity_score(lower_comp[s:e], lag) if np.isfinite(line_period) else 0.0
            dens = float(np.mean(low_density[s:e]))
            valley_like_vsync = (
                (dur_us >= 350.0) and (ratio >= 1.35) and (dens >= 0.07) and (ups <= 0.82)
            ) or ((dur_us >= 900.0) and (ratio >= 1.55) and (dens >= 0.06) and (lps >= 0.75))
            if not valley_like_vsync:
                continue

            expanded = False
            join_gap = int(fs * 0.45e-3)
            for i, (a, b) in enumerate(augmented):
                if not (e < a - join_gap or s > b + join_gap):
                    augmented[i] = (min(a, s), max(b, e))
                    expanded = True
                    break
            if not expanded:
                augmented.append((s, e))

        # Normalize augmented intervals.
        augmented.sort(key=lambda p: p[0])
        merged_aug: list[tuple[int, int]] = []
        for s, e in augmented:
            if not merged_aug or s > merged_aug[-1][1]:
                merged_aug.append((s, e))
            else:
                merged_aug[-1] = (merged_aug[-1][0], max(merged_aug[-1][1], e))

        # Final boundary refinement.
        refined: list[tuple[int, int]] = []
        if np.isfinite(line_period):
            lag = max(1, int(round(line_period)))
            ws = max(64, int(fs * 0.06e-3))
            step = max(1, ws // 3)
            for s0, e0 in merged_aug:
                s = int(s0)
                e = int(e0)
                if e - s <= ws + 4:
                    continue

                def is_broken(a: int, b: int) -> bool:
                    upper_ps_w = periodicity_score(upper_comp[a:b], lag)
                    upper_r_w = rms_energy(upper_comp[a:b])
                    lower_r_w = rms_energy(lower_comp[a:b])
                    ratio_w = lower_r_w / (upper_r_w + 1e-9)
                    dens_w = float(np.mean(low_density[a:b]))
                    return (
                        (ratio_w >= 1.55) and (dens_w >= 0.065) and (upper_r_w <= 1.25 * upper_quiet_thr)
                    ) or ((ratio_w >= 1.35) and (dens_w >= 0.065) and (upper_ps_w <= 0.72))

                # Left trim.
                good_need = 2
                good = 0
                left_hit = None
                for k in range(s, e - ws + 1, step):
                    b = k + ws
                    if is_broken(k, b):
                        good += 1
                        if good >= good_need:
                            left_hit = k - (good_need - 1) * step
                            break
                    else:
                        good = 0
                if left_hit is not None:
                    s = max(s, left_hit - int(fs * 0.03e-3))

                # Right trim.
                good = 0
                right_hit = None
                k = e - ws
                while k >= s:
                    b = k + ws
                    if is_broken(k, b):
                        good += 1
                        if good >= good_need:
                            right_hit = k + ws + (good_need - 1) * step
                            break
                    else:
                        good = 0
                    k -= step
                if right_hit is not None:
                    e = min(e, right_hit + int(fs * 0.05e-3))

                # Clamp to deep valley core.
                if e - s > int(fs * 0.20e-3):
                    base_s = s
                    base_e = e
                    seg_trend = trend[base_s:base_e]
                    if seg_trend.size:
                        p15 = float(np.percentile(seg_trend, 15.0))
                        p35 = float(np.percentile(seg_trend, 35.0))
                        core_thr = p15 + 0.35 * (p35 - p15)
                        core_mask = seg_trend <= core_thr
                        c_starts, c_ends = find_runs(core_mask)
                        min_core = int(fs * 0.08e-3)
                        good = (c_ends - c_starts) >= min_core
                        c_starts = c_starts[good]
                        c_ends = c_ends[good]
                        if c_starts.size:
                            deep_idx = int(np.argmin(seg_trend))
                            pick = np.flatnonzero((c_starts <= deep_idx) & (c_ends >= deep_idx))
                            if pick.size == 0:
                                pick = np.array([int(np.argmax(c_ends - c_starts))], dtype=np.int64)
                            j = int(pick[0])
                            rs = base_s + int(c_starts[j])
                            re = base_s + int(c_ends[j])
                            core_pad_l = int(fs * 0.03e-3)
                            core_pad_r = int(fs * 0.05e-3)
                            s = max(base_s, rs - core_pad_l)
                            e = min(base_e, re + core_pad_r)

                # Expand back from core if top periodicity remains absent.
                expand_ws = max(64, int(fs * 0.06e-3))
                expand_step = max(1, expand_ws // 3)
                left_bound = int(s0)
                right_bound = int(e0)
                no_top_ps_thr = 0.74
                no_top_amp_mul = 1.35
                no_top_density_thr = 0.05

                while s - expand_step >= left_bound:
                    a = max(left_bound, s - expand_ws)
                    b = min(e, a + expand_ws)
                    if b - a < expand_ws:
                        break
                    upper_ps_w = periodicity_score(upper_comp[a:b], lag)
                    upper_r_w = rms_energy(upper_comp[a:b])
                    dens_w = float(np.mean(low_density[a:b]))
                    if (upper_ps_w <= no_top_ps_thr) and (upper_r_w <= no_top_amp_mul * upper_quiet_thr) and (dens_w >= no_top_density_thr):
                        s = max(left_bound, s - expand_step)
                    else:
                        break

                while e + expand_step <= right_bound:
                    b = min(right_bound, e + expand_ws)
                    a = max(s, b - expand_ws)
                    if b - a < expand_ws:
                        break
                    upper_ps_w = periodicity_score(upper_comp[a:b], lag)
                    upper_r_w = rms_energy(upper_comp[a:b])
                    dens_w = float(np.mean(low_density[a:b]))
                    if (upper_ps_w <= no_top_ps_thr) and (upper_r_w <= no_top_amp_mul * upper_quiet_thr) and (dens_w >= no_top_density_thr):
                        e = min(right_bound, e + expand_step)
                    else:
                        break

                if e - s <= int(fs * 0.18e-3):
                    continue

                ups_i = periodicity_score(upper_comp[s:e], lag)
                upper_i = rms_energy(upper_comp[s:e])
                lower_i = rms_energy(lower_comp[s:e])
                ratio_i = lower_i / (upper_i + 1e-9)
                if (ups_i > 0.82) and (ratio_i < 1.7):
                    continue
                refined.append((s, e))
        else:
            refined = merged_aug

        vsync_intervals = np.array(refined, dtype=np.int64) if refined else np.empty((0, 2), dtype=np.int64)
        if vsync_intervals.size:
            vsync_starts = vsync_intervals[:, 0]
            vsync_ends = vsync_intervals[:, 1]
        else:
            vsync_starts = np.array([], dtype=np.int64)
            vsync_ends = np.array([], dtype=np.int64)
    else:
        vsync_intervals = np.empty((0, 2), dtype=np.int64)
        vsync_starts = np.array([], dtype=np.int64)
        vsync_ends = np.array([], dtype=np.int64)

    # Final HSYNC cleanup:
    # 1) HSYNC cannot be inside VSYNC intervals.
    # 2) If upper periodicity is missing, this pulse is not HSYNC.
    if h_starts.size:
        keep = np.ones(h_starts.size, dtype=bool)

        if vsync_intervals.size:
            for i, hs in enumerate(h_starts):
                if np.any((hs >= vsync_intervals[:, 0]) & (hs < vsync_intervals[:, 1])):
                    keep[i] = False

        if np.isfinite(line_period):
            lag = max(1, int(round(line_period)))
            probe = max(2 * lag, int(fs * 0.16e-3))
            upper_ps_thr = 0.38
            for i, hs in enumerate(h_starts):
                if not keep[i]:
                    continue
                s = max(0, int(hs) - probe // 2)
                e = min(demod.size, int(hs) + probe // 2)
                if e - s <= lag + 4:
                    continue
                upper_ps = periodicity_score(upper_comp[s:e], lag)
                if upper_ps < upper_ps_thr:
                    keep[i] = False

        h_starts = h_starts[keep]

    return {
        "threshold": float(thr),
        "line_period_samples": float(line_period) if np.isfinite(line_period) else None,
        "hsync_width_samples_median": float(h_med) if np.isfinite(h_med) else None,
        "hsync_starts": h_starts,
        "vsync_starts": vsync_starts,
        "vsync_ends": vsync_ends,
        "vsync_intervals": vsync_intervals,
        "pulse_starts": starts.astype(np.int64),
        "pulse_widths": widths.astype(np.int32),
    }
