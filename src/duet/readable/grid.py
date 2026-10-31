"""Beat grid: fuse the score model's beat numbering with the frame head's beat pulse.

Output
  G        times (s) of global beats 0..K (global beat 0 = downbeat of bar 0)
  shift(t) integer correction added to the model's onset q for a note at time t

Idea
  The model gives every note a position q (beats). Its integer beats with
  notes give a tempo map  M: time -> model position.  The frame head gives a
  pulse of beat times T_j.  Walking along the pulse:
    * where consecutive frame beats are one model beat apart ("agree"), the
      frame times are used (more precise than note medians);
    * where they disagree we have a gap between two agreeing stretches. Let the
      gap hold n_f frame intervals and n_m model beats.
        - n_f == n_m: same count, timing taken from the frames if the pulse is
          steady there, else from the model;
        - |n_f - n_m| == 1, the frame pulse is steady AND the frame downbeats
          after the gap support the frame count: the model miscounted (e.g.
          merged a rolled chord into the next beat). Frames win; all later
          model positions are shifted by n_f - n_m;
        - otherwise (fermata, accelerando, cadenza, unreliable frames) the
          model's beats are used.
"""

import numpy as np
from scipy.signal import find_peaks


def fix_rewinds(onsets, q, B, cfg, log, fbeats=None):
    """The model sometimes numbers a passage twice (its positions jump back by a
    few beats, often a whole bar, and carry on from there) or skips part of its
    numbering (a jump ahead far beyond what the elapsed time allows). Anchors
    built from such numbering squeeze the passage into a beat or two, or leave
    an almost empty bar. A jump that persists (the following notes continue
    from the new numbering instead of returning to the earlier line) is undone
    by shifting everything after it by whole bars; a short run of misnumbered
    notes that returns to the line is moved alone. Jumps that are not close to
    whole bars, and jumps ahead across a pause, are left to the grid.
    fbeats = (times, is_downbeat) of confident frame beats: where they show a
    steady pulse after the jump, the shift may be half a beat off a whole bar
    (the model's new numbering then starts on an off-beat), and notes just
    before the jump that sit on such a frame beat join the new numbering.
    Returns corrected q."""
    q = np.asarray(q, float).copy()
    t = np.asarray(onsets, float)
    i = 1
    while i < len(q):
        prev = np.nonzero((t < t[i] - 0.02) & (t >= t[i] - cfg.rewind_fit_s))[0]
        prev = (
            prev[q[prev] >= np.maximum.accumulate(q[prev]) - 0.5] if len(prev) else prev
        )
        # the earlier line needs several distinct attacks (a chord alone gives no tempo)
        tp = np.unique(np.round(t[prev], 2))
        if len(tp) < 4 or tp[-1] - tp[0] < 1.0:
            i += 1
            continue
        b, a = np.polyfit(t[prev], q[prev], 1)  # earlier line: q = a + b t
        if not (0 < b < 20):
            i += 1
            continue
        back = q[i] < q[prev].max() - cfg.rewind_min
        # a jump ahead of the line (more beats than the elapsed time allows) is the same
        # error the other way: the model skipped part of its numbering
        # (only right after the previous attack: across a pause, held notes or a fermata may hold the beats)
        ahead = (
            q[i] > q[prev].max() + cfg.rewind_min
            and q[i] - (a + b * t[i]) > cfg.rewind_min + 1
            and t[i] - t[prev].max() <= 0.75 / b
        )
        if not (back or ahead):
            i += 1
            continue
        nxt = np.nonzero((t >= t[i]) & (t <= t[i] + cfg.rewind_check_s))[0]
        line_new = q[i] + b * (t[nxt] - t[i])
        line_old = a + b * t[nxt]
        new_side = np.abs(q[nxt] - line_new) < np.abs(q[nxt] - line_old)
        persist = np.mean(new_side)
        if len(nxt) < 4:
            i += 1
            continue
        # offset of the new numbering from the earlier line (median over the notes that follow it;
        # chords at the jump are often labelled half in the old, half in the new numbering)
        s = float(np.median(line_old[new_side] - q[nxt][new_side]))
        # only whole bars: the model labelled a bar twice or skipped one; other amounts are
        # usually rubato the earlier line does not describe, left to the grid
        k = round(s / B)
        if (
            k == 0
            or abs(s - k * B) > min(cfg.rewind_bar_tol, B / 2 - 0.25)
            or (k > 0) != back
        ):
            i += 1
            continue
        s = k * B
        half = 0.0
        if persist >= cfg.rewind_persist and fbeats is not None:
            s, half = _frame_phase(t, q, i, nxt[new_side], s, B, fbeats, cfg)
        if persist >= cfg.rewind_persist:
            log(
                f"WARNING grid: model numbering jumps {'back' if back else 'ahead'} {abs(q[i] - q[prev].max()):.1f} beats "
                f"at {t[i]:.2f}s and continues from there ({persist:.0%} of the next notes); later positions shifted {s:+g}"
                + (" (half-beat phase from the frame beats)" if half else "")
            )
            q0 = q[i] + s
            q[nxt[new_side]] += (
                s  # notes in the window already on the earlier line stay
            )
            q[t > t[nxt[-1]]] += s
            if half:
                # the half beat is the model's local slip: it ends where the frame beats fall on
                # half beats of the shifted numbering again (the model got back into phase)
                te = _phase_end(t, q, t[i], fbeats)
                q[t >= te] -= half
                log(f"grid: half-beat phase {t[i]:.2f}-{te:.2f}s")
            # the chord at the jump is often labelled half in each numbering: one position for all of it
            ch = nxt[t[nxt] <= t[i] + 0.05]
            if len(ch) > 1:
                q[ch] = q[ch][np.argmin(np.abs(q[ch] - (a + b * t[i])))]
            if half:
                # notes in the beat before the jump that sit on a frame beat belong to the new numbering
                ft, _ = fbeats
                for tb in ft[(ft > t[i] - 1.0 / b) & (ft < t[i] - 0.05)]:
                    on_b = np.nonzero(
                        (np.abs(t - tb) < cfg.beat_refine_window) & (t < t[i])
                    )[0]
                    if len(on_b):
                        pos = q0 - round((t[i] - tb) * b * 2) / 2
                        log(
                            f"grid: {len(on_b)} notes at {tb:.2f}s on a frame beat before the jump -> position {pos:g}"
                        )
                        q[on_b] = pos
            i = int(nxt[-1]) + 1
            continue
        # a short run of notes numbered off the line, after which the numbering
        # returns to it (three old-side notes in a row): move just those notes
        end = next(
            (j for j in range(len(nxt) - 2) if not new_side[j : j + 3].any()), None
        )
        if end:
            idx = nxt[:end][new_side[:end]]
            log(
                f"grid: {len(idx)} notes at {t[i]:.2f}-{t[idx[-1]]:.2f}s numbered {abs(s)} beats too "
                f"{'low' if back else 'high'} by the model; moved"
            )
            q[idx] += s
            for j in idx:  # struck with a note that stays: that note's position
                mate = np.nonzero(
                    (np.abs(t - t[j]) < 0.03) & ~np.isin(np.arange(len(t)), idx)
                )[0]
                if len(mate):
                    q[j] = q[mate[0]]
            i = int(nxt[end])
            continue
        i += 1
    return q


def _frame_phase(t, q, i, new_idx, s, B, fbeats, cfg):
    """choose s, s - 1/2 or s + 1/2 so that the confident frame beats after the
    jump fall on whole beats and frame downbeats on bar lines"""
    ft, fd = fbeats
    idx = np.unique(
        np.concatenate(
            [
                new_idx,
                np.nonzero(
                    (t > t[new_idx].max()) & (t <= t[i] + 2 * cfg.rewind_check_s)
                )[0],
            ]
        )
    )
    o = idx[np.argsort(t[idx])]
    win = (ft > t[i] - 0.05) & (ft < t[o[-1]] + 0.05)
    if win.sum() < 3:
        return s, 0.0

    def score(c):
        pos = np.interp(ft[win], t[o], q[o] + c)
        d_beat = np.abs(pos - np.round(pos)).mean()
        dn = pos[fd[win]]
        d_bar = np.mean(np.abs(dn - B * np.round(dn / B)) > 0.25) if len(dn) else 0.0
        return d_beat + 0.5 * d_bar

    cands = {c: score(c) for c in (s, s - 0.5, s + 0.5)}
    best = min(cands, key=cands.get)
    if best != s and cands[best] < cands[s] - cfg.rewind_phase_margin:
        return best, best - s
    return s, 0.0


def _phase_end(t, q, t0, fbeats, horizon=60.0):
    """first time after t0 where two consecutive confident frame beats fall on
    half beats of q"""
    ft, _ = fbeats
    o = np.argsort(t)
    sel = ft[(ft > t0 + 0.5) & (ft < min(t0 + horizon, t[o[-1]]))]
    if not len(sel):
        return np.inf
    pos = np.interp(sel, t[o], q[o])
    off = np.abs(pos - np.round(pos)) > 0.35
    for j in range(1, len(sel)):
        if off[j] and off[j - 1]:
            # just after the last frame beat still in phase (before the model's slip)
            return (sel[j - 2] + sel[j - 1]) / 2 if j >= 2 else sel[j - 1] - 0.1
    return np.inf


def model_anchors(onsets, q):
    """integer model beats -> median onset time, monotone"""
    qi = np.round(q).astype(int)
    on_int = np.isclose(q, qi, atol=1e-6)
    ks, ts = [], []
    for k in np.unique(qi[on_int]):
        t = onsets[on_int & (qi == k)]
        ks.append(k)
        ts.append(float(np.median(t)))
    ks, ts = np.array(ks, float), np.array(ts)
    if not len(ks):
        return ks, ts
    keep = [0]
    for i in range(1, len(ks)):  # drop anchors that go back in time
        if ts[i] > ts[keep[-1]] + 0.02:
            keep.append(i)
    return ks[keep], ts[keep]


def build_grid(frames, onsets, q, B, cfg, log, pulse=1, bars=None):
    """B: quarters per bar; pulse: quarters per felt beat (3/2 in compound
    meters). Internally everything is counted in pulses; G is returned per
    quarter note and shifts in quarters."""
    pulse = float(pulse)
    if pulse != 1:
        Bp = int(round(B / pulse))
        Bp = Bp * 2 if Bp == 2 else Bp  # 6/8: judge bar phase over two bars (12/8)
        pulse_bars = (
            None if bars is None else tuple(np.asarray(v) / pulse for v in bars)
        )
        Gp, sh = _build_grid(
            frames, onsets, np.asarray(q) / pulse, Bp, cfg, log, pulse_bars
        )
        nq = int(np.floor((len(Gp) - 1) * pulse))
        G = np.interp(np.arange(nq + 1) / pulse, np.arange(len(Gp)), Gp)

        def shift_at(t):
            return sh(t) * pulse

        shift_at.free = sh.free
        return G, shift_at
    return _build_grid(frames, onsets, q, B, cfg, log, bars)


def _build_grid(frames, onsets, q, B, cfg, log, bars=None):
    ks, ts = model_anchors(onsets, q)
    if len(ks) < 2:
        # Very short pieces may contain no two distinct integer-beat attacks.
        # Use their fractional positions, then the MIDI export's 120 BPM default
        # if a single attack gives no tempo evidence. No note data is changed.
        uq = np.unique(q)
        ut = np.array([np.median(onsets[np.isclose(q, k)]) for k in uq])
        keep = [0]
        for i in range(1, len(uq)):
            if ut[i] > ut[keep[-1]] + 0.02:
                keep.append(i)
        ks, ts = uq[keep], ut[keep]
        if len(ks) < 2:
            ks, ts = np.r_[ks, ks[0] + 1], np.r_[ts, ts[0] + 0.5]
            log("WARNING grid: a single attack cannot establish tempo; using 120 BPM")
        else:
            log(
                "WARNING grid: insufficient integer-beat anchors; using fractional model attacks"
            )
    M = lambda t: np.interp(t, ts, ks, left=None, right=None)

    def Minv(k):  # model position -> time, linear extrapolation at the ends
        if k < ks[0]:
            r = (ts[1] - ts[0]) / (ks[1] - ks[0])
            return ts[0] - (ks[0] - k) * r
        if k > ks[-1]:
            r = (ts[-1] - ts[-2]) / (ks[-1] - ks[-2])
            return ts[-1] + (k - ks[-1]) * r
        return float(np.interp(k, ks, ts))

    ft = frames.performance_time_sec.values
    pk, _ = find_peaks(
        frames.p_any_beat.values,
        height=cfg.beat_peak_height,
        distance=cfg.beat_peak_distance,
    )
    sel = (ft[pk] >= ts[0] - 0.25) & (ft[pk] <= ts[-1] + 0.25)
    T = ft[pk][sel].copy()
    PDB = frames.p_downbeat.values[pk][sel]
    for i in range(len(T)):
        near = onsets[np.abs(onsets - T[i]) < cfg.beat_refine_window]
        if len(near):
            T[i] = np.median(near)
    m = M(T)
    d = np.diff(m)
    r = np.round(m).astype(int)
    # agreement is judged on integer beat numbers so a slow drift (e.g. through
    # an empty bar where the model tempo is interpolated) surfaces as a gap
    # a frame beat must also be in phase with a model beat: a pulse locked to
    # the off-beat 8ths advances at the right rate but is half a beat out
    onbeat = np.abs(m - r) < cfg.beat_phase_tol
    agree = (np.diff(r) == 1) & (np.abs(d - 1) < 0.35) & onbeat[:-1] & onbeat[1:]
    if (~onbeat).mean() > 0.02:
        log(
            f"grid: {int((~onbeat).sum())} frame beats lie between model beats (off-beat pulse); model beat times used there"
        )
    log(
        f"grid: {len(ks)} model beat anchors, {len(T)} frame beats, {agree.sum()}/{len(agree)} frame intervals agree with the model"
    )

    def steady(a, b):
        """frame intervals in the gap a..b within 20% of the agreeing intervals
        right before AND right after it (a pulse that halves or doubles is not steady)"""
        iv = np.diff(T[a : b + 1])
        D_ = np.diff(T)
        before = [D_[i] for i in range(max(0, a - 3), a) if agree[i]]
        after = [D_[i] for i in range(b, min(len(D_), b + 3)) if agree[i]]
        if not len(iv):
            return False
        for side in (before, after):
            if side and not np.all(np.abs(iv / np.median(side) - 1) < 0.2):
                return False
        return bool(before or after)

    def downbeat_support(j0, s_try):
        """(half-bar-invariant, exact) downbeat agreement over the next W bars:
        mean p_downbeat on beats labelled downbeat minus on the other beats"""
        W = cfg.phase_window_bars * B
        idx = [j for j in range(j0, min(len(T) - 1, j0 + W)) if agree[j]]
        if len(idx) < 2 * B:
            return 0.0, 0.0

        def score(unit):
            g = [int(round(m[j])) + s_try for j in idx]
            if bars is None:
                is_bar = [gg % unit == 0 for gg in g]
            else:
                starts, lengths = bars
                bi = np.clip(
                    np.searchsorted(starts, g, side="right") - 1, 0, len(starts) - 1
                )
                rel = np.asarray(g) - starts[bi]
                # The actual boundary is authoritative; an optional midpoint
                # makes the first score invariant to half-bar ambiguity.
                is_bar = np.isclose(rel, 0, atol=0.05)
                if unit != B:
                    is_bar |= np.isclose(rel, lengths[bi] / 2, atol=0.05)
            on = [PDB[j] for j, yes in zip(idx, is_bar) if yes]
            off = [PDB[j] for j, yes in zip(idx, is_bar) if not yes]
            return (np.mean(on) if on else 0) - (np.mean(off) if off else 0)

        half = B // 2 if B % 2 == 0 else B
        return score(half), score(B)

    # ---- walk the pulse --------------------------------------------------------
    beats = {}  # global beat -> time
    shifts = [(-np.inf, 0)]  # (from time, shift)
    free_zones = []  # time spans where model positions are not used
    s = 0
    j = 0
    n = len(T)
    # region before the first agreeing interval: model beats
    first_agree = int(np.argmax(agree)) if agree.any() else n
    for k in range(
        int(np.floor(ks[0])),
        int(round(m[first_agree])) if first_agree < n else int(ks[-1]) + 1,
    ):
        beats[k] = Minv(k)
    j = first_agree
    while j < n - 1:
        if agree[j]:
            g = int(r[j]) + s
            beats[g] = T[j]
            beats[g + 1] = T[j + 1]
            j += 1
            continue
        a = j
        b = a
        while b < n - 1 and not agree[b]:
            b += 1
        if b >= n - 1 and not agree[min(b, n - 2)]:
            b = n - 1
        n_f = b - a
        n_m = int(r[b] - r[a])
        ga = int(r[a]) + s
        frames_steady = steady(a, b)
        # how many beats does the gap really hold? candidates n_m + c; the
        # frame downbeats over the next bars (where frames and model agree
        # beat by beat) decide; half-bar offsets do not count as evidence
        D_ = np.diff(T)
        near_iv = [
            D_[i]
            for i in list(range(max(0, a - 3), a)) + list(range(b, min(len(D_), b + 3)))
            if agree[i]
        ]
        n_tempo = (
            (T[b] - T[a]) / np.median(near_iv) if near_iv else n_f
        )  # beats implied by the local tempo
        # candidate counts must be plausible for the local tempo (model count always allowed)
        tol_n = max(1.0, cfg.gap_tempo_tol * n_tempo)
        lo_n, hi_n = (
            max(1, int(np.ceil(n_tempo - tol_n))),
            int(np.floor(n_tempo + tol_n)),
        )
        cands = sorted(
            {0}
            | {
                n - n_m
                for n in range(lo_n, hi_n + 1)
                if abs(n - n_m) <= cfg.grid_count_max
            }
        )
        sup = {c: downbeat_support(b, s + c) for c in cands}
        best = max(
            cands,
            key=lambda c: (
                round(sup[c][0], 2),
                round(sup[c][1], 2),
                -abs(n_m + c - n_tempo),
            ),
        )
        model_ratio = n_m / n_tempo if n_tempo > 0 else 1.0
        if not cfg.grid_count_fix:
            c = 0
        elif not (0.5 <= model_ratio <= 2.0) and len(cands) > 1:
            # the model's count implies an impossible tempo here (e.g. 6 beats in
            # 1 s at 113 bpm): it is not the default; pick the best plausible count
            plaus = [c for c in cands if c != 0] or [0]
            c = max(
                plaus,
                key=lambda c: (
                    round(sup[c][0], 2),
                    round(sup[c][1], 2),
                    -abs(n_m + c - n_tempo),
                ),
            )
            if c and sup[c][0] < sup[0][0] + cfg.implausible_min_margin:
                # the later downbeats prefer the model's count after all: keep it
                log(
                    f"grid: {T[a]:.2f}-{T[b]:.2f}s model count {n_m} implies {model_ratio:.1f}x the local tempo, "
                    f"but downbeats do not support {n_m + c} ({sup[c][0]:.2f} vs {sup[0][0]:.2f}); kept"
                )
                c = 0
            else:
                log(
                    f"grid: {T[a]:.2f}-{T[b]:.2f}s model count {n_m} implies {model_ratio:.1f}x the local tempo; using {n_m + c}"
                )
        elif n_f == n_m and frames_steady and cfg.grid_keep_agreed:
            # frames and model count the same: only clear downbeat evidence overrules both
            c = (
                best
                if (best != 0 and sup[best][0] > sup[0][0] + cfg.agreed_override_margin)
                else 0
            )
        else:
            c = (
                best
                if (best != 0 and sup[best][0] > sup[0][0] + cfg.gap_support_margin)
                else 0
            )
        n_g = n_m + c
        # inside a gap the model is the suspect, so phase is checked at its ends;
        # a long gap whose frame beats mostly sit between model beats is an
        # off-beat-locked pulse, not a model error
        interior_off = float(np.mean(~onbeat[a + 1 : b])) if b - a > 1 else 0.0
        frames_in_phase = onbeat[a] and onbeat[b] and (n_f <= 4 or interior_off <= 0.3)
        if n_g == n_f and frames_steady and frames_in_phase:
            for i in range(1, n_f):
                beats[ga + i] = T[a + i]
            how = "frames"
        elif c == 0:
            for i in range(1, max(n_m, 1)):
                beats[ga + i] = Minv(ga + i - s)
            how = "model"
        else:
            # keep the model's beats and add (remove) the missing (extra) beats
            # where the model is most stretched (compressed) against the local
            # tempo, e.g. inside a fermata; notes elsewhere keep model timing
            ks_gap = list(range(int(r[a]), int(r[b]) + 1))  # model beats a..b
            if len(ks_gap) < 2:  # model has no beat inside: one interval
                ks_gap = [int(r[a]), int(r[a]) + 1]
            tk = [Minv(k) for k in ks_gap]
            tk[0], tk[-1] = T[a], T[b]
            ref = np.median(near_iv) if near_iv else (T[b] - T[a]) / max(n_g, 1)
            cnt = [1] * (len(tk) - 1)  # beats per model interval
            for _ in range(abs(c)):
                score_iv = [(tk[i + 1] - tk[i]) / ref - cnt[i] for i in range(len(cnt))]
                if c > 0:
                    i = int(np.argmax(score_iv))
                    cnt[i] += 1
                else:
                    ok_i = [i for i in range(len(cnt)) if cnt[i] > 0]
                    i = min(ok_i, key=lambda i: score_iv[i])
                    cnt[i] -= 1
            g = ga
            shift_now = s
            for i, n_i in enumerate(cnt):
                if n_i != 1:
                    # beats inside: frame beats if they fill the interval with the right count
                    inner = [t for t in T if tk[i] + 0.05 < t < tk[i + 1] - 0.05]
                    use_frames = len(inner) == n_i - 1
                for u in range(1, n_i + 1):
                    if g + u != ga + n_g:
                        if n_i != 1 and use_frames and u < n_i:
                            beats[g + u] = inner[u - 1]
                        else:
                            beats[g + u] = tk[i] + (tk[i + 1] - tk[i]) * u / n_i
                if n_i != 1:
                    # the model's positions inside this interval are meaningless:
                    # notes there are placed by timing alone (free zone)
                    shift_now += n_i - 1
                    shifts.append((tk[i] + 0.02, shift_now))
                    free_zones.append((tk[i] + 0.02, tk[i + 1] - 0.02))
                g += n_i
            how = f"{n_g} beats (frames {n_f}, model {n_m}); changed at " + ", ".join(
                f"{tk[i]:.2f}s {cnt[i] - 1:+d}" for i in range(len(cnt)) if cnt[i] != 1
            )
        if c:
            s += c
            if how.startswith("model") or how.startswith("frames"):
                shifts.append(
                    (T[a] + 0.5 * (T[a + 1] - T[a]), s)
                )  # notes at T[a] keep the old numbering
            if how == "frames" and cfg.free_zone_frames:
                # the model miscounted inside this gap, so its positions there are suspect:
                # the notes are placed on the frame beats by timing alone
                free_zones.append((T[a] + 0.02, T[b] - 0.02))
            log(
                f"WARNING grid: {T[a]:.2f}-{T[b]:.2f}s model counts {n_m} beats, frames {n_f}; frame downbeats after it "
                f"support {n_g} ({sup[c][0]:.2f} vs {sup[0][0]:.2f}) -> later model positions shifted {c:+d}"
            )
        log(
            f"grid gap {T[a]:.2f}-{T[b]:.2f}s: {n_f} frame / {n_m} model beats -> {how}"
        )
        j = b
    # after the last frame beat: model beats
    last_g = max(beats)
    for k in range(last_g + 1, int(np.ceil(ks[-1])) + 4 * B + 1):
        beats[k] = Minv(k - s)

    k0 = min(beats)
    K = max(beats)
    G = np.array([beats.get(k, np.nan) for k in range(k0, K + 1)])
    # fill holes by interpolation, keep monotone
    idx = np.arange(len(G))
    ok = ~np.isnan(G)
    G = np.interp(idx, idx[ok], G[ok])
    for i in range(1, len(G)):
        if G[i] <= G[i - 1]:
            G[i] = G[i - 1] + 1e-3
    # global beat 0 must be index 0: prepend beats if the model starts before k0
    if k0 > 0:
        step = G[1] - G[0]
        G = np.concatenate([G[0] - step * np.arange(k0, 0, -1), G])
    elif k0 < 0:
        G = G[-k0:]

    def in_free_zone(t):
        return any(a0 <= t <= b0 for a0, b0 in free_zones)

    def shift_at(t):
        sh = 0
        for t0, sv in shifts:
            if t >= t0:
                sh = sv
        return sh

    # ---- diagnostics: final bar lines vs confident frame downbeats -------------
    idxg = np.arange(len(G))
    strong = PDB > 0.75
    pos = np.interp(T[strong], G, idxg)
    if bars is not None:
        starts = np.asarray([b + shift_at(Minv(b)) for b in bars[0]])
        # Compare to explicit mapped boundaries, not q modulo a nominal meter.
        dist = (
            np.min(np.abs(pos[:, None] - starts[None, :]), axis=1)
            if len(pos)
            else np.array([])
        )
        on_bar = np.mean(dist < 0.25) if len(pos) else float("nan")
        log(
            f"grid: {strong.sum()} confident frame downbeats, {on_bar * 100:.0f}% fall on explicit model bar lines"
        )
        shift_at.free = in_free_zone
        return G, shift_at
    ph = np.round(pos) % B
    on_bar = np.mean(ph == 0) if strong.any() else float("nan")
    log(
        f"grid: {strong.sum()} confident frame downbeats, {on_bar * 100:.0f}% fall on bar lines"
    )
    W = cfg.phase_window_bars
    bars = (np.round(pos) // B).astype(int)
    half = B // 2 if B % 2 == 0 else None
    for b0 in range(0, int(bars.max()) + 1 if len(bars) else 0, W):
        sel = (bars >= b0) & (bars < b0 + W)
        if sel.sum() < 3:
            continue
        on_bar, on_half = (
            np.mean(ph[sel] == 0),
            (np.mean(ph[sel] == half) if half else 0),
        )
        if on_bar + on_half < 0.5:
            log(
                f"WARNING grid: bar index {b0}-{b0 + W - 1}: confident frame downbeats fall off the beat grid "
                f"phases {np.bincount(ph[sel].astype(int), minlength=B).tolist()} (likely a miscounted beat; check)"
            )
        elif on_half > on_bar:
            log(
                f"note grid: bar index {b0}-{b0 + W - 1}: frame downbeats sit half a bar from the bar lines "
                f"(half-bar ambiguity; kept the model's bar lines)"
            )
    shift_at.free = in_free_zone
    return G, shift_at
