"""Onsets and durations in beats (Fractions) from performance timing + model rhythm."""

from collections import defaultdict
from fractions import Fraction as F
import numpy as np

GRID_FRACS = [F(0), F(1, 4), F(1, 3), F(1, 2), F(2, 3), F(3, 4), F(1)]


def snap_frac(x, fracs=GRID_FRACS):
    b = np.floor(x)
    best = min(fracs, key=lambda fr: abs(x - (b + float(fr))))
    return F(int(b)) + best


def to_frac(v, den=24):
    return F(round(v * den), den)


def quantize(inp, G, shift_at, ctx):
    cfg, log = ctx.cfg, ctx.log
    z = inp.score
    N = z["notes"]
    P = inp.perf
    on = np.array([p[0] for p in P])
    off = np.array([p[1] for p in P])
    vel = np.array([p[3] for p in P])
    idx = np.arange(len(G))
    x = np.interp(on, G, idx)
    xo = np.interp(off, G, idx)
    # compound meters: model offsets come in whole felt beats, no triplet positions
    U = float(ctx.pulse) if ctx.compound else 1.0
    FR = [F(0), F(1, 4), F(1, 2), F(3, 4), F(1)] if ctx.compound else GRID_FRACS
    shifts = np.array([shift_at(t) for t in on])
    q = N[:, 1] + shifts
    qd = N[:, 2]
    staff = N[:, 3].astype(int)
    if ctx.compound and cfg.compound_snap:
        # 6/8 / 12/8: 4/4 16ths rewritten x1.5 land on 32nds -> back to the 16th grid.
        # The model's own triplet / sextuplet positions are kept (rounding them to
        # 16ths was a rhythm error, evaluation: gangqinpu rhythm -0.3 points)
        def snap16(v):
            trip = np.isclose(v * 12, np.round(v * 12)) & ~np.isclose(
                v * 4, np.round(v * 4)
            )
            return np.where(trip, v, np.round(v * 4) / 4)

        q = snap16(q)
        qd = np.maximum(
            snap16(qd),
            np.where(
                np.isclose(qd * 12, np.round(qd * 12)), np.minimum(qd, 0.25), 0.25
            ),
        )
    if shifts.any():
        log(
            f"model positions shifted for {int((shifts != 0).sum())} notes (grid fusion)"
        )
    grace = z["is_grace"].astype(bool) if "is_grace" in z else np.zeros(len(N), bool)

    # ---- swing: off-beat 8ths consistently played late -> measure in straight time
    ctx.swing = None
    if not ctx.compound:
        qf = q - np.floor(q)
        offb = np.isclose(qf, 0.5) & ~grace
        pos = x[offb] - np.floor(q[offb])
        if offb.sum() >= cfg.swing_min_notes:
            med = float(np.median(pos))
            iqr = float(np.subtract(*np.percentile(pos, [75, 25])))
            if cfg.swing_min <= med <= cfg.swing_max and iqr <= cfg.swing_iqr_max:
                ctx.swing = med

                def deswing(v):
                    b = np.floor(v)
                    f = v - b
                    return b + np.where(
                        f <= med, f * 0.5 / med, 0.5 + (f - med) * 0.5 / (1 - med)
                    )

                x, xo = deswing(x), deswing(xo)
                log(
                    f"WARNING swing: off-beat 8ths played at {med:.2f} of the beat (ratio {med / (1 - med):.1f}:1, "
                    f"n={int(offb.sum())}); written as straight 8ths with a 'Swing' marking"
                )
    n = len(P)
    notes = [
        dict(
            i=i,
            pitch=int(N[i, 0]),
            staff=int(staff[i]),
            x=float(x[i]),
            xo=float(xo[i]),
            q=float(q[i]),
            qd=float(qd[i]),
            vel=int(vel[i]),
            grace=bool(grace[i]),
            roll=False,
            bass_grace=False,
            t=float(on[i]),
        )
        for i in range(n)
    ]

    # ---- rolled chords ---------------------------------------------------------
    # Rolls placed by timing (the model gave the notes different positions, or a
    # position far from where the roll was played) only where the model positions
    # are unusable anyway: gaps whose beat count the frames corrected (free zones).
    # Elsewhere they mostly broke correct model rhythm (evaluation: 28 -> 17 pieces
    # worse than the raw model with them off).
    in_fz = lambda grp: any(
        getattr(shift_at, "free", lambda t: False)(on[k]) for k in grp
    )
    rolled = set()
    for st in (0, 1):
        ids = sorted(
            [k for k in range(n) if staff[k] == st and not grace[k]], key=lambda k: x[k]
        )
        a = 0
        while a < len(ids):
            b = a
            while (
                b + 1 < len(ids)
                and x[ids[b + 1]] - x[ids[b]] < cfg.roll_gap
                and x[ids[b + 1]] - x[ids[a]] <= cfg.roll_max_spread
            ):
                b += 1
            grp = ids[a : b + 1]
            spread = x[grp[-1]] - x[grp[0]]
            ok = False
            if (
                len(grp) >= 2
                and spread >= cfg.roll_min_spread
                and len({int(N[k, 0]) for k in grp}) == len(grp)
            ):
                same_q = len(set(np.round(q[grp], 3))) == 1
                offgrid = max(abs(x[k] - float(snap_frac(x[k]))) for k in grp) > 0.05
                pitches = [N[k, 0] for k in grp]
                mono = all(np.diff(pitches) > 0) or all(np.diff(pitches) < 0)
                sustain = all(xo[k] >= x[grp[-1]] + 0.5 for k in grp)
                if (
                    same_q
                    and offgrid
                    and spread <= cfg.roll_max_spread
                    and (sustain or spread <= cfg.roll_spread_free)
                ):
                    ok = True
                elif (
                    (cfg.roll_from_timing or in_fz(grp))
                    and spread <= cfg.roll_spread_free
                    and sustain
                    and mono
                ):
                    ok = True
            if ok:
                if len(set(np.round(q[grp], 3))) == 1:
                    qq = q[grp[0]]
                    xm = float(np.median(x[grp]))
                    o = to_frac(
                        qq + (U * round((xm - qq) / U) if cfg.retime_outliers else 0)
                    )
                    if (cfg.roll_from_timing or in_fz(grp)) and abs(
                        xm - float(o)
                    ) > 0.6:
                        o = to_frac(round(x[grp[0]] * 2) / 2)
                else:
                    o = to_frac(round(x[grp[0]] * 2) / 2)
                for k in grp:
                    notes[k]["on"] = o
                    notes[k]["roll"] = True
                    rolled.add(k)
                log(
                    f"roll staff{st + 1} {ctx.where(o)} pitches {[int(N[k, 0]) for k in grp]} spread {spread:.2f}"
                )
                a = b + 1
            else:
                a += 1

    # ---- onsets: model rhythm where timing agrees, timing grid otherwise ----
    # A note keeps the model's position when its timing deviation is small, or
    # when it deviates like its neighbours do (local drift = rubato, not a
    # different rhythm). Isolated outliers are re-snapped to the timing grid.
    free = [k for k in range(n) if k not in rolled and not grace[k]]
    # Model position plus the grid's beat shift. A per-note whole-beat offset
    # rounded from timing is only used when re-timing is on: against a poor grid
    # it makes neighbouring notes jump a beat (evaluation: duplicates, broken rhythm)
    if cfg.retime_outliers:
        mpos = {k: q[k] + U * round((x[k] - q[k]) / U) for k in free}
    else:
        mpos = {k: q[k] for k in free}
    fz = getattr(shift_at, "free", lambda t: False)
    protected = regular_runs(
        [k for k in free if not fz(on[k])], q, on, staff, mpos, x, cfg, ctx
    )
    err = {k: x[k] - mpos[k] for k in free}
    order = sorted(free, key=lambda k: x[k])
    xs = np.array([x[k] for k in order])
    nchanged = 0
    outliers = []
    free_zone = getattr(shift_at, "free", lambda t: False)
    for k in free:
        m = mpos[k]
        if free_zone(on[k]) and k not in protected:
            # beats were inserted/removed here: the model position is not usable
            g = snap_frac(x[k], [F(0), F(1, 4), F(1, 2), F(3, 4), F(1)])
            notes[k]["on"] = g
            if g != to_frac(m):
                outliers.append((k, m))
            continue
        if k in protected:
            notes[k]["on"] = to_frac(m)
            continue
        lo, hi = (
            np.searchsorted(xs, x[k] - cfg.drift_window),
            np.searchsorted(xs, x[k] + cfg.drift_window),
        )
        nb = [j for j in order[lo:hi] if j != k]
        if len(nb) >= 4:  # local linear drift trend (handles accelerando / ritardando)
            xs_n = np.array([x[j] for j in nb])
            es_n = np.array([err[j] for j in nb])
            if np.ptp(xs_n) > 1e-6:
                a, b = np.polyfit(xs_n, es_n, 1)
                drift = float(a * x[k] + b)
            else:
                drift = float(np.median(es_n))
        elif nb:
            drift = float(np.median([err[j] for j in nb]))
        else:
            drift = 0.0
        if (
            not cfg.retime_outliers
            or abs(err[k]) <= cfg.model_onset_tol
            or (abs(err[k] - drift) <= cfg.drift_tol and abs(err[k]) <= cfg.drift_max)
        ):
            notes[k]["on"] = to_frac(m)
            continue
        g = snap_frac(x[k], FR)
        if abs(x[k] - float(g)) > cfg.grid_onset_tol:
            g = to_frac(round(x[k] * 4) / 4)
        notes[k]["on"] = g
        if g != to_frac(m):
            outliers.append((k, m))

    # A timing-only move onto a triplet position needs triplet evidence nearby:
    # a model triplet in the same staff within 2 beats, or another outlier on a
    # triplet in the same half bar. Otherwise it is an early/late duple note.
    frac = lambda o: o - o.numerator // o.denominator
    trip_out = defaultdict(int)
    for k, m in outliers:
        if frac(notes[k]["on"]).denominator == 3:
            trip_out[(staff[k], ctx.half_index(notes[k]["on"]))] += 1
    model_trip = [k for k in range(n) if frac(to_frac(q[k])).denominator == 3]
    for k, m in outliers:
        g = notes[k]["on"]
        if frac(g).denominator == 3:
            near = any(
                staff[j] == staff[k] and abs(q[j] - q[k]) <= 2 for j in model_trip
            )
            # in a swung piece late off-beats are the swing itself: only model triplets count
            if not near and (ctx.swing or trip_out[(staff[k], ctx.half_index(g))] < 2):
                g = snap_frac(x[k], [F(0), F(1, 4), F(1, 2), F(3, 4), F(1)])
                notes[k]["on"] = g
                if g == to_frac(m):
                    continue
        nchanged += 1
        log(
            f"retimed p{N[k, 0]:.0f} staff{staff[k] + 1} x={x[k]:.3f} model->{m:.3f} grid->{float(g):.3f} ({ctx.where(g)})"
        )
    log(f"retimed {nchanged} notes")

    # ---- triplet context -------------------------------------------------------
    halves = defaultdict(list)
    for k, nt in enumerate(notes):
        if "on" in nt and not grace[k]:
            halves[(nt["staff"], ctx.half_index(nt["on"]))].append(k)
    frac = lambda o: o - o.numerator // o.denominator
    trip_count = {
        key: sum(frac(notes[k]["on"]).denominator == 3 for k in ks)
        for key, ks in halves.items()
    }
    for (st, h), ks in halves.items():
        if not cfg.triplet_context:
            break
        # per bar: with meter changes a simple-meter bar may sit in a compound piece
        if (
            ctx.bars.compound(ctx.bar_of(notes[ks[0]]["on"]))
            if ctx.bars
            else ctx.compound
        ):
            continue
        # triplet evidence: the model wrote a triplet here, timing put >= 2 notes on
        # triplets, or one note plus a triplet pattern in the neighbouring half bars
        model_trip = sum(frac(to_frac(q[k])).denominator == 3 for k in ks)
        timing_trip = trip_count[(st, h)]
        around = trip_count.get((st, h - 1), 0) + trip_count.get((st, h + 1), 0)
        if ctx.swing and model_trip == 0:
            continue
        if not (
            model_trip >= 1 or timing_trip >= 2 or (timing_trip >= 1 and around >= 2)
        ):
            continue
        for k in ks:
            if k in protected:
                continue
            if frac(notes[k]["on"]) in (F(1, 4), F(3, 4)):
                t = snap_frac(x[k], [F(0), F(1, 3), F(2, 3), F(1)])
                if abs(x[k] - float(t)) <= cfg.triplet_context_tol:
                    log(
                        f"triplet-context snap p{N[k, 0]:.0f} {float(notes[k]['on']):.3f} -> {float(t):.3f} ({ctx.where(t)})"
                    )
                    notes[k]["on"] = t

    # ---- model grace notes: attach to the next main note of the same staff -----
    for k in np.where(grace)[0]:
        cand = [
            j
            for j in range(n)
            if staff[j] == staff[k] and not grace[j] and x[j] > x[k] - 0.05
        ]
        if not cand:
            notes[k]["grace"] = False
            notes[k]["on"] = to_frac(round(x[k] * 4) / 4)
            continue
        j = min(cand, key=lambda j: x[j])
        notes[k]["on"] = notes[j]["on"]
        log(
            f"grace p{N[k, 0]:.0f} -> before p{N[j, 0]:.0f} ({ctx.where(notes[j]['on'])})"
        )

    # ---- durations: model value as prior -------------------------------------
    for nt in notes:
        qd_ = nt["qd"]
        trip = (
            abs(qd_ * 12 - round(qd_ * 12)) < 1e-6
            and abs(qd_ * 4 - round(qd_ * 4)) > 1e-6
            and qd_ >= 1 / 12
        )
        # the minimum is for stray short values; a sextuplet/triplet value is kept as the model wrote it
        nt["dur"] = (
            (to_frac(qd_) if trip else to_frac(max(qd_, cfg.min_note_dur)))
            if not nt["grace"]
            else F(0)
        )
        nt["pd"] = nt["xo"] - nt["x"]
        nt["run"] = nt["i"] in protected
    return notes


def regular_runs(free, q, on, staff, mpos, x, cfg, ctx):
    """Notes inside a regular figure: consecutive attacks of one staff whose
    model positions step evenly (all 16ths, all sextuplets...) in the same order
    as the performance. Tempo may change inside the figure (accelerando
    tremolo), so the model's rhythm is kept for the whole run instead of
    re-snapping notes one by one."""
    protected = set()
    for st in (0, 1):
        ks = [k for k in free if staff[k] == st]
        ev = {}
        for k in ks:  # attacks = model positions with their notes
            ev.setdefault(round(q[k], 6), []).append(k)
        # an attack whose notes are not played together (the model merged notes
        # far apart in time) cannot be part of a figure
        ev = {
            qq: ks_
            for qq, ks_ in ev.items()
            if np.ptp([on[k] for k in ks_]) <= cfg.run_chord_spread
        }
        keys = sorted(ev)
        times = [float(np.median([on[k] for k in ev[qq]])) for qq in keys]
        i = 0
        while i < len(keys) - 1:
            step = keys[i + 1] - keys[i]
            j = i + 1

            def smooth(a):  # interval into a+1 comparable with the previous one
                if a < i + 1:
                    return True
                if times[a] - times[a - 1] <= 0:
                    return False
                r = (times[a + 1] - times[a]) / (times[a] - times[a - 1])
                return 1 / cfg.run_max_ratio < r < cfg.run_max_ratio

            while (
                j + 1 < len(keys)
                and abs((keys[j + 1] - keys[j]) - step) < 1e-6
                and times[j + 1] > times[j]
                and smooth(j)
            ):
                j += 1
            ok = (
                times[i + 1] > times[i]
                and 0 < step <= cfg.run_max_step
                and j - i + 1 >= cfg.run_min_events
            )
            if ok:
                # one integer offset for the whole run
                U = float(ctx.pulse) if ctx.compound else 1.0
                offs = [
                    U * round((x[k] - q[k]) / U)
                    for qq in keys[i : j + 1]
                    for k in ev[qq]
                ]
                off = float(np.median(offs)) if cfg.retime_outliers else 0.0
                pos = {qq: qq for qq in keys[i : j + 1]}
                # extend backwards over attacks the model placed slightly off the
                # run's grid (e.g. an arpeggio whose first notes sit on 16ths)
                a, first, first_t = i - 1, keys[i], times[i]
                iv0 = times[i + 1] - times[i]
                while a >= 0 and a >= i - cfg.run_extend_max:
                    pred = first - step
                    gap = first_t - times[a]
                    if (
                        abs(keys[a] - pred) <= cfg.run_extend_tol
                        and 1 / cfg.run_max_ratio < gap / iv0 < cfg.run_max_ratio
                    ):
                        pos[keys[a]] = pred
                        if abs(keys[a] - pred) > 1e-6:
                            ctx.log(
                                f"regular run staff{st + 1}: model position {keys[a] + off:g} moved to "
                                f"{pred + off:g} to continue the figure ({ctx.where(pred + off)})"
                            )
                        first, first_t, a = pred, times[a], a - 1
                    else:
                        break
                for qq, pq in pos.items():
                    for k in ev[qq]:
                        mpos[k] = pq + off
                        protected.add(k)
                ctx.log(
                    f"regular run staff{st + 1} {ctx.where(min(pos.values()) + off)} - {ctx.where(keys[j] + off)}: "
                    f"{len(pos)} attacks every {step:g} beat, model rhythm kept"
                )
                i = j
            else:
                i += 1
    return protected
