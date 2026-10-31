"""End-to-end run: inputs -> grid -> quantize -> hands -> voices -> layout -> MusicXML."""

import io
import os
from collections import Counter, defaultdict
from fractions import Fraction as F

import numpy as np

from .config import Config
from .context import Ctx, Bars
from .grid import build_grid, fix_rewinds
from .hands import playability, cross_over
from .meter import analyse as analyse_meter
from .layout import (
    Plan,
    lh_clefs,
    local_clefs,
    clef_timeline,
    events_for,
    ottava_runs,
    ottava_bassa_runs,
    directions,
)
from .musicxml import write
from .quantize import quantize
from .spelling import ledgers, set_key
from .voices import dedupe, assign


def sanitize(notes, ctx):
    """onsets and ends on notatable grids: 32nds (1/8) or triplet 16ths (1/12)"""

    def snap(v):
        a, b = F(round(v * 8), 8), F(round(v * 12), 12)
        return a if abs(a - v) <= abs(b - v) else b

    fixed = 0
    for n in notes:
        on = snap(n["on"])
        if n["grace"]:
            n["on"] = on
            continue
        end = snap(n["on"] + n["dur"])
        if on != n["on"] or end != n["on"] + n["dur"]:
            fixed += 1
        n["on"] = on
        n["dur"] = max(end - on, F(1, 12))
    if fixed:
        ctx.log(f"sanitize: {fixed} notes moved onto the 1/8 or 1/12 grid")


def snap_model_bars(model_bars, q, log):
    """Bar lines on half beats. A model bar starting on a tuplet position (the
    model's numbering slipped by a fraction of a beat) is moved onto the half-beat
    grid together with its notes, so the notes keep their place in the bar and
    the bar gets a length a time signature can express. q is changed in place."""
    starts = [F(b[0]).limit_denominator(48) for b in model_bars]
    snapped = [F(round(v * 2), 2) for v in starts]
    if snapped == starts:
        return model_bars
    ends = starts[1:] + [np.inf]
    moved = 0
    for a, e, sa in zip(starts, ends, snapped):
        if sa != a:
            sel = (q >= float(a) - 1e-6) & (q < float(e) - 1e-6)
            q[sel] += float(sa - a)
            moved += int(sel.sum())
    out = []
    for i, b in enumerate(model_bars):
        L = (
            (snapped[i + 1] - snapped[i])
            if i + 1 < len(snapped)
            else F(b[3] if len(b) > 3 else b[1] * 4 / b[2]).limit_denominator(48)
        )
        out.append((float(snapped[i]), b[1], b[2], float(L)))
    n = sum(a != b for a, b in zip(starts, snapped))
    log(
        f"WARNING bars: {n} model bar lines off the half-beat grid; moved onto it with their {moved} notes"
    )
    return out


def sig_for_length(L, den):
    """a time signature for a bar of L quarters, preferring the denominator den"""
    for d in dict.fromkeys((den, 4, 8, 16)):
        n = L * d / 4
        if n.denominator == 1 and 1 <= n <= 32:
            return int(n), d
    return None


def map_model_bars(model_bars, q_model, onsets, shift_at, last_on, ctx, quiet=False):
    """Move explicit bar boundaries with the grid. The first (pickup) and last bar
    may be partial: they keep their signature and are written as implicit measures.
    A bar inside the piece whose length is not its signature's (a model bar of
    another length, or a beat added/removed by the grid) gets its own time
    signature, which reads better than a silently short or long 4/4 bar."""
    log = (lambda s: None) if quiet else ctx.log
    order = np.argsort(q_model)
    qs, ts = q_model[order], onsets[order]
    uq = np.unique(qs)
    ut = np.array([np.median(ts[qs == u]) for u in uq])

    def moved(q):
        t = float(np.interp(float(q), uq, ut))
        st = F(float(q)).limit_denominator(480) + F(shift_at(t)).limit_denominator(480)
        # bar lines on half beats (snap_model_bars put the model's there; the grid shifts whole beats)
        return F(round(st * 2), 2)

    starts, sigs, ends = [], [], []
    for bar in model_bars:
        q0, num, den = bar[:3]
        length = bar[3] if len(bar) > 3 else num * 4 / den
        st = moved(q0)
        if starts and st <= starts[-1]:
            raise ValueError("Grid correction collapsed model bar boundaries")
        starts.append(st)
        sigs.append((num, den))
        ends.append(moved(q0 + length))
    lengths = [b - a for a, b in zip(starts, starts[1:])] + [ends[-1] - starts[-1]]
    if any(L <= 0 for L in lengths):
        raise ValueError("Grid correction produced a nonpositive bar length")
    while starts[0] > 0:
        L = F(sigs[0][0] * 4, sigs[0][1])
        starts.insert(0, starts[0] - L)
        sigs.insert(0, sigs[0])
        lengths.insert(0, L)
    while starts[-1] + lengths[-1] <= last_on:
        starts.append(starts[-1] + lengths[-1])
        sigs.append(sigs[-1])
        lengths.append(F(sigs[-1][0] * 4, sigs[-1][1]))
    while len(starts) > 1 and starts[-1] > last_on:
        starts.pop()
        sigs.pop()
        lengths.pop()
    for i in range(1, len(starts) - 1):
        L = lengths[i]
        if L != F(sigs[i][0] * 4, sigs[i][1]):
            sig = sig_for_length(L, sigs[i][1])
            if sig:
                log(
                    f"bars: bar {i + 1} is {float(L):g} beats -> written {sig[0]}/{sig[1]}"
                )
                sigs[i] = sig
    bars = Bars(starts, sigs, lengths)
    irregular = sum(L != F(n * 4, d) for L, (n, d) in zip(lengths, sigs))
    log(
        f"bars: retained {bars.n} model bars, {irregular} partial measures (pickup / last bar)"
    )
    return bars


def run(inp, out_dir, title=None, cfg=None):
    cfg = cfg or Config()
    os.makedirs(out_dir, exist_ok=True)
    title = title or "Piano score"
    stem = "score"
    logf = io.StringIO()
    log = lambda s: print(s, file=logf)
    ctx = Ctx(cfg=cfg, log=log, title=title)

    z = inp.score
    on = np.array([p[0] for p in inp.perf])
    M, q_model, qd_model, to_old, model_bars = analyse_meter(z, on, log)
    z = dict(z)
    z["notes"] = z["notes"].copy()
    z["notes"][:, 1], z["notes"][:, 2] = q_model, qd_model
    if model_bars is not None:
        model_bars = snap_model_bars(model_bars, z["notes"][:, 1], log)
    inp.score = z
    ctx.B, ctx.num, ctx.beat_type = int(M.bar_q), M.num, M.den
    ctx.pulse, ctx.compound = M.pulse_q, M.compound
    B = ctx.B

    if (
        model_bars is None
    ):  # (with meter changes the model's bar list is tied to its own numbering)
        fb = None
        if ctx.pulse == 1:  # frame beats are dotted quarters in compound meters
            from scipy.signal import find_peaks

            fr = inp.frames
            pk, _ = find_peaks(
                fr.p_any_beat.values,
                height=cfg.rewind_frame_conf,
                distance=cfg.beat_peak_distance,
            )
            fb = (fr.performance_time_sec.values[pk], fr.p_downbeat.values[pk] > 0.5)
        z["notes"][:, 1] = fix_rewinds(on, z["notes"][:, 1], B, cfg, log, fb)
    # grid: model beat numbering fused with the frame pulse
    grid_bars = (
        None
        if model_bars is None
        else (
            np.array([b[0] for b in model_bars]),
            np.array([b[3] for b in model_bars]),
        )
    )
    G, shift_at = build_grid(
        inp.frames, on, z["notes"][:, 1], B, cfg, log, pulse=ctx.pulse, bars=grid_bars
    )
    first_q = min(float(z["notes"][i, 1]) + shift_at(on[i]) for i in range(len(on)))
    lead = (
        F(int(first_q / float(ctx.pulse))) * ctx.pulse
    )  # whole felt beats before the first note
    ctx.pickup = lead if 0 < lead < B else F(0)
    if model_bars is not None:
        # provisional bars (from the model positions) so that quantization's half-bar
        # windows follow the model's bar lines; rebuilt from the final onsets below
        last_q = max(float(z["notes"][i, 1]) + shift_at(on[i]) for i in range(len(on)))
        ctx.bars = map_model_bars(
            model_bars,
            z["notes"][:, 1],
            on,
            shift_at,
            F(last_q).limit_denominator(48),
            ctx,
            quiet=True,
        )
    notes = quantize(inp, G, shift_at, ctx)
    # onsets and ends on notatable grids before duplicates are judged: snapping may
    # bring two notes of one pitch together
    sanitize(notes, ctx)
    notes = dedupe(notes, ctx)
    last_on = max(n["on"] for n in notes)
    if model_bars is None:
        ctx.nbars = int(last_on // B) + 1
        ctx.bars = Bars.uniform(ctx.nbars, ctx.num, ctx.beat_type)
    else:
        ctx.bars = map_model_bars(
            model_bars, z["notes"][:, 1], on, shift_at, last_on, ctx
        )
        ctx.nbars = ctx.bars.n
        # the first bar may start before position 0: the pickup is the empty part of that bar
        first = min(n["on"] for n in notes) - ctx.bars.start(0)
        lead = F(int(first / ctx.pulse)) * ctx.pulse
        ctx.pickup = lead if 0 < lead < ctx.bars.length(0) else F(0)
    keys_model = [int(k) for k in z.get("keys", [0])]
    starts = np.asarray(z.get("bar_starts", np.arange(len(keys_model)) * B), float)
    old_pos = (
        (lambda m: to_old(float(ctx.bars.start(m)) + 1e-6))
        if model_bars is None
        else (lambda m: model_bars[min(m, len(model_bars) - 1)][0] + 1e-6)
    )
    ctx.keys = [
        keys_model[
            min(
                max(int(np.searchsorted(starts, old_pos(m), "right")) - 1, 0),
                len(keys_model) - 1,
            )
        ]
        for m in range(ctx.nbars)
    ]
    set_key(Counter(ctx.keys).most_common(1)[0][0])
    log(
        f"piece: {ctx.nbars} bars of {ctx.num}/{ctx.beat_type}, pickup {float(ctx.pickup)} beats, "
        f"keys {dict(Counter(ctx.keys))}"
    )

    notes, stats = playability(notes, ctx)
    stats["cross_over"] = cross_over(notes, ctx)
    log(f"playability: {stats}")
    layers = {st: assign(notes, st, ctx) for st in (0, 1)}

    # a very low bass grace before a treble-register LH chord reads badly in any clef
    lh_at = defaultdict(list)
    for n in layers[1]:
        lh_at[n["on"]].append(n["pitch"])
    kept = []
    for n in notes:
        ch = lh_at.get(n["on"], [])
        if (
            n["bass_grace"]
            and n["pitch"] < 50
            and ch
            and min(ch) >= 68
            and max(ch) >= 77
        ):
            log(
                f"dropped low bass grace p{n['pitch']} before high LH chord {sorted(ch)} ({ctx.where(n['on'])})"
            )
            continue
        kept.append(n)
    notes = kept

    plan = Plan()
    plan.notes, plan.layers = notes, layers
    plan.graces = [n for n in notes if n["grace"]]
    plan.clefs = {1: lh_clefs(plan, ctx, 1)}
    plan.clef2 = plan.clefs[1]
    kept = []
    for n in notes:
        if (
            n["bass_grace"]
            and plan.clef2[ctx.bar_of(n["on"])] == "G"
            and ledgers(n["pitch"], "G") > 4
        ):
            log(
                f"dropped low bass grace p{n['pitch']} in treble-clef bar ({ctx.where(n['on'])})"
            )
            continue
        kept.append(n)
    plan.notes = notes = kept
    plan.graces = [n for n in notes if n["grace"]]
    plan.clefs[0] = lh_clefs(plan, ctx, 0)
    plan.local_clefs = {st: local_clefs(plan, ctx, st) for st in (0, 1)}
    plan.clef_tls = {st: clef_timeline(plan, ctx, st) for st in (0, 1)}
    plan.local_clef, plan.clef_tl = plan.local_clefs[1], plan.clef_tls[1]
    nsw = sum(a != b for a, b in zip(plan.clefs[0], plan.clefs[0][1:]))
    if nsw or plan.clefs[0][0] != "G":
        log(
            f"RH clef: {nsw} changes, {sum(c == 'F' for c in plan.clefs[0])} bars in bass clef"
        )
    plan.EV = {(st, L): events_for(layers, st, L) for st in (0, 1) for L in ("A", "B")}
    plan.grace_by = defaultdict(list)
    for g in plan.graces:
        plan.grace_by[(g["staff"], g["on"])].append(g)
    for k in plan.grace_by:
        plan.grace_by[k].sort(key=lambda n: n["pitch"])
    plan.ott = {
        0: ottava_runs(plan, 0, *cfg.ottava_rh, ctx),
        1: ottava_runs(plan, 1, *cfg.ottava_lh, ctx),
    }
    plan.ott_bassa = ottava_bassa_runs(plan, ctx)
    plan.dirs = directions(plan, inp, G, ctx)

    # readability report
    def clef_at(t, st):
        c = plan.clefs[st][0]
        for tt, cc in plan.clef_tls[st]:
            if tt <= t:
                c = cc
        return c

    worst = []
    for n in notes:
        st = n["staff"]
        c = clef_at(n["on"], st)
        p = (
            n["pitch"]
            - (12 if any(a <= n["on"] < b for a, b in plan.ott[st]) else 0)
            + (
                12
                if st == 1 and any(a <= n["on"] < b for a, b in plan.ott_bassa)
                else 0
            )
        )
        l = ledgers(p, c)
        if l >= cfg.report_ledgers:
            worst.append((n["on"], st + 1, n["pitch"], l, n["grace"]))
    for wv in sorted(worst):
        log(
            f"LEDGER {ctx.where(wv[0])} staff {wv[1]} pitch {wv[2]} ledgers {wv[3]}{' (grace)' if wv[4] else ''}"
        )

    xml_path = os.path.join(out_dir, stem + ".musicxml")
    write(plan, ctx, xml_path)
    exact = sum(abs(float(n["on"]) - n["x"]) <= 0.06 for n in notes if not n["grace"])
    sigs = Counter(f"{a}/{b}" for a, b in ctx.bars.sigs)
    summary = dict(
        title=ctx.title,
        bars=ctx.nbars,
        meter=f"{ctx.num}/{ctx.beat_type}",
        meters=dict(sigs),
        pickup_beats=float(ctx.pickup),
        origin_beats=float(ctx.bars.start(0) + ctx.pickup),
        keys=dict(Counter(ctx.keys)),
        tempo=plan.base_bpm,
        input_notes=len(inp.perf),
        output_notes=len(notes),
        onsets_within_0_06_beat=f"{exact}/{sum(not n['grace'] for n in notes)}",
        playability=stats,
        notes_with_many_ledgers=len(worst),
        warnings=None,
    )
    summary["warnings"] = [
        line.strip()
        for line in logf.getvalue().splitlines()
        if line.startswith("WARNING")
    ]
    logf.close()
    return xml_path, summary
