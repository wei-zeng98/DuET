"""Staff-level layout decisions: LH clefs, 8va lines, events per voice layer,
and timed directions (tempo, rit., dynamics, pedal)."""

from collections import defaultdict
from fractions import Fraction as F
import numpy as np
from .spelling import ledgers


class Plan:
    """everything the writer needs"""

    pass


def clef_cost(ps, c):
    return sum(max(0, ledgers(p, c) - 2) ** 2 for p in ps)


def lh_clefs(plan, ctx, staff=1):
    cfg, NB, bars = ctx.cfg, ctx.nbars, ctx.bars
    L1 = plan.layers[staff]
    home, other = ("G", "F") if staff == 0 else ("F", "G")
    switch = cfg.clef_switch_penalty_rh if staff == 0 else cfg.clef_switch_penalty

    def bar_cost(m, c):
        o = "G" if c == "F" else "F"
        lo, hi = bars.start(m), bars.end(m)
        by = defaultdict(list)
        for n in L1:
            if n["on"] < hi and n["on"] + n["dur"] > lo:
                by[n["on"]].append(n["pitch"])
        costs = sorted(
            ((clef_cost(ps, c), clef_cost(ps, o)) for ps in by.values()),
            key=lambda t: t[1] + 6 - t[0],
        )
        tot = sum(a for a, _ in costs)
        if (
            costs and costs[0][1] + 6 < costs[0][0]
        ):  # one isolated event may take a local clef
            tot += costs[0][1] + 6 - costs[0][0]
        gr = [
            n["pitch"]
            for n in plan.notes
            if n["grace"] and n["staff"] == staff and lo <= n["on"] < hi
        ]
        bias = cfg.clef_rh_bass_bias if (staff == 0 and c == "F") else 0
        return tot + cfg.grace_clef_weight * clef_cost(gr, c) + bias

    def tie_cost(m, c):
        # notes held over the barline into bar m are read in bar m's clef: switching under
        # them stretches the ties across the clef change
        b = bars.start(m)
        held = [n["pitch"] for n in L1 if n["on"] < b < n["on"] + n["dur"]]
        return cfg.clef_tie_penalty * sum(ledgers(p, c) for p in held)

    cost = {home: bar_cost(0, home), other: bar_cost(0, other) + 3}
    back = []
    for m in range(1, NB):
        new, bk = {}, {}
        for c in "FG":
            tc = tie_cost(m, c)
            opts = {pc: cost[pc] + (0 if pc == c else switch + tc) for pc in "FG"}
            pc = min(opts, key=opts.get)
            new[c] = opts[pc] + bar_cost(m, c)
            bk[c] = pc
        cost = new
        back.append(bk)
    c = min(cost, key=cost.get)
    clef2 = [c]
    for bk in reversed(back):
        c = bk[c]
        clef2.append(c)
    return clef2[::-1]


def local_clefs(plan, ctx, staff=1):
    """bar -> [(rel onset, clef)] for LH events far outside the bar clef"""
    bars, clef2 = ctx.bars, plan.clefs[staff]
    by = defaultdict(list)
    for n in plan.layers[staff]:
        by[n["on"]].append(n["pitch"])
    for n in plan.graces:
        if n["staff"] == staff and n["on"] in by:
            by[n["on"]].append(n["pitch"])
    out = defaultdict(list)
    for on in sorted(by):
        m = bars.index(on)
        c = clef2[m]
        o = "G" if c == "F" else "F"
        lc = max(ledgers(p, c) for p in by[on])
        lo = max(ledgers(p, o) for p in by[on])
        sw = (
            lc >= ctx.cfg.local_clef_min_ledgers
            and lo <= max(3, lc - 4)
            and clef_cost(by[on], c) > clef_cost(by[on], o) + 6
        )
        out[m].append((on - bars.start(m), o if sw else c))
    res = {}
    for m, seq in out.items():
        sw, cur = [], clef2[m]
        for rel, c in seq:
            ps = by[bars.start(m) + rel]
            if c != cur and c == clef2[m] and max(ledgers(p, cur) for p in ps) <= 3:
                continue
            if c != cur:
                sw.append((rel, c))
                cur = c
        extreme = any(
            max(ledgers(p, clef2[m]) for p in by[bars.start(m) + rel]) >= 7
            for rel, _ in seq
        )
        if sw and (len(sw) <= 2 or extreme):
            res[m] = sw
            ctx.log(
                f"local clef changes staff{staff + 1} bar {m + (0 if ctx.pickup else 1)}: {[(float(r) + 1, c) for r, c in sw]}"
            )
    return res


def clef_timeline(plan, ctx, staff=1):
    """absolute (beat, clef) LH changes; a change waits while tied notes
    would read worse in the new clef"""
    bars, clef2 = ctx.bars, plan.clefs[staff]
    layer = plan.layers[staff]
    ch, state = [], clef2[0]
    for m in range(ctx.nbars):
        if clef2[m] != state:
            ch.append((bars.start(m), clef2[m]))
            state = clef2[m]
        for rel, c in plan.local_clefs[staff].get(m, []):
            if c != state:
                ch.append((bars.start(m) + rel, c))
                state = c
    attacks = sorted(
        {n["on"] for n in layer} | {n["on"] for n in plan.graces if n["staff"] == staff}
    )
    out, prev = [], clef2[0]
    for t, c in ch:
        held = [n["pitch"] for n in layer if n["on"] < t < n["on"] + n["dur"]]
        if held and clef_cost(held, prev) < clef_cost(held, c):
            nxt = [a for a in attacks if a > t]
            if nxt:
                ctx.log(
                    f"staff{staff + 1} clef change to {c} at {ctx.where(t)} delayed past tied notes to {ctx.where(nxt[0])}"
                )
                t = nxt[0]
        out.append((t, c))
        prev = c
    res, state = [], clef2[0]
    for t, c in sorted(out, key=lambda x: x[0]):
        if res and res[-1][0] == t:
            res.pop()
            state = res[-1][1] if res else clef2[0]
        if c != state:
            res.append((t, c))
            state = c
    return res


def events_for(layers, staff, layer):
    by = defaultdict(list)
    for n in layers[staff]:
        if n["layer"] == layer:
            by[n["on"]].append(n)
    ev = []
    for o in sorted(by):
        grp = sorted(by[o], key=lambda n: n["pitch"])
        d = min(n["dur"] for n in grp)
        if d <= 0:
            continue
        ev.append(dict(on=o, end=o + d, notes=grp, roll=any(n["roll"] for n in grp)))
    for a, b in zip(ev, ev[1:]):
        if a["end"] > b["on"]:
            a["end"] = b["on"]
    # JSON and diagnostics must describe the durations actually emitted by the
    # single chord event, including same-pitch/layer overlap clipping.
    for e in ev:
        for n in e["notes"]:
            n["dur"] = e["end"] - e["on"]
    return ev


def staff_events(plan, staff):
    return [e for (st, _), evs in plan.EV.items() if st == staff for e in evs]


def ottava_runs(plan, staff, lo_min, need_max, ctx):
    evs = sorted(staff_events(plan, staff), key=lambda e: e["on"])
    runs, cur = [], []
    for e in evs:
        ps = [n["pitch"] for n in e["notes"]]
        if min(ps) >= lo_min and plan.clefs[staff][ctx.bar_of(e["on"])] == "G":
            cur.append(e)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        tops = [max(n["pitch"] for n in e["notes"]) for e in r]
        if sum(t >= need_max for t in tops) >= 2 or max(tops) >= need_max + 3:
            out.append((r[0]["on"], max(e["end"] for e in r)))
    return out


def ottava_bassa_runs(plan, ctx):
    """8vb for runs of very low LH events in bass clef: at least min_run events
    with every note at or below max_top, min_low of them reaching need_low"""
    max_top, need_low, min_run, min_low = ctx.cfg.ottava_bassa
    evs = sorted(staff_events(plan, 1), key=lambda e: e["on"])
    runs, cur = [], []
    for e in evs:
        ps = [n["pitch"] for n in e["notes"]]
        if max(ps) <= max_top and plan.clefs[1][ctx.bar_of(e["on"])] == "F":
            cur.append(e)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        lows = [min(n["pitch"] for n in e["notes"]) for e in r]
        if len(r) >= min_run and sum(l <= need_low for l in lows) >= min_low:
            a, b = r[0]["on"], max(e["end"] for e in r)
            if (
                out and a - out[-1][1] <= ctx.B
            ):  # bridge short gaps: one line instead of two
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
    return out


class Directions:
    """(bar, staff) -> [(rel, order, direction-type xml, attrs, sound xml)]"""

    def __init__(self, ctx):
        self.ctx = ctx
        self.td = defaultdict(list)

    def add(self, beat, staff, dtype, attrs="", sound="", order=5, end=False):
        bars, NB = self.ctx.bars, self.ctx.nbars
        beat = F(beat)
        m = bars.index(beat)
        if end and m > 0 and beat == bars.start(m):
            m -= 1
        if beat >= self.ctx.END:
            m = NB - 1
            rel = bars.length(m)
        else:
            rel = beat - bars.start(m)
        if m == 0 and self.ctx.pickup and rel < self.ctx.pickup:
            rel = self.ctx.pickup
        self.td[(m, staff)].append((rel, order, dtype, attrs, sound))


def metronome(quarter_bpm, ctx):
    """metronome mark in the felt beat (dotted quarter in compound meters)"""
    if ctx.compound:
        return (
            f"<metronome><beat-unit>quarter</beat-unit><beat-unit-dot/>"
            f"<per-minute>{int(round(quarter_bpm / 1.5))}</per-minute></metronome>"
        )
    return f"<metronome><beat-unit>quarter</beat-unit><per-minute>{quarter_bpm}</per-minute></metronome>"


def directions(plan, inp, G, ctx):
    cfg, NB, bars = ctx.cfg, ctx.nbars, ctx.bars
    D = Directions(ctx)
    for st in (0, 1):
        for a, b in plan.ott[st]:
            D.add(
                a,
                st + 1,
                f'<octave-shift type="down" size="8" number="{st + 1}"/>',
                ' placement="above"' if st == 0 else "",
                order=1,
            )
            D.add(
                b,
                st + 1,
                f'<octave-shift type="stop" size="8" number="{st + 1}"/>',
                order=0,
                end=True,
            )
            ctx.log(f"8va staff{st + 1} {ctx.where(a)} - {ctx.where(b - F(1, 100))}")
    for a, b in plan.ott_bassa:
        D.add(
            a,
            2,
            '<octave-shift type="up" size="8" number="3"/>',
            ' placement="below"',
            order=1,
        )
        D.add(
            b, 2, '<octave-shift type="stop" size="8" number="3"/>', order=0, end=True
        )
        ctx.log(f"8vb staff2 {ctx.where(a)} - {ctx.where(b - F(1, 100))}")

    # tempo (quarter = beat for n/4). Metronome marks for sustained tempo
    # sections, "rit." / "a tempo" for short slow-downs, per-bar <sound> for playback.
    first_bar = 1 if ctx.pickup else 0
    gi = np.arange(len(G))

    def bar_beat_s(m):
        a, b = float(bars.start(m)), float(bars.end(m))
        pts = np.arange(a, b + 1e-9, 1.0)
        if len(pts) >= 2 and abs(pts[-1] - b) < 1e-9:
            return float(np.median(np.diff(np.interp(pts, gi, G))))
        return (np.interp(b, gi, G) - np.interp(a, gi, G)) / (b - a)

    beat_s = np.array([bar_beat_s(m) for m in range(NB)])
    med = float(np.median(beat_s[beat_s > 1e-3])) if np.any(beat_s > 1e-3) else 0.5
    beat_s = np.where(
        np.isfinite(beat_s) & (beat_s > 1e-3), beat_s, med
    )  # degenerate bars: use the median
    bpm = 60.0 / beat_s
    sm = np.array([np.median(bpm[max(0, m - 1) : m + 2]) for m in range(NB)])
    # playback follows a smoother curve: single bars wobble with the grid and with rubato,
    # and following them bar by bar sounds unsteady
    W = cfg.tempo_play_window // 2
    play = np.array([np.median(bpm[max(0, m - W) : m + W + 1]) for m in range(NB)])
    close_from = NB - cfg.closing_bars  # a closing rit. is followed bar by bar
    plan.base_bpm = int(round(float(np.median(bpm))))
    mark = int(round(float(np.median(sm[first_bar : first_bar + 4]))))
    D.add(
        ctx.pickup if ctx.pickup else 0,
        1,
        metronome(mark, ctx),
        ' placement="above"',
        f'<sound tempo="{mark}"/>',
        order=2,
    )
    ctx.log(f"tempo: opening mark {mark}, median {plan.base_bpm}")
    if ctx.swing:
        D.add(
            ctx.pickup if ctx.pickup else 0,
            1,
            '<words font-weight="bold">Swing</words>',
            ' placement="above"',
            order=1,
        )
    S = cfg.tempo_section_bars
    last, slow = mark, False
    m = first_bar + 1
    while m < NB:
        win = sm[m : m + S]
        # a slow-down running into the final bars is a closing rit., not a new tempo
        if (
            len(win) == S
            and m + S < NB - cfg.closing_bars
            and np.all(np.abs(win / mark - 1) > cfg.tempo_change)
            and np.all((win > 0.5 * mark) & (win < 2 * mark))
            and np.all(np.sign(win - mark) == np.sign(win[0] - mark))
        ):
            mark = int(round(float(np.median(win))))
            if slow:
                slow = False
            D.add(
                bars.start(m),
                1,
                metronome(mark, ctx),
                ' placement="above"',
                f'<sound tempo="{mark}"/>',
                order=2,
            )
            ctx.log(f"tempo: new mark {mark} at bar {m + (0 if ctx.pickup else 1)}")
            last = mark
            m += S
            continue
        if not (0.5 * mark <= bpm[m] <= 2 * mark):  # a grid artefact, not a tempo
            ctx.log(
                f"tempo: bar {m + (0 if ctx.pickup else 1)} grid tempo {int(round(bpm[m]))} ignored (mark {mark})"
            )
            m += 1
            continue
        tb = int(round(bpm[m] if m >= close_from else play[m]))
        tol = cfg.tempo_play_change
        if abs(tb - last) / last > tol or (
            abs(tb - mark) <= tol * mark and last != mark
        ):
            tb = mark if abs(tb - mark) <= tol * mark else tb
            D.add(bars.start(m), 1, "<words/>", "", f'<sound tempo="{tb}"/>', order=3)
            last = tb
        r = bpm[m] / mark
        if not slow and r < cfg.rit_ratio:
            D.add(
                bars.start(m),
                1,
                '<words font-style="italic">rit.</words>',
                ' placement="above"',
                order=3,
            )
            slow = True
        elif slow and r > cfg.a_tempo_ratio:
            D.add(
                bars.start(m),
                1,
                '<words font-style="italic">a tempo</words>',
                ' placement="above"',
                order=3,
            )
            slow = False
        m += 1

    # dynamics from mean velocity per phrase block
    blk_of = lambda on: ctx.bar_of(on) // cfg.dynamics_block_bars
    vel = defaultdict(list)
    for n in plan.notes:
        vel[blk_of(n["on"])].append(n["vel"])
    dyn = lambda v: (
        "p"
        if v < 60
        else "mp"
        if v < 70
        else "mf"
        if v < 80
        else "f"
        if v < 90
        else "ff"
    )
    lastd = None
    for blk in sorted(vel):
        d = dyn(np.mean(vel[blk]))
        if d != lastd:
            first_on = min(n["on"] for n in plan.notes if blk_of(n["on"]) == blk)
            D.add(
                first_on,
                1,
                f"<dynamics><{d}/></dynamics>",
                ' placement="below"',
                order=4,
            )
            lastd = d

    # pedal: press snapped to the attack it follows, release to the nearest 8th
    gidx = np.arange(len(G))
    all_on = sorted(set(n["on"] for n in plan.notes))
    marks = []
    for t, v in inp.pedal:
        xb = float(np.interp(t, G, gidx))
        if v >= 64:
            prevs = [o for o in all_on if float(o) <= xb + 0.05 and xb - float(o) < 1]
            pos = prevs[-1] if prevs else F(round(xb * 2), 2)
            if marks and marks[-1][1] == "stop" and pos - marks[-1][0] <= F(1, 2):
                marks[-1] = (pos, "change")
            elif not marks or marks[-1][1] == "stop":
                marks.append((pos, "start"))
        elif marks and marks[-1][1] != "stop":
            marks.append((min(F(round(xb * 2), 2), ctx.END), "stop"))
    for pos, kind in marks:
        D.add(
            pos,
            2,
            f'<pedal type="{kind}" line="yes"/>',
            ' placement="below"',
            order=6 if kind != "stop" else 0,
            end=(kind == "stop"),
        )
    return D
