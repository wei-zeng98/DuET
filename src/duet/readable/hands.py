"""Two-hand playability pass.

For each simultaneous attack that one hand cannot cover (span > hand_span or
more than hand_max_notes notes):
1. If some pitch split makes both hands playable, and every note that changes
   hands stays near what that hand is playing around it, use the split that
   moves the fewest notes.
2. Otherwise the texture has three layers. RH keeps the top note and what lies
   within reach under it; the rest goes to the LH. Pitch-class doublings are
   thinned; then the LH chord is rolled if it spans <= roll_lh_max_span, else
   its bass (with octave) becomes a grace note before the chord.
"""
from collections import defaultdict
from fractions import Fraction as F

from .spelling import ledgers


def playability(notes, ctx):
    cfg, log = ctx.cfg, ctx.log

    def ok(ps):
        if not ps:
            return True
        lim = cfg.hand_span + 2 if len(ps) == 2 else cfg.hand_span
        return len(ps) <= cfg.hand_max_notes and max(ps) - min(ps) <= lim

    def thin(grp, keep_pcs, where):
        grp = sorted(grp, key=lambda n: n['pitch'])
        dropped = []
        while not ok([n['pitch'] for n in grp]):
            pcs = [n['pitch'] % 12 for n in grp]
            cand = [n for n in grp if n['pitch'] % 12 in keep_pcs or pcs.count(n['pitch'] % 12) > 1]
            victim = cand[0] if cand else grp[0]
            grp.remove(victim); dropped.append(victim)
            log(f"thinned p{victim['pitch']} ({where})")
        return grp, dropped

    by = defaultdict(list)
    for n in notes:
        if not n['grace']:
            by[n['on']].append(n)
    removed = []
    stats = dict(reassigned=0, grace_bass=0, thinned=0, rolled=0)
    for on, grp in sorted(by.items()):
        where = ctx.where(on)
        cur = [sorted(n['pitch'] for n in grp if n['staff'] == s) for s in (0, 1)]
        if ok(cur[0]) and ok(cur[1]):
            continue
        P = sorted(grp, key=lambda n: n['pitch'])
        ctxp = {s: [n['pitch'] for n in notes if n['staff'] == s and not n['grace'] and n['on'] != on
                    and abs(n['on'] - on) <= 2] for s in (0, 1)}

        def near(n, s):
            return not ctxp[s] or min(abs(n['pitch'] - p) for p in ctxp[s]) <= cfg.hand_move_near

        best = None
        for k in range(len(P) + 1):
            lh, rh = P[:k], P[k:]
            moved_ok = all(near(n, 1) for n in lh if n['staff'] != 1) and all(near(n, 0) for n in rh if n['staff'] != 0)
            if moved_ok and ok([n['pitch'] for n in lh]) and ok([n['pitch'] for n in rh]):
                moves = sum(n['staff'] != 1 for n in lh) + sum(n['staff'] != 0 for n in rh)
                if best is None or moves < best[0]:
                    best = (moves, k)
        if best is not None:
            k = best[1]
            for n in P[:k]:
                if n['staff'] != 1: log(f"hand RH->LH p{n['pitch']} ({where})")
                n['staff'] = 1
            for n in P[k:]:
                if n['staff'] != 0: log(f"hand LH->RH p{n['pitch']} ({where})")
                n['staff'] = 0
            stats['reassigned'] += 1
            continue
        top = P[-1]['pitch']
        rh = [n for n in P if n['pitch'] >= top - cfg.hand_span and (n['staff'] == 0 or near(n, 0))]
        rest = [n for n in P if n not in rh]
        rh, d1 = thin(rh, set(), where)
        if not rest:
            removed += d1; stats['thinned'] += len(d1)
            continue
        bass = [rest[0]] + [n for n in rest[1:] if n['pitch'] - rest[0]['pitch'] <= 12][:1]
        middle = [n for n in rest if n not in bass]
        keep = {n['pitch'] % 12 for n in rh + bass}
        middle, d2 = thin(middle, keep, where) if middle else ([], [])
        removed += d1 + d2
        stats['thinned'] += len(d1) + len(d2)
        for n in rh:
            n['staff'] = 0
        for n in middle + bass:
            n['staff'] = 1
        lh = sorted(n['pitch'] for n in middle + bass)
        if ok(lh):
            stats['reassigned'] += 1
        elif lh[-1] - lh[0] <= cfg.roll_lh_max_span:
            for n in middle + bass:
                n['roll'] = True
            stats['rolled'] += 1
            log(f"LH chord {lh} rolled ({where})")
        else:
            for n in bass:
                n['grace'] = True; n['bass_grace'] = True; n['dur'] = 0
            stats['grace_bass'] += 1
            log(f"bass {[n['pitch'] for n in bass]} -> grace before LH chord {[n['pitch'] for n in middle]} ({where})")
    ids = {id(n) for n in removed}
    return [n for n in notes if id(n) not in ids], stats


def cross_over(notes, ctx):
    """LH attacks written on the RH staff:
    - an attack above every RH note sounding then (the LH crosses over the RH)
      while the LH's bass is still held, when the RH can reach it together with
      its own notes: on the LH staff the clef change would cut through those ties;
    - with cross_high, a passage too high for the bass clef in a bar whose LH
      also has a bass (below D3): attacks needing local_clef_min_ledgers ledger
      lines, together with the LH attacks within half a beat of them needing one
      fewer (the rest of the same figure), unless one doubles a RH attack. On the
      LH staff they would force a clef change in the middle of the bar, squeezed
      between the notes; on the RH staff they are still played by the LH."""
    cfg = ctx.cfg
    rh = [n for n in notes if n['staff'] == 0 and not n['grace']]
    lh_at = defaultdict(list)
    for n in notes:
        if n['staff'] == 1 and not n['grace']:
            lh_at[n['on']].append(n)
    lh_bar = defaultdict(list)
    for t, grp in lh_at.items():
        lh_bar[ctx.bar_of(t)] += [n['pitch'] for n in grp]
    led = {t: max(ledgers(n['pitch'], 'F') for n in grp) for t, grp in lh_at.items()}
    lows = {t: min(n['pitch'] for n in grp) for t, grp in lh_at.items()}
    high = set()
    if cfg.cross_high:
        high = {t for t in lh_at if lows[t] >= cfg.cross_over_min and led[t] >= cfg.local_clef_min_ledgers
                and min(lh_bar[ctx.bar_of(t)]) < 50}
        grow = True
        while grow:                      # the rest of the figure: neighbours one ledger lower
            grow = False
            for t in lh_at:
                if t not in high and lows[t] >= cfg.cross_over_min and led[t] >= cfg.local_clef_min_ledgers - 1 \
                        and any(abs(t - h) <= F(1, 2) and ctx.bar_of(h) == ctx.bar_of(t) for h in high):
                    high.add(t); grow = True
    moved = 0
    for t, grp in sorted(lh_at.items()):
        lo = lows[t]
        if lo < cfg.cross_over_min:
            continue
        held = [n for g in lh_at.values() for n in g if n['on'] < t < n['on'] + n['dur'] and n['pitch'] < lo - 12]
        if t in high:
            # written in its own voice on the RH staff: it only must not double a RH attack
            if any(n['pitch'] == r['pitch'] and r['on'] == t for n in grp for r in rh):
                continue
            # nor fall below a RH passage that is all in the 8va register: it would break the 8va line and
            # leave chords two octaves wide (performance(1).mid m1: F#4-D5 under a RH run from A5 to C#7)
            bar_rh = [r['pitch'] for r in rh if ctx.bar_of(r['on']) == ctx.bar_of(t)]
            lo_min, need = cfg.ottava_rh
            if bar_rh and min(bar_rh) >= lo_min and lo < lo_min and \
                    (sum(p >= need for p in bar_rh) >= 2 or max(bar_rh) >= need + 3):
                continue
        else:
            sounding = [n['pitch'] for n in rh if n['on'] <= t < n['on'] + n['dur']]
            if not held or not sounding or lo <= max(sounding):
                continue
            # the RH takes these notes over: it must reach them with its own
            chord = [n['pitch'] for n in rh if n['on'] == t] + [n['pitch'] for n in grp]
            if max(chord) - min(chord) > cfg.hand_span or len(chord) > cfg.hand_max_notes:
                continue
        for n in grp:
            n.setdefault('original_staff', n['staff'])
            n['staff'] = 0; n['crossed'] = 'high' if t in high else True
        moved += len(grp)
        ctx.log(f"LH {'too high for the bass clef' if t in high else 'above RH'} -> RH staff "
                f"{sorted(n['pitch'] for n in grp)} ({ctx.where(t)})")
    return moved
