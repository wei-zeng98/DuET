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


def playability(notes, ctx):
    cfg, log = ctx.cfg, ctx.log

    def ok(ps):
        if not ps:
            return True
        lim = cfg.hand_span + 2 if len(ps) == 2 else cfg.hand_span
        return len(ps) <= cfg.hand_max_notes and max(ps) - min(ps) <= lim

    def thin(grp, keep_pcs, where):
        grp = sorted(grp, key=lambda n: n["pitch"])
        dropped = []
        while not ok([n["pitch"] for n in grp]):
            pcs = [n["pitch"] % 12 for n in grp]
            cand = [
                n
                for n in grp
                if n["pitch"] % 12 in keep_pcs or pcs.count(n["pitch"] % 12) > 1
            ]
            victim = cand[0] if cand else grp[0]
            grp.remove(victim)
            dropped.append(victim)
            log(f"thinned p{victim['pitch']} ({where})")
        return grp, dropped

    by = defaultdict(list)
    for n in notes:
        if not n["grace"]:
            by[n["on"]].append(n)
    removed = []
    stats = dict(reassigned=0, grace_bass=0, thinned=0, rolled=0)
    for on, grp in sorted(by.items()):
        where = ctx.where(on)
        cur = [sorted(n["pitch"] for n in grp if n["staff"] == s) for s in (0, 1)]
        if ok(cur[0]) and ok(cur[1]):
            continue
        P = sorted(grp, key=lambda n: n["pitch"])
        ctxp = {
            s: [
                n["pitch"]
                for n in notes
                if n["staff"] == s
                and not n["grace"]
                and n["on"] != on
                and abs(n["on"] - on) <= 2
            ]
            for s in (0, 1)
        }

        def near(n, s):
            return (
                not ctxp[s]
                or min(abs(n["pitch"] - p) for p in ctxp[s]) <= cfg.hand_move_near
            )

        best = None
        for k in range(len(P) + 1):
            lh, rh = P[:k], P[k:]
            moved_ok = all(near(n, 1) for n in lh if n["staff"] != 1) and all(
                near(n, 0) for n in rh if n["staff"] != 0
            )
            if (
                moved_ok
                and ok([n["pitch"] for n in lh])
                and ok([n["pitch"] for n in rh])
            ):
                moves = sum(n["staff"] != 1 for n in lh) + sum(
                    n["staff"] != 0 for n in rh
                )
                if best is None or moves < best[0]:
                    best = (moves, k)
        if best is not None:
            k = best[1]
            for n in P[:k]:
                if n["staff"] != 1:
                    log(f"hand RH->LH p{n['pitch']} ({where})")
                n["staff"] = 1
            for n in P[k:]:
                if n["staff"] != 0:
                    log(f"hand LH->RH p{n['pitch']} ({where})")
                n["staff"] = 0
            stats["reassigned"] += 1
            continue
        top = P[-1]["pitch"]
        rh = [
            n
            for n in P
            if n["pitch"] >= top - cfg.hand_span and (n["staff"] == 0 or near(n, 0))
        ]
        rest = [n for n in P if n not in rh]
        rh, d1 = thin(rh, set(), where)
        if not rest:
            removed += d1
            stats["thinned"] += len(d1)
            continue
        bass = [rest[0]] + [n for n in rest[1:] if n["pitch"] - rest[0]["pitch"] <= 12][
            :1
        ]
        middle = [n for n in rest if n not in bass]
        keep = {n["pitch"] % 12 for n in rh + bass}
        middle, d2 = thin(middle, keep, where) if middle else ([], [])
        removed += d1 + d2
        stats["thinned"] += len(d1) + len(d2)
        for n in rh:
            n["staff"] = 0
        for n in middle + bass:
            n["staff"] = 1
        lh = sorted(n["pitch"] for n in middle + bass)
        if ok(lh):
            stats["reassigned"] += 1
        elif lh[-1] - lh[0] <= cfg.roll_lh_max_span:
            for n in middle + bass:
                n["roll"] = True
            stats["rolled"] += 1
            log(f"LH chord {lh} rolled ({where})")
        else:
            for n in bass:
                n["grace"] = True
                n["bass_grace"] = True
                n["dur"] = 0
            stats["grace_bass"] += 1
            log(
                f"bass {[n['pitch'] for n in bass]} -> grace before LH chord {[n['pitch'] for n in middle]} ({where})"
            )
    ids = {id(n) for n in removed}
    return [n for n in notes if id(n) not in ids], stats


def cross_over(notes, ctx):
    """An LH attack lying above every RH note sounding at that moment (the LH
    crosses over the RH) while the LH's bass is still held is written on the
    RH staff when the RH can still reach it; on the LH staff it would force a
    clef change under the tied bass notes."""
    cfg = ctx.cfg
    rh = [n for n in notes if n["staff"] == 0 and not n["grace"]]
    lh_at = defaultdict(list)
    for n in notes:
        if n["staff"] == 1 and not n["grace"]:
            lh_at[n["on"]].append(n)
    moved = 0
    for t, grp in sorted(lh_at.items()):
        lo = min(n["pitch"] for n in grp)
        if lo < cfg.cross_over_min:
            continue
        # only while the LH's own bass is still held from before: on the LH staff the
        # clef change would cut through those ties (plain hand crossings stay on the LH staff)
        held = [
            n
            for g in lh_at.values()
            for n in g
            if n["on"] < t < n["on"] + n["dur"] and n["pitch"] < lo - 12
        ]
        if not held:
            continue
        sounding = [n["pitch"] for n in rh if n["on"] <= t < n["on"] + n["dur"]]
        if not sounding or lo <= max(sounding):
            continue
        chord = [n["pitch"] for n in rh if n["on"] == t] + [n["pitch"] for n in grp]
        if max(chord) - min(chord) > cfg.hand_span or len(chord) > cfg.hand_max_notes:
            continue
        for n in grp:
            n["staff"] = 0
            n["crossed"] = True
        moved += len(grp)
        ctx.log(
            f"LH above RH -> RH staff {sorted(n['pitch'] for n in grp)} ({ctx.where(t)})"
        )
    return moved
