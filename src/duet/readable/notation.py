"""Turning a voice's events inside one bar into notatable pieces: rests,
readable note values, ties, triplet regions and beams. Positions are Fractions
in beats relative to the bar start; B is beats per bar."""

from collections import defaultdict
from fractions import Fraction as F

TYPES = {
    F(4): "whole",
    F(2): "half",
    F(1): "quarter",
    F(1, 2): "eighth",
    F(1, 4): "16th",
    F(1, 8): "32nd",
    F(1, 16): "64th",
}
VALUES = (F(4), F(3), F(2), F(3, 2), F(1), F(3, 4), F(1, 2), F(3, 8), F(1, 4), F(1, 8))
COMPOUND_VALUES = (
    F(6),
    F(3),
    F(3, 2),
    F(1),
    F(3, 4),
    F(1, 2),
    F(3, 8),
    F(1, 4),
    F(1, 8),
)


def compound_split(p, L, B):
    """compound meters (6/8, 12/8): the beat is a dotted quarter (3/2).
    Notes never cross a beat boundary except whole beats starting on one;
    a quarter inside a beat is allowed at its start or after its first 8th."""
    beat = F(3, 2)
    out = []
    while L > 0:
        ok = None
        k = p / beat
        on_beat = k.denominator == 1
        beat_end = (k.numerator // k.denominator + 1) * beat
        for v in COMPOUND_VALUES:
            if v > L or v > B:
                continue
            if v >= beat:
                good = (
                    on_beat
                    and (v / beat).denominator == 1
                    and (v != 3 or (p / 3).denominator == 1)
                    and (v != 6 or p == 0)
                )
            else:
                good = p + v <= beat_end
            if good:
                ok = v
                break
        if ok is None:
            ok = min(L, F(1, 8))
        out.append((p, ok))
        p += ok
        L -= ok
    return out


def duple_split(p, L, B, compound=False):
    if compound:
        return compound_split(p, L, B)
    "greedy split of [p, p+L) into values that show the beat"
    out = []
    while L > 0:
        ok = None
        for v in VALUES:
            if v > L or v > B:
                continue
            e = p + v
            if p.denominator == 1:
                ip = int(p)
                if ip == 0:
                    good = True
                elif B == 4 and ip == 2:
                    good = v <= 2
                elif B == 4 and ip == 1:
                    good = v <= 1 or v == 2
                else:
                    good = v <= min(2, B - ip) and (v.denominator == 1 or v <= 1)
            else:
                good = e <= int(p) + 1 or (
                    p - int(p) == F(1, 2)
                    and v == 1
                    and (int(p) % 2 == 0)
                    and (int(p) + 2 <= B)
                )
            if good:
                ok = v
                break
        if ok is None:
            ok = min(L, F(1, 8))
        out.append((p, ok))
        p += ok
        L -= ok
    return out


def regions(bounds, B, compound=False):
    """triplet regions: a 2-beat window whose boundaries are all on the
    quarter-triplet grid, or single beats containing 1/3 boundaries. A compound
    bar has none unless the model put notes on the triplet grid there (a bar of
    another meter in a piece with meter changes): those get quarter-beat regions"""
    regs = []
    if compound and (not any((b.denominator % 3 == 0 for b in bounds))):
        return regs
    h = F(0)
    while h < B:
        span = min(2, B - h)
        Bs = {b - h for b in bounds if h <= b <= h + span}
        if any((b.denominator % 3 == 0 for b in Bs)):
            if span == 2 and Bs <= {F(0), F(2, 3), F(4, 3), F(2)}:
                regs.append((h, h + 2, F(2, 3)))
            else:
                for bb in [
                    h + i for i in range(-(-span.numerator // span.denominator))
                ]:
                    end = min(bb + 1, h + span)
                    Bb = {b - bb for b in bounds if bb <= b <= end}
                    if any((b.denominator % 3 == 0 for b in Bb)):
                        regs.append((bb, end, F(1, 3)))
        h += 2
    return regs


def tuplet_split(p, L):
    out = []
    while L > 0:
        v = next(
            (
                c
                for c in (
                    F(4, 3),
                    F(1),
                    F(2, 3),
                    F(1, 2),
                    F(1, 3),
                    F(1, 6),
                    F(1, 12),
                    F(1, 24),
                )
                if c <= L
            ),
            L,
        )
        out.append((p, v))
        p += v
        L -= v
    return out


def type_of(v, tup):
    nv = v * F(3, 2) if tup else v
    for base, name in TYPES.items():
        if nv == base:
            return (name, 0)
        if nv == base * F(3, 2):
            return (name, 1)
    base = min(TYPES, key=lambda b: abs(b - nv))
    return (TYPES[base], 0)


def layout_voice(items, B, cut=None, compound=False, extra_cuts=()):
    """items: [(start, end, payload)] in bar coordinates, non-overlapping.
    Gaps become rests. `cut` forces a split point (pickup bars)."""
    BAR = F(B)
    filled, t = ([], F(0))
    for s, e, pl in items:
        if s > t:
            filled.append((t, s, None))
        filled.append((s, e, pl))
        t = e
    if t < BAR:
        filled.append((t, BAR, None))
    bounds = {b for s, e, _ in filled for b in (s, e)}
    regs = regions(bounds, B, compound)
    cuts = {c for a, b, _ in regs for c in (a, b)}
    cuts.update(extra_cuts)
    if cut:
        cuts.add(cut)
    pieces = []
    for s, e, pl in filled:
        extra = {F(2)} if B == 4 and pl is None and (s < 2 < e) and (s != 0) else set()
        pts = sorted({s, e} | {c for c in cuts if s < c < e} | extra)
        for a, b in zip(pts, pts[1:]):
            reg = next((r for r in regs if r[0] <= a and b <= r[1]), None)
            for ps, pv in (
                tuplet_split(a, b - a) if reg else duple_split(a, b - a, B, compound)
            ):
                pieces.append(dict(s=ps, d=pv, pl=pl, reg=reg))
    for i, pc in enumerate(pieces):
        pc["cont_prev"] = (
            i > 0 and pc["pl"] is not None and (pieces[i - 1]["pl"] is pc["pl"])
        )
        pc["cont_next"] = (
            i + 1 < len(pieces)
            and pc["pl"] is not None
            and (pieces[i + 1]["pl"] is pc["pl"])
        )
    return pieces


def beam(pieces, B, compound=False):
    """8ths beamed per half bar (per beat in odd meters); groups containing
    16ths/32nds beamed per beat; triplets per tuplet region"""
    for pc in pieces:
        pc["beam"] = {}
    span = F(3, 2) if compound else 2 if B % 2 == 0 else 1

    def win(pc):
        return ("t", pc["reg"][0]) if pc["reg"] else ("d", int(pc["s"] / span))

    groups, cur = ([], [])
    for pc in pieces:
        beamable = pc["pl"] is not None and type_of(pc["d"], pc["reg"] is not None)[
            0
        ] in ("eighth", "16th", "32nd", "64th")
        if beamable and cur and (win(cur[-1]) == win(pc)):
            cur.append(pc)
        else:
            if len(cur) > 1:
                groups.append(cur)
            cur = [pc] if beamable else []
    if len(cur) > 1:
        groups.append(cur)
    out = []
    for g in groups:
        if (
            not compound
            and (not g[0]["reg"])
            and any((type_of(p["d"], False)[0] != "eighth" for p in g))
        ):
            sub = defaultdict(list)
            for p in g:
                sub[int(p["s"])].append(p)
            out += [v for v in sub.values() if len(v) > 1]
        else:
            out.append(g)
    for g in out:
        for i, p in enumerate(g):
            p["beam"][1] = (
                "begin" if i == 0 else "end" if i == len(g) - 1 else "continue"
            )
        for lvl, names in (
            (2, ("16th", "32nd", "64th")),
            (3, ("32nd", "64th")),
            (4, ("64th",)),
        ):
            flags = [type_of(p["d"], p["reg"] is not None)[0] in names for p in g]
            i = 0
            while i < len(g):
                if not flags[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(g) and flags[j + 1]:
                    j += 1
                if j > i:
                    for k in range(i, j + 1):
                        g[k]["beam"][lvl] = (
                            "begin" if k == i else "end" if k == j else "continue"
                        )
                else:
                    g[i]["beam"][lvl] = "forward hook" if i == 0 else "backward hook"
                i = j + 1
