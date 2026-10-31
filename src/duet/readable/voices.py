"""Duplicate removal, voice layers and duration cleanup per staff."""

from bisect import bisect_left
from fractions import Fraction as F
from collections import defaultdict


def round_half(v):
    return F(max(1, round(v * 2)), 2)


def dedupe(notes, ctx):
    seen, out = {}, []
    for n in sorted(notes, key=lambda n: (n["on"], n["grace"], -n["dur"])):
        if n["grace"]:
            out.append(n)
            continue
        if (n["staff"], n["on"], n["pitch"]) in seen or (
            1 - n["staff"],
            n["on"],
            n["pitch"],
        ) in seen:
            ctx.log(f"duplicate removed p{n['pitch']} ({ctx.where(n['on'])})")
            continue
        seen[(n["staff"], n["on"], n["pitch"])] = n
        out.append(n)
    return out


def assign(notes, staff, ctx):
    """Give every note of a staff a layer (A = main voice, B = second voice)
    and a cleaned duration. Layer B holds (a) triplet notes when a half bar
    mixes triplets with duple off-beats and (b) notes sustained under moving
    notes."""
    end_beat = ctx.END
    S = sorted(
        [n for n in notes if n["staff"] == staff and not n["grace"]],
        key=lambda n: (n["on"], n["pitch"]),
    )
    onsets = sorted(set(n["on"] for n in S))
    other_on = sorted(
        set(n["on"] for n in notes if n["staff"] != staff and not n["grace"])
    )

    def next_on(o):
        for p in onsets:
            if p > o:
                return p
        return None

    frac = lambda o: o - o.numerator // o.denominator
    for n in S:
        n["layer"] = "A"
    halves = defaultdict(list)
    for n in S:
        halves[ctx.half_index(n["on"])].append(n)
    for h, grp in halves.items():
        local_compound = (
            ctx.bars.compound(ctx.bar_of(grp[0]["on"])) if ctx.bars else ctx.compound
        )
        if local_compound:
            continue
        dens = set(frac(n["on"]).denominator for n in grp)
        # on a sextuplet grid (1/6) the half beat belongs to the tuplet figure
        duple = {4, 8} if 6 in dens or 12 in dens else {2, 4, 8}
        if any(d % 3 == 0 for d in dens) and any(d in duple for d in dens):
            for n in grp:
                fr = frac(n["on"])
                if fr.denominator % 3 == 0 or (
                    fr == 0 and n["dur"].denominator % 3 == 0
                ):
                    n["layer"] = "B"
            ctx.log(
                f"staff{staff + 1} {ctx.where(min(n['on'] for n in grp))}: triplet layer split from duple layer"
            )
    for n in S:
        if n["layer"] != "A":
            continue
        nx = next_on(n["on"])
        if nx is None:
            continue
        over = n["on"] + n["dur"] - nx
        cv = ctx.cfg
        if (
            over >= F(cv.voice_b_over).limit_denominator(24)
            and n["dur"] >= F(cv.voice_b_min_dur).limit_denominator(24)
            and (n["pd"] >= float(nx - n["on"]) + cv.voice_b_hold or n.get("crossed"))
        ):
            restrike = any(
                m["pitch"] == n["pitch"] and nx <= m["on"] < n["on"] + n["dur"]
                for m in S
                if m is not n
            )
            if not restrike:
                n["layer"] = "B"

    for L in ("A", "B"):
        pool = [n for n in S if n["layer"] == L]
        ons = sorted(set(n["on"] for n in pool))
        by = defaultdict(list)
        for n in pool:
            by[n["on"]].append(n)
        for i, o in enumerate(ons):
            nx = ons[i + 1] if i + 1 < len(ons) else end_beat
            gap = nx - o
            grp = by[o]
            d = max(n["dur"] for n in grp)
            pd = max(n["pd"] for n in grp)
            if d > gap:
                d = gap
            elif d < gap:
                if (
                    ctx.cfg.extend_by_hold and pd > float(d) + 0.3
                ):  # key held longer: no rest
                    d = min(gap, max(d, round_half(pd)))
                if (
                    ctx.cfg.extend_by_hold
                    and gap - d <= F(1, 4)
                    and pd >= float(d) * 0.6
                ):
                    d = gap
                if 0 < gap - d < F(1, 4):  # never a rest shorter than a 16th
                    d = gap
                # a short rest before the next attack is a detached note, not a rest worth writing
                fill = F(ctx.cfg.rest_fill_max).limit_denominator(24)
                # (kept when the other hand attacks inside it but not with this note: alternating hands)
                if (
                    0 < gap - d <= fill
                    and gap <= ctx.cfg.rest_fill_gap
                    and d >= (gap - d) * ctx.cfg.rest_fill_ratio
                    and not (
                        bisect_left(other_on, nx) > bisect_left(other_on, o + d)
                        and bisect_left(other_on, o)
                        == bisect_left(other_on, o + F(1, 100))
                    )
                ):
                    d = gap
            # held one 16th past a beat into the next attack: end on the beat
            e = o + d
            if (
                d == gap
                and (e - int(e)) == F(1, 4)
                and int(e) - o >= F(1, 2)
                and o.denominator in (1, 2, 4)
            ):
                d = int(e) - o
            for n in grp:
                n["dur"] = d
    for n in S:  # a B note must not swallow an A note of the same pitch
        if n["layer"] == "B":
            for m in S:
                if (
                    m["layer"] == "A"
                    and m["pitch"] == n["pitch"]
                    and n["on"] < m["on"] < n["on"] + n["dur"]
                ):
                    n["dur"] = m["on"] - n["on"]
    return S
