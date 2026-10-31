"""Meter analysis.

The score model predicts a meter per bar. When the model mixes meters, a simple
-meter section whose subdivision is mostly triplets and whose 8th-note pulse
matches a compound section (e.g. "4/4 with triplet 8ths" next to 6/8) is the
same music written two ways; it is rewritten in the compound meter by scaling
its positions by 3/2 (a 4/4 bar becomes two 6/8 bars) and one meter is written.
Other meter changes keep the model's bar sequence (model_bars); the most common
meter is then the main meter used for the beat grid and quantization.

All positions stay in quarter notes. For a meter n/d:
  bar_q   = quarters per bar        (4/4 -> 4, 6/8 -> 3, 12/8 -> 6)
  pulse_q = quarters per felt beat  (simple -> 1, compound -> 3/2)
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np


@dataclass
class Meter:
    num: int
    den: int

    @property
    def compound(self):
        return self.den == 8 and self.num % 3 == 0 and self.num >= 6

    @property
    def bar_q(self):
        return F(self.num * 4, self.den)

    @property
    def pulse_q(self):
        return F(3, 2) if self.compound else F(1)

    def __str__(self):
        return f"{self.num}/{self.den}"


def parse(m):
    a, b = str(m).split("/")
    return Meter(int(a), int(b))


def analyse(score, onsets, log):
    """returns (Meter, q, qd, to_old, model_bars).
    Meter: the main meter (beat grid, quantization). q/qd possibly rewritten;
    to_old maps a rewritten position back to the model's original position.
    model_bars: None when one meter is written throughout, else the model's
    bars [(start q, num, den)] for genuine meter changes."""
    meters = [str(m) for m in score.get("meters", ["4/4"])]
    q = score["notes"][:, 1].astype(float).copy()
    qd = score["notes"][:, 2].astype(float).copy()
    counts = Counter(meters)
    if len(counts) == 1:
        M = parse(meters[0])
        check_supported(M)
        starts = np.asarray(score.get("bar_starts", []), float)
        lengths = np.asarray(
            score.get("bar_lengths", np.full(len(starts), float(M.bar_q))), float
        )
        # the bars' own length outranks the meter token: "4/4" with 3-beat bars is written 3/4
        M = meter_from_lengths(M, lengths, log)
        # A constant signature does not imply regular measures: pickups and
        # partial internal/final bars have their own actual lengths.
        regular = (
            len(starts)
            and np.allclose(starts, np.arange(len(starts)) * float(M.bar_q))
            and np.allclose(lengths, float(M.bar_q))
        )
        bars = (
            None
            if regular or not len(starts)
            else [(float(b), M.num, M.den, float(L)) for b, L in zip(starts, lengths)]
        )
        return M, q, qd, (lambda p: p), bars

    bar_starts = score["bar_starts"].astype(float)
    # contiguous sections of one meter
    sections, s0 = [], 0
    for i in range(1, len(meters) + 1):
        if i == len(meters) or meters[i] != meters[s0]:
            lo = bar_starts[s0]
            hi = bar_starts[i] if i < len(meters) else np.inf
            sections.append(
                dict(meter=parse(meters[s0]), bars=(s0, i - 1), lo=lo, hi=hi)
            )
            s0 = i
    log(
        f"meter: model sections {[(str(sc['meter']), sc['bars'][0] + 1, sc['bars'][1] + 1) for sc in sections]}"
    )

    def eighth_s(sc, triplet):
        """seconds per 8th (compound / duple) or per triplet 8th (simple)"""
        sel = (q >= sc["lo"]) & (q < sc["hi"])
        if sel.sum() < 8:
            return None
        a = np.polyfit(q[sel], onsets[sel], 1)[0]  # seconds per quarter
        return a / (3 if triplet else 2)

    compound = [sc for sc in sections if sc["meter"].compound]
    target = (
        Counter(str(sc["meter"]) for sc in compound).most_common(1)[0][0]
        if compound
        else None
    )

    def keep_changes(reason):
        """genuine meter changes: keep the model's bar sequence"""
        main = parse(counts.most_common(1)[0][0])
        if main.den == 8 and not main.compound or main.den not in (2, 4, 8):
            main = Meter(4, 4)
        log(
            f"WARNING meter: {reason}; keeping the model's meter changes {dict(counts)}, main meter {main}"
        )
        lengths = score.get("bar_lengths", [float(parse(m).bar_q) for m in meters])
        bars = [
            (float(b), *(int(v) for v in str(m).split("/")), float(L))
            for b, m, L in zip(bar_starts, meters, lengths)
        ]
        return main, q, qd, (lambda p: p), bars

    if target is None:
        return keep_changes("no compound section to unify")
    ref = np.median([v for v in (eighth_s(sc, False) for sc in compound) if v])
    maps = []  # applied in order: (lo, hi, scale) on the then-current coordinates
    for sc in sections:
        M = sc["meter"]
        if M.compound:
            continue
        sel = (q >= sc["lo"]) & (q < sc["hi"])
        fr = np.round(q[sel] % 1, 3)
        off = fr[fr != 0]
        trip = np.isin(off, [0.333, 0.667]).mean() if len(off) else 0
        e = eighth_s(sc, True)
        same_pulse = e is not None and abs(e / ref - 1) < 0.15
        log(
            f"meter: {M} bars {sc['bars'][0] + 1}-{sc['bars'][1] + 1}: {trip * 100:.0f}% of off-beat onsets on triplets, "
            f"triplet 8th {e if e is None else round(e, 3)}s vs compound 8th {ref:.3f}s"
        )
        if trip < 0.6 or not same_pulse:
            q[:] = score["notes"][:, 1]
            qd[:] = score["notes"][:, 2]
            return keep_changes(
                f"{M} -> {target} is not a triplet rewrite of the same pulse"
            )
        # rewrite: positions after the section start scale by 3/2, later sections shift accordingly
        lo, hi = sc["lo"], sc["hi"]
        inside = (q >= lo) & (q < hi)
        q[inside] = lo + (q[inside] - lo) * 1.5
        qd[inside] = qd[inside] * 1.5
        if np.isfinite(hi):
            later = q >= hi
            q[later] += (hi - lo) * 0.5
        maps.append((lo, hi))
        log(
            f"WARNING meter: bars {sc['bars'][0] + 1}-{sc['bars'][1] + 1} written by the model as {M} with triplets "
            f"are the same pulse as {target}; rewritten in {target} (positions x1.5)"
        )
    M = parse(target)
    check_supported(M)

    def to_old(p):
        for lo, hi in reversed(maps):
            new_hi = lo + (hi - lo) * 1.5 if np.isfinite(hi) else np.inf
            if p >= new_hi:
                p = p - (hi - lo) * 0.5
            elif p >= lo:
                p = lo + (p - lo) / 1.5
        return p

    return M, q, qd, to_old, None


def meter_from_lengths(M, lengths, log, share=0.6):
    """the meter implied by the length most interior bars have, when it is not
    the length of meter M (first and last bars may be partial)"""
    inner = [F(float(v)).limit_denominator(48) for v in lengths[1:-1]]
    if len(inner) < 4:
        return M
    L, n = Counter(inner).most_common(1)[0]
    if L == M.bar_q or n < share * len(inner):
        return M
    for den in (M.den, 4, 8):
        num = L * den / 4
        if num.denominator == 1:
            cand = Meter(int(num), den)
            try:
                check_supported(cand)
            except NotImplementedError:
                continue
            log(
                f"WARNING meter: model meter {M} but {n}/{len(inner)} bars are {float(L):g} beats long -> {cand}"
            )
            return cand
    return M


def check_supported(M):
    if M.bar_q.denominator != 1:
        raise NotImplementedError(
            f"meter {M}: bars must hold a whole number of quarters (supported: n/4, 6/8, 12/8)"
        )
    if M.den == 8 and not M.compound:
        raise NotImplementedError(f"meter {M}: simple x/8 meters are not supported")
    if M.den not in (2, 4, 8):
        raise NotImplementedError(f"meter {M} is not supported")
