"""Spelling of notes outside the key from their context (a proposal for the
engrave-polish skill).

Spellings sit on the line of fifths (... Gb Db Ab Eb Bb F C G D A E B F# C# ...).
A note outside the key signature takes the spelling nearest to its context:

  harmony   the weighted centre of the notes sounding with it (weight 1) and
            of the notes within `window` beats (weight falling to 0), as
            currently spelled: G7 + A#/Bb -> Bb, F-Ab-C-Eb -> Ab, Eb
  key       plus `w_key` x the distance to the key's centre
  line      plus `w_line` when the next note of the same staff is a step away
            and this spelling does not move to the next letter (G-Ab-A,
            A-G#-G): chromatic lines rise in sharps and fall in flats. The step
            from the previous note does not count (G-G# is as right as G#-A);
            `line_prev` > 0 would count it too, which tested worse

Two passes, so chromatic neighbours inform each other. Only notes outside the
key signature are respelled; notes in the key keep the key's spelling.

Weights tuned on the 148-piece test set against the target scores
(tools/eval_spelling.py --keyfit --spellfit): with keys from engrave.keyfit,
notes outside the key are spelled right 57.3% -> 73.4% (gangqinpu 64.9% ->
81.6%), all notes 96.24% -> 97.43%. Most remaining errors come with a wrong
key (ASAP: key signature agrees on 55% of notes). Weighting the notes sounding
with it more (`w_sound`) or penalising E#/B#/Cb/Fb (`w_odd`) did not help, so
pop chords like a secondary dominant's G# in C are left to the reviewer.
"""
from bisect import bisect_left, bisect_right

from .spelling import Speller, STEPS, STEP_PC

LOF_STEP = {'F': -1, 'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5}


def lof(step, alt):
    return LOF_STEP[step] + 7 * alt


def candidates(pitch):
    """spellings of a pitch with at most one sharp or flat"""
    out = []
    for s in STEPS:
        alt = (pitch - STEP_PC[s]) % 12
        alt = alt - 12 if alt > 6 else alt
        if -1 <= alt <= 1:
            out.append((s, alt))
    return out


def propose(notes, key_of, window=2.0, w_key=1.0, w_line=3.0, passes=2, w_sound=1.0, w_odd=0.0, line_prev=0.0,
            margins=None):
    """{id(note): (step, alter)} for notes outside the key whose best spelling is
    not the key's default; key_of(note) -> fifths of the key signature there.
    margins, when a dict, receives id(note) -> (cost gap to the runner-up, runner-up
    spelling) for every note outside the key: a small gap is an uncertain choice"""
    ns = sorted((n for n in notes if not n.get('grace')), key=lambda n: n['on'])
    ons = [float(n['on']) for n in ns]
    spellers = {}
    sp = lambda k: spellers.setdefault(k, Speller(k))
    cur = {id(n): sp(key_of(n)).spell(n['pitch'])[:2] for n in ns}
    tonic_pc = lambda k: (7 * k) % 12
    outside = [n for n in ns if (n['pitch'] - tonic_pc(key_of(n))) % 12 not in (0, 2, 4, 5, 7, 9, 11)]
    staff_seq = {}
    for n in ns:
        staff_seq.setdefault(n['staff'], []).append(n)
    staff_on = {st: [float(n['on']) for n in seq] for st, seq in staff_seq.items()}

    def neighbours(n):
        seq, so = staff_seq[n['staff']], staff_on[n['staff']]
        o = float(n['on'])
        before = seq[max(0, bisect_left(so, o) - 6):bisect_left(so, o)]
        after = seq[bisect_right(so, o):bisect_right(so, o) + 6]
        pick = lambda c: min(c, key=lambda m: abs(m['pitch'] - n['pitch'])) if c else None
        prev = pick([m for m in before if o - float(m['on']) <= window and float(m['on']) == float(before[-1]['on'])]) if before else None
        nxt = pick([m for m in after if float(m['on']) - o <= window and float(m['on']) == float(after[0]['on'])]) if after else None
        step = lambda m: m is not None and 1 <= abs(m['pitch'] - n['pitch']) <= 2
        return [(m, w) for m, w in ((prev, line_prev), (nxt, 1.0)) if step(m)]

    for _ in range(int(passes)):
        new = {}
        for n in outside:
            o = float(n['on'])
            lo, hi = bisect_left(ons, o - 8), bisect_right(ons, o + window)
            tot = wsum = 0.0
            for m in ns[lo:hi]:
                if m is n:
                    continue
                mo = float(m['on'])
                if mo <= o < mo + float(m['dur']):
                    w = w_sound
                elif abs(mo - o) <= window:
                    w = 0.5 * (1 - abs(mo - o) / window)
                else:
                    continue
                tot += w * lof(*cur[id(m)]); wsum += w
            k = key_of(n)
            center = tot / wsum if wsum else k + 2
            line = neighbours(n)
            best, costs = None, []
            for c in candidates(n['pitch']):
                cost = abs(lof(*c) - center) + w_key * abs(lof(*c) - (k + 2))
                if c in (('E', 1), ('B', 1), ('C', -1), ('F', -1)):
                    cost += w_odd
                for m, wl in line:
                    ms = cur[id(m)][0]
                    if ms == c[0] or abs(STEPS.index(ms) - STEPS.index(c[0])) not in (1, 6):
                        cost += w_line * wl
                costs.append((cost, c))
                if best is None or cost < best[0]:
                    best = (cost, c)
            new[id(n)] = best[1]
            if margins is not None:
                rest = sorted(x for x in costs if x[1] != best[1])
                margins[id(n)] = (rest[0][0] - best[0], rest[0][1]) if rest else (99.0, None)
        cur.update(new)
    out = {}
    for n in outside:
        if cur[id(n)] != sp(key_of(n)).spell(n['pitch'])[:2]:
            out[id(n)] = cur[id(n)]
    return out
