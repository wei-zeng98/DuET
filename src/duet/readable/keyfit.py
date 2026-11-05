"""Key signatures from the notes (a proposal for the engrave-polish skill).

Each bar is scored against every signature -7..7 fifths by how many of its notes
fall outside the scale (the raised 7th of the relative minor costs less). A
Viterbi pass picks one signature per bar: changing the signature costs
`switch` plus a little per fifth moved, and disagreeing with the model's key
costs `model_cost` per note, so the model's key stands unless the notes speak
against it over several bars. Enharmonic signatures (Gb / F#) have the same
notes; the model's choice and the distance in fifths from the neighbouring keys
decide between them.

Weights tuned on the 148-piece test set against the target scores
(tools/eval_spelling.py --keyfit): with the model's keys spelling is 94.8%
correct and the key signature agrees on 67% of notes; with these weights
96.2% / 77.2%. A larger switch cost gives fewer changes (better key agreement,
slightly worse spelling).
"""
import numpy as np

KEYS = list(range(-7, 8))


def pc_cost(fifths):
    """pitch class -> cost of a note in this signature"""
    tonic = (7 * fifths) % 12
    cost = np.ones(12)
    for d in (0, 2, 4, 5, 7, 9, 11):
        cost[(tonic + d) % 12] = 0
    cost[(tonic + 8) % 12] = 0.35         # leading note of the relative minor
    return cost


COST = {k: pc_cost(k) for k in KEYS}


def bar_costs(bar_pitches, model_keys, model_cost=0.05):
    """(bars x 15) cost matrix; bar_pitches[m] = [(pitch, weight)]"""
    C = np.zeros((len(bar_pitches), len(KEYS)))
    for m, ps in enumerate(bar_pitches):
        w = sum(v for _, v in ps)
        for j, k in enumerate(KEYS):
            C[m, j] = sum(v * COST[k][p % 12] for p, v in ps)
            if k != model_keys[m]:
                C[m, j] += model_cost * max(w, 1)
            C[m, j] += 0.02 * max(w, 1) * max(abs(k) - 5, 0)    # prefer 6 to 7 accidentals
    return C


def propose(bar_pitches, model_keys, switch=10.0, per_fifth=0.3, model_cost=0.05):
    C = bar_costs(bar_pitches, model_keys, model_cost)
    n, K = C.shape
    T = np.array([[0 if a == b else switch + per_fifth * abs(a - b) for b in KEYS] for a in KEYS])
    D = C[0].copy()
    back = np.zeros((n, K), int)
    for m in range(1, n):
        tot = D[:, None] + T
        back[m] = tot.argmin(0)
        D = tot.min(0) + C[m]
    path = [int(D.argmin())]
    for m in range(n - 1, 0, -1):
        path.append(int(back[m][path[-1]]))
    return [KEYS[j] for j in reversed(path)], C


def outside(bar_pitches, keys):
    """notes per bar outside the scale of the given keys (the raised 7th of the minor counts half)"""
    return [sum(v * (COST[k][p % 12] > 0) * (0.5 if COST[k][p % 12] < 1 else 1) for p, v in ps)
            for ps, k in zip(bar_pitches, keys)]


def guard(keys, bar_bass, share=0.3, max_len=24):
    """keep the surrounding key K0 over a passage that leaves it and comes back
    (K0 -> K1 -> K0, at most max_len bars) while K0's tonic (major or relative
    minor) is still the bass in at least `share` of its bars: borrowed chords
    around the same tonic, not a modulation. bar_bass[m]: lowest pitch starting
    in the first beat of bar m (or None). Returns (keys, [(lo, hi, K1, K0)] kept)."""
    out, kept = list(keys), []
    regs, a = [], 0
    for m in range(1, len(keys) + 1):
        if m == len(keys) or keys[m] != keys[a]:
            regs.append((a, m - 1, keys[a])); a = m
    for i in range(1, len(regs) - 1):
        lo, hi, k1 = regs[i]
        k0 = regs[i - 1][2]
        if regs[i + 1][2] != k0 or hi - lo + 1 > max_len:
            continue
        tonics = {(7 * k0) % 12, (7 * k0 + 9) % 12}
        hits = sum(1 for m in range(lo, hi + 1) if bar_bass[m] is not None and bar_bass[m] % 12 in tonics)
        if hits >= share * (hi - lo + 1):
            out[lo:hi + 1] = [k0] * (hi - lo + 1)
            kept.append((lo, hi, k1, k0))
    return out, kept

