"""Key-aware pitch spelling and staff-position helpers."""

STEPS = "CDEFGAB"
STEP_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# major tonic for each number of fifths
TONIC = {
    -7: ("C", -1),
    -6: ("G", -1),
    -5: ("D", -1),
    -4: ("A", -1),
    -3: ("E", -1),
    -2: ("B", -1),
    -1: ("F", 0),
    0: ("C", 0),
    1: ("G", 0),
    2: ("D", 0),
    3: ("A", 0),
    4: ("E", 0),
    5: ("B", 0),
    6: ("F", 1),
    7: ("C", 1),
}
MAJOR = [0, 2, 4, 5, 7, 9, 11]


def key_alters(fifths):
    """step -> alteration implied by the key signature"""
    order_sharp, order_flat = "FCGDAEB", "BEADGCF"
    alt = {s: 0 for s in STEPS}
    if fifths > 0:
        for s in order_sharp[:fifths]:
            alt[s] = 1
    elif fifths < 0:
        for s in order_flat[:-fifths]:
            alt[s] = -1
    return alt


def spelling_table(fifths):
    """pitch class -> (step, alter). Diatonic notes follow the key; chromatic
    notes use the natural letter when there is one, otherwise a sharp (sharp
    keys / C) or a flat (flat keys)."""
    fifths = max(-7, min(7, int(fifths)))
    ka = key_alters(fifths)
    table = {}
    for s in STEPS:
        table[(STEP_PC[s] + ka[s]) % 12] = (s, ka[s])
    for pc in range(12):
        if pc in table:
            continue
        nat = [s for s in STEPS if STEP_PC[s] == pc]
        if nat:
            table[pc] = (nat[0], 0)
        elif fifths >= 0:
            s = [s for s in STEPS if STEP_PC[s] == (pc - 1) % 12][0]
            table[pc] = (s, 1)
        else:
            s = [s for s in STEPS if STEP_PC[s] == (pc + 1) % 12][0]
            table[pc] = (s, -1)
    return table


class Speller:
    def __init__(self, fifths):
        self.fifths = int(fifths)
        self.table = spelling_table(fifths)
        self.key_alt = key_alters(fifths)

    def spell(self, p):
        step, alt = self.table[p % 12]
        octave = (p - alt) // 12 - 1  # B#3 sounds as C4, Cb4 as B3
        return step, alt, octave

    def diat(self, p):
        step, alt, octave = self.spell(p)
        return octave * 7 + STEPS.index(step)


_C = Speller(0)
TREBLE = (_C.diat(64), _C.diat(77))  # E4 .. F5
BASS = (_C.diat(43), _C.diat(57))  # G2 .. A3


_DEFAULT = [_C]


def set_key(fifths):
    """key used for staff-position estimates (ledger lines) in layout decisions"""
    _DEFAULT[0] = Speller(fifths)


def ledgers(p, clef, speller=None):
    lo, hi = TREBLE if clef == "G" else BASS
    d = (speller or _DEFAULT[0]).diat(p)
    return (lo - d) // 2 if d < lo else ((d - hi) // 2 if d > hi else 0)
