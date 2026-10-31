"""Shared run state."""

from bisect import bisect_right
from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Callable
from .config import Config


class Bars:
    """bar structure in quarter notes: start, length and time signature per bar"""

    def __init__(self, starts, sigs, lengths=None):
        self.starts = [F(s) for s in starts]
        self.sigs = list(sigs)  # (num, den) per bar
        self.lens = (
            [F(v) for v in lengths]
            if lengths is not None
            else [F(n * 4, d) for n, d in self.sigs]
        )
        if not (len(self.starts) == len(self.sigs) == len(self.lens)):
            raise ValueError("Bar starts, signatures and lengths must have equal size")
        if any(v <= 0 for v in self.lens) or any(
            b <= a for a, b in zip(self.starts, self.starts[1:])
        ):
            raise ValueError("Bar starts must increase and lengths must be positive")
        # first half-bar index of every bar (2-quarter windows from each bar line)
        self.half_base = [0]
        for L in self.lens:
            self.half_base.append(
                self.half_base[-1] + -(-L.numerator // (2 * L.denominator))
            )

    @classmethod
    def uniform(cls, nbars, num, den):
        L = F(num * 4, den)
        return cls([i * L for i in range(nbars)], [(num, den)] * nbars)

    @property
    def n(self):
        return len(self.starts)

    def index(self, beat):
        return max(0, min(self.n - 1, bisect_right(self.starts, beat) - 1))

    def start(self, m):
        return self.starts[m]

    def length(self, m):
        return self.lens[m]

    def end(self, m):
        return self.starts[m] + self.lens[m]

    def sig(self, m):
        return self.sigs[m]

    def compound(self, m):
        n, d = self.sigs[m]
        return d == 8 and n % 3 == 0 and n >= 6


@dataclass
class Ctx:
    cfg: Config
    log: Callable
    B: int = 4  # quarter notes per bar of the main meter (4/4 -> 4, 6/8 -> 3)
    beat_type: int = 4
    num: int = 4  # time signature numerator of the main meter
    pulse: F = F(1)  # quarters per felt beat (compound: 3/2)
    compound: bool = False
    swing: object = None  # median off-beat position when the piece is swung
    nbars: int = 0
    pickup: F = F(0)  # length of the leading empty part of bar 0 (0 = no pickup)
    keys: list = field(default_factory=list)  # fifths per bar
    title: str = ""
    bars: Bars = None

    @property
    def END(self):
        return self.bars.end(self.bars.n - 1) if self.bars else F(self.nbars * self.B)

    def bar_of(self, beat):
        return self.bars.index(beat) if self.bars else int(beat // self.B)

    def bar_start(self, m):
        return self.bars.start(m) if self.bars else F(m * self.B)

    def half_index(self, beat):
        """consecutive index of the half bar holding beat: 2-quarter windows counted
        from each bar line (a 3/4 bar is 2 + 1), so a window never crosses a bar line"""
        m = self.bar_of(beat)
        k = int((beat - self.bar_start(m)) // 2)
        if self.bars:
            return self.bars.half_base[m] + k
        return m * -(-self.B // 2) + k

    def barno(self, beat):
        """printed bar number (a pickup bar is bar 0)"""
        return self.bar_of(beat) + (0 if self.pickup else 1)

    def where(self, beat):
        b = self.bar_of(beat)
        return f"bar {self.barno(beat)} beat {float(beat - self.bar_start(b)) + 1:g}"
