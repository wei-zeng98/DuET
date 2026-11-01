"""Frozen compound-note vocabulary; times are in 480 ticks per quarter."""

import json
from pathlib import Path
import numpy as np


class Codec:
    version = 1
    representation = "note_aligned_v1"
    bar_offset = 256
    fields = (
        "bar_index",
        "onset_quarter",
        "onset_fraction",
        "duration",
        "staff",
        "bar_length",
        "meter_numerator",
        "meter_denominator",
        "key_signature",
        "voice",
        "is_grace",
    )
    width = len(fields)
    ppq = 480

    def __init__(self):
        v = json.loads(
            (Path(__file__).parent / "vocabulary.json").read_text(encoding="utf-8")
        )
        self.tick_values = {
            k: np.array(v[k + "_ticks"], dtype=np.int64)
            for k in ("duration", "bar_length")
        }
        self.fractions = np.array(v["fraction_ticks"], dtype=np.int64)
        self.sizes = (
            512,
            64,
            len(self.fractions),
            len(self.tick_values["duration"]),
            2,
            len(self.tick_values["bar_length"]),
            64,
            7,
            15,
            16,
            2,
        )
        self.denominators = (1, 2, 4, 8, 16, 32, 64)

    def attributes(self, rows, anchor=0):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] != self.width:
            raise ValueError("Invalid aligned row shape")
        for j, n in enumerate(self.sizes):
            if ((rows[:, j] < 0) | (rows[:, j] >= n)).any():
                raise ValueError("Invalid aligned class ID")
        return dict(
            bar=rows[:, 0] - self.bar_offset + anchor,
            onset_tick=rows[:, 1] * 480 + self.fractions[rows[:, 2]],
            duration_tick=self.tick_values["duration"][rows[:, 3]],
            staff=rows[:, 4],
            length_tick=self.tick_values["bar_length"][rows[:, 5]],
            numerator=rows[:, 6] + 1,
            denominator=np.array(self.denominators)[rows[:, 7]],
            key=rows[:, 8] - 7,
            voice=rows[:, 9],
            grace=rows[:, 10].astype(bool),
        )

    @staticmethod
    def meter(value):
        n, d = value.split("/")
        return (int(n), int(d))
