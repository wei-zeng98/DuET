"""Validated in-memory inputs; performance rows follow the model input order."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Inputs:
    perf: list
    pedal: list
    score: dict
    frames: pd.DataFrame

    @classmethod
    def from_prediction(cls, data, score, frames):
        perf = list(
            zip(
                data["ref_onset"],
                data["ref_offset"],
                data["ref_pitch"],
                data["ref_vel"],
            )
        )
        if len(perf) != len(score.notes) or not np.array_equal(
            data["ref_pitch"], score.notes[:, 0]
        ):
            raise ValueError(
                "Readable input notes are not aligned with the raw predictions"
            )
        raw = {
            name: np.array(getattr(score, name), copy=True)
            for name in (
                "notes",
                "bar_starts",
                "bar_lengths",
                "meters",
                "voices",
                "keys",
            )
        }
        raw["is_grace"] = raw["notes"][:, 2] == 0
        return cls(
            perf=perf,
            pedal=list(zip(data["pedal_time"], np.asarray(data["pedal_value"]) * 127)),
            score=raw,
            frames=frames,
        )
