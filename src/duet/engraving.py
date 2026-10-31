"""Internal score and frame predictions consumed by readable engraving."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import CORE_SECONDS
from .features import build_window

COLUMNS = (
    "performance_time_sec,p_background,p_ordinary_beat,p_downbeat,"
    "p_any_beat,phase_cos_raw,phase_sin_raw,bar_phase_0_1"
)


@torch.inference_mode()
def predict_engraving_frames(model, data, device):
    end = float(data["duration_sec"])
    tables = []
    for start in np.arange(0.0, end, CORE_SECONDS):
        batch, _, left = build_window(data, float(start))
        batch = {k: v.to(device) for k, v in batch.items()}
        _, _, encoded = model.encode(batch)
        hidden = encoded["frame"]
        probability = (
            model.beat_head(hidden.transpose(1, 2))
            .transpose(1, 2)[0]
            .float()
            .softmax(-1)
            .cpu()
            .numpy()
        )
        phase = (
            torch.cat(
                (
                    model.frame_phase_cos_head(hidden),
                    model.frame_phase_sin_head(hidden),
                ),
                -1,
            )[0]
            .float()
            .cpu()
            .numpy()
        )
        ticks = np.rint((left + np.arange(len(probability)) / 50.0) * 50.0).astype(int)
        times = ticks / 50.0
        own = (
            (times >= start)
            & (times < start + CORE_SECONDS)
            & (times < end)
            & encoded["frame_valid"][0].cpu().numpy().astype(bool)
        )
        angle = np.mod(np.arctan2(phase[:, 1], phase[:, 0]) / (2 * np.pi), 1.0)
        tables.append(
            np.column_stack(
                (times, probability, probability[:, 1:].sum(1), phase, angle)
            )[own]
        )
    table = np.concatenate(tables)
    if (
        not len(table)
        or not np.isfinite(table).all()
        or not np.allclose(np.diff(table[:, 0]), 0.02)
    ):
        raise ValueError("Incomplete engraving frame predictions")
    return pd.DataFrame(table, columns=COLUMNS.split(","))


def write_readable_musicxml(score, data, frames, path, *, title):
    """Publish only a complete readable XML; raw model arrays are copied first."""
    from .readable.inputs import Inputs
    from .readable.pipeline import run

    path = Path(path)
    inputs = Inputs.from_prediction(data, score, frames)
    with tempfile.TemporaryDirectory(prefix=".readable-", dir=path.parent) as temp:
        xml, summary = run(inputs, temp, title=title)
        ET.parse(xml)
        Path(xml).replace(path)
    return {
        "version": "1.0.0",
        "upstream": "0.9.1",
        "mode": "readable",
        "title": title,
        "summary": summary,
    }
