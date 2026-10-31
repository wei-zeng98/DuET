"""Readable integration tests use generated MIDI and deterministic predictions."""

import json
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import mido
import numpy as np
import pandas as pd
import torch
from duet.codec import Codec
from duet.engraving import predict_engraving_frames, write_readable_musicxml
from duet.pipeline import DuET
from duet.score import Score


class Predictions(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.codec = Codec()

    def generate_rows(self, batch, **kwargs):
        n = batch["query_note_slots"].shape[1]
        duration = int(np.flatnonzero(self.codec.tick_values["duration"] == 480)[0])
        length = int(np.flatnonzero(self.codec.tick_values["bar_length"] == 1920)[0])
        return torch.tensor(
            [
                [256 + i // 4, i % 4, 0, duration, int(i % 2), length, 3, 2, 7, 0, 0]
                for i in range(n)
            ]
        )

    def encode(self, batch):
        hidden = torch.stack(
            (batch["frame_times_rel"], torch.ones_like(batch["frame_times_rel"])), -1
        )
        return None, None, {"frame": hidden, "frame_valid": batch["frame_valid"]}

    def beat_head(self, hidden):
        return torch.zeros((1, 3, hidden.shape[-1]))

    def frame_phase_cos_head(self, hidden):
        return torch.cos(hidden[..., :1] * torch.pi)

    def frame_phase_sin_head(self, hidden):
        return torch.sin(hidden[..., :1] * torch.pi)


def performance(path):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    for i in range(8):
        track.append(
            mido.Message("note_on", note=60 + i, velocity=80, time=0 if not i else 96)
        )
        track.append(mido.Message("note_off", note=60 + i, time=384))
    # Reader cleanup must remain effective before engraving.
    track.append(mido.Message("note_on", note=90, velocity=80, time=0))
    track.append(mido.Message("note_off", note=90, time=0))
    midi.save(path)


class EngravingTests(unittest.TestCase):
    def test_midi_default_readable_and_original_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            midi = Path(d) / "曲目 & test.mid"
            performance(midi)
            original = midi.read_bytes()
            output = Path(d) / "result"
            report = DuET(Predictions(), {"weights_sha256": "test"}, "cpu").transcribe(
                midi, output
            )
            self.assertEqual(
                {p.name for p in output.iterdir()},
                {"score.mid", "score.musicxml", "notes.json", "report.json"},
            )
            xml = ET.parse(output / "score.musicxml")
            self.assertEqual(xml.findtext("work/work-title"), "曲目 & test")
            self.assertEqual(len(xml.findall("part")), 1)
            self.assertEqual(xml.findtext(".//staves"), "2")
            self.assertEqual(report["engraving"]["mode"], "readable")
            self.assertEqual(report["engraving"]["upstream"], "0.9.1")
            self.assertEqual(
                report["midi_input_cleanup"]["nonpositive_duration_notes"], 1
            )
            self.assertEqual(
                [n["pitch"] for n in json.loads((output / "notes.json").read_text())],
                list(range(60, 68)),
            )
            written = mido.MidiFile(output / "score.mid")
            self.assertEqual(len(written.tracks), 3)
            self.assertEqual(
                [t.name for t in written.tracks[1:]], ["staff_0", "staff_1"]
            )
            for track in written.tracks[1:]:
                self.assertEqual(
                    {m.channel for m in track if m.type in ("note_on", "note_off")}, {0}
                )
            self.assertEqual(midi.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                DuET(Predictions(), {"weights_sha256": "test"}, "cpu").transcribe(
                    midi, output
                )

    def test_audio_uses_original_filename_and_keeps_five_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            midi = Path(d) / "source.mid"
            performance(midi)

            def frontend(audio, output, **kwargs):
                shutil.copyfile(midi, output)
                return {"name": "test"}

            output = Path(d) / "result"
            with patch("duet.audio.audio_to_midi", side_effect=frontend):
                report = DuET(
                    Predictions(), {"weights_sha256": "test"}, "cpu"
                ).transcribe_audio("my.piano.wav", output)
            self.assertEqual(report["score_title"], "my.piano")
            self.assertEqual(
                ET.parse(output / "score.musicxml").findtext("work/work-title"),
                "my.piano",
            )
            self.assertEqual(
                {p.name for p in output.iterdir()},
                {
                    "score.mid",
                    "score.musicxml",
                    "notes.json",
                    "report.json",
                    "performance.mid",
                },
            )

    def test_frame_windows_include_silent_tail_without_duplicates(self):
        data = dict(
            ref_onset=np.array([0.0]),
            ref_offset=np.array([34.05]),
            ref_pitch=np.array([60]),
            ref_vel=np.array([80]),
            pedal_time=np.array([]),
            pedal_value=np.array([]),
            duration_sec=34.05,
        )
        frames = predict_engraving_frames(Predictions(), data, "cpu")
        np.testing.assert_allclose(
            frames.performance_time_sec, np.arange(0, 34.05, 0.02), atol=1e-10
        )
        self.assertTrue(np.isfinite(frames.to_numpy()).all())

    def test_leading_grace_and_raw_score_are_preserved(self):
        pitches = np.array([80, 81, 68, 69, 72, 74, 76, 77, 79])
        on = np.array([0.0, 0.06, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        data = dict(
            ref_onset=on,
            ref_offset=on + 0.3,
            ref_pitch=pitches,
            ref_vel=np.full(9, 80),
            pedal_time=np.array([]),
            pedal_value=np.array([]),
            duration_sec=3.8,
        )
        score = Score(
            np.column_stack(
                (pitches, np.r_[0.0, np.arange(8)], np.r_[0.0, np.ones(8)], np.zeros(9))
            ),
            np.array([0.0, 4.0]),
            np.array([4.0, 4.0]),
            ["4/4", "4/4"],
            np.zeros(9, int),
            np.zeros(2, int),
        )
        before = score.notes.copy()
        with tempfile.TemporaryDirectory() as d:
            xml = Path(d) / "score.musicxml"
            write_readable_musicxml(
                score,
                data,
                predict_engraving_frames(Predictions(), data, "cpu"),
                xml,
                title="grace",
            )
            self.assertTrue(ET.parse(xml).findall(".//grace"))
        np.testing.assert_array_equal(score.notes, before)

    def test_single_note_still_exports_readable(self):
        with tempfile.TemporaryDirectory() as d:
            midi = Path(d) / "single.mid"
            m = mido.MidiFile()
            t = mido.MidiTrack()
            m.tracks.append(t)
            t.append(mido.Message("note_on", note=60, velocity=80, time=0))
            t.append(mido.Message("note_off", note=60, time=480))
            m.save(midi)
            output = Path(d) / "result"
            report = DuET(Predictions(), {"weights_sha256": "test"}, "cpu").transcribe(
                midi, output
            )
            self.assertEqual(report["status"], "complete")
            self.assertTrue(
                ET.parse(output / "score.musicxml").findall(".//note/pitch")
            )
            self.assertTrue(
                any("120 BPM" in w for w in report["engraving"]["summary"]["warnings"])
            )

    def test_failed_engraving_does_not_publish_partial_xml(self):
        score = Score(
            np.array([[60.0, 0.0, 1.0, 0.0]]),
            np.array([0.0]),
            np.array([4.0]),
            ["4/4"],
            np.array([0]),
            np.array([0]),
        )
        data = dict(
            ref_onset=[0.0],
            ref_offset=[0.5],
            ref_pitch=[60],
            ref_vel=[80],
            pedal_time=[],
            pedal_value=[],
        )
        before = score.notes.copy()

        def fail(inputs, folder, **kwargs):
            inputs.score["notes"][0, 0] = 12
            (Path(folder) / "score.musicxml").write_text("<partial")
            raise ValueError("engraving failed")

        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "score.musicxml"
            with (
                patch("duet.readable.pipeline.run", side_effect=fail),
                self.assertRaisesRegex(ValueError, "engraving failed"),
            ):
                write_readable_musicxml(
                    score, data, pd.DataFrame(), target, title="test"
                )
            self.assertEqual(list(Path(d).iterdir()), [])
        np.testing.assert_array_equal(score.notes, before)


if __name__ == "__main__":
    unittest.main()
