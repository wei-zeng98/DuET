"""Fast runs on one scale become one wavy line from their performed positions."""
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pandas as pd

from duet.readable import gliss, pipeline
from duet.readable.config import Config
from duet.readable.inputs import Inputs
from duet.score import Score

BEAT_SECONDS = 0.5
RUN = [93, 91, 89, 88, 86, 84, 83, 81, 79, 77, 76, 74]  # A6 to D5, white keys


def fixture(run=RUN, crossing_staff=1):
    """Two 4/4 bars with a chord on the downbeat, a run played from beat 1.25 to 2.75, a crossing note, a late chord.

    The model positions of the run are stretched on 32nds, as the model does, so they overrun where it was played.
    """
    played = [0.0, 0.0] + [1.25 + i * 1.5 / (len(run) - 1) for i in range(len(run))] + [2.0, 4.0, 4.0]
    pitch = [60, 64] + run + [52, 60, 64]
    written = [0.0, 0.0] + [1.25 + i * 0.25 for i in range(len(run))] + [2.0, 4.5, 4.5]
    q = np.array(written)
    seconds = np.array(played) * BEAT_SECONDS
    data = dict(ref_onset=seconds, ref_offset=seconds + 0.2, ref_pitch=np.array(pitch), ref_vel=np.full(len(q), 80),
                pedal_time=np.array([]), pedal_value=np.array([]))
    staff = np.array([0, 0] + [0] * len(run) + [crossing_staff, 0, 0])
    notes = np.column_stack((pitch, q, np.full(len(q), 0.25), staff))
    score = Score(notes, np.array([0.0, 4.0]), np.array([4.0, 4.0]), ['4/4'] * 2, np.zeros(len(q), int),
                  np.array([4, 4]))
    frames = pd.DataFrame(dict(performance_time_sec=np.arange(0, 6, 0.02), p_any_beat=0.0, p_downbeat=0.0))
    return data, score, frames


def engrave(auto=True, **options):
    data, score, frames = fixture(**options)
    cfg = Config(auto_keys=False, auto_spelling=False, auto_runs=False, auto_glissandi=auto)
    with tempfile.TemporaryDirectory() as folder:
        path, _ = pipeline.run(Inputs.from_prediction(data, score, frames), folder, cfg=cfg)
        review = json.loads((Path(folder) / 'score.review.json').read_text(encoding='utf-8'))
        return ET.parse(path).getroot(), review


def attacks(root):
    """Written notes, without the later pieces of tied ones."""
    return [n for n in root.iter('note') if n.find('pitch') is not None and n.find('tie[@type="stop"]') is None]


def perf_of(notes):
    """(onset, offset, pitch, velocity) rows from (seconds, pitch) pairs."""
    return [(seconds, seconds + 0.5, pitch, 70) for seconds, pitch in notes]


def run_at(start, pitches, step=0.05):
    return [(start + index * step, pitch) for index, pitch in enumerate(pitches)]


def found(notes, beat=0.5):
    return gliss.find(perf_of(notes), beat)


class GlissNotationTests(unittest.TestCase):
    def test_on_by_default_and_can_be_turned_off(self):
        self.assertTrue(Config().auto_glissandi)
        root, _ = engrave(auto=False)
        self.assertEqual(list(root.iter('glissando')), [])

    def test_run_becomes_one_line_between_its_outer_notes(self):
        root, review = engrave()
        kinds = [(g.get('type'), g.get('line-type')) for g in root.iter('glissando')]
        self.assertEqual(kinds, [('start', 'wavy'), ('stop', 'wavy')])
        self.assertEqual(len(attacks(engrave(auto=False)[0])) - len(RUN) + 2, len(attacks(root)))
        self.assertEqual(len([item for item in review['items'] if item['kind'] == 'glissando']), 1)

    def test_white_key_is_written_natural_in_a_sharp_key(self):
        root, _ = engrave(run=[89, 88, 86, 84, 83, 81, 79, 77, 76, 74, 72, 71])  # starts on F6 in E major
        start = next(n for n in attacks(root) if n.findtext('pitch/step') == 'F' and n.findtext('pitch/octave') == '6')
        self.assertIsNone(start.find('pitch/alter'))
        self.assertEqual(start.findtext('accidental'), 'natural')

    def test_crossing_note_of_the_same_staff_is_a_second_voice(self):
        root, _ = engrave(crossing_staff=0)
        voices = {n.findtext('voice') for n in root.iter('note') if n.find('.//glissando') is not None}
        crossing = next(n for n in root.iter('note') if n.find('pitch') is not None
                        and n.findtext('pitch/step') == 'E' and n.findtext('pitch/octave') == '3')
        self.assertEqual(len(voices), 1)
        self.assertNotIn(crossing.findtext('voice'), voices)

    def test_ends_snap_half_up(self):
        self.assertEqual(gliss.snap(1.125), F(5, 4))
        self.assertEqual(gliss.snap(1.124), F(1))
        self.assertEqual(gliss.snap(2.74), F(11, 4))


class GlissDetectionTests(unittest.TestCase):
    def test_white_black_and_major_scale_runs(self):
        black = [90, 87, 85, 82, 80, 78, 75, 73, 70, 68, 66]
        e_major = [88, 87, 85, 83, 81, 80, 78, 76, 75, 73, 71]
        for pitches, scale in ((RUN, 'white'), (black, 'black'), (e_major, 'E major')):
            with self.subTest(scale=scale):
                runs, _ = found(run_at(0, pitches))
                self.assertEqual([run['scale'] for run in runs], [scale])
                self.assertEqual([p for _, p in runs[0]['notes']], pitches)

    def test_played_scale_short_figure_and_low_chords_are_not_runs(self):
        for notes in (run_at(0, RUN, 0.12), run_at(0, RUN[:5]), run_at(0, [55, 59, 60, 62, 64, 65, 67, 69])):
            self.assertEqual(found(notes)[0], [])

    def test_transcribed_end_out_of_order_is_one_run(self):
        # Onsets of a real run in beats at 162 BPM: the end comes out late and out of order.
        head = [1.19, 1.22, 1.28, 1.32, 1.39, 1.46, 1.53, 1.64, 1.72, 1.81, 1.86, 1.96, 2.08, 2.14]
        pitches = [93, 91, 89, 88, 86, 84, 83, 81, 79, 77, 76, 74, 72, 71]
        tail = [(2.40, 64), (2.47, 67), (2.53, 60), (2.64, 65), (2.71, 62), (2.73, 59)]
        beat = 60 / 162
        notes = [(at * beat, pitch) for at, pitch in list(zip(head, pitches)) + tail]
        self.assertEqual(len(found(notes, beat)[0][0]['notes']), 20)

    def test_chords_it_crosses_stay_out(self):
        chord = [(0.25, 60), (0.25, 63), (0.25, 68)]
        runs, _ = found(run_at(0, RUN) + chord)
        self.assertEqual(sorted(p for _, p in runs[0]['notes']), sorted(RUN))

    def test_chord_note_at_the_end_goes_to_the_chord(self):
        pitches = RUN + [72]
        end = (len(pitches) - 1) * 0.05
        runs, _ = found(run_at(0, pitches) + [(end, 45), (end, 52)])
        self.assertNotIn(72, [p for _, p in runs[0]['notes']])

    def test_note_struck_before_the_run_reaches_it_stays_out(self):
        holes = [(index * 0.05, p) for index, p in enumerate(RUN) if p != 76]
        runs, _ = found(holes + [(5 * 0.05, 76)])
        self.assertNotIn(76, [p for _, p in runs[0]['notes']])

    def test_mostly_repeated_chords_are_rejected(self):
        pitches = RUN[:10]
        notes = run_at(0, pitches)
        for index in range(1, 7):
            at = index * 0.05
            notes += [(at, 52), (at + 0.6, pitches[index]), (at + 0.6, 52)]
        runs, rejected = found(notes)
        self.assertEqual((runs, rejected), ([], 1))


if __name__ == '__main__':
    unittest.main()
