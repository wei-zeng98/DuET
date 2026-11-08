"""Regressions for unsafe gliss endpoints and crossed-staff voice allocation."""
import copy
import types
import unittest
import tempfile
import xml.etree.ElementTree as ET
from unittest.mock import patch
from fractions import Fraction as F

from duet.readable import gliss, voices
from duet.readable.config import Config


def context():
    return types.SimpleNamespace(END=F(4), cfg=Config(extend_by_hold=False), bars=None,
                                 compound=False, half_index=lambda t: int(t // 2),
                                 bar_of=lambda t: int(t // 4), where=str, log=lambda s: None)


def note(on, dur, pitch, pd=.5, crossed=None, t=None):
    n = dict(on=F(on), dur=F(dur), pitch=pitch, pd=pd, staff=0,
             grace=False, t=float(F(on)) if t is None else t)
    if crossed:
        n.update(crossed='high', original_staff=1)
    return n


class EngravingSafetyTests(unittest.TestCase):
    def test_probability_controls_reject_values_above_one(self):
        for field in ('half_bar_share', 'half_bar_max_on_bar', 'gliss_coverage'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                Config(**{field: 1.01})
            for value in (0., 1.):
                self.assertEqual(getattr(Config(**{field: value}), field), value)

    def candidate(self, following=F(2), next_staff=0):
        pitches = [60, 62, 64, 65, 67, 69, 71, 72]
        notes = [dict(t=i * .01, pitch=p, on=F(7, 4), x=1.75 + i / 140,
                      dur=F(1, 4), staff=0, grace=False) for i, p in enumerate(pitches)]
        run = dict(notes=[[n['t'], n['pitch']] for n in notes], scale='white')
        if following is not None:
            notes.append(dict(t=.2, pitch=84, on=following, x=float(following),
                              dur=F(1), staff=next_staff, grace=False))
        return notes, run

    def test_gliss_without_room_keeps_every_original_note(self):
        notes, run = self.candidate()
        before = copy.deepcopy(notes)
        ctx = context()
        gliss.apply(notes, [run], ctx)
        self.assertEqual(notes, before)
        self.assertEqual(ctx.gliss_detected, [])
        self.assertEqual(len(ctx.gliss_rejected), 1)
        self.assertEqual(ctx.gliss_spans, [])

    def test_minimum_span_precedes_room_check(self):
        notes, run = self.candidate(F(9, 4))
        ctx = context()
        gliss.apply(notes, [run], ctx)
        stop = next(n for n in notes if n.get('gliss') == 'stop')
        self.assertEqual(stop['on'], F(2))
        self.assertLess(stop['on'], notes[-1]['on'])
        self.assertLessEqual(stop['on'] + stop['dur'], notes[-1]['on'])

    def test_other_staff_does_not_block_gliss_endpoint(self):
        notes, run = self.candidate(next_staff=1)
        gliss.apply(notes, [run], context())
        self.assertEqual(len(notes), 3)
        self.assertEqual(sum(n.get('gliss') == 'stop' for n in notes), 1)

    def test_voice_cleanup_cannot_extend_stop_past_later_other_voice(self):
        stop = note(1, 2, 72, pd=2, t=.2)
        stop['gliss'] = 'stop'
        later = note(2, 1, 84, t=.3)
        later['layer'] = 'B'
        stop['layer'] = 'A'
        gliss.cap_endpoints([stop, later])
        self.assertEqual(stop['dur'], F(1))

    def test_disjoint_second_voice_leaves_crossed_figure_independent(self):
        ns = [note(0, 1, 60, pd=2), note('1/2', '1/4', 64),
              note(2, '1/2', 84), note(2, '1/2', 72, crossed=True)]
        result = voices.assign(ns, 0, context())
        self.assertEqual(result[0]['layer'], 'B')
        self.assertEqual(next(n for n in result if n['pitch'] == 72)['layer'], 'B')
        self.assertEqual(next(n for n in result if n['pitch'] == 84)['layer'], 'A')

    def test_real_second_voice_collision_restores_entire_figure_to_lh(self):
        ns = [note(0, 3, 60, pd=3), note('1/2', '1/4', 64),
              note(2, '1/2', 84), note(2, '1/2', 72, crossed=True),
              note('5/2', '1/2', 74, crossed=True)]
        result = voices.assign(ns, 0, context())
        self.assertTrue(all(n['pitch'] not in (72, 74) for n in result))
        restored = [n for n in ns if n['pitch'] in (72, 74)]
        self.assertTrue(all(n['staff'] == 1 and 'crossed' not in n for n in restored))
        left = voices.assign(ns, 1, context())
        self.assertEqual({n['pitch'] for n in left}, {72, 74})

    def test_unplaced_gliss_is_reported_without_claiming_compression(self):
        notes, run = self.candidate()
        ctx = context()
        gliss.apply(notes, [run], ctx)
        items = []
        rv = types.SimpleNamespace(bars=lambda bars: str(bars), add=lambda *args: items.append(args))
        gliss.review_items(rv, ctx)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0][0], 'glissando-unplaced')
        self.assertEqual(items[0][5], 'kept all notes')

    def test_pipeline_restarts_from_original_inputs_after_unsafe_endpoint(self):
        import numpy as np
        from test_gliss_notation import fixture
        from duet.readable import pipeline
        from duet.readable.inputs import Inputs
        data, score, frames = fixture()
        inp = Inputs.from_prediction(data, score, frames)
        raw = inp.score['notes'].copy()
        original = pipeline.write
        attempts = []

        def write(plan, ctx, path):
            attempts.append(ctx.cfg.auto_gliss)
            if ctx.cfg.auto_gliss:
                raise gliss.UnsafeGlissError('synthetic downstream endpoint conflict')
            return original(plan, ctx, path)

        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, 'write', side_effect=write):
            path, summary = pipeline.run(inp, folder)
            self.assertEqual(list(ET.parse(path).iter('glissando')), [])
        self.assertEqual(attempts, [True, False])
        self.assertTrue(summary['gliss_fallback']['applied'])
        self.assertFalse(summary['gliss_fallback']['effective_auto_gliss'])
        self.assertTrue(np.array_equal(inp.score['notes'], raw))

    def test_second_voice_from_previous_bar_is_also_reserved(self):
        ns = [note(3, 3, 60, pd=3), note('7/2', '1/4', 64),
              note('9/2', '1/2', 84), note('9/2', '1/2', 72, crossed=True)]
        ctx = context(); ctx.END = F(8)
        voices.assign(ns, 0, ctx)
        self.assertEqual(next(n for n in ns if n['pitch'] == 72)['staff'], 1)


if __name__ == '__main__':
    unittest.main()
