"""Musical invariants for grid rollback, eighth meters and public controls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fractions import Fraction as F

import numpy as np
import pandas as pd
from music21 import converter
from duet.score import Score
from duet.readable.config import Config
from duet.readable.inputs import Inputs
from duet.readable import pipeline
from duet.readable.notation import beam
from duet.engraving import write_readable_musicxml


def fixture(meter='3/8'):
    num,den=map(int,meter.split('/'));length=num*4/den
    q=np.arange(0,2*length,.5);pitch=60+np.arange(len(q))%5
    data=dict(ref_onset=q*.5,ref_offset=q*.5+.2,ref_pitch=pitch,
              ref_vel=np.full(len(q),80),pedal_time=np.array([]),pedal_value=np.array([]))
    notes=np.column_stack((pitch,q,np.full(len(q),.5),np.arange(len(q))%2))
    score=Score(notes,np.array([0.,length]),np.array([length,length]),[meter]*2,
                np.zeros(len(q),int),np.zeros(2,int))
    frames=pd.DataFrame(dict(performance_time_sec=np.arange(0,length+.5,.02),
                             p_any_beat=0.,p_downbeat=0.))
    return data,score,frames


class EngravingControlsTests(unittest.TestCase):
    def test_eighth_meters_keep_actual_measure_clocks_and_pitch(self):
        for meter in ('3/8','4/8','9/8','4/4','6/8','12/8'):
            with self.subTest(meter=meter),tempfile.TemporaryDirectory() as tmp:
                data,score,frames=fixture(meter);original=score.notes.copy()
                cfg=Config(auto_keys=False,auto_spelling=False,auto_runs=False,extend_by_hold=False)
                xml,_=pipeline.run(Inputs.from_prediction(data,score,frames),tmp,cfg=cfg)
                parsed=converter.parse(xml,forceSource=True).stripTies()
                length=score.bar_lengths[0]
                self.assertEqual(len(parsed.parts),2)
                for part in parsed.parts:
                    measures=list(part.getElementsByClass('Measure'))
                    np.testing.assert_allclose([float(m.offset) for m in measures],[0,length])
                    self.assertEqual(measures[0].timeSignature.ratioString,meter)
                actual=sorted(int(p.midi) for part in parsed.parts for n in part.flatten().notes for p in n.pitches)
                self.assertEqual(actual,sorted(data['ref_pitch'].tolist()))
                np.testing.assert_array_equal(score.notes,original)

    def test_quarter_beat_pickup_survives_half_beat_snap_proposal(self):
        data,score,frames=fixture('4/4')
        q=np.r_[0.,np.arange(.25,7.75,.5)]
        score.notes[:,1]=q;score.notes[0,2]=.25
        score.bar_starts=np.array([0.,.25,4.25]);score.bar_lengths=np.array([.25,4.,4.])
        score.meters=['4/4']*3;score.keys=np.zeros(3,int)
        data['ref_onset']=q*.5;data['ref_offset']=q*.5+.1
        original=score.notes.copy()
        with tempfile.TemporaryDirectory() as tmp:
            xml,summary=pipeline.run(Inputs.from_prediction(data,score,frames),tmp,
                                     cfg=Config(auto_runs=False,extend_by_hold=False))
            self.assertTrue(summary['grid_fallback']['applied'])
            parsed=converter.parse(xml,forceSource=True)
            for part in parsed.parts:
                np.testing.assert_allclose([float(m.offset) for m in part.getElementsByClass('Measure')],
                                           [0.,.25,4.25])
        np.testing.assert_array_equal(score.notes,original)

    def test_rejected_grid_restarts_fresh_and_does_not_catch_other_errors(self):
        data,score,frames=fixture('4/4');inp=Inputs.from_prediction(data,score,frames)
        original=inp.score['notes'].copy();seen=[]
        def attempt(fresh,out,title,cfg):
            np.testing.assert_array_equal(fresh.score['notes'],original)
            seen.append(cfg.grid_count_fix)
            fresh.score['notes'][0,1]=999
            if cfg.grid_count_fix:raise pipeline.BarMappingError('collapsed boundary')
            return 'result.xml',dict(warnings=[])
        with patch.object(pipeline,'_run',side_effect=attempt):
            _,summary=pipeline.run(inp,'unused')
        self.assertEqual(seen,[True,False]);self.assertTrue(summary['grid_fallback']['applied'])
        np.testing.assert_array_equal(inp.score['notes'],original)
        with patch.object(pipeline,'_run',side_effect=ValueError('unrelated')):
            with self.assertRaisesRegex(ValueError,'unrelated'):pipeline.run(inp,'unused')

    def test_rollback_disabled_correction_is_not_retried(self):
        data,score,frames=fixture()
        with patch.object(pipeline,'_run',side_effect=pipeline.BarMappingError('bad original bars')) as run:
            with self.assertRaises(pipeline.BarMappingError):
                pipeline.run(Inputs.from_prediction(data,score,frames),'unused',cfg=Config(grid_count_fix=False,snap_bar_lines=False))
            self.assertEqual(run.call_count,1)

    def test_independent_default_on_controls_and_recorded_parameters(self):
        self.assertTrue(Config().auto_runs)
        for auto_keys,auto_spelling,auto_runs in ((False,True,True),(True,False,True),(True,True,False)):
            with self.subTest(flags=(auto_keys,auto_spelling,auto_runs)),tempfile.TemporaryDirectory() as tmp:
                data,score,frames=fixture('4/4')
                cfg=dict(auto_keys=auto_keys,auto_spelling=auto_spelling,auto_runs=auto_runs,
                         key_switch=20.,key_model_cost=.025,spelling_window=3.,spelling_review_margin=.8)
                import duet.readable.review as review
                originals={k:getattr(review,k) for k in ('auto_keys','auto_spelling','auto_runs')}
                with patch.object(review,'auto_keys',wraps=originals['auto_keys']) as keys, \
                     patch.object(review,'auto_spelling',wraps=originals['auto_spelling']) as spelling, \
                     patch.object(review,'auto_runs',wraps=originals['auto_runs']) as runs:
                    report=write_readable_musicxml(score,data,frames,Path(tmp)/'out.musicxml',title='test',cfg=cfg)
                self.assertEqual((keys.call_count,spelling.call_count,runs.call_count),tuple(map(int,(auto_keys,auto_spelling,auto_runs))))
                for k,v in cfg.items():self.assertEqual(report['config'][k],v)

    def test_simple_eighth_beams(self):
        def pieces(n):return [dict(s=F(i,2),d=F(1,2),pl={},reg=None) for i in range(n)]
        ps=pieces(3);beam(ps,F(3,2),group_span=F(3,2),beat_unit=F(1,2))
        self.assertEqual([p['beam'][1] for p in ps],['begin','continue','end'])
        ps=pieces(4);beam(ps,F(2),group_span=F(1),beat_unit=F(1,2))
        self.assertEqual([p['beam'][1] for p in ps],['begin','end','begin','end'])

    def test_cli_flags_override_json_without_loading_a_model_for_invalid_values(self):
        import contextlib,io,sys
        from unittest.mock import Mock
        from duet import cli
        model=Mock();model.transcribe.return_value={'input_notes':0}
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'config.json'
            config.write_text(json.dumps(dict(auto_runs=False,key_switch=5.)))
            argv=['duet','input.mid','--engraving-config',str(config),'--auto-runs',
                  '--no-auto-spelling','--key-switch','20']
            with patch.object(sys,'argv',argv),patch.object(cli.DuET,'from_pretrained',return_value=model),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(),0)
            effective=model.transcribe.call_args.kwargs['engraving_config']
            self.assertTrue(effective.auto_runs);self.assertFalse(effective.auto_spelling)
            self.assertEqual(effective.key_switch,20.)
        with patch.object(sys,'argv',['duet','input.mid','--key-switch','nan']), \
             patch.object(cli.DuET,'from_pretrained') as load,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):cli.main()
            load.assert_not_called()

    def test_invalid_configuration_is_rejected(self):
        for values in ({'auto_runs':'false'},{'key_switch':float('nan')},{'spelling_window':0},
                       {'spelling_passes':1.5},{'key_model_cost':-1},{'typo':True}):
            with self.subTest(values=values),self.assertRaises(ValueError):Config.from_dict(values)

if __name__=='__main__':unittest.main()
