import unittest
from agents.ToolRuntime.warning_checks import finite_gesture_coverage_warnings
from agents.ToolRuntime.hint_checks import repeated_pitch_hints
from agents.abc_to_jianzipu.inverse_baseline import AUDIT
from agents.ToolRuntime.runtime import validate_jianzi_only


class FiniteCoverageTests(unittest.TestCase):
    def notes(self,texts,pitches=None):
        return [{'index':i,'abc':'C','jianpu':str((pitches or [1]*len(texts))[i]),'jianzi':t}
                for i,t in enumerate(texts)]

    def test_li_range(self):
        self.assertEqual(AUDIT.parse_li_string_sequence('散历七弦至二弦'),[7,6,5,4,3,2])
        w=finite_gesture_coverage_warnings(self.notes(['散历七弦至二弦']+['']*10))
        self.assertEqual([x['source_index'] for x in w],[6])

    def test_separate_qualified_endpoint(self):
        w=finite_gesture_coverage_warnings(self.notes(['散滚七弦']+['']*9+['至二弦（散音）']))
        self.assertEqual(w[0]['source_index'],6)

    def test_two_stroke_second_pitch_is_audited(self):
        r=AUDIT.audit({'metadata':{'tonic':'1=C'},'open_midi':[48,50,53,55,57,60,62],
                       'notes':self.notes(['散勾剔五弦',''],['6̣','5̣'])},50)
        d=r['details'][0]
        self.assertEqual(d['status'],'mismatched')
        self.assertEqual(d['covered_indices'],[0,1])
        self.assertEqual(d['pairs'][1]['absolute_cents'],200)

    def test_two_strokes_only_cover_two(self):
        self.assertEqual(finite_gesture_coverage_warnings(self.notes(['勾剔五弦',''])),[])
        w=finite_gesture_coverage_warnings(self.notes(['勾剔五弦','','']))
        self.assertEqual(w[0]['source_index'],2)

    def test_five_pitch_hint_replaces_reattack_warnings(self):
        for size in (4,5):
            texts=['散勾五弦']+['挑五弦']*(size-1)
            item={'score_key':'test','phrase_id':'p0001','input':{'metadata':{'tonic':'1=C'},'normalized_tuning':{'open_midi':[48,50,53,55,57,60,62]},
                          'notes_without_jianzi':self.notes(texts,[6]*size)},
                  'baseline_plan':{'actions':[{'source_index':i,'jianzi_text':t} for i,t in enumerate(texts)]}}
            hints=repeated_pitch_hints(item,item['baseline_plan']['actions'])
            self.assertEqual(bool(hints),size==5)
            _,report=validate_jianzi_only(item,[],require_complete=False)
            ordinary=[w for w in report['warnings'] if w['code']=='ordinary_open_same_string_reattack']
            self.assertEqual(len(ordinary),3 if size==4 else 0)
