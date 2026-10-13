import unittest
from agents.abc_to_jianzipu.score_structure import is_structural_note,render_structure
from agents.abc_to_jianzipu.teacher_trajectory import _render_phrase_lines
from agents.abc_to_jianzipu.teacher_gqs import render_teacher_gqs,parse_teacher_gqs
from agents.abc_to_jianzipu.phrase_splitter import split_phrase_ranges
from scripts.infer_agent_trajectories import assign_event_indices
from scripts.audit_jianpu_jianzi_pitch import audit,replay_harmonic_scope


class ScoreStructureTests(unittest.TestCase):
    def test_distinct_public_labels_and_no_event_ordinals(self):
        notes=[{'index':i,'abc':token,'jianpu':'|','duration':'小节线','structure_token':token}
               for i,token in enumerate(['|','||','||:',':||','|[1',']|[2'])]
        notes.append({'index':6,'abc':'E','jianpu':'3','duration':'四分'})
        assign_event_indices(notes)
        self.assertEqual([n['event_index'] for n in notes],[None]*6+[0])
        text=_render_phrase_lines(notes,[])
        for label in ('双小节线（||）','反复开始（||:）',':||','|[1',']|[2'):
            self.assertIn(label,text)
        self.assertEqual(render_structure({'structure_token': ':||'}), ':||')
        self.assertNotIn('反复房子', text)
        self.assertEqual(render_structure({'structure_token': ':||[2'}), ':||[2')
        self.assertIn('0｜3｜',text)

    def test_gqs_roundtrip_preserves_repeat_structure(self):
        notes=[{'index':0,'abc':'E','jianpu':'3','duration':'四分','jianzi':''},
               {'index':1,'abc':':|[2','jianpu':'|','duration':'小节线','structure_token':']:||[2','jianzi':''},
               {'index':2,'abc':'E','jianpu':'3','duration':'四分','jianzi':''}]
        text=render_teacher_gqs({'metadata':{'score_title':'test'},'notes':notes})
        self.assertIn('结构｜',text)
        restored=parse_teacher_gqs(text)['notes']
        self.assertEqual(restored[1]['structure_token'],']:||[2')
        self.assertEqual(restored[1]['abc'],':|[2')
        self.assertTrue(is_structural_note(restored[1]))

    def test_frozen_splitter_cuts_at_repeat_bar(self):
        notes=[{'abc':'E','sounding':True} for _ in range(3)]+[{'abc':':|','sounding':False}]+[{'abc':'E','sounding':True}]
        self.assertEqual(split_phrase_ranges(notes,max_sounding=2,is_sounding=lambda n:n['sounding']),[(0,4),(4,5)])

    def test_repeat_bar_does_not_end_harmonics_but_real_rest_does(self):
        metadata={'tonic':'1=C'}
        base=[{'index':0,'abc':'e','jianpu':'3̇','jianzi':'泛起食指九徽挑五弦'},
              {'index':1,'abc':':|','jianpu':'|','duration':'小节线','structure_token':':||','jianzi':''},
              {'index':2,'abc':'e','jianpu':'3̇','jianzi':'食指九徽挑五弦'}]
        data={'metadata':metadata,'open_midi':[48,50,53,55,57,60,62],'notes':base}
        self.assertEqual(audit(data,50)['details'][-1]['status'],'matched')
        self.assertTrue(replay_harmonic_scope(True,base[1],''))
        rest={**base[1],'abc':'z4','jianpu':'0（休止）','duration':'四分'}
        rest.pop('structure_token')
        self.assertFalse(replay_harmonic_scope(True,rest,''))
        self.assertEqual(audit({**data,'notes':[base[0],rest,base[2]]},50)['details'][-1]['status'],'mismatched')


if __name__=='__main__':unittest.main()
