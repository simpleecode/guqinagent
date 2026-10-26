import unittest
from agents.ToolRuntime.runtime import RealToolRuntime


class FeedbackAblationTests(unittest.TestCase):
    def make(self,on):
        item={'score_key':'s','phrase_id':'p0001','trajectory_id':'s-p0001',
              'input':{'metadata':{'tonic':'1=C'},'normalized_tuning':{'open_midi':[48,50,53,55,57,60,62]},
                       'event_range':{'start':0,'end_exclusive':5},
                       'notes_without_jianzi':[{'index':i,'event_index':i,'abc':'A,','jianpu':'6̣'} for i in range(5)]},
              'baseline_plan':{'actions':[{'source_index':i,'jianzi_text':''} for i in range(5)]}}
        return RealToolRuntime(item,{},warning_feedback=on)

    def test_hint_hidden_without_changing_plan(self):
        args={'jianzi_rows':[[i,'散勾五弦'] for i in range(5)]}
        on,off=self.make(True),self.make(False)
        a=on.invoke('edit_plan',args);b=off.invoke('edit_plan',args)
        self.assertTrue(a['result']['valid']);self.assertTrue(b['result']['valid'])
        self.assertIn(':hint:',a['result']['text'])
        self.assertNotIn(':hint:',b['result']['text'])
        self.assertNotIn(':warning:',b['result']['text'])
        self.assertEqual(on.accumulated_patches,off.accumulated_patches)

    def test_pitch_warning_hidden(self):
        args={'jianzi_rows':[[0,'散勾一弦']]}
        a=self.make(True).invoke('edit_plan',args)
        b=self.make(False).invoke('edit_plan',args)
        self.assertIn(':warning:',a['result']['text'])
        self.assertNotIn(':warning:',b['result']['text'])
        self.assertTrue(b['result']['valid'])

    def test_tool_errors_are_not_hidden(self):
        result=self.make(False).invoke('not_a_tool',{})
        self.assertFalse(result['ok'])
        self.assertIn('unknown tool',result['error'])
