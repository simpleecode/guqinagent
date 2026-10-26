import unittest
from agents.ToolRuntime.warning_checks import repeat_anchor_warnings

CODE='repeat_anchor_followed_by_blanks'


class AnchorBlankTests(unittest.TestCase):
    def check_rows(self,texts,notes=None):
        item={'score_key':'s','input':{'event_range':{'start':0},
              'notes_without_jianzi':notes or [{'index':i,'abc':'C','jianpu':'1'} for i in range(len(texts))]}}
        actions={i:{'jianzi_text':t} for i,t in enumerate(texts)}
        return [w for w in repeat_anchor_warnings(item,actions) if w['code']==CODE]

    def test_standalone_anchor_and_two_blanks(self):
        w=self.check_rows(['再作标记ㄱ','','',''])
        self.assertEqual(len(w),1)
        self.assertEqual(w[0]['source_index'],0)
        self.assertEqual(w[0]['blank_source_indices'],[1,2,3])

    def test_rest_anchor_and_intervening_bar(self):
        notes=[{'index':0,'abc':'z2','jianpu':'0（休止）'},
               {'index':1,'abc':'|','jianpu':'|'},
               {'index':2,'abc':'C','jianpu':'1'},{'index':3,'abc':'D','jianpu':'2'}]
        self.assertEqual(len(self.check_rows(['[再作标记ㄱ]','','',''],notes)),1)

    def test_explicit_instruction_is_not_flagged(self):
        self.assertEqual(self.check_rows(['再作标记ㄱ','从ㄱ再作','','']),[])

    def test_filled_anchor_and_compound_coverage_not_flagged(self):
        self.assertEqual(self.check_rows(['[再作标记ㄱ]名指七徽轮三弦','','']),[])
        self.assertEqual(self.check_rows(['再作标记ㄱ','名指七徽轮三弦','','']),[])

    def test_single_blank_or_rest_not_flagged(self):
        self.assertEqual(self.check_rows(['再作标记ㄱ','']),[])
        notes=[{'index':i,'abc':'z','jianpu':'0（休止）'} for i in range(3)]
        self.assertEqual(self.check_rows(['再作标记ㄱ','',''],notes),[])

    def test_explicit_omission_placeholders(self):
        self.assertEqual(len(self.check_rows(['再作标记ㄱ','无（由于是再作部分，省略）','无（由于是再作部分，省略）'])),1)

    def test_cross_phrase(self):
        prior={'score_key':'s','input':{'event_range':{'start':0},
              'notes_without_jianzi':[{'index':0,'abc':'C','jianpu':'1'}]},
              'reference_plan':{'actions':[{'source_index':0,'jianzi_text':'再作标记ㄱ'}]}}
        item={'score_key':'s','input':{'event_range':{'start':1},
              'notes_without_jianzi':[{'index':1,'abc':'C','jianpu':'1'}, {'index':2,'abc':'D','jianpu':'2'}]}}
        w=[w for w in repeat_anchor_warnings(item,{},historical={('s','p0001'):prior}) if w['code']==CODE]
        self.assertEqual(len(w),1)
        self.assertEqual(w[0]['source_index'],2)


if __name__=='__main__':unittest.main()
