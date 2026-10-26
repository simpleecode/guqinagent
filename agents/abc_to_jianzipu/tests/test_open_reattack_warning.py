import unittest
from agents.abc_to_jianzipu.reference_parser import AUDIT


class OpenReattackTests(unittest.TestCase):
    def warnings(self, texts):
        return AUDIT.ordinary_open_reattack_warnings({'open_midi':[48,50,53,55,57,60,62],
            'notes':[{'index':i,'abc':'C','jianpu':'1','jianzi':t} for i,t in enumerate(texts)]})

    def test_plain_and_inherited(self):
        self.assertEqual(len(self.warnings(['散挑三弦','勾三弦','挑三弦','勾三弦'])),3)

    def test_complex_different_string_and_stopped_exempt(self):
        for pair in [('散抹挑三弦','散勾三弦'),('散挑三弦','轮三弦'),
                     ('散拂一至六弦','散挑三弦'),('散挑三弦','散挑四弦'),
                     ('大指七徽挑三弦','勾三弦'),('散挑三弦',''),
                     ('散挑三弦','散挑三弦再作')]:
            self.assertEqual(self.warnings(pair),[],pair)


if __name__ == '__main__':
    unittest.main()
