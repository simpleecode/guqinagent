import unittest
from unittest.mock import patch

from agents.ToolRuntime.runtime import missing_attack_warnings


class MissingAttackTests(unittest.TestCase):
    def check_rows(self, texts, abcs=None, omitted=False):
        item = {'input': {'notes_without_jianzi': [
            {'index': i, 'abc': abc, 'notation_omitted': omitted}
            for i, abc in enumerate(abcs or ['C'] * len(texts))]}}
        actions = {i: {'jianzi_text': text} for i, text in enumerate(texts)}
        pitches = {'C': [60], 'C2': [60], 'D': [62], 'E': [64]}
        with patch('agents.ToolRuntime.warning_checks.target_midis',
                   side_effect=lambda _, i: pitches.get(item['input']['notes_without_jianzi'][i]['abc'], [])):
            return missing_attack_warnings(item, actions)

    def test_simple_attacks_and_blank_run(self):
        for attack in ('大指五徽挑五弦', '散勾三弦', '泛音名指七徽抹四弦', '搓三弦'):
            self.assertEqual([w['source_index'] for w in self.check_rows([attack, '', ''], ['C', 'D', 'E'])], [1])

    def test_same_pitch_and_duration_changes_are_exempt(self):
        self.assertEqual(self.check_rows(['挑五弦', '', ''], ['C', 'C2', 'C']), [])
        self.assertEqual([w['source_index'] for w in self.check_rows(
            ['挑五弦', '', ''], ['C', 'D', 'D'])], [1])
        self.assertEqual(self.check_rows(['挑五弦', ''], ['C', '^C']), [])

    def test_barline_and_blank_break_previous_attack_chain(self):
        self.assertEqual(self.check_rows(['挑五弦', '', ''], ['C', '|', 'D']), [])
        self.assertEqual(self.check_rows(['挑五弦', '', ''], ['C', 'C', 'D']), [])

    def test_complex_and_repeat_are_not_classified_as_single_attacks(self):
        for attack in ('大指五徽抹挑五弦', '掐撮三声', '泛音名指五徽拂一至六弦',
                       '撮（大指七徽六弦按音＋一弦散音）', '从头再作',
                       '[再作标记ㄱ]大指五徽挑五弦', '绰上七徽', '名指九徽滔起',
                       '大指五徽挑五弦再作', '大指五徽挑五弦+轮五弦'):
            self.assertEqual(self.check_rows([attack, '', '']), [], attack)

    def test_no_warning_for_rest_tie_pending_or_omission(self):
        for abc in ('z4', '-C4', '|', '[CE]', ''):
            self.assertEqual(self.check_rows(['挑五弦', ''], ['C', abc]), [])
        self.assertEqual(self.check_rows(['挑五弦', None, '']), [])
        self.assertEqual(self.check_rows(['挑五弦', ''], omitted=True), [])
        self.assertEqual(self.check_rows(['挑五弦', '无（由于是再作部分，省略）', '']), [])


if __name__ == '__main__':
    unittest.main()
