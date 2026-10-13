import unittest

from agents.ToolRuntime.runtime import (
    repeat_anchor_warnings, render_edit_preview,
)
from agents.abc_to_jianzipu.repeat_materializer import OMITTED_PLACEHOLDER


class RepeatOmissionWarningTests(unittest.TestCase):
    def item(self, texts, start=0):
        notes = [{'index': start+i, 'event_index': start+i,
                  'jianpu': '1', 'abc': 'C'} for i in range(len(texts))]
        actions = {start+i: {'source_index': start+i, 'jianzi_text': text}
                   for i, text in enumerate(texts)}
        return {'score_key': 'test', 'input': {
            'event_range': {'start': start}, 'notes_without_jianzi': notes,
        }}, actions

    def codes(self, texts):
        item, actions = self.item(texts)
        return repeat_anchor_warnings(item, actions)

    def test_only_first_omission_in_each_run_warns(self):
        warnings = self.codes(['散挑一弦', OMITTED_PLACEHOLDER,
                               OMITTED_PLACEHOLDER, '散勾二弦', OMITTED_PLACEHOLDER])
        self.assertEqual([w['source_index'] for w in warnings], [1, 4])
        self.assertTrue(all(w['code'] == 'missing_repeat_instruction' for w in warnings))

    def test_anchor_and_instruction_cannot_share_a_row(self):
        for text in ['〔再作起点〕再作', '〔再做起点〕再做',
                     '〔再作起点〕从ㄱ再作', '再作大指七徽挑七弦〔再作起点〕']:
            with self.subTest(text=text):
                warnings = self.codes([text])
                self.assertIn('repeat_anchor_instruction_overlap', [w['code'] for w in warnings])

    def test_anchor_alone_and_separate_repeat_are_legal(self):
        self.assertEqual(self.codes(['〔再作起点〕大指七徽挑七弦', '再作']), [])
        self.assertEqual(self.codes(['〔再做起点〕']), [])
        self.assertEqual(self.codes(['〔再作起点ㄱ〕大指九徽挑七弦', '从ㄱ再作']), [])
        self.assertEqual(self.codes(['〔再作起点ㄱ〕', OMITTED_PLACEHOLDER])[0]['code'],
                         'missing_repeat_instruction')

    def test_multiple_repeat_warnings_on_same_row_are_rendered(self):
        item, actions = self.item(['〔再作起点〕从ㄱ再作'])
        warnings = repeat_anchor_warnings(item, actions)
        preview = render_edit_preview(list(actions.values()), [], True, [], item, warnings)
        self.assertIn(':warning:再作起点标记与再作指令不能重合', preview)
        self.assertIn(':warning:缺少再作起点标记', preview)

    def test_instruction_allows_run(self):
        for instruction in ['再做', '再作', '从ㄱ再作', '从头再作', '从头再做', '大指七徽挑七弦再作']:
            with self.subTest(instruction=instruction):
                self.assertEqual(self.codes(['〔再作起点〕散挑一弦', instruction, OMITTED_PLACEHOLDER,
                                             OMITTED_PLACEHOLDER]), [])

    def test_repeat_from_beginning_requires_no_anchor(self):
        for instruction in ['从头再作', '从头再做']:
            self.assertEqual(self.codes([instruction, OMITTED_PLACEHOLDER,
                                         OMITTED_PLACEHOLDER]), [])

    def test_anchor_alone_or_earlier_instruction_is_insufficient(self):
        for prefix in [['〔再作起点〕散挑一弦'], ['再作', '散挑一弦']]:
            self.assertEqual(self.codes(prefix + [OMITTED_PLACEHOLDER])[-1]['code'],
                             'missing_repeat_instruction')

    def test_blank_and_structure_do_not_replace_previous_effective_text(self):
        item, actions = self.item(['再作', '', '', OMITTED_PLACEHOLDER])
        item['input']['notes_without_jianzi'][2].update(abc=':||', jianpu='|')
        self.assertEqual(repeat_anchor_warnings(item, actions), [])

    def test_history_continues_omission_run(self):
        prior, actions = self.item(['散挑一弦', OMITTED_PLACEHOLDER])
        prior['reference_plan'] = {'actions': list(actions.values())}
        current, current_actions = self.item([OMITTED_PLACEHOLDER], start=2)
        self.assertEqual(repeat_anchor_warnings(
            current, current_actions, historical={('test', 'p0001'): prior}), [])
        prior['reference_plan']['actions'][-1]['jianzi_text'] = '散挑二弦'
        self.assertEqual(repeat_anchor_warnings(
            current, current_actions, historical={('test', 'p0001'): prior})[0]['code'],
            'missing_repeat_instruction')

    def test_warning_is_visible_on_unmodified_preview_row(self):
        item, actions = self.item(['散挑一弦', OMITTED_PLACEHOLDER])
        warnings = repeat_anchor_warnings(item, actions)
        preview = render_edit_preview(list(actions.values()), [], True, [], item, warnings)
        self.assertIn('1｜1｜未修改', preview)
        self.assertIn(':warning:再作省略前缺少再作指令', preview)


if __name__ == '__main__':
    unittest.main()
