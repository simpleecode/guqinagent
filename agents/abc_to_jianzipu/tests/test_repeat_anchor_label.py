import unittest
from agents.abc_to_jianzipu.repeat_anchor import normalize_repeat_anchor_text
from agents.abc_to_jianzipu.repeat_materializer import extract_repeat_markers, materialize_repeats
from agents.ToolRuntime.runtime import repeat_anchor_warnings, repeat_content_warnings, MISSING_REPEAT_ANCHOR_TEXT
from ABC_J.scripts.generate_teacher_tool_trajectories import public_system_for


class RepeatAnchorLabelTests(unittest.TestCase):
    def test_guqinizer_explains_anchor_instruction_and_coverage(self):
        prompt=public_system_for('guqinization')
        self.assertIn('只标记重复起点，不执行再作',prompt)
        self.assertIn('实际重复处须填写“从ㄱ再作”',prompt)
        self.assertIn('只有已被有效再作指令或前序复合技法覆盖',prompt)
        self.assertNotIn('前一段或再作标记后',prompt)
        self.assertNotIn('不会删除声音或演奏状态',prompt)

    def test_normalization_idempotent_and_preserves_normal_jianzi(self):
        for label in ('〔再作起点ㄱ〕', '〔再作起点〕', '[再作标记ㄱ]', '【再做起点ㄱ】'):
            text = label + '大指七徽挑三弦'
            expected = '[再作标记ㄱ]大指七徽挑三弦'
            self.assertEqual(normalize_repeat_anchor_text(text), expected)
            self.assertEqual(normalize_repeat_anchor_text(expected), expected)
        self.assertEqual(normalize_repeat_anchor_text('从ㄱ再作，泛音四徽勾二弦'), '从ㄱ再作，泛音四徽勾二弦')

    def test_new_anchor_materializes_without_becoming_instruction(self):
        self.assertEqual(extract_repeat_markers('[再作标记ㄱ]勾三弦'), ['再作标记ㄱ'])
        notes = [{'index': 0, 'jianpu': '1', 'jianzi': '[再作标记ㄱ]勾三弦'},
                 {'index': 1, 'jianpu': '1', 'jianzi': '从ㄱ再作'},
                 {'index': 2, 'jianpu': '1', 'jianzi': ''}]
        new, report = materialize_repeats(notes, is_sounding=lambda n: True)
        self.assertEqual(report['copy_markers'][0]['anchor_index'], 0)
        self.assertTrue(new[2]['notation_omitted'])

    def test_new_anchor_runtime_recognition_and_overlap(self):
        item = {'score_key': 's', 'input': {'event_range': {'start': 0},
                'notes_without_jianzi': [{'index': 0, 'jianpu': '1'}, {'index': 1, 'jianpu': '1'}]}}
        actions = {0: {'jianzi_text': '[再作标记ㄱ]散挑一弦'}, 1: {'jianzi_text': '从ㄱ再作'}}
        self.assertEqual(repeat_anchor_warnings(item, actions), [])
        actions[0]['jianzi_text'] = '[再作标记ㄱ]从ㄱ再作'
        self.assertIn('repeat_anchor_instruction_overlap', [w['code'] for w in repeat_anchor_warnings(item, actions)])

    def test_public_prompts_do_not_inject_extra_repeat_semantic_rule(self):
        for stage in ('fingering_agent', 'guqinization', 'single_stage'):
            self.assertNotIn('不要为消除缺少标记的警告', public_system_for(stage))
            self.assertNotIn('只有标记至', public_system_for(stage))
            self.assertNotIn('再作起点', public_system_for(stage))
        self.assertIn('[再作标记ㄱ]', MISSING_REPEAT_ANCHOR_TEXT)


if __name__ == '__main__':
    unittest.main()
