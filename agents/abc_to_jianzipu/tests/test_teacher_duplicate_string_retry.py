import json
import unittest
from unittest.mock import Mock, patch
from ABC_J.scripts import generate_teacher_tool_trajectories as teacher
from agents.abc_to_jianzipu.jianzi_surface_checks import duplicate_string_prefixes


def result(text, final_text=None):
    public = {'messages': [{'role': 'assistant', 'tool_calls': [{'function': {
        'name': 'edit_plan', 'arguments': json.dumps({'jianzi_rows': [[1, text]]})}}]}]}
    private = {'teacher_private': {'accepted_plan': {'actions': [
        {'jianzi_text': final_text if final_text is not None else text}]}}}
    return public, private


class DuplicateStringRetryTests(unittest.TestCase):
    def test_detection(self):
        for text in ('泛音二弦四徽勾二弦', '名指二弦徽外剔二弦', '大指2弦七徽挑二弦'):
            self.assertTrue(duplicate_string_prefixes(text), text)
        for text in ('泛音四徽勾二弦', '勾二弦', '大指七徽历三二弦',
                     '撮（大指七徽六弦泛音＋中指七徽一弦泛音）',
                     '散勾四弦 大指七徽勾四弦', '拂一弦至五弦名指十徽八分抹五弦'):
            self.assertFalse(duplicate_string_prefixes(text), text)

    def invoke(self, failures):
        return teacher.generate_with_retries(Mock(), 'glm-5.3', {'trajectory_id': 'test-p0001'},
            'fingering_agent', [], {}, 24, 3, objective='basic_fingering', failure_traces=failures)

    def test_retries_bad_stage_then_returns_clean_result(self):
        clean = result('泛音四徽勾二弦')
        failures = []
        with patch.object(teacher, 'generate_one', side_effect=[result('泛音二弦四徽勾二弦'), clean]) as call:
            self.assertEqual(self.invoke(failures), clean)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0]['duplicate_string_cases'])

    def test_retry_cap_does_not_accept_bad_result(self):
        failures = []
        with patch.object(teacher, 'generate_one', return_value=result('泛音二弦四徽勾二弦')) as call:
            with self.assertRaisesRegex(RuntimeError, 'teacher attempts exhausted'):
                self.invoke(failures)
        self.assertEqual(call.call_count, 3)
        self.assertEqual(len(failures), 3)

    def test_bad_intermediate_still_rejected_when_final_clean(self):
        with self.assertRaises(teacher.DuplicateStringPrefixError):
            teacher.verify_teacher_string_surface(*result('泛音二弦四徽勾二弦', '泛音四徽勾二弦'))

    def test_does_not_scan_prose_or_user_prompt(self):
        public, private = result('勾二弦')
        public['messages'].insert(0, {'role': 'user', 'content': '泛音二弦四徽勾二弦'})
        public['messages'][-1]['content'] = '不要写泛音二弦四徽勾二弦'
        teacher.verify_teacher_string_surface(public, private)


if __name__ == '__main__':
    unittest.main()
