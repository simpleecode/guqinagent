import unittest
from copy import deepcopy
from unittest.mock import patch

from ABC_J.scripts.visualize_running_eval_score import (
    stage_prompts, prompt_block, recorded_stage_prompts, stage_trace,
)


class EvalPromptTests(unittest.TestCase):
    def fixture(self):
        item = {'score_key': 'test', 'phrase_id': 'p0002', 'trajectory_id': 'test-p0002',
                'harmonic_region_at_start': False,
                'input': {'metadata': {'tonic': '1=C'},
                          'normalized_tuning': {'open_midi': [48, 50, 53, 55, 57, 60, 62]},
                          'event_range': {'start': 2, 'end_exclusive': 3},
                          'notes_without_jianzi': [{'index': 2, 'event_index': 1,
                                                   'abc': 'G', 'jianpu': '5', 'duration': '四分'}]},
                'baseline_plan': {'actions': [{'source_index': 2, 'jianzi_text': ''}]}}
        prior = {'score_key': 'test', 'phrase_id': 'p0001',
                 'input': {'event_range': {'start': 0, 'end_exclusive': 2},
                           'notes_without_jianzi': [{'index': 0, 'abc': 'G', 'jianpu': '5'},
                                                   {'index': 1, 'abc': ':|', 'jianpu': '|'}]},
                 'reference_plan': {'actions': [{'source_index': 0,
                                                'jianzi_text': '泛起食指七徽挑四弦'}]}}
        return item, {('test', 'p0001'): prior}

    def test_legacy_prompt_replays_harmonics_from_model_history(self):
        item, history = self.fixture()
        result = stage_prompts({}, item, None, historical=history)
        self.assertIn('泛音区间｜此段开始时仍然处于泛音区间', result[1])
        self.assertFalse(item['harmonic_region_at_start'])

    def test_actual_record_overrides_new_template_and_rebuilt_state(self):
        item, _ = self.fixture()
        stage = {'trace': [{'request_messages': [
            {'role': 'system', 'content': 'original system'},
            {'role': 'user', 'content': 'original user'},
        ]}]}
        with patch('ABC_J.scripts.visualize_running_eval_score.render_public_prompt',
                   side_effect=AssertionError('must not rebuild stored prompts')):
            result = stage_prompts({}, item, stage, guqinizer=stage)
        self.assertEqual(result, ('original system', 'original user',
                                  'original system', 'original user'))

    def test_prompt_origin_is_explicit(self):
        self.assertIn('重建提示词，非原始请求', prompt_block('base', 's', 'u'))
        self.assertIn('实际请求记录', prompt_block('base', 's', 'u', recorded=True))
        self.assertIsNone(recorded_stage_prompts({'trace': [{'raw_output': 'reply'}]}))

    def test_round_snapshot_remains_visible_including_tool_messages(self):
        messages = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'},
                    {'role': 'tool', 'content': 'tool reply'}]
        snapshot = deepcopy(messages)
        messages[-1]['content'] = 'changed later'
        html = stage_trace({'ok': True, 'trace': [{'round': 2,
                            'request_messages': snapshot, 'raw_output': 'done'}]}, 'Base')
        self.assertIn('本轮实际请求 messages', html)
        self.assertIn('tool reply', html)
        self.assertNotIn('changed later', html)


if __name__ == '__main__':
    unittest.main()
