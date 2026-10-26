import unittest
from unittest.mock import patch
from agents.ToolRuntime.warning_checks import repeat_content_warnings
from agents.abc_to_jianzipu.repeat_materializer import OMITTED_PLACEHOLDER


class RepeatAndZhiTests(unittest.TestCase):
    def check(self, texts, pitches, jianpu=None):
        item = {'score_key': 's', 'input': {'event_range': {'start': 0},
                'normalized_tuning': {'open_midi': [36,38,41,43,45,48,50]},
                'notes_without_jianzi': [{'index': i, 'abc': 'C', 'jianpu': jianpu[i] if jianpu else '1'}
                                        for i in range(len(texts))]}}
        with patch('agents.ToolRuntime.warning_checks.target_midis',
                   side_effect=lambda row, i: [pitches[i]]):
            return {w['code'] for w in repeat_content_warnings(item,
                    {i: {'jianzi_text': t} for i,t in enumerate(texts)})}

    def test_attack_cannot_supply_endpoint(self):
        self.assertIn('zhi_missing_continuous_gesture', self.check(
            ['散滚三弦', '', '至一弦', '散挑七弦', '至一弦'], [41,38,36,50,36]))

    def test_downward_sweep(self):
        self.assertFalse(self.check(['散滚三弦', '', '至一弦'], [41,38,36]))

    def test_upward_sweep(self):
        self.assertFalse(self.check(['散拂一弦', '', '至三弦'], [36,38,41]))

    def test_six_string_sweep(self):
        self.assertFalse(self.check(['散滚七弦','','','','','至二弦'], [50,48,45,43,41,38]))

    def test_wrong_sweep(self):
        self.assertIn('zhi_sweep_pitch_mismatch', self.check(
            ['散滚三弦', '', '至一弦'], [41,43,36]))

    def test_endpoint_can_precede_blank_tail(self):
        self.assertFalse(self.check(
            ['散滚七弦','','至二弦','','',''], [50,48,45,43,41,38]))

    def test_endpoint_tail_does_not_cross_new_attack(self):
        self.assertIn('zhi_sweep_pitch_mismatch', self.check(
            ['散滚七弦','','至二弦','散勾四弦','',''], [50,48,45,43,41,38]))

    def test_matching_incomplete_tail_is_unknown(self):
        self.assertFalse(self.check(['散滚七弦','','至二弦'], [50,48,45]))

    def test_suoling_endpoint(self):
        self.assertFalse(self.check(['索铃','','至四弦'], [50,48,43]))

    def test_intermediate_string_marker(self):
        self.assertFalse(self.check(['大指九徽滚七弦','六弦','','至二弦'], [60,58,55,48]))

    def test_walk_modifier_inside_sweep(self):
        self.assertFalse(self.check(['散拂二弦','','浒上七徽','至六弦'], [38,41,43,48]))

    def test_walk_without_sweep_still_warns(self):
        self.assertIn('zhi_missing_continuous_gesture', self.check(
            ['散挑七弦','浒上七徽','至六弦'], [50,52,48]))

    def test_bare_repeat_mismatch(self):
        self.assertIn('repeat_plain_pitch_mismatch', self.check(
            ['散勾一弦','散勾二弦','再作',OMITTED_PLACEHOLDER], [36,38,36,41]))

    def test_bare_repeat_match(self):
        self.assertFalse(self.check(
            ['散勾一弦','散勾二弦','再作',OMITTED_PLACEHOLDER], [36,38,36,38]))

    def test_repeated_short_motif(self):
        self.assertFalse(self.check(['散勾一弦','散勾二弦','再作','','',''],
                                    [36,38,36,38,36,38]) & {'repeat_plain_pitch_mismatch'})

    def test_packed_sweep_is_not_scalar_pitch(self):
        self.assertFalse(self.check(['散拂一弦','至三弦'], [0,41],
                                    ['1 2（饰）','3']))

    def test_endpoint_control_on_rest(self):
        self.assertFalse(self.check(['散滚三弦','至一弦'], [41,0],
                                    ['3','0（休止）']))
