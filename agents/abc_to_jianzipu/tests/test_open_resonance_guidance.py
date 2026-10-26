import unittest

from ABC_J.scripts.generate_teacher_tool_trajectories import (
    OPEN_RESONANCE_REPAIR_GUIDE,
    PUBLIC_SYSTEM,
)
from agents.abc_to_jianzipu.reference_parser import AUDIT


class OpenResonanceGuidanceTests(unittest.TestCase):
    def test_all_stages_allow_neighbor_and_harmonic_repairs(self):
        for stage, prompt in PUBLIC_SYSTEM.items():
            self.assertIn(OPEN_RESONANCE_REPAIR_GUIDE, prompt, stage)
            self.assertNotIn('后一个按音改到不同弦', prompt, stage)
        self.assertIn('调整前音', OPEN_RESONANCE_REPAIR_GUIDE)
        self.assertIn('泛音', OPEN_RESONANCE_REPAIR_GUIDE)

    def test_same_pitch_realizations_can_be_swapped_or_use_harmonic(self):
        original = ['名指九徽勾四弦', '散挑七弦', '大指七徽挑七弦']
        alternatives = [
            ['散挑七弦', '名指九徽勾四弦', '大指七徽挑七弦'],
            ['名指九徽勾四弦', '散挑七弦', '泛音大指七徽挑七弦'],
        ]
        def warnings(texts):
            return AUDIT.open_to_stopped_transition_warnings({
                'open_midi': [48, 50, 53, 55, 57, 60, 62],
                'notes': [{'index': i, 'abc': abc, 'jianpu': '2', 'jianzi': text}
                          for i, (abc, text) in enumerate(zip(['D', 'D', 'd'], texts))],
            })
        self.assertEqual(len(warnings(original)), 1)
        for alternative in alternatives:
            self.assertEqual(warnings(alternative), [], alternative)


if __name__ == '__main__':
    unittest.main()
