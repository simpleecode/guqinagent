import unittest
from train.scripts.eval_two_stage_score import warning_stop_action


class StopActionTests(unittest.TestCase):
    def test_clean_noop(self):
        self.assertEqual(warning_stop_action(False, False, False), 'accept')

    def test_warned_noop_gets_one_reminder(self):
        self.assertEqual(warning_stop_action(True, False, False), 'remind')
        self.assertEqual(warning_stop_action(True, False, True), 'unresolved')

    def test_attempt_does_not_require_zero_warnings(self):
        self.assertEqual(warning_stop_action(True, True, False), 'unresolved')

    def test_budget_exhausted(self):
        self.assertEqual(warning_stop_action(True, False, False, False), 'unresolved')


if __name__ == '__main__':
    unittest.main()
