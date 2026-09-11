from __future__ import annotations

import unittest

from scripts.visualize_agent_trajectories import (
    parse_annotation_gqs,
    render_cross_stage_comparison,
    render_final_annotation_comparison,
)


class TrajectoryComparisonTests(unittest.TestCase):
    def test_parses_compact_annotation_rows(self) -> None:
        rows = parse_annotation_gqs(
            'GQS｜annotation-phrase-1.0\n列｜index｜jianpu｜jianzi\n'
            '音｜[3,"5","散挑7弦"]'
        )
        self.assertEqual(rows[3], {"jianpu": "5", "jianzi": "散挑7弦"})

    def test_renders_equal_and_different_rows(self) -> None:
        private = {"teacher_private": {
            "annotation_gqs": (
                'GQS｜annotation-phrase-1.0\n列｜index｜jianpu｜jianzi\n'
                '音｜[3,"5","散挑7弦"]\n音｜[4,"6","散勾6弦"]'
            ),
            "accepted_plan": {"actions": [
                {"source_index": 3, "mode": "open", "string": 7,
                 "right_finger": "挑", "left_finger": None, "attack": True,
                 "techniques": [], "jianzi_text": "散挑7弦"},
                {"source_index": 4, "mode": "open", "string": 6,
                 "right_finger": "挑", "left_finger": None, "attack": True,
                 "techniques": [], "jianzi_text": "散挑6弦"},
            ]},
        }}

        rendered = render_final_annotation_comparison(private)

        self.assertIn("最终版本 vs 标注版本", rendered)
        self.assertIn('class="same"', rendered)
        self.assertIn('class="different"', rendered)

    def test_renders_one_combined_stage_comparison(self) -> None:
        annotation = (
            'GQS｜annotation-phrase-1.0\n列｜index｜jianpu｜jianzi\n'
            '音｜[3,"5","散挑7弦"]\n音｜[4,"6","散勾6弦"]'
        )
        fingering = {"teacher_private": {
            "annotation_gqs": annotation,
            "accepted_plan": {"actions": [
                {"source_index": 3, "jianzi_text": "散挑7弦"},
                {"source_index": 4, "jianzi_text": "散勾6弦"},
            ]},
        }}
        guqinizer = {"teacher_private": {
            "annotation_gqs": annotation,
            "accepted_plan": {"actions": [
                {"source_index": 3, "jianzi_text": "散摘七弦"},
                {"source_index": 4, "jianzi_text": "散勾6弦"},
            ]},
        }}
        samples = [
            {"sample_id": "t-f", "agent_stage": "fingering_agent",
             "provenance": {"source_trajectory_id": "trajectory-a"}},
            {"sample_id": "t-g", "agent_stage": "guqinization",
             "provenance": {"source_trajectory_id": "trajectory-a"}},
        ]

        rendered = render_cross_stage_comparison(
            samples, {"t-f": fingering, "t-g": guqinizer}
        )

        self.assertIn("两阶段最终版本 vs 标注版本｜trajectory-a", rendered)
        self.assertIn("Fingering 最终版", rendered)
        self.assertIn("Guqinizer 最终版", rendered)
        self.assertIn("[散挑7弦]", rendered)
        self.assertIn("[散摘七弦]", rendered)
        self.assertEqual(rendered.count('class="match"'), 3)


if __name__ == "__main__":
    unittest.main()
