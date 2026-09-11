import unittest

from agents.abc_to_jianzipu.repeat_materializer import materialize_repeats


class RepeatMaterializerTests(unittest.TestCase):
    def test_anchor_and_from_marker_flag_existing_copy_without_inserting_rows(self):
        notes = [
            {"index": 0, "jianpu": "1", "jianzi": "〔再作起点〕勾3弦"},
            {"index": 1, "jianpu": "2", "jianzi": "挑4弦"},
            {"index": 2, "jianpu": "|", "jianzi": ""},
            {"index": 3, "jianpu": "1", "jianzi": "从ㄱ再作"},
            {"index": 4, "jianpu": "2", "jianzi": ""},
            {"index": 5, "jianpu": "|", "jianzi": ""},
            {"index": 6, "jianpu": "3", "jianzi": "勾5弦"},
        ]
        materialized, report = materialize_repeats(notes, is_sounding=lambda n: n["jianpu"] != "|")
        self.assertEqual(len(materialized), len(notes))
        self.assertFalse(materialized[0].get("notation_omitted"))
        self.assertTrue(materialized[3].get("notation_omitted"))
        self.assertTrue(materialized[4].get("notation_omitted"))
        self.assertTrue(materialized[5].get("notation_omitted"))
        self.assertEqual(report["copy_markers"][0]["anchor_index"], 0)
        self.assertEqual(report["spans_materialized"], 1)

    def test_unpaired_copy_marker_is_reported_without_guessing(self):
        notes = [{"index": 0, "jianpu": "1", "jianzi": "从ㄱ再作"}]
        materialized, report = materialize_repeats(notes, is_sounding=lambda n: True)
        self.assertEqual(len(materialized), 1)
        self.assertFalse(materialized[0].get("notation_omitted"))
        self.assertEqual(report["unpaired_markers"][0]["marker"], "从ㄱ再作")

    def test_anchor_can_be_reused_by_multiple_from_markers(self):
        notes = [
            {"index": 0, "jianpu": "1", "jianzi": "〔再作起点〕勾3弦"},
            {"index": 1, "jianpu": "2", "jianzi": "挑4弦"},
            {"index": 2, "jianpu": "1", "jianzi": "从ㄱ再作"},
            {"index": 3, "jianpu": "2", "jianzi": ""},
            {"index": 4, "jianpu": "1", "jianzi": "从ㄱ再作"},
            {"index": 5, "jianpu": "2", "jianzi": ""},
            {"index": 6, "jianpu": "3", "jianzi": "勾5弦"},
        ]
        materialized, report = materialize_repeats(notes, is_sounding=lambda n: True)
        self.assertEqual(len(report["copy_markers"]), 2)
        self.assertEqual([item["anchor_index"] for item in report["copy_markers"]], [0, 0])
        self.assertTrue(materialized[2].get("notation_omitted"))
        self.assertTrue(materialized[4].get("notation_omitted"))


if __name__ == "__main__":
    unittest.main()
