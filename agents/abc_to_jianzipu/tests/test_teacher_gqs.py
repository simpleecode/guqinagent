from __future__ import annotations

import unittest

from agents.abc_to_jianzipu.teacher_gqs import parse_teacher_gqs, render_teacher_gqs


class TeacherGqsTests(unittest.TestCase):
    def test_round_trip_is_lossless(self) -> None:
        data = {
            "metadata": {"score_title": "测试｜曲", "tuning": {"name": "正调"}},
            "notes": [{
                "index": 0, "jianpu": "5̇", "jianpu_alt": None, "abc": "g2",
                "duration": "八分", "lyric": None, "jianzi": "泛起挑4弦",
                "jianzi_text": "泛起挑4弦",
                "section": {"number": 1, "marker": "<一>"},
            }],
        }
        self.assertEqual(parse_teacher_gqs(render_teacher_gqs(data)), data)

    def test_invalid_schema_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_teacher_gqs("GQS｜unknown\n")

    def test_both_chuo_spellings_round_trip_losslessly(self) -> None:
        data = {
            "metadata": {"score_title": "绰测试", "tuning": {"name": "正调"}},
            "notes": [
                {"index": 0, "jianpu": "1", "jianpu_alt": None, "abc": "C",
                 "duration": "四分", "lyric": None, "jianzi": "绰上五徽",
                 "jianzi_text": "绰上五徽",
                 "section": {"number": 1, "marker": "<一>"}},
                {"index": 1, "jianpu": "2", "jianpu_alt": None, "abc": "D",
                 "duration": "四分", "lyric": None,
                 "jianzi": "绰大指五徽六分挑6弦",
                 "jianzi_text": "绰大指五徽六分挑6弦",
                 "section": {"number": 1, "marker": None}},
            ],
        }
        self.assertEqual(parse_teacher_gqs(render_teacher_gqs(data)), data)


if __name__ == "__main__":
    unittest.main()
